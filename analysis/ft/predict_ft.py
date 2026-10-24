"""Prediction / locked test evaluation for fine-tuned Borzoi runs (sequence-reading) and ensembles of them.  hg38.
Usage (see README_predict.md):
  predict_ft.py --run FT-S05 --split val [--tta rc] [--force]
  predict_ft.py --runs FT-S05,FT-S06 --split val             (per-run val predictions, one file each)
  predict_ft.py --runs FT-S05,FT-S06 --split test --config-id ENS:FT-S05+FT-S06      (LOCKED; see below)
  predict_ft.py --split test --config-id 'BLEND:FT-S05|FT-C61|w=0.4'   (LOCKED fixed-w blend of FT/ENS:A+B with a ridge config; w from the id)
Model reconstruction: results_ft/runs/<RUN>/run_cmd.json argv -> train_ft.parse() -> train_ft.build() (incl. --replicate), then the
trainable state of best_model.pt is loaded and checked KEY-FOR-KEY (checkpoint keys == model requires_grad keys; all remaining model
keys are the frozen pretrained weights re-loaded from the HF cache).
Prediction transform = train_ft.evaluate() exactly: forward strand, unshifted window, no RC, bf16 autocast, central 4096 bins, crop 256 bins/side
(896*4 bins), mean-pool x4 -> 896 x 128 bp, float64 sums.  --tta rc additionally averages with the reverse-complement prediction
(re-reversed along the bin axis); default off; the TTA choice must be selected on VALIDATION (val outputs are written to *_rc files).
val : results_ft/preds/<RUN>/val_pred[_rc].npy fp16 (tiles x 896 x 183), val_gid.npy (pooled-128 val order = tiles.csv val rows sorted by
      split_idx), val_stats[_rc].npz (per-tile n,sx,sy,sxy,sx2,sy2,sse; gid), val_repro[_rc].json (r vs logged best full-val r).
test: LOCKED.  Refuses unless the config id is a row of TEST_LOCK.csv and <ft_test_eval>/<CONFIG>/test_stats.npz does not exist (same rule
      as test_eval_cache.py).  Writes test_stats.npz (same keys) + run_info.json and runs the shared compare stage (compare_ft.py).
Exit codes: 0 ok, 1 refused (test lock; message on stderr), 3 val reproduction failed."""
from __future__ import annotations
import os, sys, csv, json, time, argparse, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
RUNS = ROOT / "results_ft/runs"
PREDS = ROOT / "results_ft/preds"
TEST_OUT = ROOT / "results_ft/ft_test_eval"
LOCK = ROOT / "research_management/finetuning/TEST_LOCK.csv"
TILES_CSV = ROOT / "results_phase2/trunk_cache/tiles.csv"
CROP, POOL, G_BINS, N_TGT = 256, 4, 896, 183
KEYS = ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]
REPRO_TOL = 1e-3


# ---------------------------------------------------------------------------------------------------------------- lock logic
def safe_name(config_id: str) -> str:
    return config_id.replace(":", "_").replace("/", "_")


def ens_config_id(runs):
    return runs[0] if len(runs) == 1 else "ENS:" + "+".join(runs)


def check_test_allowed(config_id: str, lock_path=LOCK, test_root=TEST_OUT) -> Path:
    """Raise SystemExit(2-style message) unless config_id is locked and has no prior test_stats.npz. Returns the output dir (not created)."""
    rows = list(csv.DictReader(open(lock_path)))
    if not any(r["config_id"] == config_id for r in rows):
        raise SystemExit(f"{config_id} is not in TEST_LOCK.csv; refusing to evaluate on test")
    out = Path(test_root) / safe_name(config_id)
    if (out / "test_stats.npz").exists():
        raise SystemExit("test_stats.npz already exists for this locked config: at most one test evaluation per config")
    return out


# ---------------------------------------------------------------------------------------------------------------- tables
def split_gids(split):
    """pooled-128 order of the split (tiles.csv rows of the split sorted by split_idx) + chrom/start arrays"""
    import pandas as pd
    tt = pd.read_csv(TILES_CSV)
    t = tt[tt.split == split].sort_values("split_idx")
    return tt, t.gid.to_numpy(), t


def r_from_stats(st):
    from estats import pearson_from_sums
    g = {k: st[k].sum(0) for k in ["n", "sx", "sy", "sxy", "sx2", "sy2"]}
    return pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])


def check_repro(r_mean, logged, tol=REPRO_TOL):
    """True iff |recomputed - logged| <= tol"""
    return abs(float(r_mean) - float(logged)) <= tol


# ---------------------------------------------------------------------------------------------------------------- models
def load_run_model(run, max_gb=0.0):
    """-> (model.eval() on cuda, args, payload meta).  Strict check of the trainable-state keys."""
    import torch
    import train_ft as T
    rd = RUNS / run
    cmd = json.load(open(rd / "run_cmd.json"))["argv"]
    old = sys.argv
    sys.argv = ["train_ft.py"] + cmd
    try:
        args = T.parse()
    finally:
        sys.argv = old
    assert args.run_id == run, (args.run_id, run)
    repo_id = f"johahi/borzoi-replicate-{args.replicate}"
    model, opt = T.build(args, None, repo_id)
    del opt
    payload = torch.load(rd / "best_model.pt", map_location="cpu", weights_only=False)
    assert payload.get("kind") == "best_model", payload.get("kind")
    want = {n for n, p in model.named_parameters() if p.requires_grad}
    got = set(payload["trainable"].keys())
    assert got == want, f"trainable-state key mismatch: only in ckpt {sorted(got - want)[:3]}, only in model {sorted(want - got)[:3]}"
    missing, unexpected = model.load_state_dict(payload["trainable"], strict=False)
    assert not unexpected and not (set(missing) & want), (unexpected[:3], list(set(missing) & want)[:3])
    model.eval()
    return model, args, payload["meta"]


def _logged_best(run, meta):
    return float(meta["val_mean_pearson_r_full"])


# ---------------------------------------------------------------------------------------------------------------- prediction core
def _rc_codes(c):
    import torch
    return torch.where(c < 4, 3 - c, c).flip(-1)


def iter_tile_predictions(models_args, gids, tiles, tta="none", num_workers=4):
    """yields (i, gid, P (896,183) float32 cuda, Y (896,183) float64 cuda) with P = ensemble mean over models (and TTA)"""
    import torch
    import data_ft as D
    from torch.utils.data import DataLoader
    dl = DataLoader(D.TileJobs(tiles, [(int(g), False, 0) for g in gids]), batch_size=1, num_workers=num_workers)
    with torch.no_grad():
        for i, b in enumerate(dl):
            codes, tgt = b["codes"].cuda(), b["tgt"].cuda()
            acc = None
            for model, args in models_args:
                ac = args.precision == "bf16"
                p = model(codes, autocast=ac)
                if tta == "rc":
                    p = 0.5 * (p + model(_rc_codes(codes), autocast=ac).flip(1))
                acc = p if acc is None else acc + p
            pred = acc / len(models_args)
            P_ = pred[0].double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
            Y_ = tgt[0].double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
            yield i, int(b["gid"][0]), P_, Y_


def tile_stats(P, Y):
    import torch
    return dict(sx=P.sum(0), sy=Y.sum(0), sxy=(P * Y).sum(0), sx2=(P * P).sum(0), sy2=(Y * Y).sum(0), sse=((P - Y) ** 2).sum(0))


def run_prediction(runs, split, tta, save_pred_path=None, tgt_check=None, max_gb=20.0, log=print):
    """returns stats dict (tiles x 183 arrays, n, gid). If save_pred_path: writes fp16 predictions (tiles x 896 x 183) via memmap.
    tgt_check: optional (n,896,183) array of pooled targets (fp32) -> asserts max |loader target - pooled target| and returns it."""
    import numpy as np, torch
    import data_ft as D
    if max_gb:
        torch.cuda.set_per_process_memory_fraction(max_gb / (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    tiles, gids, _ = split_gids(split)
    models, metas = [], []
    for r in runs:
        m, a, meta = load_run_model(r); models.append((m, a)); metas.append(meta)
        log(f"loaded {r}: best epoch {meta['epoch']} step {meta['global_step']} logged full-val r {meta['val_mean_pearson_r_full']:.4f}")
    n = len(gids)
    mm = None
    if save_pred_path is not None:
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(str(save_pred_path) + ".tmp.npy", "w+", np.float16, (n, G_BINS, N_TGT))
    st = {k: np.zeros((n, N_TGT)) for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    tdiff, t0 = 0.0, time.time()
    for i, gid, P, Y in iter_tile_predictions(models, gids, tiles, tta):
        assert gid == int(gids[i]), (gid, gids[i])
        s = tile_stats(P, Y)
        for k, v in s.items():
            st[k][i] = v.cpu().numpy()
        if mm is not None:
            mm[i] = P.float().cpu().numpy().astype(np.float16)
        if tgt_check is not None:
            tdiff = max(tdiff, float(np.abs(Y.cpu().numpy() - tgt_check[i]).max()))
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{n} tiles {time.time() - t0:.0f}s")
    if mm is not None:
        mm.flush(); del mm
        os.replace(str(save_pred_path) + ".tmp.npy", save_pred_path)
    st["n"] = np.full((n, N_TGT), float(G_BINS)); st["gid"] = gids
    return st, tdiff, metas, time.time() - t0


# ---------------------------------------------------------------------------------------------------------------- main
def do_val(run, tta, force, max_gb, log=print):
    import numpy as np
    suf = "" if tta == "none" else f"_{tta}"
    od = PREDS / run
    pred_p = od / f"val_pred{suf}.npy"
    if pred_p.exists() and not force:
        log(f"{pred_p} exists; skipping (use --force)"); return 0
    pooled = None
    try:
        from poolcache import Pooled
        pooled = Pooled("val")
    except Exception as e:                               # pooled copy is node-local; target check only possible when present
        log(f"WARNING: pooled-128 copy unavailable ({e}); loader-vs-pooled target check skipped")
    tgt = np.asarray(pooled.tgt) if pooled is not None else None
    st, tdiff, metas, secs = run_prediction([run], "val", tta, pred_p, tgt, max_gb, log)
    if pooled is not None:
        assert (pooled.gid == st["gid"]).all()
        assert tdiff < 1e-4, f"loader targets differ from pooled-128 val targets: max abs diff {tdiff}"
    np.save(od / "val_gid.npy", st["gid"])
    np.savez(od / f"val_stats{suf}.npz", **st)
    r = r_from_stats(st)
    from ftlib import masks
    mk = masks()
    mean183, mean152 = float(np.nanmean(r)), float(np.nanmean(r[mk["not_identical152"]]))
    logged = _logged_best(run, metas[0])
    ok = check_repro(mean183, logged)
    rep = dict(run=run, tta=tta, split="val", n_tiles=len(st["gid"]), val_mean_pearson_r_all183=mean183, val_mean_pearson_r_152=mean152,
               logged_best_full_val_r_all183=logged, abs_diff=abs(mean183 - logged), tol=REPRO_TOL,
               reproduces=bool(ok) if tta == "none" else None, max_abs_target_diff_vs_pooled=tdiff if pooled is not None else None,
               seconds=round(secs), pred_file=str(pred_p), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (od / f"val_repro{suf}.json").write_text(json.dumps(rep, indent=1))
    log(f"{run} val[{tta}] mean r all183 {mean183:.4f} r152 {mean152:.4f} | logged {logged:.4f} | diff {abs(mean183 - logged):.5f} "
        f"{'OK' if ok else ('FAIL' if tta == 'none' else '(tta: not comparable)')} | target maxdiff {tdiff:.2e} | {secs:.0f}s")
    return 0 if (ok or tta != "none") else 3


def do_test(runs, config_id, tta, max_gb, n_boot, log=print):
    import numpy as np
    out = check_test_allowed(config_id)                  # BEFORE any model / data is touched
    lock = [r for r in csv.DictReader(open(LOCK)) if r["config_id"] == config_id][0]
    out.mkdir(parents=True, exist_ok=True)
    st, _, metas, secs = run_prediction(runs, "test", tta, None, None, max_gb, log)
    np.savez(out / "test_stats.npz", **st)
    json.dump(dict(config_id=config_id, runs=runs, tta=tta, lock=lock, seconds=round(secs),
                   logged_val_r=[m["val_mean_pearson_r_full"] for m in metas]), open(out / "run_info.json", "w"), indent=1)
    log(f"wrote {out / 'test_stats.npz'} ({secs:.0f}s)")
    subprocess.check_call([sys.executable, str(HERE / "compare_ft.py"), "--stats", str(out / "test_stats.npz"), "--out", str(out), "--n-boot", str(n_boot)])
    return 0


# ---------------------------------------------------------------------------------------------------------------- locked blend (FT + ridge)
COMPONENTS = TEST_OUT / "_components"


def parse_blend_id(config_id):
    """'BLEND:<FTpart>|<RIDGEcfg>|w=0.4' -> (ft_runs list, ridge cfg name, w float). FTpart = run id or 'ENS:A+B'. w is FIXED by the lock row id."""
    assert config_id.startswith("BLEND:"), config_id
    parts = config_id[6:].split("|")
    if len(parts) != 3 or not parts[2].startswith("w="):
        raise SystemExit(f"bad blend id {config_id!r}: expected BLEND:<FTpart>|<RIDGEcfg>|w=<float>")
    ft, ridge, w = parts[0], parts[1], float(parts[2][2:])
    if not 0.0 <= w <= 1.0:
        raise SystemExit(f"w={w} outside [0,1]")
    runs = ft[4:].split("+") if ft.startswith("ENS:") else [ft]
    return runs, ridge, w


def do_test_blend(config_id, tta, max_gb, n_boot, log=print, lock_path=LOCK, test_root=TEST_OUT):
    """LOCKED test evaluation of w*FT + (1-w)*ridge with w taken from the (locked) config id; component test predictions are cached under
    <test_root>/_components/ as a by-product (reused by other locked blends sharing a component; each config id is still evaluated once)."""
    import numpy as np
    import blend_val as BV
    out = check_test_allowed(config_id, lock_path, test_root)       # BEFORE anything is touched
    runs, ridge, w = parse_blend_id(config_id)
    lock = [r for r in csv.DictReader(open(lock_path)) if r["config_id"] == config_id][0]
    comp = Path(test_root) / "_components"
    ft_dir = comp / (safe_name(ens_config_id(runs)) + ("" if tta == "none" else f"_{tta}"))
    rd_dir = comp / ("RIDGE_" + safe_name(ridge))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (ft_dir / "test_pred.npy").exists():
        ft_dir.mkdir(parents=True, exist_ok=True)
        st, _, _, _ = run_prediction(runs, "test", tta, ft_dir / "test_pred.npy", None, max_gb, log)
        np.savez(ft_dir / "test_stats.npz", **st); np.save(ft_dir / "test_gid.npy", st["gid"])
    else:
        log(f"reusing cached FT component {ft_dir}")
    if not (rd_dir / "test_pred.npy").exists():
        BV.ridge_split_pred(ridge, "test", rd_dir / "test_pred.npy", max_gb, log)
    else:
        log(f"reusing cached ridge component {rd_dir}")
    from poolcache import Pooled
    te = Pooled("test")
    assert (np.load(ft_dir / "test_gid.npy") == te.gid).all() and (np.load(rd_dir / "test_gid.npy") == te.gid).all(), "component tile order != pooled test order"
    A, B = np.load(ft_dir / "test_pred.npy", mmap_mode="r"), np.load(rd_dir / "test_pred.npy", mmap_mode="r")
    st = BV.blend_tile_stats(A, B, te.tgt, w); st["gid"] = te.gid
    np.savez(out / "test_stats.npz", **st)
    json.dump(dict(config_id=config_id, runs=runs, ridge=ridge, w_fixed_from_lock=w, tta=tta, lock=lock, components=[str(ft_dir), str(rd_dir)],
                   seconds=round(time.time() - t0)), open(out / "run_info.json", "w"), indent=1)
    log(f"wrote {out / 'test_stats.npz'}")
    subprocess.check_call([sys.executable, str(HERE / "compare_ft.py"), "--stats", str(out / "test_stats.npz"), "--out", str(out), "--n-boot", str(n_boot)])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--runs", help="comma-separated run ids (ensemble for test; per-run files for val)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--config-id", default=None)
    ap.add_argument("--tta", default="none", choices=["none", "rc"])
    ap.add_argument("--force", action="store_true"); ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    runs = a.runs.split(",") if a.runs else ([a.run] if a.run else [])
    if a.split == "test" and (a.config_id or "").startswith("BLEND:"):
        return do_test_blend(a.config_id, a.tta, a.max_gb, a.n_boot)
    if not runs:
        raise SystemExit("give --run or --runs")
    if a.split == "test":
        return do_test(runs, a.config_id or ens_config_id(runs), a.tta, a.max_gb, a.n_boot)
    rc = 0
    for r in runs:
        rc = max(rc, do_val(r, a.tta, a.force, a.max_gb))
    return rc


if __name__ == "__main__":
    sys.exit(main())
