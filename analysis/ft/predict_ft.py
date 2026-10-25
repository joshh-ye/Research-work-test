"""Prediction / locked test evaluation for fine-tuned Borzoi runs (sequence-reading) and ensembles of them.  hg38.
Usage (see README_predict.md):
  predict_ft.py --run FT-S05 --split val [--tta rc] [--force]
  predict_ft.py --runs FT-S05,FT-S06 --split val             (per-run val predictions, one file each)
  predict_ft.py --split test --config-id 'ENS:FT-S05+FT-S06#tta=rc' [--runs FT-S05,FT-S06]   (LOCKED; ids end in #tta=none|rc; --runs/--tta must match the id)
  predict_ft.py --split test --config-id 'BLEND:ENS:A+B|FT-C61|w=0.7#tta=rc'   (LOCKED fixed-w blend; w from the id)
  predict_ft.py --split val --config-id '<same id>'   (self-check of the exact test code path on VALIDATION; no lock; results_ft/selfcheck_val/<id>/)
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
SELFCHECK = ROOT / "results_ft/selfcheck_val"
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


def check_lock_tta(lock, tta):
    """the lock row's notes must declare the TTA choice ('tta=none' or 'tta=rc'), so --tta cannot differ from what was locked"""
    if f"tta={tta}" not in lock.get("notes", ""):
        raise SystemExit(f"{lock['config_id']}: lock notes do not declare tta={tta}; refusing")


# ---------------------------------------------------------------------------------------------------------------- config ids (bind lock -> models, tta)
def parse_id(config_id):
    """FT id : '<FTpart>#tta=<none|rc>'                 FTpart = run id or 'ENS:A+B+...'
    blend id: 'BLEND:<FTpart>|<RIDGEcfg>|w=<float>#tta=<none|rc>'   (w is FIXED by the id; never chosen on test)
    -> dict(kind 'ft'|'blend', runs [...] (order as written), ridge, w, tta, id)"""
    body, sep, t = config_id.rpartition("#tta=")
    if not sep or t not in ("none", "rc") or not body:
        raise SystemExit(f"bad config id {config_id!r}: must end with '#tta=none' or '#tta=rc'")
    kind, ridge, w = "ft", None, None
    if body.startswith("BLEND:"):
        kind = "blend"
        parts = body[6:].split("|")
        if len(parts) != 3 or not parts[2].startswith("w="):
            raise SystemExit(f"bad blend id {config_id!r}: expected BLEND:<FTpart>|<RIDGEcfg>|w=<float>#tta=<none|rc>")
        body, ridge = parts[0], parts[1]
        try:
            w = float(parts[2][2:])
        except ValueError:
            raise SystemExit(f"bad w in {config_id!r}")
        if not 0.0 <= w <= 1.0:
            raise SystemExit(f"w={w} outside [0,1]")
    runs = body[4:].split("+") if body.startswith("ENS:") else [body]
    if not all(runs) or len(set(runs)) != len(runs):
        raise SystemExit(f"bad / duplicated member list in {config_id!r}")
    return dict(kind=kind, runs=runs, ridge=ridge, w=w, tta=t, id=config_id)


def parse_blend_id(config_id):
    p = parse_id(config_id)
    if p["kind"] != "blend":
        raise SystemExit(f"{config_id!r} is not a BLEND id")
    return p["runs"], p["ridge"], p["w"]


def bind_checks(config_id, runs_arg, tta_arg):
    """lock <-> models/tta binding; raises SystemExit BEFORE anything is loaded. runs_arg: list or None; tta_arg: 'none'|'rc'|None. Returns parsed id."""
    p = parse_id(config_id)
    if tta_arg is not None and tta_arg != p["tta"]:
        raise SystemExit(f"--tta {tta_arg} differs from the tta in the config id ({p['tta']}); refusing")
    if runs_arg is not None and list(runs_arg) != p["runs"]:
        raise SystemExit(f"--runs {list(runs_arg)} != members of the config id {p['runs']} (same order required); refusing")
    return p


def check_lock_committed(lock_path=LOCK):
    """refuse unless TEST_LOCK.csv is tracked and has no uncommitted changes vs HEAD; returns git commit hash"""
    def g(*a):
        return subprocess.run(["git", "-C", str(ROOT)] + list(a), capture_output=True, text=True)
    rel = str(Path(lock_path).resolve().relative_to(ROOT.resolve()))
    if g("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit("TEST_LOCK.csv is not tracked by git; refusing (commit the lock first)")
    if g("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit("TEST_LOCK.csv has uncommitted changes vs HEAD; refusing (commit the lock first)")
    return g("rev-parse", "HEAD").stdout.strip()


def _lock_row(config_id, lock_path):
    return [r for r in csv.DictReader(open(lock_path)) if r["config_id"] == config_id][0]


def _guard_test(config_id, runs_arg, tta_arg, lock_path, test_root, require_committed):
    """all refusals, in order; nothing is created/loaded before they pass"""
    p = bind_checks(config_id, runs_arg, tta_arg)
    out = check_test_allowed(config_id, lock_path, test_root)
    lock = _lock_row(config_id, lock_path)
    check_lock_tta(lock, p["tta"])
    commit = check_lock_committed(lock_path) if require_committed else None
    return p, out, lock, commit


def _git_commit():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------------------------------------------- stats producers (shared by test and val self-check)
def ft_stats(p, split, max_gb, log=print):
    st, _, metas, secs = run_prediction(p["runs"], split, p["tta"], None, None, max_gb, log)
    return st, dict(logged_val_r=[m["val_mean_pearson_r_full"] for m in metas], seconds=round(secs))


def blend_stats(p, split, comp_root, max_gb, log=print):
    """w*FT + (1-w)*ridge on `split`; components cached under comp_root (a by-product of the calling evaluation only)"""
    import numpy as np
    import blend_val as BV
    from poolcache import Pooled
    t0 = time.time()
    runs, ridge, w, tta = p["runs"], p["ridge"], p["w"], p["tta"]
    ft_dir = Path(comp_root) / (safe_name(ens_config_id(runs)) + ("" if tta == "none" else f"_{tta}"))
    rd_dir = Path(comp_root) / ("RIDGE_" + safe_name(ridge))
    pf, pr = ft_dir / f"{split}_pred.npy", rd_dir / f"{split}_pred.npy"
    if not pf.exists():
        ft_dir.mkdir(parents=True, exist_ok=True)
        st, _, _, _ = run_prediction(runs, split, tta, pf, None, max_gb, log)
        np.savez(ft_dir / f"{split}_stats.npz", **st); np.save(ft_dir / f"{split}_gid.npy", st["gid"])
    else:
        log(f"reusing cached FT component {ft_dir}")
    if not pr.exists():
        BV.ridge_split_pred(ridge, split, pr, max_gb, log)
    else:
        log(f"reusing cached ridge component {rd_dir}")
    te = Pooled(split)
    assert (np.load(ft_dir / f"{split}_gid.npy") == te.gid).all() and (np.load(rd_dir / f"{split}_gid.npy") == te.gid).all(), "component tile order != pooled order"
    st = BV.blend_tile_stats(np.load(pf, mmap_mode="r"), np.load(pr, mmap_mode="r"), te.tgt, w); st["gid"] = te.gid
    return st, dict(components=[str(ft_dir), str(rd_dir)], seconds=round(time.time() - t0))


def _finish_test(out, st, info, n_boot, log):
    import numpy as np
    np.savez(out / "test_stats.npz", **st)
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    log(f"wrote {out / 'test_stats.npz'}")
    subprocess.check_call([sys.executable, str(HERE / "compare_ft.py"), "--stats", str(out / "test_stats.npz"), "--out", str(out), "--n-boot", str(n_boot)])
    return 0


def do_test(config_id, runs_arg, tta_arg, max_gb, n_boot, log=print, lock_path=LOCK, test_root=TEST_OUT, require_committed=None):
    """LOCKED test evaluation of an FT run / ensemble id ('...#tta=..')"""
    rc_ = (Path(lock_path) == LOCK) if require_committed is None else require_committed
    p, out, lock, commit = _guard_test(config_id, runs_arg, tta_arg, lock_path, test_root, rc_)
    if p["kind"] != "ft":
        raise SystemExit("use a BLEND id with the blend path")
    out.mkdir(parents=True, exist_ok=True)
    st, info = ft_stats(p, "test", max_gb, log)
    info.update(config_id=config_id, runs=p["runs"], tta=p["tta"], lock=lock, git_commit=_git_commit(), lock_committed_unchanged_at_head=bool(commit))
    return _finish_test(out, st, info, n_boot, log)


def do_test_blend(config_id, runs_arg, tta_arg, max_gb, n_boot, log=print, lock_path=LOCK, test_root=TEST_OUT, require_committed=None):
    """LOCKED test evaluation of w*FT + (1-w)*ridge, w from the id; component test predictions cached under <test_root>/_components/ as a by-product"""
    rc_ = (Path(lock_path) == LOCK) if require_committed is None else require_committed
    p, out, lock, commit = _guard_test(config_id, runs_arg, tta_arg, lock_path, test_root, rc_)
    if p["kind"] != "blend":
        raise SystemExit("not a BLEND id")
    out.mkdir(parents=True, exist_ok=True)
    st, info = blend_stats(p, "test", Path(test_root) / "_components", max_gb, log)
    info.update(config_id=config_id, runs=p["runs"], ridge=p["ridge"], w_fixed_from_lock=p["w"], tta=p["tta"], lock=lock, git_commit=_git_commit(),
                lock_committed_unchanged_at_head=bool(commit))
    return _finish_test(out, st, info, n_boot, log)


def do_selfcheck_val(config_id, runs_arg, tta_arg, max_gb, log=print, root=SELFCHECK):
    """SAME code path as the locked test evaluation (ft_stats / blend_stats), on VALIDATION tiles; no lock; writes under results_ft/selfcheck_val/<id>/"""
    import numpy as np
    from ftlib import masks
    p = bind_checks(config_id, runs_arg, tta_arg)
    out = Path(root) / safe_name(config_id); out.mkdir(parents=True, exist_ok=True)
    st, info = ft_stats(p, "val", max_gb, log) if p["kind"] == "ft" else blend_stats(p, "val", Path(root) / "_components", max_gb, log)
    np.savez(out / "val_stats.npz", **st)
    r = r_from_stats(st); mk = masks()
    res = dict(config_id=config_id, split="val", val_mean_pearson_r_all183=float(np.nanmean(r)), val_mean_pearson_r_152=float(np.nanmean(r[mk["not_identical152"]])),
               n_tiles=len(st["gid"]), time=time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: v for k, v in info.items() if k != "logged_val_r"})
    json.dump(res, open(out / "selfcheck.json", "w"), indent=1)
    log(f"SELFCHECK VAL {config_id}: r183 {res['val_mean_pearson_r_all183']:.4f} r152 {res['val_mean_pearson_r_152']:.4f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--runs", help="comma-separated run ids (must equal the members of the config id for test / id self-check; per-run files for plain val)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--config-id", default=None, help="test: id with '#tta=..' suffix (or BLEND:...); val + id: self-check of the test code path on validation (no lock)")
    ap.add_argument("--tta", default=None, choices=["none", "rc"], help="must equal the tta in the id if both given; default none for plain val")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    runs = a.runs.split(",") if a.runs else ([a.run] if a.run else None)
    if a.split == "test":
        cid = a.config_id
        if cid is None:
            if not runs:
                raise SystemExit("give --config-id (or --run/--runs and --tta)")
            cid = f"{ens_config_id(runs)}#tta={a.tta or 'none'}"
        fn = do_test_blend if cid.startswith("BLEND:") else do_test
        return fn(cid, runs, a.tta, a.max_gb, a.n_boot)
    if a.config_id:
        return do_selfcheck_val(a.config_id, runs, a.tta, a.max_gb)
    if not runs:
        raise SystemExit("give --run or --runs")
    rc = 0
    for r in runs:
        rc = max(rc, do_val(r, a.tta or "none", a.force, a.max_gb))
    return rc


if __name__ == "__main__":
    sys.exit(main())
