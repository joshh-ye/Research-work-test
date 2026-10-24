"""LOCKED test evaluation of a cache-based configuration (ridge or head) on the pooled-128 copy.
Refuses to run unless the config id is a row of research_management/finetuning/TEST_LOCK.csv, and refuses a second
evaluation of the same id (output dir must not contain test_stats.npz) except 'E4-ridge-repro' (public gate config).
Stage predict: refit (ridge: deterministic Gram+eigh on TRAIN tiles; head: load checkpoint) -> per-tile sufficient stats on the
359 test tiles (same keys as the E4 stats: n,sx,sy,sxy,sx2,sy2,sse; shape tiles x 183; gid) -> OUT/test_stats.npz.
Stage compare (CPU only): mean Pearson r / r^2 / prediction R^2 with tile-block CI, and paired delta vs ridge[pre_raw:union1888]
and vs Borzoi-TL with tile-block, TF-cluster, cell-line-cluster and two-way CIs (delta_ci of priorityD/07_e4_sensitivity.py;
10,000 resamples; seed 20261020), for all183 and not_identical152.
Usage: test_eval_cache.py --config-id FT-C61 [--stage both|predict|compare] [--n-boot 10000]"""
import sys, json, csv, argparse, subprocess, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research")
LOCK = ROOT / "research_management/finetuning/TEST_LOCK.csv"
OUTROOT = ROOT / "results_ft/cache_exps/test_eval"
E4_CONFIG = dict(family="ridge", src=["pre"], pre_tf="raw", offsets=[], ctx=["pre"], ytf="raw", backtransform=False, lam_mode="shared", lam=1e-4)

ap = argparse.ArgumentParser(); ap.add_argument("--config-id", required=True); ap.add_argument("--stage", default="both", choices=["both", "predict", "compare"])
ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--max-gb", type=float, default=29.0)
ap.add_argument("--split", default="test", choices=["test", "val"], help="'val' = self-check of the predict path on VALIDATION tiles (no lock needed; no comparison)")
a = ap.parse_args()
out = (OUTROOT / a.config_id) if a.split == "test" else (OUTROOT.parent / "selfcheck_val" / a.config_id); out.mkdir(parents=True, exist_ok=True)


def check_lock(cid):
    rows = list(csv.DictReader(open(LOCK)))
    ok = [r for r in rows if r["config_id"] == cid]
    if not ok: raise SystemExit(f"{cid} is not in TEST_LOCK.csv; refusing to evaluate on test")
    return ok[0]


def load_config(cid):
    if cid == "E4-ridge-repro": return dict(E4_CONFIG)
    p = ROOT / "results_ft/cache_exps/configs" / f"{cid}.json"
    if not p.exists(): raise SystemExit(f"no config json {p}")
    return json.load(open(p))


def predict():
    import numpy as np, torch, time
    from ftheads import make_head, predict_stats, raw_prediction
    from ftlib import (Spec, Pooled, base_stats, accumulate, RidgeEig, final_coefs, eval_stream, cap_gpu, LAMBDAS, N_TGT, G_BINS, DEV)
    lock = check_lock(a.config_id) if a.split == "test" else {}; cfg = load_config(a.config_id)
    if (out / "test_stats.npz").exists() and a.config_id != "E4-ridge-repro":
        raise SystemExit("test_stats.npz already exists for this locked config: at most one test evaluation per config")
    cap_gpu(a.max_gb); t0 = time.time()
    spec = Spec.from_dict(cfg); tr, te = Pooled("train"), Pooled(a.split)
    bmu, bsd = base_stats(tr, spec)
    if cfg["family"] == "ridge":
        acc = accumulate(tr, spec, bmu, bsd, (cfg["ytf"],)); eig = RidgeEig(acc)
        if cfg["lam_mode"] == "shared":
            li = cfg.get("lam_idx", None)
            if li is None or isinstance(li, list): li = int(np.argmin(np.abs(np.log10(LAMBDAS) - np.log10(cfg["lam"]))))
            assert abs(np.log10(LAMBDAS[li]) - np.log10(cfg["lam"])) < 1e-6 if cfg.get("lam") else True
        else: li = np.asarray(cfg["lam_idx"])
        B = final_coefs(eig, cfg["ytf"], LAMBDAS, li); mode = (cfg["ytf"], bool(cfg["backtransform"]))
        sm, ys, (tiles, ytile) = eval_stream(te, spec, bmu, bsd, eig, [mode], LAMBDAS[:1], collect_tiles=True, Bfixed={mode: B})
        st = {k: np.stack([t[k][0] for t in tiles[mode]]) for k in ["sx", "sx2", "sxy", "sse"]}
        st.update({k: np.stack([t[k] for t in ytile]) for k in ["sy", "sy2"]}); st["n"] = np.full(st["sx"].shape, float(G_BINS))
    else:
        ck = cfg["checkpoint"]; head = make_head(cfg["kind"], spec.dim(), cfg.get("hidden", 1024), cfg.get("dropout", 0.2), cfg["loss"] == "poisson").to(DEV)
        head.load_state_dict(torch.load(ck, map_location=DEV)); head.eval()
        ymu = None if cfg.get("ymu") is None else torch.tensor(cfg["ymu"], device=DEV); ysd = None if cfg.get("ysd") is None else torch.tensor(cfg["ysd"], device=DEV)
        st = predict_stats(head, te, spec, bmu, bsd, cfg, ymu, ysd)
    st["gid"] = te.gid
    np.savez(out / "test_stats.npz", **st)
    json.dump(dict(config=cfg, lock=lock, seconds=round(time.time() - t0)), open(out / "run_info.json", "w"), indent=1)
    print("wrote", out / "test_stats.npz", f"{time.time()-t0:.0f}s")
    if a.split == "val":
        from ftheads import r_from_tile_stats
        r = r_from_tile_stats(st); print("selfcheck VAL mean r all183", float(np.nanmean(r)), "recorded", float(np.nanmean(np.load(ROOT / "results_ft/cache_exps/per_track_val_r" / f"{a.config_id}.npz")["r"])))


def compare():
    check_lock(a.config_id)
    from compare_ft import run_compare      # shared compare stage (moved verbatim from here; numerics unchanged)
    run_compare(out / "test_stats.npz", out, a.n_boot)


if a.stage in ("both", "predict"): predict()
if a.split == "test" and a.stage in ("both", "compare"):
    if a.stage == "both": subprocess.check_call([sys.executable, __file__, "--config-id", a.config_id, "--stage", "compare", "--n-boot", str(a.n_boot)])
    else: compare()
