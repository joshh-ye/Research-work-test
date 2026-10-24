"""Shared COMPARE stage for locked test evaluations (CPU only).  Moved verbatim from test_eval_cache.compare() (numerics unchanged);
used by test_eval_cache.py (cache configs) and predict_ft.py (sequence-reading fine-tuned models / ensembles).
Input: per-tile sufficient stats npz (keys n,sx,sy,sxy,sx2,sy2,sse; tiles x 183; gid) in the test-tile order of the pooled-128 copy
(tiles.csv test rows sorted by split_idx).  Output: <out>/test_comparison.csv with mean Pearson r / r^2 / prediction R^2 (tile-block CI) and
paired deltas "new - ridge[pre_raw:union1888]" and "TL - new" with tile-block, TF-cluster, cell-line-cluster and two-way CIs,
10,000 resamples, seed 20261020 (cmp_lib.block_w / 07_e4_sensitivity.delta_ci); track sets all183 and not_identical152.
The lock check is the caller's responsibility (this module never decides whether a test evaluation is allowed).
CLI: compare_ft.py --stats PATH --out DIR [--n-boot 10000]"""
import sys, argparse, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "analysis/priorityD"))
sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))


def test_gid_order():
    """test-tile order of the pooled-128 copy (= /tmp/yejz1_ft/pooled128/test_gid.npy): tiles.csv test rows sorted by split_idx"""
    import pandas as pd
    tt = pd.read_csv(ROOT / "results_phase2/trunk_cache/tiles.csv")
    return tt[tt.split == "test"].sort_values("split_idx").gid.values


def run_compare(stats_path, out, n_boot=10000):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np, pandas as pd
    out = Path(out)
    sp = importlib.util.spec_from_file_location("e4sens", ROOT / "analysis/priorityD/07_e4_sensitivity.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    from cmp_lib import load_stats, block_w, summarize, metrics
    new = load_stats(stats_path)
    refs = {"ridge_union1888_raw": m.TL_PATHS["ridge_union1888_raw"], "TL": m.TL_PATHS["TL"]}
    S = {k: load_stats(p) for k, p in refs.items()}
    tt = pd.read_csv(ROOT / "results_phase2/trunk_cache/tiles.csv").set_index("gid")
    te_gid = test_gid_order()
    f = Path("/tmp/yejz1_ft/pooled128/test_gid.npy")
    if f.exists():
        assert (np.load(f) == te_gid).all(), "tiles.csv test order != pooled-128 test_gid.npy"
    assert (new["gid"] == te_gid).all() and (S["ridge_union1888_raw"]["gid"] == te_gid).all(), "tile order mismatch with E4 stats"
    tiles = tt.loc[te_gid].reset_index()
    tl_tiles = pd.read_csv(m.EV / "baseline_borzoi_rerun/tiles_test.csv")
    assert (tiles.chrom.values == tl_tiles.chrom.values).all() and (tiles.start.values == tl_tiles.start.values).all()
    for k in ["sy", "sy2"]:        # same targets as E4
        assert np.allclose(new[k], S["ridge_union1888_raw"][k], rtol=1e-5), k
    masks, tf, cl = m.masks_and_labels(); B = n_boot; w = m.block_w(tiles, B)
    rows = []
    for mk, mask in masks.items():
        met = metrics(new, w[:1]); r = np.nanmean(met["pearson_r"][0][mask]); r2 = np.nanmean(met["pearson_r_squared"][0][mask]); pr2 = np.nanmean(met["prediction_R2"][0][mask])
        _, lo, hi = summarize(new, w, mask)
        rows.append(dict(comparison="new (absolute)", track_set=mk, mean_pearson_r=r, tileblock_lo=lo, tileblock_hi=hi, mean_pearson_r_squared=r2, mean_prediction_R2=pr2))
        for nm, a_, b_ in [("new - ridge[pre_raw:union1888]", new, S["ridge_union1888_raw"]), ("TL - new", S["TL"], new)]:
            rows.append(dict(comparison=nm, track_set=mk, **m.delta_ci(a_, b_, w, mask, tf, cl, B)))
    df = pd.DataFrame(rows); df.to_csv(out / "test_comparison.csv", index=False); print(df.round(4).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(); run_compare(a.stats, a.out, a.n_boot)
