"""Exposure-stratified re-analysis of Borzoi-TL vs pretrained Borzoi (test chr9+chr22, 359 tiles, 183 tracks, hg38,
shared 896x128bp grid). Reuses Qing's per-tile sufficient statistics; nothing is re-run on GPU.
Usage: python run_exposure_stratified.py [--n-boot 10000]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from estats import *

ROOT = Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research")
EV = ROOT / "reports/qing_deliverables/evaluation_evidence"; BR = EV / "baseline_borzoi_rerun"
OUT = ROOT / "results_phase2/exposure_stratified"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--block", type=int, default=10)
ap.add_argument("--seed", type=int, default=20261020); args = ap.parse_args()
B, SEED, BLOCK = args.n_boot, args.seed, args.block
CLASSES = ["identical_experiment", "same_cellline_same_target", "same_cellline_other_target", "same_target_other_cellline", "unexposed"]
UNEXP_CL = ["GP5d", "LS174T", "LS180", "DLD-1"]

# ---------------------------------------------------------------- load
tl = dict(np.load(EV / "tl_per_tile_sufficient_stats_shared128.npz"))
bs = np.load(BR / "per_tile_stats_test_shared128.npz"); elig = bs["eligible_track_index"]
tiles = pd.read_csv(BR / "tiles_test.csv"); zero = bs["tile_zero_signal"]
assert np.array_equal(zero, tl["tile_zero_signal"]) and (tiles.zero_signal_all_tracks.values == zero).all()
assert (bs["chrom"] == tiles.chrom.values).all() and (bs["start"] == tiles.start.values).all()
exp = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
assert (exp.track_idx.values == np.arange(183)).all()
assert [f"{d}.bw" for d in exp.DCid] == tl["bw_filename"].tolist()
maps = pd.read_csv(BR / "track_mappings.csv"); assert (maps.track_idx.values == np.arange(183)).all()
pt = pd.read_csv(BR / "baseline_vs_tl_per_track.csv"); pt = pt[pt.tile_set == "A_all_359_tiles"]
bench = pd.read_csv(ROOT / "results_benchmark/benchmark_comparison_test.csv")
assert (bench.bw_filename.values == tl["bw_filename"]).all()
T = 183; ar = np.arange(T)
base = {}
for m in ["oracle_test", "val_selected"]:
    sel = np.searchsorted(elig, maps[f"{m}_track_idx"].values); assert (elig[sel] == maps[f"{m}_track_idx"].values).all()
    q = pt[pt.mapping == m].sort_values("track_idx"); a = q.baseline_calib_a.values; c = q.baseline_calib_c.values
    n = bs["n"][:, None]; sT, sT2 = bs["sT"], bs["sT2"]; sP, sP2, sTP = bs["sP"][:, sel], bs["sP2"][:, sel], bs["sTP"][:, ar, sel]
    sse_cal = sT2 - 2 * a * sT - 2 * c * sTP + n * a ** 2 + 2 * a * c * sP + c ** 2 * sP2   # validation-calibrated
    base[m] = dict(n=np.broadcast_to(n, sT.shape), sx=sP, sy=sT, sxy=sTP, sx2=sP2, sy2=sT2, sse=sse_cal)
    assert np.allclose(sT, tl["sy"], atol=1e-6 * abs(tl["sy"]).max())
tlS = dict(n=tl["n"], sx=tl["sx"], sy=tl["sy"], sxy=tl["sxy"], sx2=tl["sx2"], sy2=tl["sy2"], sse=tl["sse"])
np.savez_compressed(OUT / "per_tile_selected_pair_stats.npz", **{f"tl_{k}": v for k, v in tlS.items()},
                    **{f"{m}_{k}": v for m in base for k, v in base[m].items()}, tile_zero=zero)

def metrics(S, w):
    """w: (B,tiles) weights -> dict pearson_r, pearson_r_squared, prediction_R2 as (B,T)."""
    g = {k: w @ S[k] for k in ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    return dict(pearson_r=r, pearson_r_squared=pearson_r2(r), prediction_R2=prediction_r2(g["sse"], g["sy"], g["sy2"], g["n"]))

MODELS = dict(tl=tlS, oracle=base["oracle_test"], valsel=base["val_selected"])
def all_metrics(w): return {k: metrics(S, w) for k, S in MODELS.items()}
METRICS = ["pearson_r", "pearson_r_squared", "prediction_R2"]
MAPS = {"oracle": "oracle_test", "valsel": "val_selected"}

# ---------------------------------------------------------------- reproduction of recorded numbers (stop if not)
w_all = np.ones((1, 359)); M0 = all_metrics(w_all)
rep = dict(TL_r=float(M0["tl"]["pearson_r"].mean()), oracle_r=float(M0["oracle"]["pearson_r"].mean()), valsel_r=float(M0["valsel"]["pearson_r"].mean()))
d_or = M0["tl"]["pearson_r"][0] - M0["oracle"]["pearson_r"][0]
bt = np.random.default_rng(0).integers(0, T, (B, T)); bd = d_or[bt].mean(1); rep["delta_oracle"] = float(d_or.mean()); rep["delta_oracle_track_boot_ci"] = pct_ci(bd)
rep["TL_R2_pred"] = float(M0["tl"]["prediction_R2"].mean()); rep["TL_r2_sq"] = float(M0["tl"]["pearson_r_squared"].mean())
ok = (abs(rep["TL_r"] - .682) < 5e-4 and abs(rep["oracle_r"] - .664) < 5e-4 and abs(rep["valsel_r"] - .661) < 5e-4
      and abs(rep["delta_oracle_track_boot_ci"][0] - .009) < 1e-3 and abs(rep["delta_oracle_track_boot_ci"][1] - .026) < 1e-3
      and abs(rep["delta_oracle"] - .018) < 5e-4)
rep["reproduced"] = bool(ok); print("REPRODUCTION", rep)
if not ok:
    json.dump(rep, open(OUT / "reproduction_FAILED.json", "w"), indent=1); sys.exit("recorded numbers NOT reproduced")

# ---------------------------------------------------------------- subsets / clusters
tf = exp.TF.values; cl = exp.cell_line.values
subsets = {"all_183": np.ones(T, bool), "primary_152_non_identical": (exp.Borzoi_class != "identical_experiment").values,
           "identical_experiment_31": (exp.Borzoi_class == "identical_experiment").values}
for c in CLASSES: subsets[f"class:{c}"] = (exp.Borzoi_class == c).values
subsets["unexposed_celllines_GP5d_LS174T_LS180_DLD1"] = np.isin(cl, UNEXP_CL)
assert subsets["identical_experiment_31"].sum() == 31 and subsets["unexposed_celllines_GP5d_LS174T_LS180_DLD1"].sum() == 17

bid = block_ids(tiles.chrom.values, tiles.start.values, BLOCK)
def analyse(keep, tag):
    rng = np.random.default_rng(SEED)
    wpt = keep.astype(float)[None]; Mp = all_metrics(wpt)
    wb = block_weights(bid, B, rng) * keep[None]
    Mb = all_metrics(wb)                                  # tile-block bootstrap, (B,T)
    rows = []
    for sname, mask in subsets.items():
        is_unexp = sname.startswith("unexposed_cell")
        draws = {}
        for sch, lab in [("TF", tf), ("cell", cl)]:
            draws[sch] = cluster_multiplicity(lab, B, np.random.default_rng(SEED + (1 if sch == "TF" else 8)), mask)
        for mp, mk in MAPS.items():
            for met in METRICS:
                if met == "prediction_R2" and mp not in ("oracle", "valsel"): continue
                dpt = Mp["tl"][met][0] - Mp[mp][met][0]; db = Mb["tl"][met] - Mb[mp][met]
                ok_t = mask & np.isfinite(dpt)
                row = dict(tile_set=tag, mapping=MAPS[mp], metric=met, pretrained_metric_note="validation_calibrated_R2" if met == "prediction_R2" else "",
                           subset=sname, n_tracks=int(ok_t.sum()), n_TF_clusters=len(np.unique(tf[ok_t])), n_cellline_clusters=len(np.unique(cl[ok_t])),
                           mean_TL=float(np.nanmean(Mp["tl"][met][0][ok_t])), mean_pretrained=float(np.nanmean(Mp[mp][met][0][ok_t])), delta_TL_minus_pretrained=float(dpt[ok_t].mean()))
                dd = np.where(np.isfinite(dpt), dpt, 0.0)
                one = np.broadcast_to(ok_t.astype(float), (B, T))
                row["ci_tracks_only_lo"], row["ci_tracks_only_hi"] = pct_ci(weighted_mean(dd, rng_track(B, ok_t)))
                row["ci_tileblock_lo"], row["ci_tileblock_hi"] = pct_ci(weighted_mean(np.nan_to_num(db), one))
                for sch in ["TF", "cell"]:
                    mult = draws[sch] * ok_t[None]
                    ok_b = mult.sum(1) > 0
                    row[f"ci_{sch}cluster_lo"], row[f"ci_{sch}cluster_hi"] = pct_ci(weighted_mean(dd, mult)[ok_b])
                    row[f"ci_twoway_tileblock_x_{sch}_lo"], row[f"ci_twoway_tileblock_x_{sch}_hi"] = pct_ci(weighted_mean(np.nan_to_num(db), mult)[ok_b])
                    labs = (tf if sch == "TF" else cl)[ok_t]; cm = pd.Series(dpt[ok_t]).groupby(labs).mean().values
                    p, mode = signflip_p(cm, rng=np.random.default_rng(SEED + 99))
                    row[f"signflip_p_{sch}_cluster_means"] = p; row[f"signflip_mode_{sch}"] = mode; row[f"mean_of_{sch}_cluster_means"] = float(cm.mean())
                    row[f"n_clusters_{sch}_positive"] = int((cm > 0).sum())
                rows.append(row)
    df = pd.DataFrame(rows)
    return df, Mp

_rt = {}
def rng_track(B_, mask):
    k = mask.sum(); r = np.random.default_rng(SEED + 5)
    out = np.zeros((B_, T)); idx = np.where(mask)[0]; d = r.integers(0, k, (B_, k))
    for j in range(k): out[:, idx[j]] = (d == j).sum(1)
    return out

res = []; per_track = None
for tag, keep in [("all_359_tiles", np.ones(359, bool)), ("zero_signal_51_excluded_308_tiles", ~zero)]:
    df, Mp = analyse(keep, tag); res.append(df)
    pt_df = pd.DataFrame(dict(track_idx=ar, bw_filename=tl["bw_filename"], TF=tf, cell_line=cl, Borzoi_class=exp.Borzoi_class.values,
                              Enformer_class=exp.Enformer_class.values, tile_set=tag))
    for k in ["tl", "oracle", "valsel"]:
        for met in METRICS:
            if met == "prediction_R2" and k != "tl":
                pt_df[f"{k}_prediction_R2_validation_calibrated"] = Mp[k][met][0]
            else:
                pt_df[f"{k}_{met}"] = Mp[k][met][0]
    pt_df["delta_r_oracle"] = pt_df.tl_pearson_r - pt_df.oracle_pearson_r; pt_df["delta_r_valsel"] = pt_df.tl_pearson_r - pt_df.valsel_pearson_r
    pt_df["target_variance_per_128bp_bin"] = ((tl["sy2"] * keep[:, None]).sum(0) - (tl["sy"] * keep[:, None]).sum(0) ** 2 / (tl["n"] * keep[:, None]).sum(0)) / (tl["n"] * keep[:, None]).sum(0)
    per_track = pd.concat([per_track, pt_df]) if per_track is not None else pt_df
R = pd.concat(res); R.to_csv(OUT / "exposure_stratified_stats.csv", index=False)
per_track.to_csv(OUT / "per_track_metrics_by_exposure.csv", index=False)

# ---------------------------------------------------------------- unexposed cell lines: descriptive per cluster
pa = per_track[per_track.tile_set == "all_359_tiles"]
u = pa[pa.cell_line.isin(UNEXP_CL)]
ucl = u.groupby("cell_line").agg(n_tracks=("track_idx", "size"), n_TFs=("TF", "nunique"), mean_TL_pearson_r=("tl_pearson_r", "mean"),
        mean_oracle_pearson_r=("oracle_pearson_r", "mean"), mean_valsel_pearson_r=("valsel_pearson_r", "mean"),
        mean_delta_r_oracle=("delta_r_oracle", "mean"), mean_delta_r_valsel=("delta_r_valsel", "mean"),
        Borzoi_classes=("Borzoi_class", lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().items()))).reset_index()
ucl["caveat"] = "descriptive only; 4 clusters; TL head trained on these tracks (train chromosomes) -> tests pretrained exposure, not TL generalisation"
ucl.to_csv(OUT / "unexposed_cellline_per_cluster.csv", index=False)

# ---------------------------------------------------------------- signal variance sensitivity
sens = []
for tag, g in per_track.groupby("tile_set"):
    for col in ["tl_pearson_r", "oracle_pearson_r", "valsel_pearson_r", "delta_r_oracle", "delta_r_valsel"]:
        rho, p = stats.spearmanr(g.target_variance_per_128bp_bin, g[col], nan_policy="omit")
        sens.append(dict(tile_set=tag, y=col, x="target_variance_per_128bp_bin", spearman_rho=rho, p=p, n=int(g[col].notna().sum())))
    for vq, sub in [("variance_tertile_low", 0), ("variance_tertile_mid", 1), ("variance_tertile_high", 2)]:
        q = pd.qcut(g.target_variance_per_128bp_bin, 3, labels=False)
        s = g[q == sub]; sens.append(dict(tile_set=tag, y="mean delta_r_oracle | mean delta_r_valsel | mean TL r", x=vq, spearman_rho=np.nan, p=np.nan, n=len(s),
              mean_delta_r_oracle=s.delta_r_oracle.mean(), mean_delta_r_valsel=s.delta_r_valsel.mean(), mean_tl_r=s.tl_pearson_r.mean()))
pd.DataFrame(sens).to_csv(OUT / "signal_variance_sensitivity.csv", index=False)

# ---------------------------------------------------------------- TL vs Enformer / AlphaGenome (oracle per-track r, benchmark csv), by their own class
ben = []
for comp, col, ccol in [("enformer_oracle", "enformer_r", "Enformer_class"), ("alphagenome_oracle", "alphagenome_r", "AlphaGenome_class")]:
    d = bench.borzoi_tl_r.values - bench[col].values
    for lv in ["ALL"] + sorted(exp[ccol].unique()):
        m = np.ones(T, bool) if lv == "ALL" else (exp[ccol] == lv).values
        bm = weighted_mean(d, cluster_multiplicity(tf, B, np.random.default_rng(SEED + 3), m) * m[None])
        ben.append(dict(comparator=comp, metric="pearson_r", exposure_class_source=ccol, class_=lv, n_tracks=int(m.sum()), n_TF_clusters=len(np.unique(tf[m])),
                        mean_TL=bench.borzoi_tl_r[m].mean(), mean_comparator=bench[col][m].mean(), delta=d[m].mean(),
                        ci_TFcluster_lo=pct_ci(bm)[0], ci_TFcluster_hi=pct_ci(bm)[1]))
pd.DataFrame(ben).to_csv(OUT / "tl_vs_enformer_alphagenome_by_class.csv", index=False)

summ = dict(n_boot=B, seed=SEED, tile_block_size_tiles=BLOCK, n_tile_blocks=int(len(np.unique(bid))), reproduction=rep,
            ci="percentile 2.5/97.5", tile_resampling="contiguous non-overlapping blocks of tiles within chromosome (~%.1f Mb), chr9 and chr22 blocks pooled" % (BLOCK * .524288),
            n_cell_line_clusters=int(len(np.unique(cl))), n_TF_clusters=int(len(np.unique(tf))))
json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
print("done")
