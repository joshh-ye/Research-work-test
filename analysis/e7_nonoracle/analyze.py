"""E7 analysis: non-oracle (validation-selected) vs oracle Enformer / AlphaGenome baselines and TL - model delta-r with bootstraps.
Usage: analyze.py enformer|alphagenome [--n-boot 10000]. Inputs: results_phase2/e7_nonoracle/<model>/tile_stats_{val,test}.npz (+ selection_valsel.csv, frozen before test),
results_phase2/exposure_stratified/per_tile_selected_pair_stats.npz (TL per-tile stats), exposure_audit/track_exposure.tsv."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "exposure_stratified"))
from e7lib import *
from estats import *
from select_valsel import candidates

ap = argparse.ArgumentParser(); ap.add_argument("model", choices=["enformer", "alphagenome"]); ap.add_argument("--n-boot", type=int, default=10000)
ap.add_argument("--block", type=int, default=10); ap.add_argument("--seed", type=int, default=20261020); a = ap.parse_args()
M, B, SEED = a.model, a.n_boot, a.seed
OUT = ROOT / f"results_phase2/e7_nonoracle/{M}"; T = 183; ar = np.arange(T)
zv = np.load(OUT / "tile_stats_val.npz"); zt = np.load(OUT / "tile_stats_test.npz", allow_pickle=True)
tl = np.load(ROOT / "results_phase2/exposure_stratified/per_tile_selected_pair_stats.npz")
tiles = pd.read_csv(BR / "tiles_test.csv"); assert (zt["chrom"] == tiles.chrom.values).all() and (zt["start"] == tiles.start.values).all()
assert (zt["tile_zero_signal"] == tl["tile_zero"]).all()
exp = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t"); bench = pd.read_csv(ROOT / "results_benchmark/benchmark_comparison_test.csv")
meta, ident = candidates(M); C = len(meta); assert zt["sTP"].shape == (359, T, C)
sel_df = pd.read_csv(OUT / "selection_valsel.csv")
# val-based quantities
nV = zv["n"].sum(); V = {k: zv[k].sum(0) for k in ["sT", "sT2", "sP", "sP2", "sTP"]}
# the frozen selection must equal a fresh recomputation from validation only
rV = pearson_matrix(nV, V["sT"], V["sT2"], V["sP"], V["sP2"], V["sTP"])
sel = {"valsel": select_argmax(rV), "valsel_excl_ident": select_argmax(rV, ~ident)}
assert (sel["valsel"] == sel_df.valsel_cand_idx.values).all() and (sel["valsel_excl_ident"] == sel_df.valsel_excl_ident_cand_idx.values).all()
nT = zt["n"].sum(); Tt = {k: zt[k].sum(0) for k in ["sT", "sT2", "sP", "sP2", "sTP"]}
rT = pearson_matrix(nT, Tt["sT"], Tt["sT2"], Tt["sP"], Tt["sP2"], Tt["sTP"])
sel["oracle"] = select_argmax(rT); sel["oracle_excl_ident"] = select_argmax(rT, ~ident)
# reproduction gate for the August oracle values
col = {"enformer": "enformer_r", "alphagenome": "alphagenome_r"}[M]
rep = dict(oracle_mean_r_new=float(rT[ar, sel["oracle"]].mean()), oracle_mean_r_benchmark=float(bench[col].mean()),
           max_abs_diff_per_target=float(np.abs(rT[ar, sel["oracle"]] - bench[col].values).max()))
rep["reproduced"] = bool(abs(rep["oracle_mean_r_new"] - rep["oracle_mean_r_benchmark"]) < 5e-4 and rep["max_abs_diff_per_target"] < 2e-3)
print("REPRO", rep)
if not rep["reproduced"]:
    json.dump(rep, open(OUT / "reproduction_FAILED.json", "w"), indent=1); sys.exit("oracle not reproduced")

def pair_stats(mapping):
    s = sel[mapping]; av, cv = fit_calibration(nV, V["sT"][ar], V["sP"][s], V["sP2"][s], V["sTP"][ar, s])
    n = zt["n"][:, None]; sP, sP2, sTP, sT, sT2 = zt["sP"][:, s], zt["sP2"][:, s], zt["sTP"][:, ar, s], zt["sT"], zt["sT2"]
    return dict(n=np.broadcast_to(n, sT.shape), sx=sP, sy=sT, sxy=sTP, sx2=sP2, sy2=sT2, sse=calibrated_sse(n, sT, sT2, sP, sP2, sTP, av, cv))
MAPS = ["oracle", "valsel", "valsel_excl_ident", "oracle_excl_ident"]
PS = {m: pair_stats(m) for m in MAPS}
TL = {k: tl[f"tl_{k}"] for k in ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]}
assert np.allclose(PS["valsel"]["sy"], TL["sy"], atol=1e-6 * abs(TL["sy"]).max())
np.savez_compressed(OUT / "per_tile_selected_pair_stats.npz", **{f"tl_{k}": v for k, v in TL.items()},
                    **{f"{m}_{k}": v for m in MAPS for k, v in PS[m].items()}, tile_zero=tl["tile_zero"], chrom=zt["chrom"], start=zt["start"])
def metrics(S, w):
    g = {k: w @ S[k] for k in ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    return dict(pearson_r=r, pearson_r_squared=pearson_r2(r), prediction_R2=prediction_r2(g["sse"], g["sy"], g["sy2"], g["n"]))
ALL = dict(tl=TL, **PS)
def all_metrics(w): return {k: metrics(S_, w) for k, S_ in ALL.items()}
rng0 = np.random.default_rng(SEED)
Mp = all_metrics(np.ones((1, 359)))
bid = block_ids(tiles.chrom.values, tiles.start.values, a.block)
Mb = all_metrics(block_weights(bid, B, np.random.default_rng(SEED)))
tf = exp.TF.values; cl = exp.cell_line.values
subsets = {"all_183": np.ones(T, bool), "primary_152_non_identical": (exp.Borzoi_class != "identical_experiment").values,
           "identical_experiment_31": (exp.Borzoi_class == "identical_experiment").values}
assert subsets["identical_experiment_31"].sum() == 31
# per-track table
pt = pd.DataFrame(dict(track_idx=ar, bw_filename=bench.bw_filename, TF=tf, cell_line=cl, Borzoi_class=exp.Borzoi_class, tl_pearson_r=Mp["tl"]["pearson_r"][0], tl_prediction_R2=Mp["tl"]["prediction_R2"][0]))
for m in MAPS:
    s = sel[m]; pt[f"{m}_cand_idx"] = s; pt[f"{m}_track"] = meta.track_id.values[s]; pt[f"{m}_description"] = meta.description.values[s]
    pt[f"{m}_pearson_r"] = Mp[m]["pearson_r"][0]; pt[f"{m}_pearson_r_squared"] = Mp[m]["pearson_r_squared"][0]
    pt[f"{m}_prediction_R2_validation_calibrated"] = Mp[m]["prediction_R2"][0]; pt[f"{m}_val_pearson_r"] = rV[ar, s]
pt.to_csv(OUT / "per_track_metrics.csv", index=False)
# per-tile r and r^2 sanity: per-tile Pearson per target for valsel (descriptive)
# ---- deltas
def ci_rows(dpt, db, mask, label_extra):
    row = dict(label_extra); ok = mask & np.isfinite(dpt); dd = np.where(np.isfinite(dpt), dpt, 0.0); one = np.broadcast_to(ok.astype(float), (B, T))
    row.update(n_tracks=int(ok.sum()), n_TF_clusters=len(np.unique(tf[ok])), n_cellline_clusters=len(np.unique(cl[ok])), delta=float(dpt[ok].mean()))
    row["ci_tileblock_lo"], row["ci_tileblock_hi"] = pct_ci(weighted_mean(np.nan_to_num(db), one))
    for sch, lab in [("TF", tf), ("cell", cl)]:
        mult = cluster_multiplicity(lab, B, np.random.default_rng(SEED + (1 if sch == "TF" else 8)), ok) * ok[None]
        okb = mult.sum(1) > 0
        row[f"ci_{sch}cluster_lo"], row[f"ci_{sch}cluster_hi"] = pct_ci(weighted_mean(dd, mult)[okb])
        row[f"ci_twoway_tileblock_x_{sch}_lo"], row[f"ci_twoway_tileblock_x_{sch}_hi"] = pct_ci(weighted_mean(np.nan_to_num(db), mult)[okb])
        cm = pd.Series(dpt[ok]).groupby(lab[ok]).mean().values; p, mode = signflip_p(cm, rng=np.random.default_rng(SEED + 99))
        row[f"signflip_p_{sch}_cluster_means"] = p; row[f"n_{sch}_clusters_positive"] = int((cm > 0).sum())
    return row
rows = []
for m in MAPS:
    for met in ["pearson_r", "pearson_r_squared", "prediction_R2"]:
        dpt = Mp["tl"][met][0] - Mp[m][met][0]; db = Mb["tl"][met] - Mb[m][met]
        for sn, mask in subsets.items():
            r_ = ci_rows(dpt, db, mask, dict(model=M, mapping=m, metric=met + ("_vs_validation_calibrated_baseline" if met == "prediction_R2" else ""), subset=sn))
            ok = mask & np.isfinite(dpt)
            r_["mean_TL"] = float(np.nanmean(Mp["tl"][met][0][ok])); r_["mean_model"] = float(np.nanmean(Mp[m][met][0][ok]))
            r_["TL_wins"] = int((dpt[ok] > 0).sum()); r_["TL_losses"] = int((dpt[ok] < 0).sum()); rows.append(r_)
        # contrast: identical_31 minus primary_152 mean delta (same resamples for both)
        mp_, mi_ = subsets["primary_152_non_identical"], subsets["identical_experiment_31"]; dd = np.nan_to_num(dpt)
        row = dict(model=M, mapping=m, metric=met, subset="contrast_identical31_minus_primary152", n_tracks=T,
                   delta=float(dd[mi_].mean() - dd[mp_].mean()))
        for sch, lab in [("TF", tf), ("cell", cl)]:
            mult = cluster_multiplicity(lab, B, np.random.default_rng(SEED + (1 if sch == "TF" else 8)))
            wi, wp = weighted_mean(dd, mult * mi_[None]), weighted_mean(dd, mult * mp_[None])
            okb = (mult * mi_[None]).sum(1) > 0; okb &= (mult * mp_[None]).sum(1) > 0
            row[f"ci_{sch}cluster_lo"], row[f"ci_{sch}cluster_hi"] = pct_ci((wi - wp)[okb])
        dbn = np.nan_to_num(db); row["ci_tileblock_lo"], row["ci_tileblock_hi"] = pct_ci(dbn[:, mi_].mean(1) - dbn[:, mp_].mean(1))
        rows.append(row)
R = pd.DataFrame(rows); R.to_csv(OUT / "delta_stats.csv", index=False)
# model-only summary
msum = []
for m in MAPS:
    for sn, mask in subsets.items():
        msum.append(dict(model=M, mapping=m, subset=sn, n_tracks=int(mask.sum()), mean_pearson_r=float(Mp[m]["pearson_r"][0][mask].mean()), median_pearson_r=float(np.median(Mp[m]["pearson_r"][0][mask])),
                         mean_pearson_r_squared=float(Mp[m]["pearson_r_squared"][0][mask].mean()), mean_prediction_R2_validation_calibrated=float(np.nanmean(Mp[m]["prediction_R2"][0][mask])),
                         mean_tl_pearson_r=float(Mp["tl"]["pearson_r"][0][mask].mean())))
pd.DataFrame(msum).to_csv(OUT / "model_summary.csv", index=False)
json.dump(dict(reproduction=rep, n_boot=B, seed=SEED, block=a.block, n_candidates=C, n_identical_candidates_removed_any_target=int(ident.any(0).sum()),
               n_targets_valsel_pick_identical=int(ident[ar, sel["valsel"]].sum()), selection_agreement_valsel_vs_oracle=int((sel["valsel"] == sel["oracle"]).sum())),
          open(OUT / "summary.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(msum).query("subset=='all_183'").to_string())
print(R[R.metric == "pearson_r"][["mapping", "subset", "n_tracks", "delta", "ci_TFcluster_lo", "ci_TFcluster_hi", "ci_cellcluster_lo", "ci_cellcluster_hi", "ci_tileblock_lo", "ci_tileblock_hi"]].to_string())
