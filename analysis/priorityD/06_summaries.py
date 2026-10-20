"""Aggregate E4 / E5 results into tables with tile-block bootstrap CIs and write SUMMARY.md + CSVs (re-runnable; sections
are skipped if their inputs do not exist yet). Pearson r / r^2 / prediction R^2 are kept separate.
Models are compared on identical tiles; CI = percentile of 10,000 tile-block (10 tiles, within chromosome) bootstraps with the
track set fixed. No track-level or seed-level p-values are used as primary inference."""
import argparse, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cmp_lib import *
from folds import split_gids, FOLDS

ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=10000); a = ap.parse_args()
import os
R2 = Path(os.environ.get("PD_RESULTS", ROOT / "results_phase2")); EV = ROOT / "reports/qing_deliverables/evaluation_evidence"
masks, exp = exposure_masks(); tt = tile_table().set_index("gid")


def tiles_of(gids): return tt.loc[gids].reset_index()

import importlib.util
_sp = importlib.util.spec_from_file_location("e4sens", Path(__file__).parent / "07_e4_sensitivity.py"); _e4s = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e4s)
_m3, TFL, CLL = _e4s.masks_and_labels()   # all183 / not_identical152 masks, TF and cell-line labels (cluster CIs; 07 helper delta_ci)
PENDING = []


def vs_table(heads, others, gids, B):
    """heads: name -> TL stats; others: name -> comparator stats. Paired TL - other mean-r deltas with tile-block (06 convention), TF-cluster and
    cell-line-cluster CIs (07_e4_sensitivity.delta_ci, same seeds). Pearson r."""
    w = block_w(tiles_of(gids), B); rows = []
    for hn, H in heads.items():
        for on, O in others.items():
            for mk, m in _m3.items():
                r = _e4s.delta_ci(H, O, w, m, TFL, CLL, B)
                rows.append(dict(head=hn, comparator=on, track_set=mk, delta_r_TL_minus_comparator=r["delta"], tileblock_lo=r["tileblock_lo"], tileblock_hi=r["tileblock_hi"],
                                 TFcluster_lo=r["TFcluster_lo"], TFcluster_hi=r["TFcluster_hi"], cellcluster_lo=r["cellcluster_lo"], cellcluster_hi=r["cellcluster_hi"],
                                 n_TF_clusters=r["n_TF_clusters"], n_cellline_clusters=r["n_cellline_clusters"]))
    return pd.DataFrame(rows)


def pending_note():
    q = [str(Path(f).parent.relative_to(R2)) for f in glob.glob(str(R2 / "e5_folds/F*/seed*/metrics.json"))]
    allq = sorted(str(Path(f).relative_to(R2)) for f in glob.glob(str(R2 / "e5_folds/F*/seed*")))
    done = sorted(str(Path(f).parent.relative_to(R2)) for f in glob.glob(str(R2 / "e5_folds/F*/seed*/test_stats.npz")))
    pend = [d for d in allq if d not in done]
    return (f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Heads included: {len(done)} fold x seed runs ({', '.join(done) or 'none'}). "
            f"Runs present but without test_stats at generation time (still pending/running): {', '.join(pend) or 'none'}. Further queued runs not yet started are not listed; re-run 06_summaries.py after the queue ends.")


def cmp_table(models, gids, ref, primary_mask="all183"):
    """models: name -> stats. rows: mean r (+CI) for both masks, paired diff vs ref, mean r^2 and prediction R^2."""
    w = block_w(tiles_of(gids), a.n_boot); rows = []
    for nm, S in models.items():
        for mk, m in masks.items():
            r, lo, hi = summarize(S, w, m, "pearson_r"); r2 = summarize(S, w, m, "pearson_r_squared")[0]; R2_ = summarize(S, w, m, "prediction_R2")[0]
            row = dict(model=nm, track_set=mk, mean_pearson_r=r, r_lo=lo, r_hi=hi, mean_pearson_r_squared=r2, mean_prediction_R2=R2_)
            if ref in models and nm != ref:
                d, dl, dh = paired(S, models[ref], w, m); row.update(delta_r_vs_ref=d, delta_lo=dl, delta_hi=dh)
            rows.append(row)
    return pd.DataFrame(rows)


def tl_orig_qing():
    d = dict(np.load(EV / "tl_per_tile_sufficient_stats_shared128.npz")); return {k: d[k].astype(float) for k in KEYS}


def check_same(models, gids):
    for k, S in models.items():
        if "gid" in S: assert (np.asarray(S["gid"]) == np.asarray(gids)).all(), k


# ------------------------------------------------------------------ E4 (orig split)
tr, va, te = split_gids("orig")
E4 = R2 / "e4_ridge"; models = {}
pv = E4 / "pretrained_valsel_orig"
if (pv / "valsel_test_stats.npz").exists():
    models["pretrained_Borzoi_valsel_nonoracle (REF; R2 val-calibrated)"] = load_stats(pv / "valsel_test_stats.npz")
    models["pretrained_Borzoi_oracle_test_selected (optimistic; R2 uncalibrated)"] = load_stats(pv / "oracle_test_stats.npz")
models["Borzoi-TL original checkpoint (Qing stats)"] = tl_orig_qing()
p_cache = R2 / "e5_seeds/orig_ckpt_cache_eval/test_stats.npz"
if p_cache.exists(): models["Borzoi-TL original ckpt re-evaluated from fp16 cache (gate)"] = load_stats(p_cache)
for d in sorted(E4.glob("orig_*")):
    for f in sorted(d.glob("*_test_stats.npz")):
        sub = f.name.replace("_test_stats.npz", ""); models[f"ridge[{d.name.replace('orig_','')}:{sub}]"] = load_stats(f)
ref = "pretrained_Borzoi_valsel_nonoracle (REF; R2 val-calibrated)"
if len(models) > 2 and ref in models:
    check_same(models, te)
    t = cmp_table(models, te, ref); t.to_csv(E4 / "e4_comparison.csv", index=False)
    # TL vs ridge paired
    w = block_w(tiles_of(te), a.n_boot); tlq = models["Borzoi-TL original checkpoint (Qing stats)"]; rows = []
    for nm, S in models.items():
        if nm.startswith("ridge"):
            for mk, m in masks.items():
                d, lo, hi = paired(tlq, S, w, m); rows.append(dict(ridge_model=nm, track_set=mk, delta_r_TL_minus_ridge=d, lo=lo, hi=hi))
    tr_ = pd.DataFrame(rows); tr_.to_csv(E4 / "e4_TL_minus_ridge.csv", index=False)
    lam = []
    for d in sorted(E4.glob("orig_*/summary.json")): lam += [dict(run=d.parent.name, **x) for x in json.loads(d.read_text())]
    lam = pd.DataFrame(lam)
    L = ["# E4 supervised baseline (SD-008) - auto-generated by 06_summaries.py", "",
         "Split: train = all chromosomes except chr1, chr8 (val) and chr9, chr22 (test); hg38; 183 CRC ChIP targets; shared 896 x 128 bp grid; 359 test tiles.",
         "Ridge: primary features = pretrained 4-replicate Borzoi outputs, union1888 channels (1,882 ChIP-TF + 6 CRC DNase; SD-018), raw (sensitivity variants: 1,882 ChIP-TF channels, 132 CRC tracks, log1p; the 7,611-track variant was dropped with the slim cache, SD-018) or frozen-trunk 1,920-d embedding (linear probe), standardised on train; lambda selected on validation chromosomes only (mean per-track Pearson r); trained on the same tiles as Borzoi-TL.",
         "Prediction R^2 is 1-SSE/SST on raw predictions (ridge/TL raw; pretrained valsel is validation-calibrated). Pearson r is primary.", "",
         "## Mean Pearson r over tracks, with tile-block bootstrap CI and paired difference vs pretrained val-selected", "", md(t), "",
         "## TL minus ridge (paired, tile-block bootstrap)", "", md(tr_), "", "## Selected lambda / validation r", "", md(lam) if len(lam) else "(none)", "",
         "Interpretation rule (set before results): if TL - best pre-specified ridge CI includes 0 or is negative, the 'CRC-specific adaptation' claim is not supported beyond a supervised linear readout of pretrained features (SD-008).",
         "Limitations: tile-block bootstrap treats tracks as fixed; chr9/22 mostly inside Borzoi/Enformer training folds; ridge uses fp16-cached features; the 1,888 primary ridge features include the identical-experiment Borzoi training tracks (>=31 of 183 targets), so ridge and TL are stratified by Borzoi exposure class (primary = not_identical152; identical31 and per-class rows shown; small classes have wide, descriptive CIs)."]
    (E4 / "SUMMARY.md").write_text("\n".join(L)); print("wrote E4 summary")

# ------------------------------------------------------------------ E5 seeds (orig split)
S5 = R2 / "e5_seeds"; seeds = sorted(glob.glob(str(S5 / "seed*/test_stats.npz")))
if seeds and ref in models:
    mm = {f"TL seed {Path(f).parent.name}": load_stats(f) for f in seeds}
    mm.update({ref: models[ref], "Borzoi-TL original checkpoint (Qing stats)": models["Borzoi-TL original checkpoint (Qing stats)"]})
    check_same({k: v for k, v in mm.items() if "gid" in v}, te)
    t = cmp_table(mm, te, ref); t.to_csv(S5 / "e5_seed_comparison.csv", index=False)
    sd = []
    for mk in masks:
        v = t[(t.track_set == mk) & t.model.str.startswith("TL seed")]
        sd.append(dict(track_set=mk, n_seeds=len(v), mean_of_mean_r=v.mean_pearson_r.mean(), sd_across_seeds=v.mean_pearson_r.std(ddof=1) if len(v) > 1 else np.nan,
                       min=v.mean_pearson_r.min(), max=v.mean_pearson_r.max(), mean_delta_vs_pretrained=v.delta_r_vs_ref.mean(), min_delta=v.delta_r_vs_ref.min(), max_delta=v.delta_r_vs_ref.max()))
    sd = pd.DataFrame(sd); sd.to_csv(S5 / "e5_seed_spread.csv", index=False)
    nrc = sorted(glob.glob(str(R2 / "e5_seeds/norc_seed*/test_stats.npz")))
    ablt = ""
    if nrc:
        m2 = {f"TL no-RC {Path(f).parent.name}": load_stats(f) for f in nrc}; m2[ref] = models[ref]
        tn = cmp_table(m2, te, ref); tn.to_csv(S5 / "e5_noRC_ablation.csv", index=False); ablt = "\n## No-RC augmentation ablation\n\n" + md(tn) + "\n"
    seedn = [Path(f).parent.name for f in seeds]
    heads = {f"TL {n}": load_stats(str(S5 / n / "test_stats.npz")) for n in seedn}
    heads["Borzoi-TL original ckpt (Qing stats; gate = E4 TL-ridge)"] = models["Borzoi-TL original checkpoint (Qing stats)"]
    oth = {"ridge[pre_raw:union1888]": models["ridge[pre_raw:union1888]"]}
    if "ridge[probe:trunk1920]" in models: oth["linear probe[trunk1920]"] = models["ridge[probe:trunk1920]"]
    sv = vs_table(heads, oth, te, a.n_boot); sv.to_csv(S5 / "e5_seed_vs_ridge.csv", index=False)
    vstxt = ("\n## Paired TL - ridge[pre_raw:union1888] and TL - linear probe (orig split, Pearson r; delta = TL - comparator)\n\n"
             "Tile-block CI = 10,000 resamples, 10-tile blocks, seed 20261020 (as everywhere); TF-cluster and cell-line-cluster CIs from 07_e4_sensitivity.delta_ci (seeds +1/+8). The probe is a ridge readout on the same frozen-trunk embedding the TL head uses, i.e. the exposure-symmetric comparison. Seed heads use the cache recipe (no RC; DESIGN.md section 8).\n\n" + md(sv) + "\n")
    L = ["# E5 head-seed sensitivity (SD-009) - auto-generated by 06_summaries.py", "",
         "Heads retrained on the frozen-trunk cache with explicit seeds, identical hyperparameters to the original run EXCEPT no RC augmentation (slim cache; see DESIGN.md section 8: shift-emulated, no-RC, fp16). Same split/tiles/grid as the original; Pearson r.", "",
         "## Spread across seeds", "", md(sd), "", "## Per model", "", md(t), vstxt, ablt, "Run status: " + pending_note(), "",
         "Note: seeds vary head initialisation, dropout and data order/augmentation only; the frozen backbone and the chr9/22 backbone-exposure are unchanged, so this does not remove sequence exposure (SD-009 clarification)."]
    (S5 / "SUMMARY.md").write_text("\n".join(L)); print("wrote E5 seeds summary")

# ------------------------------------------------------------------ E5 folds
rows = []
for fold in [f for f in FOLDS if f != "orig"]:
    _, _, te_f = split_gids(fold)
    pvf = E4 / f"pretrained_valsel_{fold}/valsel_test_stats.npz"
    tls = sorted(glob.glob(str(R2 / f"e5_folds/{fold}/seed*/test_stats.npz")))
    rg = E4 / f"{fold}_pre_raw/union1888_test_stats.npz"
    if not (pvf.exists() and tls): continue
    mm = {"pretrained_valsel (REF)": load_stats(pvf)}
    for f in tls: mm[f"TL {Path(f).parent.name}"] = load_stats(f)
    if rg.exists(): mm["ridge pre_raw union1888"] = load_stats(rg)
    check_same(mm, te_f)
    t = cmp_table(mm, te_f, "pretrained_valsel (REF)"); t.insert(0, "fold", fold); t["test_chroms"] = ",".join(FOLDS[fold]["test"]); rows.append(t)
if rows:
    F = pd.concat(rows); F.to_csv(R2 / "e5_folds/e5_fold_comparison.csv", index=False)
    vr = []
    for fold in [f for f in FOLDS if f != "orig"]:
        _, _, te_f = split_gids(fold); rg = E4 / f"{fold}_pre_raw/union1888_test_stats.npz"
        tls = sorted(glob.glob(str(R2 / f"e5_folds/{fold}/seed*/test_stats.npz")))
        if not (rg.exists() and tls): continue
        h = {f"TL {Path(f).parent.name}": load_stats(f) for f in tls}
        v = vs_table(h, {"ridge[pre_raw:union1888]": load_stats(rg)}, te_f, a.n_boot); v.insert(0, "fold", fold); vr.append(v)
    FV = pd.concat(vr) if vr else pd.DataFrame(); FV.to_csv(R2 / "e5_folds/e5_fold_vs_ridge.csv", index=False)
    tl = F[F.model.str.startswith("TL ") & (F.track_set == "all183")].groupby("fold").agg(tl_mean_r=("mean_pearson_r", "mean"), tl_delta=("delta_r_vs_ref", "mean")).reset_index()
    L = ["# E5 chromosome-fold sensitivity (SD-009) - auto-generated", "",
         "Folds hold out different chromosomes (folds.py), validation chromosomes per fold used only for early stopping, lambda and pretrained-track selection. Fold 'orig' (chr9/22) is in e5_seeds. Folds test head robustness only; the backbone saw most of these chromosomes during pretraining (exposure_audit).", "",
         "## Per fold (mean over seeds available)", "", md(tl), "",
         "R^2 labelling: in the Details table, mean_prediction_R2 for 'pretrained_valsel (REF)' is **validation-calibrated R^2** (linear calibration fitted on the fold's validation chromosomes); TL and ridge rows are raw prediction R^2 = 1 - SS_res/SS_tot. Pearson r is the primary metric.", "",
         "## Paired TL - ridge[pre_raw:union1888] per fold x seed (Pearson r; TL - ridge; tile-block / TF-cluster / cell-line-cluster CIs)", "",
         "No linear-probe fold stats exist (probe was run on the orig split only). Fold-specific cluster CIs use the same track clusters as orig.", "", md(FV), "",
         "## Details", "", md(F), "", "Run status: " + pending_note()]
    (R2 / "e5_folds/SUMMARY.md").write_text("\n".join(L)); print("wrote E5 folds summary")
