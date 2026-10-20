"""E2/E6 pre-specified validation analysis (variant_scoring/E2_ANALYSIS_PLAN.md). JOINS MODEL SCORES TO LABELS.

Refuses to run without --unblind (a reviewer gates unblinding). Prerequisites (label-only / score-only, already allowed):
  ./torch-venv/bin/python analysis/e2_validation/labels.py              -> results_phase2/e2_validation/labels/
  ./ag-venv/bin/python analysis/e2_validation/scores.py build-ag-cache  -> results_phase2/e2_validation/scores/
Run: ./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scores as S  # noqa: E402
import stats as ST  # noqa: E402

ROOT = Path("/home/yejz1/experimental_research")
OUT = ROOT / "results_phase2/e2_validation"
LABELS = OUT / "labels"
HEAD = S.HEADLINE


# ----------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    if df is None or len(df) == 0:
        return "(empty)\n"
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "NA" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def prepare_covariates(lab: pd.DataFrame, tss: dict | None) -> pd.DataFrame:
    lab = lab.copy()
    if "dist_tss" not in lab and tss is not None:
        lab["dist_tss"] = ST.distance_to_tss(lab["chrom"], lab["pos"], tss)
    lab["neg_log10_dist"] = -np.log10(lab["dist_tss"] + 1.0) if "dist_tss" in lab else np.nan   # higher = closer to a TSS
    lab["log10_dist"] = -lab["neg_log10_dist"]
    return lab


def _subset(lab, model_tab, label_col, score_col="primary", cluster_col="chen_region", drop_excl=False, extra_mask=None):
    """Join one model's score onto the labelled (pos/neg) variants. Returns df with y, s, region (+signed)."""
    d = lab[lab[label_col].isin(["pos", "neg"])].copy()
    if extra_mask is not None:
        d = d[extra_mask.reindex(d.index, fill_value=True) if isinstance(extra_mask, pd.Series) else extra_mask]
    m = model_tab.reindex(d["S6_CCV_input_key"])
    d["s"] = m[score_col].to_numpy()
    d["signed"] = m["signed"].to_numpy() if "signed" in m else np.nan
    d["excl_window"] = m["excl_window"].to_numpy() if "excl_window" in m else False
    d["y"] = (d[label_col] == "pos").astype(int)
    d["cluster"] = d[cluster_col].to_numpy()
    if drop_excl:
        d = d[~d["excl_window"].astype(bool)]
    return d


def p1_row(name, d, B, n_perm, family, desc=""):
    d = d.dropna(subset=["s"])
    r = ST.p1_stratified(d["s"].to_numpy(), d["y"].to_numpy(), d["cluster"].to_numpy(), B=B, n_perm=n_perm)
    return dict(family=family, model=name, description=desc, n_scored=len(d), **r)


def run_analysis(lab: pd.DataFrame, models: dict, out: Path, B: int = ST.N_BOOT, n_perm: int = ST.N_PERM,
                 tss: dict | None = None, eqtl_pairs: pd.DataFrame | None = None, ag_rna: pd.DataFrame | None = None,
                 extra_scores: dict | None = None, partial: bool = False) -> dict:
    """Pure function of the label table + score tables (all in memory). Writes tables and SUMMARY.md to `out`."""
    out = Path(out); (out / "tables").mkdir(parents=True, exist_ok=True)
    lab = prepare_covariates(lab, tss)
    res: dict = {}
    # ------------------------------------------------- power statement (label only)
    prim = lab[lab["mpra_primary"].isin(["pos", "neg"])]
    pw = ST.minimum_detectable_auroc(  # label-only, random scores
        (prim["mpra_primary"] == "pos").astype(int), prim["chen_region"], n_sim=60, B=200)
    cnt = pd.DataFrame([dict(label="mpra_primary", n_pos=int((lab.mpra_primary == "pos").sum()), n_neg=int((lab.mpra_primary == "neg").sum()),
                             n_excluded=int((lab.mpra_primary == "excluded").sum()), **pw)])
    cnt.to_csv(out / "tables/power_statement.tsv", sep="\t", index=False)
    res["power"] = cnt

    # ------------------------------------------------- primary P1 and P2 (Holm over 4 headline models each)
    p1, p2, region_tabs = [], [], {}
    for m in HEAD:
        d = _subset(lab, models[m], "mpra_primary")
        p1.append(p1_row(m, d, B, n_perm, "primary_P1", "max|score| stratified AUROC, region bootstrap, within-region permutation"))
        region_tabs[m] = ST.region_table(d.dropna(subset=["s"])["s"], d.dropna(subset=["s"])["y"], d.dropna(subset=["s"])["cluster"])
        dd = d[(d["y"] == 1)].dropna(subset=["signed"])
        lk_ = lab.set_index("S6_CCV_input_key")
        dd = dd.assign(logfc=lk_.loc[dd["S6_CCV_input_key"], "logFC_alt_vs_ref"].to_numpy(),
                       orient=lk_.loc[dd["S6_CCV_input_key"], "orientation"].to_numpy())
        dd = dd[dd["orient"] == "keep"]           # A1 item 5: allele-swapped ST4 rows (rs34400960, rs36079436) excluded from P2
        r = ST.direction_test(dd["signed"], dd["logfc"], dd["cluster"], B=B)
        p2.append(dict(family="primary_P2", model=m, **r))
    p1 = pd.DataFrame(p1); p2 = pd.DataFrame(p2)
    p1["p_used_for_holm"] = p1["perm_p_greater"]          # one-sided: score higher in MPRA positives
    p1["holm_p"] = ST.holm(p1["p_used_for_holm"])
    p2["p_used_for_holm"] = p2["boot_p_two_sided"]
    p2["holm_p"] = ST.holm(p2["p_used_for_holm"])
    for k, t in region_tabs.items():
        t.to_csv(out / f"tables/P1_per_region_auroc_{k}.tsv", sep="\t")
    res["p1"], res["p2"] = p1, p2

    # ------------------------------------------------- baselines (distance, annotation count)
    base_scores = {"baseline_neg_log10_dist_tss": lab.set_index("S6_CCV_input_key")["neg_log10_dist"],
                   "baseline_chen_annot_count": lab.set_index("S6_CCV_input_key")["chen_annot_count"]}
    base_tabs = {k: pd.DataFrame({"primary": v, "signed": np.nan, "excl_window": False}) for k, v in base_scores.items()}
    base_rows = [p1_row(k, _subset(lab, t, "mpra_primary"), B, n_perm, "baseline_P1", "trivial baseline") for k, t in base_tabs.items()]
    res["baselines"] = pd.DataFrame(base_rows)

    # ------------------------------------------------- secondary / exploratory families
    sec = []
    allm = dict(models); allm.update(extra_scores or {})
    for m, t in allm.items():                                       # mean|score| sensitivity, raw AG, TL-all183, per-cell-line extras
        for col, desc in [("primary", "primary score"), ("mean_abs", "mean |score| over tracks")]:
            if (col == "primary" and m in HEAD) or col not in t:
                continue
            sec.append(p1_row(m, _subset(lab, t, "mpra_primary", col), B, n_perm, "sens_score_variants", desc))
    for m in HEAD:
        t = models[m]
        sec.append(p1_row(m, _subset(lab, t, "mpra_S1"), B, n_perm, "sens_labels", "S1: positives = any ST2 flag>=1"))
        for L in ("HT29", "SW403", "HCEC"):
            sec.append(p1_row(m, _subset(lab, t, f"mpra_S2_{L}"), B, n_perm, "sens_labels", f"S2: per-cell-line label {L}"))
        sec.append(p1_row(m, _subset(lab, t, "mpra_primary", extra_mask=~lab["palindromic"]), B, n_perm, "sens_exclusions", "S3: non-palindromic only"))
        sec.append(p1_row(m, _subset(lab, t, "mpra_primary", drop_excl=True), B, n_perm, "sens_exclusions",
                          "S4: exclude padded/N windows (AG: shifted windows)"))
        sec.append(p1_row(m, _subset(lab, t, "mpra_primary", cluster_col="chen_lead_snp"), B, n_perm, "sens_clustering", "S5: signal (lead SNP) clusters"))
        for lcol, desc in (("eqtl_gtex_v10_colon_any", "GTEx v10 colon eQTL (any gene) vs not [NOT independent of training]"),
                           ("eqtl_cedar_credible", "CEDAR credible-set eQTL vs not")):
            if lcol in lab:
                l2 = lab.assign(_l=np.where(lab[lcol] == 1, "pos", "neg"))
                sec.append(p1_row(m, _subset(l2, t, "_l"), B, n_perm, "eqtl_P1", desc))
    for name, t in base_tabs.items():
        sec.append(p1_row(name, _subset(lab, t, "mpra_primary", cluster_col="chen_lead_snp"), B, n_perm, "sens_clustering", "S5 baseline"))
    sec = pd.DataFrame(sec)
    p2s = []
    lk2 = lab.set_index("S6_CCV_input_key")
    for m, t in allm.items():                      # A1 item 3: P2 repeated for non-headline scores that have a signed companion
        if m in HEAD or "signed" not in t or t["signed"].notna().sum() == 0:
            continue
        d = _subset(lab, t, "mpra_primary"); d = d[(d["y"] == 1)].dropna(subset=["signed"])
        d = d.assign(logfc=lk2.loc[d["S6_CCV_input_key"], "logFC_alt_vs_ref"].to_numpy(), orient=lk2.loc[d["S6_CCV_input_key"], "orientation"].to_numpy())
        d = d[d["orient"] == "keep"]
        p2s.append(dict(family="secondary_P2", model=m, **ST.direction_test(d["signed"], d["logfc"], d["cluster"], B=B)))
    res["secondary_P2"] = pd.DataFrame(p2s)
    # A1 item 7: held-out-chromosome subset (chr1/8/9/22), DESCRIPTIVE only (point estimates, no CI/p)
    ho = []
    hmask = lab["chrom"].astype(str).str.replace("chr", "", regex=False).isin(["1", "8", "9", "22"])
    for m in HEAD + [k for k in allm if k.startswith("borzoi_pretrained_trackmatched")]:
        d = _subset(lab, allm[m], "mpra_primary", extra_mask=hmask).dropna(subset=["s"])
        t_ = ST.region_table(d["s"], d["y"], d["cluster"])
        ho.append(dict(model=m, n_pos=int(d["y"].sum()), n_neg=int((1 - d["y"]).sum()), n_informative_regions=int((t_["w"] > 0).sum()),
                       auroc_stratified=ST._wmean(t_["auc"].to_numpy(), t_["w"].to_numpy()), note="descriptive, no inference"))
    res["heldout_chr_descriptive"] = pd.DataFrame(ho)
    if len(sec):
        sec["bh_q_within_family"] = np.nan
        for fam, idx in sec.groupby("family").groups.items():
            sec.loc[idx, "bh_q_within_family"] = ST.bh(sec.loc[idx, "perm_p_greater"])
        sec["label"] = "EXPLORATORY (secondary; BH within family)"
    res["secondary_P1"] = sec

    # ------------------------------------------------- paired comparisons (models and vs distance baseline)
    pair = []
    dsets = {m: _subset(lab, models[m], "mpra_primary") for m in HEAD}
    dsets.update({k: _subset(lab, t, "mpra_primary") for k, t in base_tabs.items()})
    for k in allm:
        if k.startswith("borzoi_pretrained_trackmatched"):
            dsets[k] = _subset(lab, allm[k], "mpra_primary")
    names = list(dsets)
    for a, b in [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if ((x in HEAD or y in HEAD) and not (x.startswith('borzoi_pretrained_trackmatched') and y.startswith('baseline')))]:
        da, db = dsets[a], dsets[b]
        common = da.index.intersection(db.index)
        da, db = da.loc[common].dropna(subset=["s"]), db.loc[common].dropna(subset=["s"])
        common = da.index.intersection(db.index)
        da, db = da.loc[common], db.loc[common]
        r = ST.paired_delta(da["s"].to_numpy(), db["s"].to_numpy(), da["y"].to_numpy(), da["cluster"].to_numpy(), B=B)
        pair.append(dict(family="paired_delta_vs_distance" if b == "baseline_neg_log10_dist_tss" or a == "baseline_neg_log10_dist_tss" else "paired_delta_models",
                         model_a=a, model_b=b, n=len(common), **r))
    pair = pd.DataFrame(pair)
    if len(pair):
        pair["bh_q_within_family"] = np.nan
        for fam, idx in pair.groupby("family").groups.items():
            pair.loc[idx, "bh_q_within_family"] = ST.bh(pair.loc[idx, "boot_p_two_sided"])
    res["paired"] = pair

    # ------------------------------------------------- conditional logistic regression (strata = region)
    cl_rows = []
    lk = lab.set_index("S6_CCV_input_key")
    for m in HEAD:
        d = _subset(lab, models[m], "mpra_primary").dropna(subset=["s"])
        d = d.assign(log10_dist=lk.loc[d["S6_CCV_input_key"], "log10_dist"].to_numpy(),
                     annot=lk.loc[d["S6_CCV_input_key"], "chen_annot_count"].to_numpy()).dropna(subset=["log10_dist", "annot"])
        zs = ST.zscore(pd.Series(d["s"].to_numpy()).rank(pct=True).to_numpy())     # rank-based z of the score (monotone-invariant like AUROC)
        zd, za = ST.zscore(d["log10_dist"]), ST.zscore(d["annot"])
        y, st = d["y"].to_numpy(), d["cluster"].to_numpy()
        specs = {"score+log10dist": (np.c_[zs, zd], ["score_rankz", "log10_dist_z"], np.c_[zd]),
                 "score+annot_count": (np.c_[zs, za], ["score_rankz", "annot_count_z"], np.c_[za]),
                 "score+log10dist+annot_count": (np.c_[zs, zd, za], ["score_rankz", "log10_dist_z", "annot_count_z"], np.c_[zd, za])}
        for sname, (X, nm, Xred) in specs.items():
            f = ST.conditional_logit(X, y, st, nm)
            lr = ST.clogit_lr_test(X, Xred, y, st)
            for i, n in enumerate(nm):
                cl_rows.append(dict(family="clogit", model=m, spec=sname, term=n, coef=f["coef"][i], se=f["se"][i], odds_ratio=np.exp(f["coef"][i]),
                                    wald_p=f["p"][i], n=f["n"], n_strata=f["n_strata"],
                                    lr_p_score_added=lr["p"] if n == "score_rankz" else np.nan))
    res["clogit"] = pd.DataFrame(cl_rows)

    # ------------------------------------------------- eQTL direction (AG RNA-seq, colon, gene-level; secondary)
    if eqtl_pairs is not None and ag_rna is not None and len(ag_rna):
        g = eqtl_pairs[eqtl_pairs["source"].str.startswith("gtex_v10_")].copy()
        g["tissue"] = g["source"].str.replace("gtex_v10_", "", regex=False)
        g["gid"] = g["gene_id"].astype(str).str.split(".").str[0]
        r = ag_rna.copy(); r["gid"] = r["gene_id"].astype(str).str.split(".").str[0]
        j = g.merge(r, left_on=["S6_CCV_input_key", "gid", "tissue"], right_on=["S6_CCV_input_key", "gid", "gtex_tissue"], how="inner")
        j = j.merge(lab[["S6_CCV_input_key", "chen_region"]], on="S6_CCV_input_key")
        rows = []
        for col in ("quantile_score", "raw_score"):
            dr = ST.direction_test(j[col], j["effect_alt_vs_ref"], j["chen_region"], B=B)
            rows.append(dict(family="eqtl_direction", model=f"alphagenome_rna_colon_{col}", n_pairs=len(j), **dr))
        res["eqtl_direction"] = pd.DataFrame(rows)

    # ------------------------------------------------- Framing rule (sec 7)
    dv = res["paired"][res["paired"]["family"] == "paired_delta_vs_distance"] if len(res["paired"]) else pd.DataFrame()
    verdicts = []
    for _, r in p1.iterrows():
        beats = False
        if len(dv):
            x = dv[((dv.model_a == r.model) & (dv.model_b == "baseline_neg_log10_dist_tss")) | ((dv.model_b == r.model) & (dv.model_a == "baseline_neg_log10_dist_tss"))]
            if len(x):
                x = x.iloc[0]
                beats = bool(x.ci_lo > 0) if x.model_a == r.model else bool(x.ci_hi < 0)
        ok = bool(r.holm_p < 0.05 and r.ci_lo > 0.5 and beats)
        verdicts.append(dict(model=r.model, holm_p_lt_0_05=bool(r.holm_p < 0.05), ci_excludes_0_5=bool(r.ci_lo > 0.5), beats_distance_baseline_ci=beats,
                             framing_B_criteria_met=ok))
    res["framing"] = pd.DataFrame(verdicts)

    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"tables/{k}.tsv", sep="\t", index=False)
    write_summary(res, out, B, n_perm, partial)
    return res


def write_summary(res: dict, out: Path, B: int, n_perm: int, partial: bool) -> None:
    f = res["framing"]
    framing = "B" if f["framing_B_criteria_met"].any() else "A"
    lines = ["# E2/E6 validation summary (auto-generated by analysis/e2_validation/run_e2.py)", "",
             f"Bootstrap resamples B={B}, permutations={n_perm}, seed={ST.SEED}. AUROC = stratified (within-region, weight n_pos*n_neg) Mann-Whitney AUROC "
             "(not Pearson r, r^2 or R^2). Build hg38. All numbers below were computed by the script; none are hand-edited.", ""]
    if partial:
        lines += ["**WARNING: PARTIAL RUN - not every labelled variant was scored for every model; not valid for pre-registered inference.**", ""]
    lines += ["## Power statement (label only)", md_table(res["power"]),
              "## Primary P1 (within-region discrimination; Holm across 4 headline models; one-sided permutation p)", md_table(res["p1"].drop(columns=["family", "description"])),
              "## Primary P2 (direction vs MPRA logFC; Holm across 4 models; bootstrap two-sided p)", md_table(res["p2"].drop(columns=["family"])),
              "## Held-out-chromosome subset chr1/8/9/22 (S6; descriptive, no inference)", md_table(res["heldout_chr_descriptive"]),
              "## Secondary P2 (non-headline scores incl. track-matched Borzoi)", md_table(res["secondary_P2"]),
              "## Trivial baselines (P1)", md_table(res["baselines"].drop(columns=["family", "description"])),
              "## Framing rule (plan sec 7)", md_table(f), f"Result: **Framing {framing}** (B only if >=1 model meets all three criteria).", "",
              "## Paired differences in stratified AUROC (secondary; BH within family)", md_table(res["paired"]),
              "## Conditional logistic regression (strata = region; exploratory)", md_table(res["clogit"]),
              "## Secondary / exploratory P1 variants (BH within family)", md_table(res["secondary_P1"].drop(columns=["label"]) if len(res["secondary_P1"]) else res["secondary_P1"])]
    if "eqtl_direction" in res:
        lines += ["## eQTL direction, AlphaGenome colon RNA-seq gene-level vs GTEx v10 slope (secondary; NOT independent of training)", md_table(res["eqtl_direction"])]
    lines += ["## Limitations", "- MPRA negatives are 'tested, ST2 flag 0'; per-variant effects for non-hits are not public.",
              "- eQTL labels: GTEx colon is not independent of AlphaGenome/Borzoi/Enformer training (RNA-seq of the same tissue); non-significant = not in the significant-pairs file (includes untested variants).",
              "- Larger predicted effects do not imply greater biological accuracy; no causal or clinical claims."]
    (Path(out) / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unblind", action="store_true", help="REQUIRED: acknowledges that real scores are joined to real labels")
    ap.add_argument("--allow-partial", action="store_true", help="allow models scored for only part of the labelled set (marks output PARTIAL)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--boot", type=int, default=ST.N_BOOT)
    ap.add_argument("--perm", type=int, default=ST.N_PERM)
    a = ap.parse_args()
    if not a.unblind:
        sys.exit("REFUSING TO RUN: pass --unblind (reviewer-gated: this joins model scores to real MPRA/eQTL labels).")
    lab = pd.read_csv(LABELS / "variant_labels.tsv", sep="\t", dtype={"chrom": str})
    models = S.load_all()
    keys = set(lab["S6_CCV_input_key"])
    partial = False
    for m in HEAD:
        miss = len(keys - set(models[m].index))
        if miss:
            if not a.allow_partial:
                sys.exit(f"model {m} lacks scores for {miss} labelled variants (run unfinished?); use --allow-partial to proceed")
            partial = True
    extra = {k: models[k] for k in ("alphagenome_raw", "borzoi_tl_all183") if k in models}
    try:                                                                         # A1 item 3: track-matched pretrained Borzoi
        extra["borzoi_pretrained_trackmatched"] = S.track_matched_borzoi()
    except Exception as e:
        print(f"[warn] track-matched comparator skipped: {e}")
    for m in ("borzoi_pretrained", "borzoi_tl", "enformer"):                      # per-cell-line sensitivity (HCT-116 / LoVo / other)
        try:
            cl = S.cellline_primary(m)
            for c in cl.columns:
                extra[f"{m}__{c}"] = pd.DataFrame({"primary": cl[c], "signed": np.nan, "excl_window": False})
        except Exception as e:
            print(f"[warn] per-cell-line sensitivity skipped for {m}: {e}")
    eq = pd.read_csv(LABELS / "eqtl_pairs_aligned.tsv.gz", sep="\t")
    rna = pd.read_csv(S.CACHE / "alphagenome_rna_colon.tsv.gz", sep="\t")
    res = run_analysis(lab, models, Path(a.out), a.boot, a.perm, tss=ST.load_tss(), eqtl_pairs=eq, ag_rna=rna, extra_scores=extra, partial=partial)
    print(f"[run_e2] wrote {a.out} (tables/, SUMMARY.md)")


if __name__ == "__main__":
    main()
