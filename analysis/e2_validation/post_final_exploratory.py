"""POST-HOC EXPLORATORY (RV-007) analyses on the E2 final run. NOT pre-registered; nothing here changes
results_phase2/e2_validation/final/ (read-only record). Outputs -> results_phase2/e2_validation/post_final_exploratory/.

Tasks (RV-007 minor findings 3, 5, 7):
  clogit  : region-bootstrap CI of the conditional-logit score_rankz coefficient (score + log10dist + annot_count spec), 4 headline models.
  weights : P1 region-weight concentration, equal-weight-region AUROC with region bootstrap, leave-region-out stratified AUROC.
  eqtl    : AlphaGenome colon RNA-seq vs GTEx v10 slope direction: join checks and restricted concordance.

Usage (cwd = repo root; <=4 threads; CPU only):
  ./ag-venv/bin/python    analysis/e2_validation/post_final_exploratory.py prep-ag      # pyarrow needed: gene TSS table + raw join check
  ./torch-venv/bin/python analysis/e2_validation/post_final_exploratory.py run [--tasks weights clogit eqtl]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labels as L  # noqa: E402
import scores as S  # noqa: E402
import stats as ST  # noqa: E402

ROOT = Path("/home/yejz1/experimental_research")
E2 = ROOT / "results_phase2/e2_validation"
OUT = E2 / "post_final_exploratory"
TAG = "post-hoc exploratory (RV-007)"
SEED = ST.SEED   # 20261020
HEAD = S.HEADLINE


def _w(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.insert(0, "label", TAG)
    df.to_csv(OUT / name, sep="\t", index=False)


# ================================================================================ prep-ag (ag-venv)
def prep_ag() -> None:
    """(1) gene TSS table from GENCODE v46 (hg38, 1-based TSS; gene_id without version); (2) raw-source join check on known pairs."""
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.read_feather(S.ROOT / "alphagenome_variant_effects/input/annotations/gencode.v46.annotation.gtf.gz.feather",
                        columns=["Chromosome", "Feature", "Start", "End", "Strand", "gene_id", "gene_name"])
    g = g[g["Feature"] == "gene"]
    tss = np.where(g["Strand"].astype(str) == "+", g["Start"] + 1, g["End"])   # pyranges: Start 0-based, End exclusive
    t = pd.DataFrame({"gene_id": g["gene_id"].astype(str).str.split(".").str[0], "gene_name": g["gene_name"].astype(str),
                      "chrom": g["Chromosome"].astype(str).str.replace("^chr", "", regex=True), "tss_1based": tss,
                      "strand": g["Strand"].astype(str)}).drop_duplicates("gene_id")
    t.to_csv(OUT / "gene_tss_gencode_v46_hg38.tsv", sep="\t", index=False)
    print(f"[prep-ag] {len(t)} genes -> gene_tss_gencode_v46_hg38.tsv")

    # ---- raw join check: choose strongest pairs + named CRC-locus genes; compare raw GTEx parquet and raw AG parquet with the aligned/cached tables
    pairs = pd.read_csv(E2 / "labels/eqtl_pairs_aligned.tsv.gz", sep="\t")
    pairs = pairs[pairs["source"].str.startswith("gtex_v10_")].copy()
    pairs["tissue"] = pairs["source"].str.replace("gtex_v10_", "", regex=False)
    pairs["gid"] = pairs["gene_id"].astype(str).str.split(".").str[0]
    pairs["z"] = (pairs["effect_alt_vs_ref"] / pairs["se"]).abs()
    rna = pd.read_csv(E2 / "scores/alphagenome_rna_colon.tsv.gz", sep="\t")
    sym = rna.drop_duplicates("gene_id").set_index("gene_id")["gene_symbol"]
    pairs["symbol"] = pairs["gid"].map(sym)
    pick = [pairs[pairs.tissue == "Colon_Transverse"].nlargest(4, "z"), pairs[pairs.tissue == "Colon_Sigmoid"].nlargest(2, "z")]
    for gname in ("POU5F1B", "GREM1", "SMAD7", "BMP4", "CDX2", "MYC", "TCF7L2", "CCND2"):
        x = pairs[pairs.symbol == gname]
        if len(x):
            pick.append(x.nlargest(1, "z"))
    pick = pd.concat(pick).drop_duplicates(["S6_CCV_input_key", "gid", "tissue"]).reset_index(drop=True)
    rows = []
    for _, r in pick.iterrows():
        c, p, ref, alt = r["S6_CCV_input_key"].split(":")
        raw = pd.read_parquet(L.GTEX_DIR / f"{r['tissue']}.v10.eQTLs.signif_pairs.parquet")
        raw = raw[raw["gene_id"].str.split(".").str[0] == r["gid"]]
        vid = raw["variant_id"]
        hit = raw[vid.isin([f"chr{c}_{p}_{ref}_{alt}_b38", f"chr{c}_{p}_{alt}_{ref}_b38"])]
        ag = pd.read_parquet(S.AG_OUT / "variant_all_scores_per_track" / f"chr{c}.parquet",
                             filters=[("input_variant", "==", r["S6_CCV_input_key"]), ("gene_id", "==", r["gid"])],
                             columns=["input_variant", "modality", "scorer_kind", "gene_id", "gtex_tissue", "raw_score", "quantile_score"])
        ag = ag[(ag.modality == "RNA_SEQ") & (ag.scorer_kind == "diff") & (ag.gtex_tissue == r["tissue"])]
        cache = rna[(rna.S6_CCV_input_key == r["S6_CCV_input_key"]) & (rna.gene_id.str.split(".").str[0] == r["gid"]) & (rna.gtex_tissue == r["tissue"])]
        h = hit.iloc[0] if len(hit) else None
        swapped_src = (h is not None) and h["variant_id"].split("_")[2] != ref
        exp_eff = None if h is None else float(h["slope"]) * (-1 if swapped_src else 1)
        rows.append(dict(key=r["S6_CCV_input_key"], gene_id=r["gid"], symbol=r["symbol"], tissue=r["tissue"],
                         raw_gtex_variant_id=None if h is None else h["variant_id"], raw_gtex_slope=None if h is None else float(h["slope"]),
                         raw_gtex_pval=None if h is None else float(h["pval_nominal"]),
                         source_allele_order_swapped=swapped_src, expected_effect_alt_vs_ref=exp_eff,
                         aligned_effect_alt_vs_ref=float(r["effect_alt_vs_ref"]),
                         effect_match=bool(exp_eff is not None and np.isclose(exp_eff, r["effect_alt_vs_ref"], rtol=1e-4)),
                         n_raw_ag_rows=len(ag), raw_ag_raw_score=float(ag["raw_score"].mean()) if len(ag) else np.nan,
                         cache_ag_raw_score=float(cache["raw_score"].iloc[0]) if len(cache) else np.nan,
                         ag_match=bool(len(ag) == 1 and len(cache) == 1 and np.isclose(ag["raw_score"].iloc[0], cache["raw_score"].iloc[0], rtol=1e-4, atol=1e-8)),
                         sign_ag_vs_eqtl=("concordant" if np.sign(cache["raw_score"].iloc[0]) == np.sign(r["effect_alt_vs_ref"]) else "discordant") if len(cache) else "NA"))
    out = pd.DataFrame(rows)
    _w(out, "eqtl_join_check_raw.tsv")
    print(out.to_string())


# ================================================================================ shared setup (torch-venv)
def setup():
    import run_e2 as RE
    lab = pd.read_csv(E2 / "labels/variant_labels.tsv", sep="\t", dtype={"chrom": str})
    models = S.load_all(models=HEAD, finalised_only=True)
    lab = RE.prepare_covariates(lab, ST.load_tss())
    base = {"baseline_neg_log10_dist_tss": lab.set_index("S6_CCV_input_key")["neg_log10_dist"],
            "baseline_chen_annot_count": lab.set_index("S6_CCV_input_key")["chen_annot_count"]}
    tabs = {m: models[m] for m in HEAD}
    tabs.update({k: pd.DataFrame({"primary": v, "signed": np.nan, "excl_window": False}) for k, v in base.items()})
    return RE, lab, tabs


# ================================================================================ task 5: weights / equal-weight / leave-region-out
def task_weights(RE, lab, tabs) -> None:
    rts = {}
    for m, t in tabs.items():
        d = RE._subset(lab, t, "mpra_primary").dropna(subset=["s"])
        rts[m] = ST.region_table(d["s"], d["y"], d["cluster"])
    ref = rts[HEAD[0]]
    for m, t in rts.items():
        assert (t["w"].to_numpy() == ref["w"].to_numpy()).all(), "weights differ between models"
    info = ref[ref["w"] > 0].sort_values("w", ascending=False)
    wtot = info["w"].sum()
    reg = info[["n_pos", "n_neg", "w"]].copy()
    reg["share_of_weight"] = reg["w"] / wtot
    reg["cum_share"] = reg["share_of_weight"].cumsum()
    reg["rank"] = np.arange(1, len(reg) + 1)
    for m, t in rts.items():
        reg[f"auc_{m}"] = t.loc[reg.index, "auc"]
    reg = reg.reset_index().rename(columns={"cluster": "region"})
    _w(reg, "weights_per_region.tsv")
    conc = pd.DataFrame([dict(k=k, regions=";".join(reg["region"][:k]), n_pos_in_topk=int(reg["n_pos"][:k].sum()),
                              weight_share=float(reg["share_of_weight"][:k].sum())) for k in (1, 3, 5)])
    conc.insert(0, "n_informative_regions", len(reg)); conc.insert(1, "total_weight", int(wtot))
    conc["effective_n_regions_1_over_sum_share2"] = 1.0 / float((reg["share_of_weight"] ** 2).sum())
    _w(conc, "weights_concentration.tsv")

    # equal-weight region mean AUROC, region bootstrap (resample all regions; uninformative weight 0)
    rows, lo_rows = [], []
    top3 = list(reg["region"][:3])
    for m, t in rts.items():
        tt = t.copy()
        tt["w_eq"] = (tt["w"] > 0).astype(float)
        eq = ST._wmean(tt["auc"].to_numpy(), tt["w_eq"].to_numpy())
        strat = ST._wmean(tt["auc"].to_numpy(), tt["w"].to_numpy())
        cnt = ST.boot_counts(len(tt), ST.N_BOOT, SEED)
        te = tt.assign(w=tt["w_eq"])
        bs = ST.boot_stratified(te, cnt)
        lo, hi = ST.percentile_ci(bs)
        bst = ST.boot_stratified(tt, cnt)
        slo, shi = ST.percentile_ci(bst)
        rows.append(dict(model=m, n_informative_regions=int((tt["w"] > 0).sum()), auroc_equal_weight_regions=eq, eq_ci_lo=lo, eq_ci_hi=hi,
                         auroc_stratified_prereg=strat, strat_ci_lo_recomputed=slo, strat_ci_hi_recomputed=shi,
                         frac_regions_auc_gt_0_5=float((tt.loc[tt["w"] > 0, "auc"] > 0.5).mean()),
                         median_region_auc=float(tt.loc[tt["w"] > 0, "auc"].median()), n_boot=ST.N_BOOT, seed=SEED))
        # leave-region-out
        def strat_without(drop):
            u = tt.drop(index=drop)
            return ST._wmean(u["auc"].to_numpy(), u["w"].to_numpy())
        for i, r in enumerate(top3, 1):
            lo_rows.append(dict(model=m, scenario=f"drop_top{i}_alone ({r})", n_regions_left=int((tt["w"] > 0).sum()) - 1, auroc_stratified=strat_without([r])))
        lo_rows.append(dict(model=m, scenario=f"drop_top3_jointly ({';'.join(top3)})", n_regions_left=int((tt["w"] > 0).sum()) - 3,
                            auroc_stratified=strat_without(top3)))
        allloo = np.array([strat_without([r]) for r in tt.index[tt["w"] > 0]])
        lo_rows.append(dict(model=m, scenario="all_58_single_region_LOO_min", n_regions_left=int((tt["w"] > 0).sum()) - 1, auroc_stratified=allloo.min()))
        lo_rows.append(dict(model=m, scenario="all_58_single_region_LOO_max", n_regions_left=int((tt["w"] > 0).sum()) - 1, auroc_stratified=allloo.max()))
    _w(pd.DataFrame(rows), "equal_weight_auroc.tsv")
    lo_df = pd.DataFrame(lo_rows)
    _w(lo_df, "leave_region_out_auroc.tsv")
    print(conc.to_string()); print(pd.DataFrame(rows).round(4).to_string()); print(lo_df.round(4).to_string())
    # consistency with the pre-registered table
    p1 = pd.read_csv(E2 / "final/tables/p1.tsv", sep="\t").set_index("model")
    for r in rows:
        if r["model"] in p1.index:
            assert abs(p1.loc[r["model"], "auroc_stratified"] - r["auroc_stratified_prereg"]) < 1e-9, r["model"]
            assert abs(p1.loc[r["model"], "ci_lo"] - r["strat_ci_lo_recomputed"]) < 1e-9, r["model"]


# ================================================================================ task 3: clogit region bootstrap
def _wnll(beta, groups):
    v = 0.0
    for Xs, ys, k, c in groups:
        eta = Xs @ beta
        v -= c * (eta[ys == 1].sum() - ST._log_ek(eta, k))
    return v


def _fit(groups, b0):
    r = optimize.minimize(_wnll, b0, args=(groups,), method="BFGS", options=dict(gtol=1e-4))
    # status 2 = "precision loss" of the line search with the numerical gradient at an (already) flat optimum; accepted when the
    # final central-difference gradient is small
    g = np.array([(_wnll(r.x + e, groups) - _wnll(r.x - e, groups)) / 2e-5 for e in np.eye(len(b0)) * 1e-5])
    return r.x, bool(np.isfinite(r.fun) and np.abs(g).max() < 1e-2)


def _boot_worker(args):
    model, X, y, st, counts, b0 = args
    strata = [(X[st == s], y[st == s], int(y[st == s].sum())) for s in pd.unique(st)]
    strata = [g for g in strata if 0 < g[2] < len(g[1])]
    out = np.full((len(counts), len(b0)), np.nan); ok = np.zeros(len(counts), bool)
    for i, c in enumerate(counts):
        groups = [(Xs, ys, k, ci) for (Xs, ys, k), ci in zip(strata, c) if ci > 0]
        out[i], ok[i] = _fit(groups, b0)
    return out, ok


def task_clogit(RE, lab, tabs, B: int = 2000, nproc: int = 4) -> None:
    import multiprocessing as mp
    lk = lab.set_index("S6_CCV_input_key")
    jobs, obs, meta = [], {}, {}
    rng = np.random.default_rng(SEED)
    cnt = None
    for m in HEAD:
        d = RE._subset(lab, tabs[m], "mpra_primary").dropna(subset=["s"])
        d = d.assign(log10_dist=lk.loc[d["S6_CCV_input_key"], "log10_dist"].to_numpy(),
                     annot=lk.loc[d["S6_CCV_input_key"], "chen_annot_count"].to_numpy()).dropna(subset=["log10_dist", "annot"])
        zs = ST.zscore(pd.Series(d["s"].to_numpy()).rank(pct=True).to_numpy())
        X = np.c_[zs, ST.zscore(d["log10_dist"]), ST.zscore(d["annot"])]
        y, st = d["y"].to_numpy(), d["cluster"].to_numpy()
        f = ST.conditional_logit(X, y, st, ["score_rankz", "log10_dist_z", "annot_count_z"])
        lr = ST.clogit_lr_test(X, np.c_[X[:, 1], X[:, 2]], y, st)
        obs[m] = (f, lr)
        informative = [s for s in pd.unique(st) if 0 < y[st == s].sum() < (st == s).sum()]
        if cnt is None:      # resample the informative regions (the only ones that contribute to the conditional likelihood); same resamples for all models
            n_inf = len(informative)
            cnt = rng.multinomial(n_inf, np.full(n_inf, 1.0 / n_inf), size=B)
            inf_order = informative
        assert set(informative) == set(inf_order)
        # arrange counts in this model's strata order
        order = [s for s in pd.unique(st)]
        full = np.zeros((B, len(order)))
        pos = {s: j for j, s in enumerate(order)}
        for j, s in enumerate(inf_order):
            full[:, pos[s]] = cnt[:, j]
        meta[m] = (X, y, st, order, full, f["coef"])
    rows = []
    for m in HEAD:
        X, y, st, order, full, b0 = meta[m]
        strata_ids = pd.unique(st)
        # _boot_worker iterates pd.unique(st) filtered to informative; map counts accordingly
        inf_mask = [0 < y[st == s].sum() < (st == s).sum() for s in strata_ids]
        cm = full[:, np.flatnonzero(inf_mask)]
        chunks = np.array_split(cm, nproc)
        with mp.get_context("fork").Pool(nproc) as pool:
            res = pool.map(_boot_worker, [(m, X, y, st, c, b0) for c in chunks])
        coefs = np.vstack([r[0] for r in res]); ok = np.concatenate([r[1] for r in res])
        s = coefs[:, 0]
        good = ok & np.isfinite(s)
        lo, hi = np.quantile(s[good], [0.025, 0.975])
        f, lr = obs[m]
        rows.append(dict(model=m, spec="score+log10dist+annot_count", term="score_rankz", n=f["n"], n_regions=f["n_strata"],
                         coef_obs=f["coef"][0], se_model_based=f["se"][0], wald_p=f["p"][0], lr_p_score_added=lr["p"],
                         boot_ci_lo=lo, boot_ci_hi=hi, boot_se=float(np.std(s[good], ddof=1)),
                         or_obs=float(np.exp(f["coef"][0])), or_ci_lo=float(np.exp(lo)), or_ci_hi=float(np.exp(hi)),
                         boot_frac_coef_gt0=float((s[good] > 0).mean()), n_boot=B, n_boot_converged=int(good.sum()), seed=SEED,
                         note="LR/Wald p treat the variants within a region as independent despite LD; CI resamples regions (cluster bootstrap), covariates z-scored once on the full sample"))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    pre = pd.read_csv(E2 / "final/tables/clogit.tsv", sep="\t")
    pre = pre[(pre.spec == "score+log10dist+annot_count") & (pre.term == "score_rankz")].set_index("model")
    for r in rows:   # reproduction of the pre-registered point estimate/LR p
        assert abs(pre.loc[r["model"], "coef"] - r["coef_obs"]) < 1e-4, r["model"]
        assert abs(np.log10(pre.loc[r["model"], "lr_p_score_added"]) - np.log10(r["lr_p_score_added"])) < 0.05, r["model"]
    _w(df, "clogit_region_bootstrap.tsv")


# ================================================================================ task 7: eQTL direction
def _binom_ci(k: int, n: int) -> tuple[float, float]:
    ci = sps.binomtest(int(k), int(n), 0.5).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def _row(name, j, B, note=""):
    """Direction concordance of AG RNA (quantile & raw sign) vs GTEx slope sign for the pair subset j."""
    out = dict(subset=name, n_pairs=len(j), n_unique_variants=int(j["S6_CCV_input_key"].nunique()), n_unique_genes=int(j["gid"].nunique()))
    for col in ("quantile_score", "raw_score"):
        if len(j) == 0:
            out.update({f"{col}_n": 0})
            continue
        dr = ST.direction_test(j[col], j["effect_alt_vs_ref"], j["chen_region"], B=B)
        b_lo, b_hi = _binom_ci(dr["n_concordant"], dr["n"]) if dr["n"] else (np.nan, np.nan)
        out.update({f"{col}_n": dr["n"], f"{col}_k": dr["n_concordant"], f"{col}_frac": dr["frac"],
                    f"{col}_binom_ci_lo": b_lo, f"{col}_binom_ci_hi": b_hi,
                    f"{col}_regionboot_ci_lo": dr["ci_lo"], f"{col}_regionboot_ci_hi": dr["ci_hi"], f"{col}_binom_p": dr["binom_p"]})
    out["note"] = note
    return out


def task_eqtl(RE, lab, tabs, B: int = 2000) -> None:
    eq = pd.read_csv(E2 / "labels/eqtl_pairs_aligned.tsv.gz", sep="\t")
    rna = pd.read_csv(S.CACHE / "alphagenome_rna_colon.tsv.gz", sep="\t")
    g = eq[eq["source"].str.startswith("gtex_v10_")].copy()
    g["tissue"] = g["source"].str.replace("gtex_v10_", "", regex=False)
    g["gid"] = g["gene_id"].astype(str).str.split(".").str[0]
    r = rna.copy(); r["gid"] = r["gene_id"].astype(str).str.split(".").str[0]
    # --- join diagnostics
    diag = dict(n_gtex_pairs_aligned=len(g), n_pairs_variants_unique=int(g.S6_CCV_input_key.nunique()),
                n_dup_key_gene_tissue_in_pairs=int(g.duplicated(["S6_CCV_input_key", "gid", "tissue"]).sum()),
                n_dup_in_ag_rna=int(r.duplicated(["S6_CCV_input_key", "gid", "gtex_tissue"]).sum()), n_swapped_pairs=int(g["swapped"].sum()))
    j = g.merge(r, left_on=["S6_CCV_input_key", "gid", "tissue"], right_on=["S6_CCV_input_key", "gid", "gtex_tissue"], how="inner", validate="m:1")
    diag["n_joined"] = len(j)
    diag["n_pairs_unjoined"] = len(g) - len(j)
    # which pairs fail to join: gene absent from AG scored window vs variant missing
    un = g.merge(r[["S6_CCV_input_key", "gid", "gtex_tissue"]], left_on=["S6_CCV_input_key", "gid", "tissue"], right_on=["S6_CCV_input_key", "gid", "gtex_tissue"], how="left", indicator=True)
    un = un[un["_merge"] == "left_only"]
    diag["n_unjoined_gene_not_in_AG_gene_set"] = int((~un["gid"].isin(set(r["gid"]))).sum())
    diag["n_unjoined_gene_known_but_not_scored_for_variant"] = int(un["gid"].isin(set(r["gid"])).sum())
    j = j.merge(lab[["S6_CCV_input_key", "chen_region", "chrom", "pos", "ref", "alt"]], on="S6_CCV_input_key", how="left")
    # reproduce the pre-registered row
    pre = pd.read_csv(E2 / "final/tables/eqtl_direction.tsv", sep="\t").set_index("model")
    for col in ("quantile_score", "raw_score"):
        dr = ST.direction_test(j[col], j["effect_alt_vs_ref"], j["chen_region"], B=200)
        p = pre.loc[f"alphagenome_rna_colon_{col}"]
        assert dr["n"] == p["n"] and dr["n_concordant"] == p["n_concordant"], (col, dr, p.to_dict())
        diag[f"reproduced_{col}_n_k"] = f"{dr['n']}/{dr['n_concordant']}"
    # attach nominal p, gene TSS distance
    pv = []
    for t in L.GTEX_TISSUES:
        d = L._read_gtex(t)
        k, s = L.harmonise(d["variant_id"], set(lab["S6_CCV_input_key"]))
        d = d.assign(S6_CCV_input_key=k, tissue=t)
        d = d[k != ""]
        d["gid"] = d["gene_id"].astype(str).str.split(".").str[0]
        pv.append(d[["S6_CCV_input_key", "gid", "tissue", "pval_nominal", "slope"]])
    pv = pd.concat(pv).drop_duplicates(["S6_CCV_input_key", "gid", "tissue"])
    j = j.merge(pv[["S6_CCV_input_key", "gid", "tissue", "pval_nominal", "slope"]], on=["S6_CCV_input_key", "gid", "tissue"], how="left")
    diag["n_pval_missing_after_merge"] = int(j["pval_nominal"].isna().sum())
    diag["slope_vs_aligned_effect_sign_flips_n"] = int(((np.sign(j["slope"]) != np.sign(j["effect_alt_vs_ref"])) & j["slope"].notna()).sum())
    tss = pd.read_csv(OUT / "gene_tss_gencode_v46_hg38.tsv", sep="\t", dtype={"chrom": str}).set_index("gene_id")
    j["gene_chrom"] = j["gid"].map(tss["chrom"]); j["gene_tss"] = j["gid"].map(tss["tss_1based"])
    diag["n_gene_on_other_chrom_than_variant"] = int((j["gene_chrom"].astype(str) != j["chrom"].astype(str)).sum())
    j["dist_to_gene_tss"] = (j["pos"] - j["gene_tss"]).abs()
    diag["joined_dist_to_tss_max_bp"] = float(j["dist_to_gene_tss"].max())
    uu = un[un["gid"].isin(set(r["gid"]))].merge(lab[["S6_CCV_input_key", "pos"]], on="S6_CCV_input_key")
    uu["d"] = (uu["pos"] - uu["gid"].map(tss["tss_1based"])).abs()
    diag["unjoined_known_gene_dist_to_tss_min_bp"] = float(uu["d"].min())
    diag["unjoined_known_gene_dist_to_tss_median_bp"] = float(uu["d"].median())
    diag["n_pairs_dist_ge_500kb_joined"] = int((j["dist_to_gene_tss"] >= 500_000).sum())
    pd.DataFrame([diag]).T.reset_index().rename(columns={"index": "check", 0: "value"}).pipe(_w, "eqtl_join_diagnostics.tsv")
    print(diag)

    absraw = j["raw_score"].abs(); absq = j["quantile_score"].abs()
    rows = [_row("all_pairs (reproduces final)", j, B)]
    for tis in L.GTEX_TISSUES:
        rows.append(_row(f"tissue_{tis}", j[j.tissue == tis], B, "same variant-gene pairs partly shared with the other tissue"))
    # (a) top decile of |AG score|
    for lab_, ser in (("raw", absraw), ("quantile", absq)):
        for q in (0.9, 0.95, 0.99):
            thr = ser.quantile(q)
            rows.append(_row(f"(a) top {int(round((1-q)*100))}% by |AG {lab_}| (pooled over pairs; thr {thr:.4g})", j[ser >= thr], B,
                             "quantile score is saturated/tie-heavy" if lab_ == "quantile" else ""))
    # deciles of |raw|
    dec = pd.qcut(absraw.rank(method="first"), 10, labels=False)
    dec_rows = []
    for k in range(10):
        s = j[dec == k]
        rr = _row(f"decile_{k+1}_of_|AG raw| (1=weakest)", s, B)
        rr["median_abs_raw"] = float(absraw[dec == k].median())
        dec_rows.append(rr)
    # (b) lead = smallest nominal p per gene (and tissue), within the CCV-position pairs
    jl = j.copy()
    jl["minp"] = jl.groupby(["gid", "tissue"])["pval_nominal"].transform("min")
    lead = jl[jl["pval_nominal"] == jl["minp"]]
    rows.append(_row("(b) lead per (gene,tissue): smallest GTEx nominal p among CCV-position pairs (ties kept)", lead, B,
                     "lead among CCVs only, not necessarily the gene's global lead eVariant"))
    lead1 = jl.sort_values(["pval_nominal", "S6_CCV_input_key"]).drop_duplicates(["gid", "tissue"])
    rows.append(_row("(b') lead per (gene,tissue), ties broken arbitrarily (1 variant)", lead1, B))
    rows.append(_row("(b'') lead per (gene,tissue) AND in top 25% |AG raw|", lead[absraw.reindex(lead.index) >= absraw.quantile(0.75)], B))
    # distance strata (positive control: variants within the gene promoter should show AG effect direction)
    bins = [(0, 2_000), (2_000, 10_000), (10_000, 100_000), (100_000, 1e9)]
    for lo_, hi_ in bins:
        s = j[(j["dist_to_gene_tss"] >= lo_) & (j["dist_to_gene_tss"] < hi_)]
        rows.append(_row(f"distance to gene TSS [{int(lo_)}, {int(hi_) if hi_ < 1e8 else 'inf'}) bp", s, B, "positive control: promoter-proximal pairs"))
    s = j[j["dist_to_gene_tss"] < 10_000]
    top = s[s["raw_score"].abs() >= s["raw_score"].abs().quantile(0.5)]
    rows.append(_row("distance <10kb and above-median |AG raw| within that stratum", top, B))
    # (c) fine-mapping: PIP not provided by the source -> GWAS Catalog p as a clearly labelled proxy
    pip_col = pd.read_csv(ROOT / "variant_provenance/allele_matched_variants.tsv", sep="\t", usecols=["S6_CCV_input_key", "PIP", "gwascat_p", "chen_region"])
    pip_num = pd.to_numeric(pip_col["PIP"], errors="coerce")
    diag_c = dict(pip_numeric_values=int(pip_num.notna().sum()), pip_example=str(pip_col["PIP"].iloc[0]))
    print(diag_c)
    pip_col["gwascat_p"] = pd.to_numeric(pip_col["gwascat_p"], errors="coerce")
    jp = j.merge(pip_col[["S6_CCV_input_key", "gwascat_p"]], on="S6_CCV_input_key", how="left")
    rows.append(dict(subset="(c) high-PIP CCVs: NOT POSSIBLE - PIP column holds 'not_provided_by_source' for all 5,729 rows", n_pairs=0, note="no PIP; Chen 2024 CCV definition = conditional P within 2 orders of magnitude of the top"))
    thr = pip_col["gwascat_p"].quantile(0.10)
    rows.append(_row(f"(c-proxy) CCVs in the lowest 10% of GWAS Catalog p (p<={thr:.2g}); NOT PIP", jp[jp["gwascat_p"] <= thr], B, "proxy for fine-mapping rank; GWAS p is LD-confounded"))
    lead_reg = pip_col.loc[pip_col.groupby("chen_region")["gwascat_p"].idxmin(), "S6_CCV_input_key"]
    rows.append(_row("(c-proxy) per-region minimum-GWAS-p variant (region lead); NOT PIP", jp[jp["S6_CCV_input_key"].isin(set(lead_reg))], B))
    tab = pd.DataFrame(rows)
    _w(tab, "eqtl_direction_restricted.tsv")
    _w(pd.DataFrame(dec_rows), "eqtl_direction_deciles.tsv")
    # correlation of AG score with slope (continuous)
    cr = []
    for nm, s in (("all", j), ("dist<2kb", j[j.dist_to_gene_tss < 2000]), ("dist>=100kb", j[j.dist_to_gene_tss >= 100_000])):
        for col in ("raw_score", "quantile_score"):
            rho = sps.spearmanr(s[col], s["effect_alt_vs_ref"])
            cr.append(dict(subset=nm, score=col, n=len(s), spearman_rho=float(rho.statistic), p=float(rho.pvalue)))
    _w(pd.DataFrame(cr), "eqtl_spearman_score_vs_slope.tsv")
    show = [c for c in tab.columns if c in ("subset", "n_pairs", "raw_score_n", "raw_score_k", "raw_score_frac", "raw_score_binom_ci_lo", "raw_score_binom_ci_hi",
                                            "quantile_score_n", "quantile_score_frac", "quantile_score_binom_ci_lo", "quantile_score_binom_ci_hi")]
    print(tab[show].round(4).to_string())
    print(pd.DataFrame(dec_rows)[["subset", "n_pairs", "raw_score_frac", "median_abs_raw"]].round(4).to_string())
    print(pd.DataFrame(cr).round(4).to_string())
    j.to_csv(OUT / "cache_eqtl_joined_pairs.tsv.gz", sep="\t", index=False)


# ================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep-ag", "run"])
    ap.add_argument("--tasks", nargs="+", default=["weights", "clogit", "eqtl"], choices=["weights", "clogit", "eqtl"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "prep-ag":
        prep_ag(); return
    OUT.mkdir(parents=True, exist_ok=True)
    RE, lab, tabs = setup()
    if "weights" in a.tasks:
        task_weights(RE, lab, tabs)
    if "clogit" in a.tasks:
        task_clogit(RE, lab, tabs, B=a.boot)
    if "eqtl" in a.tasks:
        task_eqtl(RE, lab, tabs, B=a.boot)


if __name__ == "__main__":
    main()
