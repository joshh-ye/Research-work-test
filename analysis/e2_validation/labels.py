"""E2/E6 label tables (E2_ANALYSIS_PLAN.md sec 2). LABEL-ONLY: never touches model scores.

Build: hg38, 1-based. Matched set = variant_provenance/allele_matched_variants.tsv (REF/ALT from GWAS Catalog GCST90129505).
Usage (torch-venv):  ./torch-venv/bin/python analysis/e2_validation/labels.py
Parquet->TSV cache (ag-venv, has pyarrow): ./ag-venv/bin/python analysis/e2_validation/labels.py convert-gtex
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path("/home/yejz1/experimental_research")
MATCHED = ROOT / "variant_provenance/allele_matched_variants.tsv"
LAW = ROOT / "external_data/validation/law2024_nat_genet"
VAL = ROOT / "external_data/validation"
CHEN_XLSX = ROOT / "external_data/chen2024/MOESM4_ESM.xlsx"
OUT = ROOT / "results_phase2/e2_validation/labels"
LINES = {"HT29": "MPRA_HT29", "SW403": "MPRA_SW403", "HCEC": "MPRA_HCEC-1CT"}   # ST4 prefix -> ST2 column
FDR_THR = 1e-3
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def load_matched() -> pd.DataFrame:
    m = pd.read_csv(MATCHED, sep="\t", dtype=str)
    m["pos"] = m["hg38_pos_1based"].astype(int)
    m["chrom"] = m["hg38_chr"].astype(str)
    m["palindromic"] = m["palindromic"].astype(str).str.lower().eq("true")
    return m


def orientation(m_ref, m_alt, s_ref, s_alt) -> str:
    """keep / flip / exclude for a Law ST4 REF/ALT pair vs the matched (REF, ALT)."""
    if (m_ref, m_alt) == (s_ref, s_alt):
        return "keep"
    if (m_ref, m_alt) == (s_alt, s_ref):
        return "flip"
    return "exclude"


def build_mpra_labels(matched: pd.DataFrame, st2: pd.DataFrame, st4: pd.DataFrame) -> pd.DataFrame:
    """Per-variant MPRA labels. st2 needs rsid, Chr, 'bp (b38)', MPRA_* flags; st4 rsid, REF, ALT, <L>_FDR, <L>_logFC."""
    st2 = st2.copy()
    st2["Chr"] = st2["Chr"].astype(str)
    st2_by_rs = st2.drop_duplicates("rsid").set_index("rsid")
    st2_by_pos = st2.assign(_k=st2["Chr"] + ":" + st2["bp (b38)"].astype(str)).drop_duplicates("_k").set_index("_k")
    st4_by_rs = st4.drop_duplicates("rsid").set_index("rsid")
    rows = []
    for r in matched.itertuples(index=False):
        d = dict(S6_CCV_input_key=r.S6_CCV_input_key, rsid=r.rsid, chrom=r.chrom, pos=r.pos, ref=r.ref, alt=r.alt,
                 chen_region=r.chen_region, chen_lead_snp=r.chen_lead_snp, palindromic=r.palindromic)
        # ST2 join: rsID first, hg38-position fallback
        s2 = None
        d["st2_join"] = "none"
        if r.rsid in st2_by_rs.index:
            s2, d["st2_join"] = st2_by_rs.loc[r.rsid], "rsid"
        elif f"{r.chrom}:{r.pos}" in st2_by_pos.index:
            s2, d["st2_join"] = st2_by_pos.loc[f"{r.chrom}:{r.pos}"], "pos"
        d["st2_rsid"] = s2.name if (s2 is not None and d["st2_join"] == "rsid") else (s2["rsid"] if s2 is not None else "")
        # ST4 join: matched rsid, else the ST2 rsid of the position-joined row
        s4, d["st4_join"] = None, "none"
        if r.rsid in st4_by_rs.index:
            s4, d["st4_join"] = st4_by_rs.loc[r.rsid], "rsid"
        elif d["st2_rsid"] and d["st2_rsid"] in st4_by_rs.index:
            s4, d["st4_join"] = st4_by_rs.loc[d["st2_rsid"]], "st2_rsid"
        flags = {L: (int(s2[c]) if s2 is not None else np.nan) for L, c in LINES.items()}
        for L in LINES:
            d[f"flag_{L}"] = flags[L]
        d["tested"] = s2 is not None or s4 is not None
        d["in_st4"] = s4 is not None
        d["orientation"] = "NA"
        d["best_line"], d["best_fdr"], d["logFC_alt_vs_ref"] = "", np.nan, np.nan
        for L in LINES:
            d[f"fdr_{L}"], d[f"logFC_{L}"] = np.nan, np.nan
        if s4 is not None:
            o = orientation(r.ref, r.alt, s4["REF"], s4["ALT"])
            d["orientation"] = o
            sgn = -1.0 if o == "flip" else 1.0
            for L in LINES:
                d[f"fdr_{L}"] = float(s4[f"{L}_FDR"])
                d[f"logFC_{L}"] = sgn * float(s4[f"{L}_logFC"])
            best = min(LINES, key=lambda L: d[f"fdr_{L}"])
            d["best_line"], d["best_fdr"], d["logFC_alt_vs_ref"] = best, d[f"fdr_{best}"], d[f"logFC_{best}"]
        rows.append(d)
    t = pd.DataFrame(rows)
    fl = t[[f"flag_{L}" for L in LINES]]
    anyflag1 = (fl == 1).any(axis=1)
    anyflag2 = (fl == 2).any(axis=1)
    anyflag = (fl >= 1).any(axis=1)
    allzero = (fl == 0).all(axis=1)
    bad_orient = t["orientation"] == "exclude"

    def lab(pos, neg, why_ex):
        out = np.where(pos, "pos", np.where(neg, "neg", "excluded"))
        out = np.where(~t["tested"], "untested", out)
        out = np.where(bad_orient & t["tested"], "excluded", out)
        return out

    # primary (plan 2a)
    t["mpra_primary"] = lab(t["in_st4"], allzero & ~t["in_st4"], None)
    reason = np.full(len(t), "", dtype=object)
    reason[(t["mpra_primary"] == "excluded") & anyflag1.to_numpy() & ~t["in_st4"].to_numpy()] = "st2_flag1_not_in_ST4"
    reason[(t["mpra_primary"] == "excluded") & anyflag2.to_numpy() & ~t["in_st4"].to_numpy()] = "st2_flag2_not_in_ST4"
    reason[bad_orient.to_numpy() & t["tested"].to_numpy()] = "st4_allele_orientation_mismatch"
    reason[(t["mpra_primary"] == "excluded") & (reason == "")] = "in_ST2_without_complete_flags"
    t["exclusion_reason"] = reason
    # S1: any flag>=1 or ST4 = positive; all flags 0 and not ST4 = negative
    t["mpra_S1"] = lab(t["in_st4"] | anyflag, allzero & ~t["in_st4"], None)
    # S2 per line: ST4 FDR<thr in that line = pos; ST2 flag 0 in that line = neg; otherwise excluded
    for L in LINES:
        sig = (t[f"fdr_{L}"] < FDR_THR)
        t[f"mpra_S2_{L}"] = lab(sig, (t[f"flag_{L}"] == 0) & ~sig, None)
    return t


def read_annotation_counts() -> pd.DataFrame:
    d = pd.read_excel(CHEN_XLSX, sheet_name="Supplementary Data 6", header=1)
    flagcols = list(d.columns[5:12])
    out = d[["CCV"] + flagcols].rename(columns={"CCV": "rsid"}).dropna(subset=["rsid"])
    out = out[out["rsid"].astype(str).str.startswith("rs")]
    out["chen_annot_count"] = out[flagcols].astype(int).sum(axis=1)
    out.columns = ["rsid"] + [f"annot_{i}" for i in range(len(flagcols))] + ["chen_annot_count"]
    return out.drop_duplicates("rsid")


# ------------------------- eQTL -------------------------
GTEX_DIR = VAL / "gtex_v10/GTEx_Analysis_v10_eQTL_updated"
CACHE = OUT / "cache"
GTEX_TISSUES = ["Colon_Transverse", "Colon_Sigmoid"]


def _keyset(matched: pd.DataFrame) -> set:
    s = set()
    for c, p, r, a in zip(matched["chrom"], matched["pos"], matched["ref"], matched["alt"]):
        s.add(f"chr{c}_{p}_{r}_{a}"); s.add(f"chr{c}_{p}_{a}_{r}")
    return s


def convert_gtex(matched: pd.DataFrame) -> None:
    """ag-venv: parquet -> TSV restricted to matched positions (either allele order)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ks = _keyset(matched)
    for t in GTEX_TISSUES:
        d = pd.read_parquet(GTEX_DIR / f"{t}.v10.eQTLs.signif_pairs.parquet",
                            columns=["gene_id", "variant_id", "slope", "slope_se", "pval_nominal"])
        d = d[d["variant_id"].str.replace("_b38", "", regex=False).isin(ks)]
        d.to_csv(CACHE / f"gtex_v10_{t}.ccv.tsv.gz", sep="\t", index=False)
        print(f"[convert-gtex] {t}: {len(d)} pairs at matched positions", flush=True)


def _read_gtex(t: str) -> pd.DataFrame:
    cache = CACHE / f"gtex_v10_{t}.ccv.tsv.gz"
    if cache.exists():
        return pd.read_csv(cache, sep="\t")
    return pd.read_parquet(GTEX_DIR / f"{t}.v10.eQTLs.signif_pairs.parquet")


def harmonise(var_id: pd.Series, matched_by_key: dict) -> tuple[pd.Series, pd.Series]:
    """var_id 'chrN_pos_ref_alt' (suffix _b38 stripped) -> (matched key or '', sign +1 / -1 for the effect (ALT of source))."""
    key, sign = [], []
    for v in var_id:
        c, p, r, a = v.replace("_b38", "").split("_")[:4]
        k1, k2 = f"{c[3:]}:{p}:{r}:{a}", f"{c[3:]}:{p}:{a}:{r}"
        if k1 in matched_by_key:
            key.append(k1); sign.append(1)
        elif k2 in matched_by_key:
            key.append(k2); sign.append(-1)      # swapped: source ALT is matched REF -> flip effect sign
        else:
            key.append(""); sign.append(0)
    return pd.Series(key, index=var_id.index), pd.Series(sign, index=var_id.index)


def build_eqtl_pairs(matched: pd.DataFrame) -> pd.DataFrame:
    mk = set(matched["S6_CCV_input_key"])
    parts = []
    for t in GTEX_TISSUES:
        d = _read_gtex(t)
        k, s = harmonise(d["variant_id"], mk)
        d = d.assign(S6_CCV_input_key=k, sign=s)
        d = d[d["sign"] != 0]
        parts.append(pd.DataFrame(dict(S6_CCV_input_key=d["S6_CCV_input_key"], source=f"gtex_v10_{t}", gene_id=d["gene_id"],
                                       effect_alt_vs_ref=d["slope"] * d["sign"], se=d["slope_se"], swapped=d["sign"] < 0)))
    cs = pd.read_csv(VAL / "eqtl_catalogue_cedar/QTD000068.credible_sets.tsv.gz", sep="\t")
    k, s = harmonise(cs["variant"], mk)
    cs = cs.assign(S6_CCV_input_key=k, sign=s)
    cs = cs[cs["sign"] != 0]
    parts.append(pd.DataFrame(dict(S6_CCV_input_key=cs["S6_CCV_input_key"], source="cedar_QTD000068_credible_set",
                                   gene_id=cs["gene_id"], effect_alt_vs_ref=cs["beta"] * cs["sign"], se=cs["se"],
                                   swapped=cs["sign"] < 0)))
    return pd.concat(parts, ignore_index=True)


def eqtl_labels(matched: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    t = matched[["S6_CCV_input_key"]].copy()
    gt = set(pairs.loc[pairs.source.str.startswith("gtex"), "S6_CCV_input_key"])
    ce = set(pairs.loc[pairs.source.str.startswith("cedar"), "S6_CCV_input_key"])
    t["eqtl_gtex_v10_colon_any"] = t["S6_CCV_input_key"].isin(gt).astype(int)
    t["eqtl_cedar_credible"] = t["S6_CCV_input_key"].isin(ce).astype(int)
    return t


# ------------------------- power / counts -------------------------
def label_counts(t: pd.DataFrame, col: str, region_col: str = "chen_region") -> dict:
    vc = t[col].value_counts()
    sub = t[t[col].isin(["pos", "neg"])]
    g = sub.groupby(region_col)[col].agg(npos=lambda x: (x == "pos").sum(), nneg=lambda x: (x == "neg").sum())
    inf = g[(g.npos > 0) & (g.nneg > 0)]
    return dict(label=col, n_pos=int(vc.get("pos", 0)), n_neg=int(vc.get("neg", 0)), n_excluded=int(vc.get("excluded", 0)),
                n_untested=int(vc.get("untested", 0)), n_informative_regions=len(inf),
                n_pos_in_informative=int(inf.npos.sum()), n_neg_in_informative=int(inf.nneg.sum()),
                n_pairs_pos_x_neg=int((inf.npos * inf.nneg).sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matched = load_matched()
    if len(sys.argv) > 1 and sys.argv[1] == "convert-gtex":
        convert_gtex(matched); return
    st2 = pd.read_csv(LAW / "ST2_variant_annotation.tsv", sep="\t")
    st4 = pd.read_csv(LAW / "ST4_mpra_hits.tsv", sep="\t")
    t = build_mpra_labels(matched, st2, st4)
    t = t.merge(read_annotation_counts()[["rsid", "chen_annot_count"]], on="rsid", how="left")
    pairs = build_eqtl_pairs(matched)
    t = t.merge(eqtl_labels(matched, pairs), on="S6_CCV_input_key", how="left")
    t.to_csv(OUT / "variant_labels.tsv", sep="\t", index=False)
    pairs.to_csv(OUT / "eqtl_pairs_aligned.tsv.gz", sep="\t", index=False)
    cols = ["mpra_primary", "mpra_S1"] + [f"mpra_S2_{L}" for L in LINES]
    cnt = pd.DataFrame([label_counts(t, c) for c in cols])
    # eQTL counts (binary 0/1 -> pos/neg)
    for c in ["eqtl_gtex_v10_colon_any", "eqtl_cedar_credible"]:
        tt = t.assign(**{c + "_lab": np.where(t[c] == 1, "pos", "neg")})
        cnt = pd.concat([cnt, pd.DataFrame([label_counts(tt, c + "_lab")])], ignore_index=True)
    cnt.to_csv(OUT / "label_counts.tsv", sep="\t", index=False)
    # ST4 resolution report
    rep = st4.assign(in_matched_by_rsid=st4.rsid.isin(matched.rsid), in_ST2_by_rsid=st4.rsid.isin(st2.rsid))
    rep = rep.merge(t[["rsid", "st4_join", "tested", "orientation", "mpra_primary"]].rename(columns={"rsid": "m_rsid"}),
                    left_on="rsid", right_on="m_rsid", how="left")
    rep.to_csv(OUT / "st4_resolution.tsv", sep="\t", index=False)
    unres = st4[~st4.rsid.isin(st2.rsid)]
    print("ST4 rsIDs absent from ST2 by rsID:", len(unres), "; of these present in matched set by rsid:",
          int(unres.rsid.isin(matched.rsid).sum()))
    print("ST4 hits in matched set (tested universe):", int(t.in_st4.sum()), "; orientation:", t.orientation.value_counts().to_dict())
    print("ST2 join methods:", t.st2_join.value_counts().to_dict(), "; exclusion reasons:", t.exclusion_reason.replace("", np.nan).value_counts().to_dict())
    print("pos with palindromic alleles:", int(((t.mpra_primary == 'pos') & t.palindromic).sum()))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(cnt.to_string(index=False))
    # sec 6 power statement (label-only: random scores, no model output)
    import stats as ST
    prim = t[t.mpra_primary.isin(["pos", "neg"])]
    pw = ST.minimum_detectable_auroc((prim.mpra_primary == "pos").astype(int), prim.chen_region, n_sim=100, B=300)
    pw.update({k: v for k, v in cnt.iloc[0].to_dict().items() if k != "label"})
    pd.DataFrame([pw]).to_csv(OUT / "power_statement.tsv", sep="\t", index=False)
    ho = t[(t.mpra_primary.isin(["pos", "neg"])) & t.chrom.isin(["1", "8", "9", "22"])]
    inf_regions = set(prim.groupby("chen_region").mpra_primary.agg(lambda x: (x == "pos").any() and (x == "neg").any()).loc[lambda x: x].index)
    pw["n_pos_in_noninformative_regions"] = int(((prim.mpra_primary == "pos") & ~prim.chen_region.isin(inf_regions)).sum())
    pw["n_tested_ST2"] = int(t.tested.sum())
    pw["heldout_chr1_8_9_22_n_tested_pos_neg"] = len(ho); pw["heldout_chr1_8_9_22_n_pos"] = int((ho.mpra_primary == "pos").sum())
    (OUT / "POWER.md").write_text("# Label-only power statement (E2 plan sec 6 / Amendment A1.1)\n\nNo model score used. Generated by labels.py.\n\n"
        + "".join(f"- {k}: {v:.4f}\n" if isinstance(v, float) else f"- {k}: {v}\n" for k, v in pw.items())
        + "\nMinimum detectable AUROC = 0.5 + (1.645+0.842) x null region-bootstrap SE (random scores; one-sided alpha 0.05, 80% power, normal approx; no Holm correction).\n")
    print("power statement (label-only):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pw.items()})
    print(f"[labels] wrote {OUT}")


if __name__ == "__main__":
    main()
