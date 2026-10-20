"""Supplementary tables ST1-ST4 (TSV) in figures/tables/. Values copied/reshaped from saved result files only.
Run: ./torch-venv/bin/python figures/scripts/make_supp_tables.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import style as S

R = S.ROOT / "results_phase2"
rd = lambda p: pd.read_csv(p, sep="\t")
te = rd(S.ROOT / "exposure_audit/track_exposure.tsv")
# ST2: exposure audit per track (verbatim)
te.to_csv(S.TABLES / "ST2_exposure_audit_per_track.tsv", sep="\t", index=False)
cls = te[["track_idx", "Borzoi_class", "Enformer_class", "AlphaGenome_class"]]

# ST1: per-track metrics, long format
pt = pd.read_csv(R / "exposure_stratified/per_track_metrics_by_exposure.csv")
pt = pt[pt.tile_set == "all_359_tiles"]
base = pt[["track_idx", "bw_filename", "TF", "cell_line"]]
L = []
L.append(base.assign(model="Borzoi-TL", mapping="fine-tuned head", pearson_r=pt.tl_pearson_r, pearson_r_squared=pt.tl_pearson_r_squared,
                     prediction_R2=pt.tl_prediction_R2, prediction_R2_kind="uncalibrated prediction R2", selected_track=""))
for m, tag in (("oracle", "oracle_test"), ("valsel", "val_selected")):
    L.append(base.assign(model="Borzoi (pretrained)", mapping=m, pearson_r=pt[f"{m}_pearson_r"], pearson_r_squared=pt[f"{m}_pearson_r_squared"],
                         prediction_R2=pt[f"{m}_prediction_R2_validation_calibrated"], prediction_R2_kind="validation-calibrated R2", selected_track=""))
for key, name in (("enformer", "Enformer"), ("alphagenome", "AlphaGenome")):
    d = pd.read_csv(R / f"e7_nonoracle/{key}/per_track_metrics.csv")
    assert (d.track_idx.values == pt.track_idx.values).all()
    for m in ("oracle", "valsel", "valsel_excl_ident"):
        desc = d[f"{m}_description"] if f"{m}_description" in d else d[f"{m}_track"]
        L.append(base.assign(model=name, mapping=m, pearson_r=d[f"{m}_pearson_r"].values, pearson_r_squared=d[f"{m}_pearson_r_squared"].values,
                             prediction_R2=d[f"{m}_prediction_R2_validation_calibrated"].values, prediction_R2_kind="validation-calibrated R2",
                             selected_track=(d[f"{m}_track"].astype(str) + " | " + desc.astype(str)).values))
st1 = pd.concat(L, ignore_index=True).merge(cls, on="track_idx")
st1["zero_signal_tiles"] = "all 359 test tiles"
st1.to_csv(S.TABLES / "ST1_per_track_metrics_all_models.tsv", sep="\t", index=False)
# sanity: means equal source summaries
ms = pd.read_csv(R / "e7_nonoracle/enformer/model_summary.csv")
chk = st1[(st1.model == "Enformer") & (st1.mapping == "oracle")].pearson_r.mean()
assert np.isclose(chk, ms[(ms.mapping == "oracle") & (ms.subset == "all_183")].mean_pearson_r.iloc[0])
assert np.isclose(st1[st1.model == "Borzoi-TL"].pearson_r.mean(), 0.681988226530368)

# ST3: allele-matched variant set summary
P = S.ROOT / "variant_provenance"
allr = rd(P / "exploratory_all_rows.tsv"); am = rd(P / "allele_matched_variants.tsv"); un = rd(P / "unmatched_or_excluded.tsv")
lc = rd(R / "e2_validation/labels/label_counts.tsv")
rows = [("input_rows", len(allr), "exploratory_all_rows.tsv"), ("input_unique_rsids", allr.rsid.nunique(), "exploratory_all_rows.tsv"),
        ("S6_CCV_absent_from_input", int((un.match_status == "S6_CCV_absent_from_input").sum()), "unmatched_or_excluded.tsv"),
        ("allele_matched_rows", len(am), "allele_matched_variants.tsv"),
        ("allele_matched_palindromic", int(am.palindromic.astype(str).eq("True").sum()), "allele_matched_variants.tsv"),
        ("allele_matched_strand_flip", int(am.strand_flip.astype(str).eq("True").sum()), "allele_matched_variants.tsv"),
        ("alt_is_gwascat_effect_allele", int(am.alt_is_gwascat_effect_allele.astype(str).eq("True").sum()), "allele_matched_variants.tsv")]
for k, v in allr.match_status.value_counts().items(): rows.append((f"input_rows_{k}", int(v), "exploratory_all_rows.tsv"))
for c, v in am.hg38_chr.astype(str).value_counts().items(): rows.append((f"allele_matched_chr{c}", int(v), "allele_matched_variants.tsv"))
st3 = pd.DataFrame(rows, columns=["item", "count", "source_file"])
st3["genome_build"] = "hg38 (S6 source hg19, lifted/looked up upstream; see PROVENANCE.md)"
st3.to_csv(S.TABLES / "ST3_allele_matched_variant_summary.tsv", sep="\t", index=False)
lc.to_csv(S.TABLES / "ST3b_mpra_label_counts.tsv", sep="\t", index=False)

# ST4: E2 interim tables stacked
parts = []
for k in ("alphagenome", "enformer"):
    for t in ("p1", "baselines", "secondary_P1", "p2"):
        d = rd(R / f"e2_validation/interim_{k}/tables/{t}.tsv")
        d.insert(0, "table", t); d.insert(0, "run", f"interim_{k}"); parts.append(d)
st4 = pd.concat(parts, ignore_index=True)
st4["caveat"] = np.where(st4.table.isin(["p2"]), "P2 sign convention unresolved (SD-017)", "INTERIM, unadjusted, Holm deferred; Borzoi/Borzoi-TL pending")
st4.to_csv(S.TABLES / "ST4_e2_interim_tables.tsv", sep="\t", index=False)
for f in sorted(S.TABLES.glob("ST*.tsv")): print(f.name, len(pd.read_csv(f, sep="\t")))
