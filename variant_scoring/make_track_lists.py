"""Build the frozen track-list files for SD-005 (variant_scoring/track_lists/*.tsv).

Selection is by METADATA KEYWORDS ONLY (cell-line names, assay prefix, histone exclusion);
no performance, correlation or validation-label information is used.

Run (CPU):  ./torch-venv/bin/python variant_scoring/make_track_lists.py
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "track_lists"
OUT.mkdir(exist_ok=True)

# CRC (colorectal carcinoma) cell lines. Non-malignant colon lines/tissues (HCEC 1CT, colonic mucosa)
# are deliberately NOT included.
CRC_LINE_RE = (r"HCT[- ]?116|HT[- ]?29|LoVo|Caco[- ]?2|SW[- ]?480|SW[- ]?620|RKO|COLO[- ]?205|DLD[- ]?1"
               r"|T84|LS[- ]?174T|LS180|GP5d")
HISTONE_RE = re.compile(r"(?<![A-Za-z0-9])H(1|2A|2B|3|4)(K\d|\.Z|FZ|$|[^A-Za-z0-9])", re.I)
CRC_LINE_NORM = [("HCT116", r"HCT[- ]?116"), ("HT-29", r"HT[- ]?29"), ("LoVo", r"LoVo"), ("Caco-2", r"Caco[- ]?2"),
                 ("SW480", r"SW[- ]?480"), ("SW620", r"SW[- ]?620"), ("RKO", r"RKO"), ("COLO205", r"COLO[- ]?205"),
                 ("DLD-1", r"DLD[- ]?1"), ("T84", r"T84"), ("LS174T", r"LS[- ]?174T"), ("LS180", r"LS180"),
                 ("GP5d", r"GP5d")]


def norm_line(s: str) -> str:
    for name, rx in CRC_LINE_NORM:
        if re.search(rx, s, re.I):
            return name
    return ""


def pretrained_list(df: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df.copy()
    df["assay"] = df["description"].str.split(":").str[0]
    sel = df["description"].str.contains(CRC_LINE_RE, case=False, regex=True) & df["assay"].isin(["CHIP", "DNASE", "ATAC"])
    d = df[sel].copy()
    d["is_histone"] = d["description"].apply(lambda s: bool(HISTONE_RE.search(s)) and s.startswith("CHIP"))
    d = d[~d["is_histone"]].copy()
    d["cell_line"] = d["description"].apply(norm_line)

    def factor(desc):
        p = desc.split(":")
        if p[0] != "CHIP":
            return ""
        if p[1] not in (".", ""):
            return p[1]
        m = re.search(r"chrom\d+_[^_]+_([^_]+)_", p[2])   # Enformer GEO tracks: batch1_chrom1_LoVo_MYC_...
        return m.group(1) if m else ""
    d["factor"] = d["description"].apply(factor)
    d["model"] = model
    return d[["model", "track_index", "identifier", "assay", "factor", "cell_line", "description"]]


def main():
    b = pd.read_csv(ROOT / "borzoi_data/targets_human.csv")
    assert (b["track_index"].values == np.arange(len(b))).all() and len(b) == 7611
    e = pd.read_csv(ROOT / "enformer_data/targets_human.txt", sep="\t").rename(columns={"index": "track_index"})
    assert (e["track_index"].values == np.arange(len(e))).all() and len(e) == 5313

    pb = pretrained_list(b, "borzoi_pretrained"); pb.to_csv(OUT / "borzoi_pretrained_crc_tracks.tsv", sep="\t", index=False)
    pe = pretrained_list(e, "enformer_pretrained"); pe.to_csv(OUT / "enformer_pretrained_crc_tracks.tsv", sep="\t", index=False)

    # ---- Borzoi-TL: 183 heads, order = results_full/test_per_track_metrics.csv = borzoi_ckpt_full/bw_files.json
    cf = pd.read_csv(ROOT / "reports/qing_deliverables/evaluation_evidence/contamination_flags.csv")
    bw = [Path(x).name for x in json.load(open(ROOT / "borzoi_ckpt_full/bw_files.json"))]
    assert bw == cf["bw_filename"].tolist() and (cf["track_idx"].values == np.arange(183)).all()
    cf = cf.rename(columns={"track_idx": "tl_head_index"})
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    # track_idx is 0-based head index; verify order against bw_files.json via DCid
    assert (ex["track_idx"].values == np.arange(183)).all()
    assert [int(Path(x).stem) for x in bw] == ex["DCid"].tolist()
    cf["contam_lovo_gsm"] = cf["gsm_in_borzoi_training_targets"] | cf["gsm_in_enformer_training_targets"]
    cf["contam_encsr_145_148"] = cf["tl_head_index"].between(145, 148)
    cf["borzoi_exposure_class"] = ex["Borzoi_class"].values
    cf["enformer_exposure_class"] = ex["Enformer_class"].values
    cf["alphagenome_exposure_class"] = ex["AlphaGenome_class"].values
    cf["contaminated"] = cf["borzoi_exposure_class"] == "identical_experiment"   # exposure audit (31)
    assert cf["contam_lovo_gsm"].sum() == 24 and cf["contaminated"].sum() == 31
    assert set(cf.loc[cf.contam_lovo_gsm | cf.contam_encsr_145_148, "tl_head_index"]) <= set(cf.loc[cf.contaminated, "tl_head_index"])
    assert cf.loc[cf.contaminated, "tl_head_index"].tolist() == list(range(13, 37)) + [78, 81, 93, 145, 146, 147, 148]
    cf["primary"] = ~cf["contaminated"]
    assert cf["primary"].sum() == 152
    cf[["tl_head_index", "bw_filename", "GSMID", "factor", "cell_line", "contam_lovo_gsm", "contam_encsr_145_148",
        "borzoi_exposure_class", "enformer_exposure_class", "alphagenome_exposure_class", "contaminated", "primary"]
       ].to_csv(OUT / "borzoi_tl_tracks.tsv", sep="\t", index=False)

    # ---- AlphaGenome (from the existing per-track output; tracks identical for every variant)
    code = f"""
import pandas as pd
d=pd.read_parquet('{ROOT}/alphagenome_variant_effects/output/variant_all_scores_per_track/chr8.parquet',
  columns=['modality','scorer_kind','track_name','track_strand','assay_title','ontology_curie','biosample_name','transcription_factor','histone_mark','variant_scorer'])
d=d[(d.scorer_kind=='diff')&d.modality.isin(['DNASE','ATAC','CHIP_TF'])].drop_duplicates(['modality','track_name','track_strand'])
d.to_csv('{OUT}/_ag_all_diff_tracks.tmp.tsv',sep='\\t',index=False)
"""
    subprocess.run([str(ROOT / "ag-venv/bin/python"), "-c", code], check=True)
    ag = pd.read_csv(OUT / "_ag_all_diff_tracks.tmp.tsv", sep="\t", keep_default_na=False)
    (OUT / "_ag_all_diff_tracks.tmp.tsv").unlink()
    ag["cell_line"] = ag["biosample_name"].apply(norm_line)
    ag = ag[ag["cell_line"] != ""].rename(columns={"transcription_factor": "factor"})
    ag["model"] = "alphagenome"
    ag[["model", "modality", "track_name", "track_strand", "assay_title", "ontology_curie", "biosample_name",
        "cell_line", "factor", "variant_scorer"]].to_csv(OUT / "alphagenome_crc_tracks.tsv", sep="\t", index=False)

    for n, d in [("borzoi_pretrained", pb), ("enformer_pretrained", pe), ("alphagenome", ag)]:
        print(n, len(d), d.groupby(["assay"] if "assay" in d else ["modality"]).size().to_dict(),
              d["cell_line"].value_counts().to_dict())
    print("borzoi_tl", len(cf), "primary", int(cf.primary.sum()))


if __name__ == "__main__":
    main()
