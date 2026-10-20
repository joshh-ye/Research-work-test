"""Per-model score loaders for E2/E6 -> common table keyed by S6_CCV_input_key ('chr:pos:ref:alt', hg38, 1-based, no 'chr').
Scores only; no labels. Definitions follow variant_scoring/SCORING_SPEC.md (SD-005, SD-014).

Standard columns of every loaded model table (index = S6_CCV_input_key):
  primary (max_t |score_t|), signed (score at arg-max track), argmax_track, mean_abs, mean_signed, excl_window (bool).
AlphaGenome cache (ag-venv; pyarrow):  ./ag-venv/bin/python analysis/e2_validation/scores.py build-ag-cache
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "variant_scoring"))
AG_OUT = ROOT / "alphagenome_variant_effects/output"
AG_TRACKS = ROOT / "variant_scoring/track_lists/alphagenome_crc_tracks.tsv"
TL_DIR = ROOT / "variant_scoring/track_lists"
SCORES = ROOT / "results_phase2/variant_scores"
CACHE = ROOT / "results_phase2/e2_validation/scores"
S6_INPUT = ROOT / "alphagenome_variant_effects/input/S6_CCV_ChrPosRefAlt_rsid.tsv"
PSEUDOCOUNT = 1.0
STD = ["primary", "signed", "argmax_track", "mean_abs", "mean_signed", "excl_window"]

# headline model definitions: name -> (results dir, head, track-set name, summary-key prefix)
SUMMARY_MODELS = {
    "borzoi_pretrained": ("borzoi", "pretrained", "crc_dnase_tfchip"),
    "borzoi_tl": ("borzoi", "tl", "primary152"),
    "borzoi_tl_all183": ("borzoi", "tl", "all183"),          # sensitivity (includes 31 training-contaminated tracks)
    "enformer": ("enformer", "pretrained", "crc_dnase_tfchip"),
}
HEADLINE = ["alphagenome_quantile", "borzoi_pretrained", "borzoi_tl", "enformer"]


# ---------------- summarising (same maths as variant_scoring.common.summarize; kept local so tests need no torch) -------------
def summarize_matrix(x: np.ndarray, track_ids: np.ndarray | None = None) -> pd.DataFrame:
    """x (n_rows, n_tracks) signed per-track scores -> primary/signed/argmax/mean_abs/mean_signed. NaN rows propagate as NaN."""
    a = np.abs(x)
    am = np.nanargmax(np.where(np.isnan(a), -1, a), axis=1)
    r = np.arange(len(x))
    out = pd.DataFrame({"primary": a[r, am], "signed": x[r, am],
                        "argmax_track": (track_ids[am] if track_ids is not None else am),
                        "mean_abs": a.mean(axis=1), "mean_signed": x.mean(axis=1)})
    bad = np.isnan(x).any(axis=1)
    out.loc[bad, ["primary", "signed", "mean_abs", "mean_signed"]] = np.nan
    return out


def log2fc(ref_sum: np.ndarray, alt_sum: np.ndarray, pc: float = PSEUDOCOUNT) -> np.ndarray:
    return np.log2((alt_sum + pc) / (ref_sum + pc))


# ---------------- Borzoi / Enformer ----------------
def load_summary(model: str, results_dir: Path = SCORES) -> pd.DataFrame:
    """Finalised summary.tsv (runner.finalize): columns '<head>__<set>__<stat>'. Returns the STD table indexed by input_variant."""
    d, head, sset = SUMMARY_MODELS[model]
    f = Path(results_dir) / d / "summary.tsv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found (scoring run not finalised); use load_from_chunks")
    s = pd.read_csv(f, sep="\t")
    p = f"{head}__{sset}__"
    out = pd.DataFrame({"primary": s[p + "primary_max_abs_log2fc"], "signed": s[p + "signed_at_argmax"],
                        "argmax_track": s[p + "argmax_track_pos"], "mean_abs": s[p + "mean_abs_log2fc"],
                        "mean_signed": s[p + "mean_signed_log2fc"],
                        "excl_window": s["window_padded"].astype(bool) | (s["n_N_in_agg_window"] > 0)})
    out.index = pd.Index(s["input_variant"], name="S6_CCV_input_key")
    assert out.index.is_unique
    return out


def track_mask(model: str) -> np.ndarray:
    """Boolean track mask over the full output axis for the frozen list (SCORING_SPEC sec 3)."""
    _, head, sset = SUMMARY_MODELS[model]
    if head == "tl":
        tl = pd.read_csv(TL_DIR / "borzoi_tl_tracks.tsv", sep="\t")
        m = np.zeros(183, bool)
        m[tl["tl_head_index"].to_numpy()] = tl["primary"].to_numpy(bool) if sset == "primary152" else True
        return m
    f = "borzoi_pretrained_crc_tracks.tsv" if model.startswith("borzoi") else "enformer_pretrained_crc_tracks.tsv"
    t = pd.read_csv(TL_DIR / f, sep="\t")
    m = np.zeros(7611 if model.startswith("borzoi") else 5313, bool)
    m[t["track_index"].to_numpy()] = True
    return m


def load_from_chunks(model: str, results_dir: Path = SCORES, s6_input: Path = S6_INPUT) -> pd.DataFrame:
    """Same table from chunk_*.npz (payload per variant_scoring/runner.py: row_ids, pos_of_row, ref_<head> (n_pos,T), alt_<head> (n_rows,T),
    n_pad_left/right, n_N_in_agg); only rows of chunks present. input_variant = line `row_id` of the S6 input file."""
    d, head, _ = SUMMARY_MODELS[model]
    files = sorted(f for f in (Path(results_dir) / d / "chunks").glob("chunk_*.npz") if ".tmp" not in f.name)
    keys = pd.read_csv(s6_input, sep="\t").iloc[:, 0].to_numpy()
    mask = track_mask(model)
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        ref = z[f"ref_{head}"][z["pos_of_row"]]
        lfc = log2fc(ref, z[f"alt_{head}"])[:, mask]
        t = summarize_matrix(lfc, np.flatnonzero(mask))
        t["excl_window"] = ((z["n_pad_left"] + z["n_pad_right"]) > 0) | (z["n_N_in_agg"] > 0)
        t.index = pd.Index(keys[z["row_ids"]], name="S6_CCV_input_key")
        parts.append(t)
    if not parts:
        raise FileNotFoundError(f"no chunks in {Path(results_dir)/d/'chunks'}")
    out = pd.concat(parts)
    assert out.index.is_unique
    return out


def load_model(model: str, results_dir: Path = SCORES, finalised_only: bool = False) -> pd.DataFrame:
    """finalised_only=True (used by run_e2 --unblind, Amendment A3): only summary.tsv of a finalised run; NO partial-chunk fallback."""
    try:
        return load_summary(model, results_dir)
    except FileNotFoundError:
        if finalised_only:
            raise FileNotFoundError(f"model {model} is not finalised (no summary.tsv in {Path(results_dir)/SUMMARY_MODELS[model][0]}); "
                                    "refusing to use partial chunks for an unblinded analysis")
        return load_from_chunks(model, results_dir)


QING_MAP = ROOT / "reports/qing_deliverables/evaluation_evidence/baseline_borzoi_rerun/track_mappings.csv"


def lfc_columns(model: str, cols: np.ndarray, results_dir: Path = SCORES, s6_input: Path = S6_INPUT, finalised_only: bool = False) -> pd.DataFrame:
    """Signed log2fc (pc=1) for output channels `cols` of the model's head, indexed by S6_CCV_input_key. Uses the saved whole-vocabulary
    log2fc_<head>.npy of a finalised run if present, else recomputes from chunk npz (which store the FULL head: ref_<head>/alt_<head>, all T channels)."""
    d, head, _ = SUMMARY_MODELS[model]
    base = Path(results_dir) / d
    cols = np.asarray(cols)
    if (base / f"log2fc_{head}.npy").exists() and (base / "variants.tsv").exists():
        lfc = np.load(base / f"log2fc_{head}.npy", mmap_mode="r")[:, cols]
        keys = pd.read_csv(base / "variants.tsv", sep="\t")["input_variant"].to_numpy()
        return pd.DataFrame(np.asarray(lfc), index=pd.Index(keys, name="S6_CCV_input_key"), columns=cols)
    if finalised_only:
        raise FileNotFoundError(f"model {model} is not finalised (no log2fc_{head}.npy/variants.tsv); partial chunks refused")
    files = sorted(f for f in (base / "chunks").glob("chunk_*.npz") if ".tmp" not in f.name)
    allkeys = pd.read_csv(s6_input, sep="\t").iloc[:, 0].to_numpy()
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        lfc = log2fc(z[f"ref_{head}"][z["pos_of_row"]][:, cols], z[f"alt_{head}"][:, cols])
        parts.append(pd.DataFrame(lfc, index=pd.Index(allkeys[z["row_ids"]], name="S6_CCV_input_key"), columns=cols))
    if not parts:
        raise FileNotFoundError(f"no scores for {model}")
    return pd.concat(parts)


def _norm_cl(x: str) -> str:
    return str(x).replace("-", "").upper()


def _group(cl: str) -> str:
    n = _norm_cl(cl)
    return "HCT116" if n == "HCT116" else ("LoVo" if n == "LOVO" else "other")


def cellline_primary(model: str, results_dir: Path = SCORES, by: str = "group", finalised_only: bool = False) -> pd.DataFrame:
    """Per-cell-line (by='cell_line') or HCT-116 / LoVo / other (by='group') max|log2fc| over the frozen track set of `model`
    (TL: the 152 primary tracks). Amendment A1 item 3."""
    d, head, _ = SUMMARY_MODELS[model]
    if head == "tl":
        t = pd.read_csv(TL_DIR / "borzoi_tl_tracks.tsv", sep="\t"); t = t[t["primary"]]; t = t.assign(idx=t["tl_head_index"])
    else:
        f = "borzoi_pretrained_crc_tracks.tsv" if model.startswith("borzoi") else "enformer_pretrained_crc_tracks.tsv"
        t = pd.read_csv(TL_DIR / f, sep="\t"); t = t.assign(idx=t["track_index"])
    t = t.assign(g=t["cell_line"].map(_group) if by == "group" else t["cell_line"].map(_norm_cl))
    lfc = lfc_columns(model, t["idx"].to_numpy(), results_dir, finalised_only=finalised_only)
    lfc.columns = t["idx"].to_numpy()
    out = {}
    for g, grp in t.groupby("g"):
        out[g] = lfc[grp["idx"].to_numpy()].abs().max(axis=1)
    return pd.DataFrame(out)


def track_matched_borzoi_cols(same_tf: bool = False) -> tuple:
    """(unique pretrained-Borzoi output channels, n_tl_tracks_dropped) for the 'validation-selected-mapping comparator' (A1.3, relabelled by A3).
    Source: Qing track_mappings.csv (0-based targets_human track_index; identifiers verified against borzoi_data/targets_human.csv) for the 152 TL
    primary tracks. same_tf=False: column val_selected_track_idx (73 channels). same_tf=True: val_selected_sameTF_track_idx, keeping only tracks with
    sameTF_fell_back_to_val_selected == False (a same-TF candidate existed); the others are dropped and counted."""
    m = pd.read_csv(QING_MAP)
    tl = pd.read_csv(TL_DIR / "borzoi_tl_tracks.tsv", sep="\t")
    assert (tl["tl_head_index"].to_numpy() == m["track_idx"].to_numpy()).all()
    m = m[tl["primary"].to_numpy(bool)]
    if not same_tf:
        return np.unique(m["val_selected_track_idx"].to_numpy()), 0
    keep = ~m["sameTF_fell_back_to_val_selected"].astype(bool)
    return np.unique(m.loc[keep, "val_selected_sameTF_track_idx"].astype(int).to_numpy()), int((~keep).sum())


def track_matched_borzoi(results_dir: Path = SCORES, same_tf: bool = False, finalised_only: bool = False) -> pd.DataFrame:
    """Validation-selected-mapping comparator (or same-TF variant): pretrained Borzoi max|log2fc| over the mapped channels (STD columns)."""
    cols, _ = track_matched_borzoi_cols(same_tf)
    lfc = lfc_columns("borzoi_pretrained", cols, results_dir, finalised_only=finalised_only)
    out = summarize_matrix(lfc.to_numpy(np.float64), cols)
    out.index = lfc.index
    out["excl_window"] = load_model("borzoi_pretrained", results_dir, finalised_only).reindex(out.index)["excl_window"].to_numpy()
    return out


# ---------------- AlphaGenome ----------------
def build_ag_cache() -> None:
    """ag-venv. Reads variant_all_scores_per_track/*.parquet, keeps the frozen 33 tracks (modality, track_name, track_strand; scorer_kind diff),
    writes alphagenome_scores.tsv (raw + quantile versions) and colon gene-level RNA-seq (alphagenome_rna_colon.tsv.gz). No labels used."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(AG_TRACKS, sep="\t", dtype=str, keep_default_na=False)
    tkey = set(zip(tr["modality"], tr["track_name"], tr["track_strand"]))
    cols = ["input_variant", "modality", "scorer_kind", "track_name", "track_strand", "gene_id", "gene_symbol", "gtex_tissue",
            "raw_score", "quantile_score"]
    rows, rna = [], []
    for f in sorted((AG_OUT / "variant_all_scores_per_track").glob("*.parquet")):
        d = pd.read_parquet(f, columns=cols)
        dd = d[(d["scorer_kind"] == "diff") & d["modality"].isin(["DNASE", "ATAC", "CHIP_TF"])].copy()
        dd["track_strand"] = dd["track_strand"].fillna(".")
        dd = dd[[k in tkey for k in zip(dd["modality"], dd["track_name"], dd["track_strand"])]]
        n = dd.groupby("input_variant").size()
        assert (n == len(tkey)).all(), f"{f.name}: variants without exactly {len(tkey)} tracks: {n.value_counts().to_dict()}"
        assert not dd.duplicated(["input_variant", "modality", "track_name", "track_strand"]).any()
        dd = dd.sort_values(["input_variant", "modality", "track_name", "track_strand"])
        for ver, col in (("raw", "raw_score"), ("quantile", "quantile_score")):
            x = dd.pivot(index="input_variant", columns=["modality", "track_name", "track_strand"], values=col)
            s = summarize_matrix(x.to_numpy(dtype=np.float64))
            s.insert(0, "S6_CCV_input_key", x.index.to_numpy()); s["version"] = ver
            rows.append(s)
        r = d[(d["modality"] == "RNA_SEQ") & (d["scorer_kind"] == "diff") & d["gtex_tissue"].isin(["Colon_Transverse", "Colon_Sigmoid"])]
        rna.append(r.groupby(["input_variant", "gene_id", "gene_symbol", "gtex_tissue"], as_index=False)[["raw_score", "quantile_score"]].mean())
        print(f"[ag-cache] {f.name}: {dd.input_variant.nunique()} variants", flush=True)
    s = pd.concat(rows)
    st = pd.read_csv(AG_OUT / "variant_run_status.tsv", sep="\t", usecols=["input_variant", "status", "interval_shifted"])
    st = st.rename(columns={"input_variant": "S6_CCV_input_key"})
    s = s.merge(st, on="S6_CCV_input_key", how="left")
    s.to_csv(CACHE / "alphagenome_scores.tsv", sep="\t", index=False)
    pd.concat(rna).rename(columns={"input_variant": "S6_CCV_input_key"}).to_csv(CACHE / "alphagenome_rna_colon.tsv.gz", sep="\t", index=False)
    print(f"[ag-cache] wrote {CACHE}", flush=True)


def load_alphagenome(version: str, cache: Path = CACHE) -> pd.DataFrame:
    """version in {'quantile' (SD-014 headline), 'raw'}."""
    s = pd.read_csv(Path(cache) / "alphagenome_scores.tsv", sep="\t")
    s = s[s["version"] == version].copy()
    assert (s["status"] == "SUCCESS").all()
    s["excl_window"] = s["interval_shifted"].astype(bool)
    s = s.set_index("S6_CCV_input_key")
    assert s.index.is_unique
    return s[STD]


def load_all(results_dir: Path = SCORES, cache: Path = CACHE, models: list | None = None, finalised_only: bool = False) -> dict:
    """models: subset of HEADLINE (default all). Only the listed models (and their own sensitivity variants) are loaded (Amendment A3)."""
    models = list(models) if models is not None else list(HEADLINE)
    bad = [m for m in models if m not in HEADLINE]
    if bad:
        raise ValueError(f"unknown model(s) {bad}; choose from {HEADLINE}")
    out = {}
    if "alphagenome_quantile" in models:
        out["alphagenome_quantile"] = load_alphagenome("quantile", cache)
        out["alphagenome_raw"] = load_alphagenome("raw", cache)          # sensitivity of the same model
    for m in ("borzoi_pretrained", "borzoi_tl", "enformer"):
        if m in models:
            out[m] = load_model(m, results_dir, finalised_only)
    if "borzoi_tl" in models:
        out["borzoi_tl_all183"] = load_model("borzoi_tl_all183", results_dir, finalised_only)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build-ag-cache":
        build_ag_cache()
    else:
        print(__doc__)
