"""
Shared utilities for the filtered-Borzoi baseline and the transfer-vs-Borzoi
comparison. Refactored out of baseline_borzoi.py so the filtering, correlation,
best-match, metadata and statistics logic lives in one reusable place.

No torch import here — this module is importable on a laptop for the analysis /
figure steps. The Borzoi model inference itself lives in baseline_borzoi_filtered.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — the transfer model and CRC targets are 32 bp bins over the
# 131,072 bp center window (4096 bins); Borzoi takes a 524,288 bp input.
# ---------------------------------------------------------------------------
CENTER_BINS = 4096
SEQ_LEN     = 524288

# Borzoi output-channel range searched for CRC-relevant ChIP tracks (inclusive).
INDEX_LO = 2186
INDEX_HI = 6069

# Histone-mark terms to exclude (case-insensitive). We want TF / DNA-binding
# ChIP, not histone-modification assays. Kept as an ordered list so the *first*
# matching term can be logged as the exclusion reason (auditable, not silent).
HISTONE_TERMS = [
    "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K27me3", "H3K36me3",
    "H3K9ac", "H3K9me3", "H3K79me2", "H3K9me2", "H3K4ac", "H3K56ac",
    "H2AK", "H2BK", "H4K", "H3K", "H2A.Z", "H2AFZ",
    # generic single-letter histone families as a last-resort catch:
    "H3", "H4", "H2A", "H2B",
]
_HISTONE_RE = re.compile("|".join(re.escape(t) for t in HISTONE_TERMS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Metadata loading / parsing
# ---------------------------------------------------------------------------
def load_borzoi_targets(targets_csv: str | Path) -> pd.DataFrame:
    """Load targets_human.csv. `track_index` is 0-based and equals the row
    position / Borzoi output channel. We validate that invariant."""
    df = pd.read_csv(targets_csv)
    assert "track_index" in df.columns, "expected a track_index column"
    if not (df["track_index"].to_numpy() == np.arange(len(df))).all():
        raise ValueError("track_index does not equal row position — indexing assumption broken")
    return df


def parse_borzoi_description(desc: str) -> dict:
    """Parse a Borzoi description like 'CHIP:POLR2A:Raji treated with ...' into
    assay / factor / cell_tissue components (best effort; genomics descriptions
    are not perfectly structured)."""
    parts = [p.strip() for p in str(desc).split(":")]
    assay = parts[0] if parts else ""
    factor = parts[1] if len(parts) > 1 else ""
    cell = ":".join(parts[2:]).strip() if len(parts) > 2 else ""
    return {"assay": assay, "factor": factor, "cell_tissue": cell}


def is_histone(desc: str) -> str | None:
    """Return the matched histone term (exclusion reason) or None."""
    m = _HISTONE_RE.search(str(desc))
    return m.group(0) if m else None


def filter_borzoi_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the filtering strategy. Returns (included, excluded) frames, each
    with parsed metadata columns and an inclusion/exclusion reason."""
    in_range = df[(df["track_index"] >= INDEX_LO) & (df["track_index"] <= INDEX_HI)].copy()

    parsed = in_range["description"].apply(parse_borzoi_description)
    in_range["assay"]       = [p["assay"] for p in parsed]
    in_range["factor"]      = [p["factor"] for p in parsed]
    in_range["cell_tissue"] = [p["cell_tissue"] for p in parsed]

    is_chip   = in_range["description"].str.contains("chip", case=False, na=False)
    histone   = in_range["description"].apply(is_histone)

    included_mask = is_chip & histone.isna()

    included = in_range[included_mask].copy()
    included["reason"] = "ChIP, non-histone, in 2186-6069"

    excluded = in_range[~included_mask].copy()
    excl_reason = []
    for _, row in excluded.iterrows():
        if not str(row["description"]).lower().__contains__("chip"):
            excl_reason.append("not ChIP")
        else:
            h = is_histone(row["description"])
            excl_reason.append(f"histone mark: {h}" if h else "excluded")
    excluded["reason"] = excl_reason
    return included, excluded


def load_crc_target_meta(bw_filenames: list[str], tf_csv: str | Path) -> pd.DataFrame:
    """Map each CRC target (by bw filename '<DCid>.bw') to its TF_184tracks.csv row.
    Returns a frame indexed by target position with Factor / Cell_line / Tissue."""
    tf = pd.read_csv(tf_csv)
    dcid_col = "DCid"
    rows = []
    for i, fname in enumerate(bw_filenames):
        dcid = int(Path(str(fname)).stem)
        match = tf[tf[dcid_col] == dcid]
        if len(match) == 0:
            rows.append({"target_idx": i, "bw_filename": fname, "DCid": dcid,
                         "Factor": "", "Cell_line": "", "Cell_type": "", "Tissue_type": ""})
        else:
            m = match.iloc[0]
            rows.append({
                "target_idx": i, "bw_filename": fname, "DCid": dcid,
                "Factor": m.get("Factor", ""), "Cell_line": m.get("Cell_line", ""),
                "Cell_type": m.get("Cell_type", ""), "Tissue_type": m.get("Tissue_type", ""),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streaming Pearson (moved verbatim from baseline_borzoi.py) — computes Pearson
# R between every (target_track, borzoi_track) pair without holding all preds.
# ---------------------------------------------------------------------------
class StreamingPearson:
    def __init__(self, n_targets: int, n_borzoi: int):
        self.n   = 0
        self.sx  = np.zeros(n_targets, dtype=np.float64)
        self.sy  = np.zeros(n_borzoi,  dtype=np.float64)
        self.sxy = np.zeros((n_targets, n_borzoi), dtype=np.float64)
        self.sx2 = np.zeros(n_targets, dtype=np.float64)
        self.sy2 = np.zeros(n_borzoi,  dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n   += x.shape[0]
        self.sx  += x.sum(axis=0)
        self.sy  += y.sum(axis=0)
        self.sxy += x.T @ y
        self.sx2 += (x ** 2).sum(axis=0)
        self.sy2 += (y ** 2).sum(axis=0)

    def result(self) -> np.ndarray:
        n = self.n
        num   = n * self.sxy - np.outer(self.sx, self.sy)
        denom = np.sqrt(
            np.maximum(n * self.sx2 - self.sx ** 2, 0)[:, None] *
            np.maximum(n * self.sy2 - self.sy ** 2, 0)[None, :]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom > 0, num / denom, 0.0)
        return r.astype(np.float32)


def per_track_pearson(preds: np.ndarray, targets: np.ndarray,
                      chunk: int = 100) -> np.ndarray:
    """Flattened per-track Pearson R (same convention as StreamingPearson): each
    track correlated over all interval*bin observations. preds/targets:
    (n_intervals, n_bins, n_tracks).

    Accumulated in float64 over interval chunks so peak memory stays ~one chunk
    rather than upcasting the whole (multi-GB) array at once."""
    n_int, n_bins, n_tracks = preds.shape
    sx = np.zeros(n_tracks); sy = np.zeros(n_tracks)
    sxy = np.zeros(n_tracks); sx2 = np.zeros(n_tracks); sy2 = np.zeros(n_tracks)
    n = 0
    for lo in range(0, n_int, chunk):
        p = preds[lo:lo + chunk].reshape(-1, n_tracks).astype(np.float64)
        t = targets[lo:lo + chunk].reshape(-1, n_tracks).astype(np.float64)
        n   += p.shape[0]
        sx  += p.sum(0); sy  += t.sum(0)
        sxy += (p * t).sum(0)
        sx2 += (p * p).sum(0); sy2 += (t * t).sum(0)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sx2 - sx ** 2, 0) * np.maximum(n * sy2 - sy ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


# ---------------------------------------------------------------------------
# Best-match selection from a (n_targets, n_eligible) Pearson matrix
# ---------------------------------------------------------------------------
def select_best_match(
    pmat: np.ndarray,
    eligible_track_indices: list[int],
    borzoi_meta: pd.DataFrame,
    target_meta: pd.DataFrame,
) -> pd.DataFrame:
    """For each CRC target row, pick the eligible Borzoi track with highest R.
    Records rank (=1 for the best) and #eligible evaluated. borzoi_meta indexed
    by track_index; target_meta gives per-target Factor/cell metadata."""
    n_targets, n_elig = pmat.shape
    assert n_elig == len(eligible_track_indices)
    best_local = pmat.argmax(axis=1)
    best_r     = pmat.max(axis=1)
    # rank of the selected track among eligible (1 = highest); always 1 here but
    # kept explicit so the column is meaningful if selection logic changes.
    order = (-pmat).argsort(axis=1)
    rank_of_best = np.array([int(np.where(order[i] == best_local[i])[0][0]) + 1
                             for i in range(n_targets)])

    rows = []
    for i in range(n_targets):
        bidx = eligible_track_indices[best_local[i]]
        bmeta = borzoi_meta.loc[bidx]
        tmeta = target_meta.iloc[i] if i < len(target_meta) else {}
        rows.append({
            "target_idx": i,
            "target_name": tmeta.get("Factor", "") if hasattr(tmeta, "get") else "",
            "target_cell_line": tmeta.get("Cell_line", "") if hasattr(tmeta, "get") else "",
            "target_tissue": tmeta.get("Tissue_type", "") if hasattr(tmeta, "get") else "",
            "bw_filename": tmeta.get("bw_filename", "") if hasattr(tmeta, "get") else "",
            "best_borzoi_track_idx": bidx,
            "best_borzoi_identifier": bmeta.get("identifier", ""),
            "best_borzoi_description": bmeta.get("description", ""),
            "best_borzoi_factor": bmeta.get("factor", ""),
            "best_borzoi_cell_tissue": bmeta.get("cell_tissue", ""),
            "pearson_r": float(best_r[i]),
            "rank_of_selected": int(rank_of_best[i]),
            "n_eligible_evaluated": int(n_elig),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_model_summary(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    lo, hi = bootstrap_ci(r, np.mean)
    n = len(r)
    return {
        "n": n, "mean": float(r.mean()), "median": float(np.median(r)),
        "sd": float(r.std(ddof=1)), "iqr": float(np.subtract(*np.percentile(r, [75, 25]))),
        "min": float(r.min()), "max": float(r.max()),
        "ci95_lo": lo, "ci95_hi": hi,
        "pct_r_gt_0":  float((r > 0).mean() * 100),
        "pct_r_ge_0.3": float((r >= 0.3).mean() * 100),
        "pct_r_ge_0.5": float((r >= 0.5).mean() * 100),
        "pct_r_ge_0.7": float((r >= 0.7).mean() * 100),
    }


def paired_comparison(transfer_r: np.ndarray, borzoi_r: np.ndarray,
                      tie_tol: float = 0.01) -> dict:
    from scipy import stats
    a = np.asarray(transfer_r, float); b = np.asarray(borzoi_r, float)
    d = a - b
    n = len(d)
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    cohens_d = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    lo, hi = bootstrap_ci(d, np.mean)
    return {
        "n": n, "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "diff_ci95_lo": lo, "diff_ci95_hi": hi,
        "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "cohens_d_paired": cohens_d,
        "n_transfer_wins": int((d > tie_tol).sum()),
        "n_borzoi_wins":   int((d < -tie_tol).sum()),
        "n_tie":           int((np.abs(d) <= tie_tol).sum()),
        "pct_transfer_wins": float((d > tie_tol).mean() * 100),
        "pct_borzoi_wins":   float((d < -tie_tol).mean() * 100),
        "pct_tie":           float((np.abs(d) <= tie_tol).mean() * 100),
    }


# TF family grouping for a light biological-plausibility heuristic.
_TF_FAMILY_PREFIXES = ["FOX", "GATA", "HOX", "SOX", "KLF", "STAT", "IRF", "SMAD",
                       "TCF", "LEF", "NFK", "NR", "ZNF", "E2F", "ETS", "ELF",
                       "CEBP", "JUN", "FOS", "MYC", "RUNX", "TEAD", "TP", "SP"]


def _family(factor: str) -> str:
    f = str(factor).upper()
    for p in _TF_FAMILY_PREFIXES:
        if f.startswith(p):
            return p
    return f[:3]  # crude fallback


def classify_plausibility(target_factor: str, borzoi_factor: str,
                          target_tissue: str, borzoi_cell: str) -> str:
    tf_t = str(target_factor).upper().strip()
    tf_b = str(borzoi_factor).upper().strip()
    if not tf_b or tf_b in ("", "NAN"):
        return "unclear from metadata"
    if tf_t and tf_t == tf_b:
        return "same TF"
    if tf_t and _family(tf_t) == _family(tf_b) and _family(tf_t):
        return "related TF/family"
    if tf_t and tf_t != tf_b:
        return "different TF"
    return "unclear from metadata"
