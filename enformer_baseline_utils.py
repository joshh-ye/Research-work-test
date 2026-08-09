"""
Enformer-specific utilities for the original-Enformer best-match baseline.

Mirrors borzoi_baseline_utils.py, which stays the home of everything that is
model-agnostic (StreamingPearson, per_track_pearson, statistics, plausibility).
Only the pieces that genuinely differ for Enformer live here:

  * track metadata format (tab-separated targets_human.txt, `index` column)
  * eligible-track filtering (assay prefix instead of a hand-picked index range)
  * resolution / span bookkeeping, since Enformer and the CRC targets do not
    share a bin size or a window length.

Resolution note (the crux of a fair comparison)
-----------------------------------------------
The CRC targets are 4096 bins x 32 bp = 131,072 bp centered on each interval.
Enformer consumes 196,608 bp and predicts 896 bins x 128 bp = 114,688 bp, also
centered. So the Enformer output span is a strict subset of the target span. We
therefore compare on Enformer's grid: crop the targets to the central 3584 bins
(114,688 bp) and mean-pool groups of 4 to reach 128 bp bins. Mean-pooling is the
right aggregation because both signals are coverage means, not counts.

No torch import here, so this module is importable on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from borzoi_baseline_utils import is_histone, parse_borzoi_description

# --- Enformer geometry -----------------------------------------------------
ENF_SEQ_LEN   = 196_608   # model input length
ENF_BINS      = 896       # model output bins
ENF_BIN_SIZE  = 128       # bp per output bin
ENF_SPAN      = ENF_BINS * ENF_BIN_SIZE          # 114,688 bp predicted

# --- CRC target geometry (from the transfer-model pipeline) ----------------
TGT_BINS      = 4096
TGT_BIN_SIZE  = 32
TGT_SPAN      = TGT_BINS * TGT_BIN_SIZE          # 131,072 bp

POOL          = ENF_BIN_SIZE // TGT_BIN_SIZE     # 4 target bins per Enformer bin
TGT_CROP_BINS = ENF_SPAN // TGT_BIN_SIZE         # 3584 target bins kept
TGT_CROP_OFF  = (TGT_BINS - TGT_CROP_BINS) // 2  # 256 bins dropped each side

assert TGT_CROP_BINS % POOL == 0
assert TGT_CROP_BINS // POOL == ENF_BINS

# Enformer's human head assays, as they appear at the head of `description`.
ELIGIBLE_ASSAY = "CHIP"


def load_enformer_targets(targets_txt: str | Path) -> pd.DataFrame:
    """Load the Basenji/Enformer human targets file (tab-separated).

    `index` is 0-based and equals the human-head output channel; we validate
    that it also equals row position, the same invariant the Borzoi loader
    checks, because every downstream selection indexes the model output by it.
    """
    df = pd.read_csv(targets_txt, sep="\t")
    if "index" not in df.columns:
        raise ValueError("expected an `index` column in the Enformer targets file")
    df = df.rename(columns={"index": "track_index"})
    if not (df["track_index"].to_numpy() == np.arange(len(df))).all():
        raise ValueError("track_index does not equal row position — indexing assumption broken")
    return df


def filter_enformer_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep non-histone ChIP tracks; return (included, excluded) with reasons.

    Unlike Borzoi, no index range is applied: Enformer's targets file is already
    sorted by assay and the assay prefix identifies ChIP unambiguously, so the
    range heuristic the Borzoi baseline needed would only add a second, weaker
    way of saying the same thing.
    """
    out = df.copy()
    parsed = out["description"].apply(parse_borzoi_description)
    out["assay"]       = [p["assay"].upper() for p in parsed]
    out["factor"]      = [p["factor"] for p in parsed]
    out["cell_tissue"] = [p["cell_tissue"] for p in parsed]

    is_chip = out["assay"] == ELIGIBLE_ASSAY
    histone = out["description"].apply(is_histone)

    included = out[is_chip & histone.isna()].copy()
    included["reason"] = "ChIP, non-histone"

    excluded = out[~(is_chip & histone.isna())].copy()
    reasons = []
    for _, row in excluded.iterrows():
        if row["assay"] != ELIGIBLE_ASSAY:
            reasons.append(f"not ChIP: {row['assay']}")
        else:
            reasons.append(f"histone mark: {is_histone(row['description'])}")
    excluded["reason"] = reasons
    return included, excluded


def targets_to_enformer_grid(targets: np.ndarray) -> np.ndarray:
    """Crop + mean-pool CRC targets onto Enformer's 896 x 128 bp grid.

    targets: (..., 4096, n_tracks) -> (..., 896, n_tracks). Accepts a single
    interval (2-D) or a stack (3-D), and works on a memory-mapped slice.
    """
    arr = np.asarray(targets)
    if arr.shape[-2] != TGT_BINS:
        raise ValueError(f"expected {TGT_BINS} target bins, got {arr.shape[-2]}")
    cropped = arr[..., TGT_CROP_OFF: TGT_CROP_OFF + TGT_CROP_BINS, :]
    n_tracks = cropped.shape[-1]
    reshaped = cropped.reshape(*cropped.shape[:-2], ENF_BINS, POOL, n_tracks)
    return reshaped.astype(np.float64).mean(axis=-2)
