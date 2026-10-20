"""Shared helpers for Priority D (E4 ridge, E5 seeds/folds, E8). hg38; tiles = borzoi_code.genome_tiler logic
(524,288 bp non-overlapping windows from 0); targets = 183 BigWig tracks at 32 bp over the central 131,072 bp.
Cache layout: see 01_build_cache.py."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research")
CACHE = ROOT / "results_phase2/trunk_cache"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "borzoi_code")); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))

SEQ_LEN = 524288
CENTER_BINS = 4096          # 32 bp bins, central 131,072 bp
MARGIN = 4                  # extra 32 bp bins each side cached (shift augmentation +-128 bp = +-4 bins)
CBINS = CENTER_BINS + 2 * MARGIN   # 4104
EMB_DIM = 1920
N_TGT = 183
N_HUMAN = 7611
# shared 128 bp grid (identical to enformer_baseline_utils.targets_to_enformer_grid): crop 256 bins/side, mean-pool x4
CROP, POOL, G_BINS = 256, 4, 896


def to_grid128(a):
    """(..., 4096, C) at 32 bp -> (..., 896, C) at 128 bp. float64 mean-pool, same as targets_to_enformer_grid."""
    a = np.asarray(a)
    assert a.shape[-2] == CENTER_BINS, a.shape
    c = a[..., CROP:CROP + G_BINS * POOL, :]
    return c.reshape(*c.shape[:-2], G_BINS, POOL, c.shape[-1]).astype(np.float64).mean(-2)


def central(a_margin):
    """(..., 4104, C) -> central (..., 4096, C)."""
    return a_margin[..., MARGIN:MARGIN + CENTER_BINS, :]


def tile_table():
    """DataFrame of all tiles: gid, split, split_idx, chrom, start, end (train, val, test order as tile_genome)."""
    import pandas as pd
    from genome_tiler import tile_genome
    sp = tile_genome(str(ROOT / "borzoi_data/hg38/hg38.ml.fa"))
    rows = []
    for s in ["train", "val", "test"]:
        for i, iv in enumerate(sp[s]):
            rows.append((s, i, iv.chrom, iv.start, iv.end))
    df = pd.DataFrame(rows, columns=["split", "split_idx", "chrom", "start", "end"])
    df.insert(0, "gid", np.arange(len(df)))
    return df


def tile_path(gid, kind):
    return CACHE / kind / f"t{gid:05d}.npy"


def bw_bin_means(bw, chrom, start, end, bin_size=32):
    """Mean over covered bases per bin, 0 where a bin has no covered base. Equivalent (max rel. err ~1e-7, tested) to
    pyBigWig stats(type='mean', nBins) as used by borzoi_code/bigwig_loader.py, but ~50x faster via bw.intervals()."""
    n = end - start
    nb = n // bin_size
    iv = bw.intervals(chrom, start, end) or []
    v = np.full(n, np.nan)
    for a, b, x in iv:
        v[max(a, start) - start:min(b, end) - start] = x
    v = v.reshape(nb, bin_size)
    cov = np.isfinite(v).sum(1)
    s = np.nansum(v, 1)
    return np.where(cov > 0, s / np.maximum(cov, 1), 0.0).astype(np.float32)
