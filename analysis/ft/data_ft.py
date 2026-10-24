"""Sequence-reading dataset for Borzoi fine-tuning.  hg38 (borzoi_data/hg38/hg38.ml.fa), 0-based half-open tile coordinates from
results_phase2/trunk_cache/tiles.csv (same table/order as borzoi_code.genome_tiler.tile_genome; hash recorded in checkpoints).

Sequence: window [center - 262144, center + 262144) with center = tile centre + shift*32 bp (true shift, whole bins, +-4 bins =
+-128 bp), upper-cased, out-of-chromosome padded with N (all-zero one-hot, as FastaReader / one_hot_encode).  Encoded as uint8
codes A0 C1 G2 T3 N4 (one-hot on the GPU).  RC: reverse + complement (3-c; N stays) and the 4,096 target bins reversed
(as borzoi_code/dataset.py).
Targets: results_phase2/trunk_cache/tgt/tNNNNN.npy (4104 x 183 float32, = central 4096 bins + 4 margin bins/side of the UNSHIFTED
tile; READ ONLY).  Shifted window = rows [4 + shift, 4 + shift + 4096).  Cache targets were verified == original BigWig loader
(analysis/priorityD/tests/test_cache.py; and analysis/ft/tests/test_ft_data.py re-checks with shifted windows).
Per-epoch order / shift / RC are a pure function of (seed, epoch) (same recipe as analysis/priorityD/headlib.epoch_jobs) so that
resume inside an epoch is exact and independent of DataLoader worker RNG.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "borzoi_code"))
FASTA = ROOT / "borzoi_data/hg38/hg38.ml.fa"
CACHE = ROOT / "results_phase2/trunk_cache"
TILES_CSV = CACHE / "tiles.csv"
SEQ_LEN, CENTER_BINS, MARGIN, BIN = 524288, 4096, 4, 32

_LUT = np.full(256, 4, np.uint8)
for i, b in enumerate(b"ACGT"):
    _LUT[b] = i
    _LUT[b + 32] = i          # lower case (fetch upper-cases anyway)


def load_tiles(limit_train=0, limit_val=0):
    df = pd.read_csv(TILES_CSV)
    return df


def tiles_hash() -> str:
    return hashlib.sha256(TILES_CSV.read_bytes()).hexdigest()[:16]


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii"), np.uint8)]


def revcomp_codes(c: np.ndarray) -> np.ndarray:
    return np.where(c < 4, 3 - c, c)[::-1].copy()


def epoch_jobs(gids, epoch, seed, rc_prob=0.5, max_shift_bins=4):
    """List of (gid, rc, shift_bins) - same RNG recipe as analysis/priorityD/headlib.epoch_jobs."""
    rs = np.random.RandomState((seed * 1000003 + epoch * 7919) % (2 ** 31 - 1))
    order = rs.permutation(len(gids))
    shifts = rs.randint(-max_shift_bins, max_shift_bins + 1, len(gids)) if max_shift_bins > 0 else np.zeros(len(gids), int)
    rcs = rs.rand(len(gids)) < rc_prob
    return [(int(gids[o]), bool(rcs[k]), int(shifts[k])) for k, o in enumerate(order)]


class TileJobs(Dataset):
    """jobs: list of (gid, rc, shift_bins). Returns dict(codes uint8 (L,), tgt float32 (4096,183), gid)."""

    def __init__(self, tiles: pd.DataFrame, jobs, with_targets=True):
        self.t = tiles.set_index("gid")
        self.jobs, self.with_targets = list(jobs), with_targets
        self.fa = None

    def __len__(self):
        return len(self.jobs)

    def window(self, gid, shift):
        r = self.t.loc[gid]
        return r.chrom, (int(r.start) + int(r.end)) // 2 + shift * BIN

    def seq_codes(self, gid, rc, shift):
        from fasta_reader import FastaReader
        if self.fa is None:
            self.fa = FastaReader(str(FASTA))
        chrom, center = self.window(gid, shift)
        c = encode(self.fa.fetch(chrom, center - SEQ_LEN // 2, center + SEQ_LEN // 2))
        return revcomp_codes(c) if rc else c

    def target(self, gid, rc, shift):
        t = np.load(CACHE / "tgt" / f"t{gid:05d}.npy", mmap_mode="r")
        lo = MARGIN + shift
        t = np.array(t[lo:lo + CENTER_BINS], dtype=np.float32)
        return t[::-1].copy() if rc else t

    def __getitem__(self, i):
        gid, rc, shift = self.jobs[i]
        d = dict(codes=torch.from_numpy(self.seq_codes(gid, rc, shift)), gid=gid)
        if self.with_targets:
            d["tgt"] = torch.from_numpy(self.target(gid, rc, shift))
        return d
