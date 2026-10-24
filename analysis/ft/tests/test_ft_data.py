"""Data path of train_ft == original loader (borzoi_code GenomicDataset + BigWigLoader). Run: ./torch-venv/bin/python -m pytest -q analysis/ft/tests/test_ft_data.py"""
import json, sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/ft")); sys.path.insert(0, str(ROOT / "borzoi_code"))
import data_ft as D
from dataset import GenomicDataset
from genome_tiler import Interval
from bigwig_loader import BigWigLoader
from sequence_utils import reverse_complement, one_hot_encode

tiles = D.load_tiles()
SEL = np.arange(0, 183, 12)
BW = [str(ROOT / b) for b in json.load(open(ROOT / "borzoi_ckpt_full/bw_files.json"))]


def orig(gid, shift):
    r = tiles.set_index("gid").loc[gid]
    ds = GenomicDataset(str(D.FASTA), [Interval(r.chrom, int(r.start) + shift * 32, int(r.end) + shift * 32)],
                        bigwig_loader=BigWigLoader([BW[i] for i in SEL]), training=False)
    return ds[0]


def onehot(codes):
    oh = np.zeros((len(codes), 4), np.float32); m = codes < 4
    oh[np.where(m)[0], codes[m]] = 1
    return oh


def test_tiles_split_counts():
    vc = tiles.split.value_counts().to_dict()
    assert vc == {"train": 4659, "val": 750, "test": 359}
    assert set(tiles[tiles.split == "val"].chrom) == {"chr1", "chr8"} and set(tiles[tiles.split == "test"].chrom) == {"chr9", "chr22"}


@pytest.mark.parametrize("gid,shift", [(120, -3), (120, 4), (0, -4)])   # (0,-4): window runs off the chromosome start -> N padding
def test_seq_and_targets_equal_original(gid, shift):
    o = orig(gid, shift)
    ds = D.TileJobs(tiles, [])
    assert np.array_equal(onehot(ds.seq_codes(gid, False, shift)), o["sequence"].numpy())
    assert np.array_equal(ds.target(gid, False, shift)[:, SEL], o["targets"].numpy())


def test_rc_matches_original_convention():
    gid, shift = 120, 2
    o = orig(gid, shift)
    ds = D.TileJobs(tiles, [])
    seq_oh = o["sequence"].numpy()
    # original RC: reverse_complement(str) and reverse target bins
    s = "".join("ACGT"[i] if row.any() else "N" for i, row in zip(seq_oh.argmax(1), seq_oh))
    assert np.array_equal(onehot(ds.seq_codes(gid, True, shift)), one_hot_encode(reverse_complement(s)))
    assert np.array_equal(ds.target(gid, True, shift)[:, SEL], o["targets"].numpy()[::-1])


def test_epoch_jobs_deterministic_and_shift_range():
    g = np.arange(100)
    a, b = D.epoch_jobs(g, 3, 7), D.epoch_jobs(g, 3, 7)
    assert a == b and a != D.epoch_jobs(g, 4, 7)
    assert all(-4 <= s <= 4 for _, _, s in a) and sorted(x[0] for x in a) == list(range(100))
