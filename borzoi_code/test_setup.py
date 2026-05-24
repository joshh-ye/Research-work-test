"""
Early-failure diagnostic. Run before training to catch environment issues.
  python test_setup.py
"""

import sys

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        failures.append(name)


# ── 0. Data files ────────────────────────────────────────────────────────────
print("\n[0] Data files  (edit config.py to set paths)")
import os
from config import FASTA, BW_FILES

def _fasta_exists():
    assert os.path.exists(FASTA), f"not found: {FASTA}"
check("FASTA exists", _fasta_exists)

for bw in BW_FILES:
    def _bw_exists(path=bw):
        assert os.path.exists(path), f"not found: {path}"
    check(f"BigWig exists: {os.path.basename(bw)}", _bw_exists)


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1] Imports")

def _torch():
    import torch
check("torch", _torch)

def _numpy():
    import numpy as np
check("numpy", _numpy)

def _pyfaidx():
    import pyfaidx
check("pyfaidx", _pyfaidx)

def _pybigwig():
    import pyBigWig
check("pyBigWig", _pybigwig)

def _borzoi():
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks
check("borzoi_pytorch", _borzoi)


# ── 2. Device ─────────────────────────────────────────────────────────────────
print("\n[2] Device")
import torch

def _mps_available():
    assert torch.backends.mps.is_available(), "MPS not available — will fall back to CPU"
check("MPS available", _mps_available)

def _mps_tensor():
    assert torch.backends.mps.is_available(), "skipped (no MPS)"
    t = torch.tensor([1.0, 2.0], device="mps")
    assert t.device.type == "mps"
check("MPS tensor op", _mps_tensor)

def _device_selection():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"         → selected device: {device}", end="  ")
check("device selection", _device_selection)


# ── 3. sequence_utils ─────────────────────────────────────────────────────────
print("\n[3] sequence_utils")
from sequence_utils import one_hot_encode, reverse_complement

def _ohe_shape():
    arr = one_hot_encode("ACGTN")
    assert arr.shape == (5, 4), f"expected (5,4) got {arr.shape}"
check("one_hot_encode shape", _ohe_shape)

def _ohe_values():
    import numpy as np
    arr = one_hot_encode("ACGT")
    expected = np.eye(4, dtype="float32")
    assert np.allclose(arr, expected), f"unexpected values:\n{arr}"
check("one_hot_encode values (A/C/G/T)", _ohe_values)

def _ohe_n():
    import numpy as np
    arr = one_hot_encode("N")
    assert np.allclose(arr, [[0, 0, 0, 0]]), "N should encode to all zeros"
check("one_hot_encode N → zeros", _ohe_n)

def _rc():
    assert reverse_complement("AACGT") == "ACGTT", \
        f"got {reverse_complement('AACGT')}"
check("reverse_complement", _rc)


# ── 4. bigWigLoader ───────────────────────────────────────────────────────────
print("\n[4] bigWigLoader  (using first .w5 from config)")
from bigwig_loader import bigWigLoader
from config import BW_FILES

_w5 = BW_FILES[:1]  # just one track for speed

def _bw_load_shape():
    loader = bigWigLoader(_w5, bin_size=32)
    arr = loader.load("chr1", 1_000_000, 1_131_072)  # 131072 bp → 4096 bins
    loader.close()
    assert arr.shape == (4096, 1), f"expected (4096, 1) got {arr.shape}"
check("load shape (4096, 1)", _bw_load_shape)

def _bw_load_values():
    import numpy as np
    loader = bigWigLoader(_w5, bin_size=32)
    arr = loader.load("chr1", 1_000_000, 1_131_072)
    loader.close()
    assert arr.dtype == np.float32, f"unexpected dtype {arr.dtype}"
    assert np.all(arr >= 0), "negative signal values"
check("load values non-negative", _bw_load_values)

def _bw_bad_chrom():
    import numpy as np
    loader = bigWigLoader(_w5)
    arr = loader.load("chrFAKE", 0, 32000)
    loader.close()
    assert np.all(arr == 0), "bad chrom should return zeros"
check("bad chrom → zeros (no crash)", _bw_bad_chrom)


# ── 5. dataset shapes ─────────────────────────────────────────────────────────
print("\n[5] Dataset __getitem__ shapes")

def _dataset_shapes():
    import numpy as np
    from unittest.mock import MagicMock
    from dataset import GenomicsDatabase
    from genome_tiler import Interval

    # stub FastaReader so we don't need a real FASTA
    fake_seq = "ACGT" * (524288 // 4)
    mock_fasta = MagicMock()
    mock_fasta.fetch.return_value = fake_seq

    bw_loader = bigWigLoader(_w5, bin_size=32)
    ivs = [Interval("chr1", 1_000_000, 1_524_288)]

    ds = GenomicsDatabase.__new__(GenomicsDatabase)
    ds.fasta         = mock_fasta
    ds.intervals     = ivs
    ds.bigwig_loader = bw_loader
    ds.seq_len       = 524288
    ds.center_bin_size = 131072
    ds.bin_size      = 32

    item = ds[0]
    seq  = item["sequence"]
    tgt  = item["targets"]

    assert seq.shape == (524288, 4), f"sequence shape {seq.shape}"
    assert tgt.shape == (4096, 1),   f"targets shape {tgt.shape}"
    bw_loader.close()

check("sequence (524288, 4) + targets (4096, 1)", _dataset_shapes)


# ── 6. Model head (no backbone load) ─────────────────────────────────────────
print("\n[6] Model head (isolated, skips HuggingFace download)")

def _head_forward():
    import torch.nn as nn
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    head = nn.Sequential(nn.Linear(7611, 2), nn.Softplus()).to(device)
    x    = torch.rand(4096, 7611, device=device)
    out  = head(x)
    assert out.shape == (4096, 2), f"expected (4096, 2) got {out.shape}"
    assert (out >= 0).all(), "Softplus output should be non-negative"
check("head forward on selected device", _head_forward)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"  {len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
else:
    print("  All checks passed.")
