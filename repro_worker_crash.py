"""
Minimal reproduction of the DataLoader worker crash.

The full training run processes only chr2-chr7, chr10-chr21, chrX, chrY
(everything except chr1, chr8, chr9, chr22).  Validation is the FIRST time
any worker tries to call bw.stats("chr1", ...) or bw.stats("chr8", ...).

Two failure modes reproduced here without loading the Borzoi model:

  Mode A -- hidden exception in _open():
    BigWigLoader._open() is called lazily inside __getitem__.
    Any exception it raises propagates UNHANDLED through the worker's
    _worker_loop.  PyTorch catches Python exceptions and re-raises them,
    but a C-level crash (SIGSEGV from libBigWig on malformed data) kills
    the process outright.  filter_bad_bigwigs only calls bw.chroms(), so
    a file that is broken only in its chr1/chr8 data passes the filter.

  Mode B -- too many simultaneous file opens:
    8 spawn workers each open all 183 BigWig handles on their first item.
    That is 8 x 183 = 1464 concurrent opens against the network filesystem.
    Under filesystem pressure this can stall past PyTorch's 5-second
    watchdog, which then kills any workers that have not yet responded.

Usage:
    source torch-venv/bin/activate
    python repro_worker_crash.py          # tests val chromosomes (chr1/chr8)
    python repro_worker_crash.py --train  # tests train chromosomes as control
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))

from bigwig_loader import BigWigLoader
from dataset import GenomicDataset
from genome_tiler import tile_genome


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./borzoi_data")
    p.add_argument("--n-bw", type=int, default=None,
                   help="Limit BigWig tracks (default: all 183)")
    p.add_argument("--n-intervals", type=int, default=20,
                   help="Number of intervals to test (default: 20)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--train", action="store_true",
                   help="Use train split (control) instead of val split")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")

    bw_files = sorted(glob.glob(str(data_root / "CRC_TFs_bw" / "*.bw")))
    if args.n_bw:
        bw_files = bw_files[: args.n_bw]
    print(f"BigWig tracks : {len(bw_files)}")
    print(f"num_workers   : {args.num_workers}")

    splits = tile_genome(fasta_path)
    split_name = "train" if args.train else "val"
    intervals = splits[split_name][: args.n_intervals]
    chroms = sorted({iv.chrom for iv in intervals})
    print(f"Split         : {split_name}  ({len(intervals)} intervals, chroms={chroms})")

    # Fresh dataset — handles are None, opened lazily inside each worker.
    dataset = GenomicDataset(
        fasta_path,
        intervals,
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
    )

    # Reproduce the exact DataLoader config used in train_resumable.
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    print("\nStarting DataLoader iteration — if workers crash you will see:")
    print("  RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly\n")

    try:
        for i, batch in enumerate(tqdm(loader, total=len(intervals))):
            if i == 0:
                print(f"  First batch OK: seq={batch['sequence'].shape}, "
                      f"targets={batch['targets'].shape}")
        print("\nAll batches loaded successfully.")
    except RuntimeError as e:
        print(f"\nCRASH REPRODUCED:\n  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
