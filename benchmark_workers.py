"""
Benchmark DataLoader num_workers for BigWig I/O throughput.

The bottleneck being tested is BigWigLoader.load() — reading signal from
184 BigWig files per genomic interval. Window tiling (tile_genome) is fast
arithmetic and is not the bottleneck.

Run from the project root:
    python benchmark_workers.py
    python benchmark_workers.py --n-intervals 20 --max-workers 64
"""

import argparse
import glob
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))

from bigwig_loader import BigWigLoader
from dataset import GenomicDataset
from genome_tiler import tile_genome
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    default="./borzoi_data")
    p.add_argument("--n-intervals",  type=int, default=10,
                   help="Intervals to load per trial (more = more stable result)")
    p.add_argument("--max-workers",  type=int, default=64,
                   help="Highest num_workers value to test")
    p.add_argument("--n-bw",         type=int, default=None,
                   help="Limit to first N BigWig files (default: all)")
    return p.parse_args()


def run_trial(fasta_path, intervals, bw_files, num_workers):
    loader = BigWigLoader(bw_files, bin_size=32)
    ds = GenomicDataset(fasta_path, intervals, bigwig_loader=loader)
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers, shuffle=False)

    t0 = time.perf_counter()
    n = sum(1 for _ in tqdm(dl, desc=f"  workers={num_workers}", leave=False))
    elapsed = time.perf_counter() - t0

    loader.close()
    return elapsed, n


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")

    bw_files = sorted(glob.glob(str(data_root / "CRC_TFs_bw" / "*.bw")))
    if not bw_files:
        sys.exit(f"No BigWig files found under {data_root}/CRC_TFs_bw/")
    if args.n_bw:
        bw_files = bw_files[: args.n_bw]
    print(f"BigWig files : {len(bw_files)}")

    splits = tile_genome(fasta_path)
    intervals = splits["train"][: args.n_intervals]
    print(f"Intervals    : {len(intervals)}")
    print(f"Max workers  : {args.max_workers}\n")

    # build worker counts: 0, 1, 2, 4, 8, ... up to max_workers
    candidates = [0]
    w = 1
    while w <= args.max_workers:
        candidates.append(w)
        w *= 2
    if args.max_workers not in candidates:
        candidates.append(args.max_workers)

    results = []
    for nw in candidates:
        print(f"  num_workers={nw:3d} ...", end="", flush=True)
        elapsed, n = run_trial(fasta_path, intervals, bw_files, nw)
        tp = n / elapsed
        print(f"  {elapsed:6.1f}s  |  {tp:.3f} intervals/s")
        results.append((nw, elapsed, tp))

    print("\n── Results ──────────────────────────────────────")
    print(f"{'workers':>8}  {'time(s)':>8}  {'intervals/s':>12}")
    for nw, elapsed, tp in results:
        print(f"{nw:>8}  {elapsed:>8.1f}  {tp:>12.3f}")

    best = max(results, key=lambda x: x[2])
    print(f"\nOptimal num_workers: {best[0]}  ({best[2]:.3f} intervals/s)")


if __name__ == "__main__":
    main()
