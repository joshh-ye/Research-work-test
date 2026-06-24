"""
Benchmark DataLoader num_workers throughput, memory, and CPU utilization.

Usage:
    python benchmark_workers.py [--data-root ./borzoi_data] [--n-bw 10]
                                [--n-batches 50] [--workers 0 1 2 4 8]
                                [--batch-size 1]
"""

from __future__ import annotations

import argparse
import glob
import os
import resource
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "hf_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))

from bigwig_loader import BigWigLoader
from dataset import GenomicDataset
from genome_tiler import tile_genome

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("psutil not available — CPU % column will be skipped\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DataLoader num_workers")
    p.add_argument("--data-root",  default="./borzoi_data")
    p.add_argument("--n-bw",       type=int, default=10,
                   help="Number of BigWig tracks to use")
    p.add_argument("--n-batches",  type=int, default=50,
                   help="Batches to time per workers config")
    p.add_argument("--workers",    type=int, nargs="+", default=[0, 1, 2, 4, 8, 16],
                   help="num_workers values to test")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def cpu_sampler(samples: list[float], stop: threading.Event) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        samples.append(proc.cpu_percent(interval=None))
        time.sleep(0.5)


def run_config(
    fasta_path: str,
    bw_files: list[str],
    intervals: list,
    num_workers: int,
    batch_size: int,
    n_batches: int,
) -> tuple[float, float, float | None]:
    """Returns (batches_per_sec, peak_rss_mb, avg_cpu_pct)."""

    # Fresh dataset each call — spawn can't pickle open file handles.
    dataset = GenomicDataset(
        fasta_path,
        intervals,
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
        training=False,
    )

    dl_context = "spawn" if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        multiprocessing_context=dl_context,
    )

    cpu_samples: list[float] = []
    stop_event = threading.Event()

    if HAVE_PSUTIL:
        psutil.Process().cpu_percent(interval=None)  # prime the counter
        t = threading.Thread(target=cpu_sampler, args=(cpu_samples, stop_event), daemon=True)
        t.start()

    t0 = time.perf_counter()
    count = 0
    for batch in tqdm(loader, total=n_batches, desc=f"workers={num_workers}", leave=False):
        count += 1
        if count >= n_batches:
            break
    elapsed = time.perf_counter() - t0

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # On Linux ru_maxrss is in KB; on macOS it's in bytes
    if sys.platform == "darwin":
        peak_rss_mb = peak_rss_mb / 1024

    stop_event.set()
    avg_cpu = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None

    throughput = count / elapsed if elapsed > 0 else 0.0
    return throughput, peak_rss_mb, avg_cpu


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")

    bw_files = sorted(glob.glob(str(data_root / "CRC_TFs_bw" / "*.bw")))
    if not bw_files:
        sys.exit(f"No BigWig files found under {data_root}/CRC_TFs_bw/")
    bw_files = bw_files[: args.n_bw]
    print(f"Using {len(bw_files)} BigWig tracks, {args.n_batches} batches per config\n")

    splits = tile_genome(fasta_path)
    intervals = splits["train"]

    results: list[tuple[int, float, float, float | None]] = []

    for nw in args.workers:
        print(f"Testing num_workers={nw} ...")
        throughput, rss, cpu = run_config(fasta_path, bw_files, intervals, nw, args.batch_size, args.n_batches)
        results.append((nw, throughput, rss, cpu))
        print(f"  {throughput:.2f} batches/s  |  {rss:.0f} MB RSS"
              + (f"  |  {cpu:.1f}% CPU" if cpu is not None else ""))

    # Summary table
    print("\n" + "=" * 60)
    header = f"{'workers':>8}  {'batches/s':>10}  {'peak_RSS_MB':>12}"
    if HAVE_PSUTIL:
        header += f"  {'avg_cpu_%':>10}"
    print(header)
    print("-" * (len(header)))
    for nw, tput, rss, cpu in results:
        row = f"{nw:>8}  {tput:>10.2f}  {rss:>12.0f}"
        if HAVE_PSUTIL:
            row += f"  {cpu:>10.1f}" if cpu is not None else f"  {'N/A':>10}"
        print(row)

    best_nw, best_tput, _, _ = max(results, key=lambda r: r[1])
    print(f"\nBest throughput: num_workers={best_nw} ({best_tput:.2f} batches/s)")
    print("=" * 60)

    # Plot batches/s vs num_workers
    nw_vals  = [r[0] for r in results]
    tput_vals = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nw_vals, tput_vals, marker="o", linewidth=1.5, color="steelblue")
    ax.axvline(best_nw, color="darkorange", linestyle="--", linewidth=1,
               label=f"best: {best_nw} workers ({best_tput:.2f} b/s)")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title(f"DataLoader throughput vs num_workers  (n_bw={len(bw_files)}, n_batches={args.n_batches})")
    ax.set_xticks(nw_vals)
    ax.legend()
    plt.tight_layout()
    plot_path = Path("benchmark_workers.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
