"""
Plot comparison: actual targets vs Borzoi baseline vs transfer-learned model.

Requires outputs from both main.py and baseline_borzoi.py to exist.

Usage:
    python plot_comparison.py [--targets-dir ./results_full]
                              [--model-dir ./results_full]
                              [--baseline-dir ./results_baseline]
                              [--output-dir ./results_comparison]
                              [--split val] [--n-tracks 5] [--interval 0]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare model vs baseline predictions")
    p.add_argument("--targets-dir",  default="./results_full")
    p.add_argument("--model-dir",    default="./results_full")
    p.add_argument("--baseline-dir", default="./results_baseline")
    p.add_argument("--output-dir",   default="./results_comparison")
    p.add_argument("--split",        default="val", choices=["val", "test"])
    p.add_argument("--n-tracks",     type=int, default=5,
                   help="Number of top tracks to show in trace plot")
    p.add_argument("--interval",     type=int, default=0,
                   help="Interval index to use for trace plots")
    return p.parse_args()


def pearson_per_track(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n_tracks = preds.shape[2]
    p_flat = preds.reshape(-1, n_tracks)
    t_flat = targets.reshape(-1, n_tracks)
    return np.array([pearsonr(p_flat[:, i], t_flat[:, i])[0] for i in range(n_tracks)])


def load_track_names(baseline_dir: Path, split: str, n_tracks: int) -> list[str]:
    csv_path = baseline_dir / f"baseline_best_match_{split}.csv"
    if not csv_path.exists():
        return [str(i) for i in range(n_tracks)]
    names = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get("best_borzoi_description", str(len(names))))
    return names[:n_tracks] if len(names) >= n_tracks else names + [str(i) for i in range(len(names), n_tracks)]


def plot_pearson_comparison(
    model_r: np.ndarray,
    baseline_r: np.ndarray,
    out_dir: Path,
    split: str,
) -> None:
    order = np.argsort(model_r)[::-1]
    x = np.arange(len(model_r))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(model_r) * 0.06 + 2), 4))
    ax.bar(x - w / 2, baseline_r[order], width=w, label="Borzoi baseline", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, model_r[order],    width=w, label="Our model",        color="darkorange", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Track (sorted by model R)")
    ax.set_ylabel("Pearson R")
    ax.set_title(
        f"{split} per-track Pearson R — baseline mean={baseline_r.mean():.3f}  "
        f"model mean={model_r.mean():.3f}"
    )
    ax.set_xticks([])
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "pearson_r_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> pearson_r_comparison.png")


def plot_traces(
    targets: np.ndarray,
    baseline_preds: np.ndarray,
    model_preds: np.ndarray,
    model_r: np.ndarray,
    baseline_r: np.ndarray,
    track_names: list[str],
    interval: int,
    n_tracks: int,
    out_dir: Path,
    split: str,
) -> None:
    top = np.argsort(model_r)[::-1][:n_tracks]
    x = np.arange(targets.shape[1])

    fig, axes = plt.subplots(n_tracks, 1, figsize=(13, 4 * n_tracks), sharex=True)
    if n_tracks == 1:
        axes = [axes]

    for ax, ti in zip(axes, top):
        name = track_names[ti] if ti < len(track_names) else str(ti)
        ax.plot(x, targets[interval, :, ti],       color="black",      linewidth=0.9, label="actual",   alpha=0.9)
        ax.plot(x, baseline_preds[interval, :, ti], color="steelblue",  linewidth=0.8, label="baseline", alpha=0.8, linestyle="--")
        ax.plot(x, model_preds[interval, :, ti],    color="darkorange", linewidth=0.8, label="model",    alpha=0.8, linestyle="--")
        ax.set_ylabel("signal")
        ax.set_title(f"Track {ti} — {name[:60]}  "
                     f"(baseline R={baseline_r[ti]:.3f}, model R={model_r[ti]:.3f})")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Genomic bin (32 bp)")
    fig.suptitle(f"{split} signal traces — top {n_tracks} tracks by model R, interval {interval}")
    plt.tight_layout()
    fig.savefig(out_dir / f"traces_top{n_tracks}.png", dpi=150)
    plt.close(fig)
    print(f"  -> traces_top{n_tracks}.png")


def plot_scatter(
    model_r: np.ndarray,
    baseline_r: np.ndarray,
    out_dir: Path,
    split: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(baseline_r, model_r, s=8, alpha=0.4, color="steelblue", rasterized=True)
    lim_min = min(baseline_r.min(), model_r.min()) - 0.05
    lim_max = max(baseline_r.max(), model_r.max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel(f"Baseline Pearson R  (mean={baseline_r.mean():.3f})")
    ax.set_ylabel(f"Model Pearson R  (mean={model_r.mean():.3f})")
    ax.set_title(f"{split} model vs baseline per-track R\n"
                 f"model wins: {(model_r > baseline_r).sum()}/{len(model_r)} tracks")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_baseline_vs_model.png", dpi=150)
    plt.close(fig)
    print(f"  -> scatter_baseline_vs_model.png")


def main() -> None:
    args = parse_args()
    targets_dir  = Path(args.targets_dir)
    model_dir    = Path(args.model_dir)
    baseline_dir = Path(args.baseline_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split

    print("Loading arrays ...")
    targets        = np.load(targets_dir  / f"{split}_targets.npy")
    model_preds    = np.load(model_dir    / f"{split}_preds.npy")
    baseline_preds = np.load(baseline_dir / f"baseline_preds_{split}.npy")

    assert targets.shape == model_preds.shape == baseline_preds.shape, (
        f"Shape mismatch: targets={targets.shape} model={model_preds.shape} "
        f"baseline={baseline_preds.shape}"
    )
    n_intervals, center_bins, n_tracks = targets.shape
    print(f"Arrays loaded: {n_intervals} intervals x {center_bins} bins x {n_tracks} tracks")

    print("Computing Pearson R ...")
    model_r    = pearson_per_track(model_preds,    targets)
    baseline_r = pearson_per_track(baseline_preds, targets)

    print(f"  Baseline mean R : {baseline_r.mean():.4f}")
    print(f"  Model    mean R : {model_r.mean():.4f}")
    print(f"  Model wins on   : {(model_r > baseline_r).sum()}/{n_tracks} tracks\n")

    track_names = load_track_names(baseline_dir, split, n_tracks)

    print("Saving plots ...")
    plot_pearson_comparison(model_r, baseline_r, out_dir, split)
    plot_traces(targets, baseline_preds, model_preds, model_r, baseline_r,
                track_names, args.interval, args.n_tracks, out_dir, split)
    plot_scatter(model_r, baseline_r, out_dir, split)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
