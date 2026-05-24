"""
Generate figures from a completed trial run.
  python plot.py
Reads results/history.json and results/eval_*.npy, writes figures/ directory.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from config import TARGETS_TXT


def _load_labels():
    labels = []
    with open(TARGETS_TXT) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                labels.append(parts[8])
    return labels


def main():
    results = Path("results")
    figures = Path("figures")
    figures.mkdir(exist_ok=True)

    with open(results / "history.json") as fh:
        history = json.load(fh)

    preds   = np.load(results / "eval_preds.npy")    # (n_intervals, center_bins, n_tracks)
    targets = np.load(results / "eval_targets.npy")

    n_intervals, center_bins, n_tracks = preds.shape
    labels = _load_labels()
    if len(labels) < n_tracks:
        labels = [str(i) for i in range(n_tracks)]

    # Figure 1: loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], marker="o", label="train")
    ax.plot(epochs, history["val_loss"],   marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Poisson NLL loss")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "loss_curve.png", dpi=150)
    plt.close(fig)
    print("Saved figures/loss_curve.png")

    # Figure 2: per-track Pearson R
    p_flat    = preds.reshape(-1, n_tracks)
    t_flat    = targets.reshape(-1, n_tracks)
    pearson_r = np.array([pearsonr(p_flat[:, i], t_flat[:, i])[0] for i in range(n_tracks)])
    order     = np.argsort(pearson_r)[::-1]

    fig, ax = plt.subplots(figsize=(max(8, n_tracks * 0.15), 5))
    ax.bar(range(n_tracks), pearson_r[order], width=1.0, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Track (sorted by R)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"Per-track Pearson R  (mean={pearson_r.mean():.3f})")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(figures / "pearson_per_track.png", dpi=150)
    plt.close(fig)
    print("Saved figures/pearson_per_track.png")

    # Figure 3: example predicted vs actual (top-3 tracks, first interval)
    top3 = order[:3]
    x    = np.arange(center_bins)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, ti in zip(axes, top3):
        pred_signal   = preds[0, :, ti]
        target_signal = targets[0, :, ti]
        r, _          = pearsonr(pred_signal, target_signal)
        short_label   = labels[ti].split(":")[-1] if ":" in labels[ti] else labels[ti]
        ax.plot(x, target_signal, label="actual",    alpha=0.8, linewidth=0.8)
        ax.plot(x, pred_signal,   label="predicted", alpha=0.8, linewidth=0.8, linestyle="--")
        ax.set_ylabel("signal")
        ax.set_title(f"{short_label}  (R={r:.3f})")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Genomic bin")
    fig.suptitle("Predicted vs actual  —  top-3 tracks, interval 0")
    fig.tight_layout()
    fig.savefig(figures / "example_tracks.png", dpi=150)
    plt.close(fig)
    print("Saved figures/example_tracks.png")

    # Figure 4: scatter — mean signal per interval×track
    p_mean   = preds.mean(axis=1).ravel()
    t_mean   = targets.mean(axis=1).ravel()
    r_all, _ = pearsonr(p_mean, t_mean)
    fig, ax  = plt.subplots(figsize=(5, 5))
    ax.scatter(t_mean, p_mean, s=4, alpha=0.3, rasterized=True)
    lim = max(t_mean.max(), p_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Actual mean signal")
    ax.set_ylabel("Predicted mean signal")
    ax.set_title(f"Predicted vs actual  (R={r_all:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "scatter.png", dpi=150)
    plt.close(fig)
    print("Saved figures/scatter.png")

    print(f"\nDone. Mean Pearson R across {n_tracks} tracks: {pearson_r.mean():.4f}")


if __name__ == "__main__":
    main()
