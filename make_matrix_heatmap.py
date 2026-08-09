#!/usr/bin/env python3
"""Render the raw best-match correlation matrix (183 targets x N eligible Borzoi
tracks) as a heatmap, plus a labeled CSV copy of the same matrix.

Both axes are sorted alphabetically by factor for display ONLY — the saved .npy
keeps its construction order (rows = bigwig load order, cols = ascending Borzoi
track_index). The sort is what makes the same-TF blocks visible; it is also why
the best-match markers appear to trace a diagonal, so the slide says so.

Usage: python make_matrix_heatmap.py --split test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRS = {"val": ("results_baseline_filtered", "results_comparison"),
        "test": ("results_baseline_filtered_test", "results_comparison_test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    base, out = (Path(d) for d in DIRS[args.split])
    split = args.split

    m = np.load(base / f"borzoi_filtered_pearson_matrix_{split}.npy")
    bm = pd.read_csv(base / f"borzoi_filtered_best_match_{split}.csv").sort_values("target_idx")
    inc = pd.read_csv(base / "included_borzoi_tracks.csv")

    rows = (bm["target_name"].astype(str) + " (" +
            bm["target_cell_line"].fillna("?").astype(str) + ")").values
    cols = (inc["factor"].astype(str) + ":" +
            inc["cell_tissue"].astype(str).str[:18]).values
    pd.DataFrame(m, index=rows, columns=cols).to_csv(
        out / f"pearson_matrix_{split}_labeled.csv")

    ro = np.argsort(bm["target_name"].astype(str).values, kind="stable")
    co = np.argsort(inc["factor"].astype(str).values, kind="stable")
    M = m[np.ix_(ro, co)]
    rown, coln = rows[ro], inc["factor"].astype(str).values[co]
    best = np.argmax(M, axis=1)
    same = float(np.mean([coln[best[i]] == rown[i].split(" (")[0] for i in range(len(best))]))

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=0.9, interpolation="nearest")
    ax.scatter(best, np.arange(len(best)), s=5, c="#00e5ff",
               label="best match (argmax) — the baseline for that target")
    ax.set_xlabel(f"{M.shape[1]:,} eligible Borzoi ChIP tracks (sorted by factor for display)")
    ax.set_ylabel(f"{M.shape[0]} CRC targets (sorted by TF for display)")
    ax.set_title(f"{split} split: Pearson r — measured target signal vs. Borzoi predicted track")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, label="Pearson r")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pearson_matrix_heatmap_{split}.{ext}", dpi=160)

    print(f"[{split}] heatmap + labeled CSV -> {out}/  "
          f"({M.shape[0]}x{M.shape[1]}, same-TF best match {same:.0%})")


if __name__ == "__main__":
    main()
