"""
Identify CRC targets where the transfer-learned head underperforms the
filtered-original-Borzoi best-match baseline, and emit a filtered target
list (excluding those targets) for higher aggregate performance.

Inputs:
  results_comparison_{split}/transfer_vs_filtered_borzoi_{split}.csv
    (produced by compare_transfer_borzoi.py --split {split})

Outputs (written to a new directory, existing results_comparison_* left
untouched):
  underperforming_targets_{split}.csv   rows where winner == "borzoi"
  kept_targets_{split}.csv              remaining target_idx (transfer + tie)
  summary_before_after_{split}.csv      aggregate transfer/borzoi r, before vs.
                                         after dropping underperforming targets

Usage:
  python identify_underperforming_targets.py --split test \
      --out-dir results_targets_filtered_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
TRANSFER_C = "#D1495B"   # transfer model
BORZOI_C   = "#00798C"   # borzoi baseline


def savefig(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=None,
                    help="dir holding transfer_vs_filtered_borzoi_{split}.csv "
                         "(default: results_comparison_{split})")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out-dir", default=None,
                    help="default: results_targets_filtered_{split}")
    return p.parse_args()


def summarize(df: pd.DataFrame, label: str) -> dict:
    n_wins = int((df["winner"] == "transfer").sum())
    return {
        "stage": label,
        "n_targets": len(df),
        "mean_transfer_r": df["transfer_r"].mean(),
        "median_transfer_r": df["transfer_r"].median(),
        "mean_borzoi_r": df["borzoi_r"].mean(),
        "median_borzoi_r": df["borzoi_r"].median(),
        "mean_delta": df["delta_transfer_minus_borzoi"].mean(),
        "pct_transfer_wins": 100.0 * n_wins / len(df),
    }


def main():
    args = parse_args()
    split = args.split
    default_in_dir = "results_comparison_test" if split == "test" else "results_comparison"
    in_dir = Path(args.in_dir) if args.in_dir else Path(default_in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"results_targets_filtered_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(in_dir / f"transfer_vs_filtered_borzoi_{split}.csv")

    excluded = table[table["winner"] == "borzoi"].copy()
    kept = table[table["winner"] != "borzoi"].copy()

    cols = ["target_idx", "target_name", "target_cell_line", "target_tissue",
            "transfer_r", "borzoi_r", "delta_transfer_minus_borzoi",
            "biological_match"]
    excluded[cols].sort_values("delta_transfer_minus_borzoi").to_csv(
        out_dir / f"underperforming_targets_{split}.csv", index=False)
    kept[["target_idx", "target_name"]].to_csv(
        out_dir / f"kept_targets_{split}.csv", index=False)

    summary = pd.DataFrame([
        summarize(table, "before (all targets)"),
        summarize(kept, "after (underperforming targets removed)"),
    ])
    summary.to_csv(out_dir / f"summary_before_after_{split}.csv", index=False)

    print("=" * 72)
    print(f"Underperforming-target filtering — {split} split")
    print("=" * 72)
    print(f"  total targets            : {len(table)}")
    print(f"  underperforming (removed): {len(excluded)}")
    print(f"  kept                     : {len(kept)}")
    print("-" * 72)
    print(summary.to_string(index=False))
    print("-" * 72)
    print(f"Wrote {out_dir / f'underperforming_targets_{split}.csv'} ({len(excluded)} rows)")
    print(f"Wrote {out_dir / f'kept_targets_{split}.csv'} ({len(kept)} rows)")
    print(f"Wrote {out_dir / f'summary_before_after_{split}.csv'}")

    # ============================ FIGURES (kept set) ========================
    borzoi_r = kept["borzoi_r"].to_numpy()
    transfer_r = kept["transfer_r"].to_numpy()
    d = kept["delta_transfer_minus_borzoi"].to_numpy()

    # 1. scatter transfer vs borzoi with identity line
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(borzoi_r, transfer_r, s=18, alpha=0.6, color=TRANSFER_C, edgecolor="none")
    lim = [min(borzoi_r.min(), transfer_r.min()) - 0.05, 1.0]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Filtered Borzoi best-match Pearson r")
    ax.set_ylabel("Transfer-model Pearson r")
    ax.set_title(f"{split}: transfer vs. filtered Borzoi, underperformers removed (n={len(kept)})")
    for _, r in kept.nlargest(3, "delta_transfer_minus_borzoi").iterrows():
        ax.annotate(r["target_name"], (r["borzoi_r"], r["transfer_r"]), fontsize=7)
    for _, r in kept.nsmallest(3, "delta_transfer_minus_borzoi").iterrows():
        ax.annotate(r["target_name"], (r["borzoi_r"], r["transfer_r"]), fontsize=7)
    savefig(fig, out_dir, f"scatter_transfer_vs_borzoi_{split}")

    # 2. paired dot plot (sorted by borzoi_r)
    order = np.argsort(borzoi_r)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    ax.vlines(x, borzoi_r[order], transfer_r[order], color="lightgray", lw=0.6, zorder=1)
    ax.scatter(x, borzoi_r[order], s=8, color=BORZOI_C, label="Borzoi", zorder=2)
    ax.scatter(x, transfer_r[order], s=8, color=TRANSFER_C, label="Transfer", zorder=2)
    ax.set_xlabel("Target (sorted by Borzoi r)"); ax.set_ylabel("Pearson r")
    ax.set_title(f"{split}: paired per-target performance, underperformers removed")
    ax.legend()
    savefig(fig, out_dir, f"paired_performance_{split}")

    # 3. violin/box of per-target r
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.violinplot([borzoi_r, transfer_r], showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Filtered\nBorzoi", "Transfer"])
    ax.set_ylabel("Per-target Pearson r")
    ax.set_title(f"{split}: distribution, underperformers removed")
    savefig(fig, out_dir, f"violin_distribution_{split}")

    # 4. histogram of paired differences
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(d, bins=30, color=TRANSFER_C, alpha=0.8)
    ax.axvline(0, color="gray", ls="--"); ax.axvline(d.mean(), color="black",
               label=f"mean Δ={d.mean():.3f}")
    ax.set_xlabel("Δ (transfer − Borzoi) Pearson r"); ax.set_ylabel("# targets")
    ax.set_title(f"{split}: paired differences, underperformers removed"); ax.legend()
    savefig(fig, out_dir, f"paired_difference_hist_{split}")

    # 5. empirical CDF
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r, c, lab in ((borzoi_r, BORZOI_C, "Borzoi"), (transfer_r, TRANSFER_C, "Transfer")):
        xs = np.sort(r); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, color=c, label=lab, where="post")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{split}: ECDF, underperformers removed"); ax.legend()
    savefig(fig, out_dir, f"ecdf_{split}")

    # 6. biological-plausibility bar
    fig, ax = plt.subplots(figsize=(5, 3.4))
    vc = kept["biological_match"].value_counts()
    ax.bar(vc.index, vc.values, color=BORZOI_C)
    ax.set_ylabel("# targets")
    ax.set_title(f"{split}: best-match biological plausibility, underperformers removed")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, out_dir, f"biological_plausibility_{split}")

    print(f"\nWrote 6 figure sets (PNG+PDF) to {out_dir}/")


if __name__ == "__main__":
    main()
