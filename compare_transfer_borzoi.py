"""
Compare the transfer-learned head against the filtered-Borzoi best-match baseline,
per CRC target, with paper-ready statistics, robustness checks, biological-
plausibility review, and publication figures.

Inputs (per split):
  results_full_2/{split}_preds.npy, {split}_targets.npy, {split}_per_track_metrics.csv
  results_baseline_filtered/borzoi_filtered_best_match_{split}.csv
  results_baseline_filtered/borzoi_filtered_pearson_matrix_{split}.npy

Outputs:
  results_comparison/transfer_vs_filtered_borzoi_{split}.csv     per-target table
  results_comparison/summary_stats_{split}.csv                   aggregate + paired stats
  results_comparison/stratified_{split}.csv                      by Factor/Cell/Tissue
  results_comparison/*_{split}.{png,pdf}                         figures

Usage:
  python compare_transfer_borzoi.py --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from borzoi_baseline_utils import (
    per_track_pearson, per_model_summary, paired_comparison,
    classify_plausibility, bootstrap_ci,
)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
TRANSFER_C = "#D1495B"   # transfer model
BORZOI_C   = "#00798C"   # borzoi baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",    default="./results_full_2")
    p.add_argument("--baseline-dir", default="./results_baseline_filtered")
    p.add_argument("--out-dir",      default="./results_comparison")
    p.add_argument("--split",        default="val", choices=["val", "test"])
    p.add_argument("--tie-tol",      type=float, default=0.01)
    return p.parse_args()


def savefig(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def main():
    args = parse_args()
    split = args.split
    model_dir = Path(args.model_dir); base_dir = Path(args.baseline_dir)
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    # --- transfer model per-target r (recomputed for identical numerics) -----
    # memory-map the multi-GB arrays; per_track_pearson streams over chunks.
    preds   = np.load(model_dir / f"{split}_preds.npy", mmap_mode="r")
    targets = np.load(model_dir / f"{split}_targets.npy", mmap_mode="r")
    transfer_r = per_track_pearson(preds, targets)

    # sanity: reproduce the saved per_track_metrics
    saved = pd.read_csv(model_dir / f"{split}_per_track_metrics.csv")
    max_dev = float(np.abs(transfer_r - saved["pearson_r"].to_numpy()).max())
    print(f"[check] transfer r vs saved per_track_metrics max abs dev: {max_dev:.4f}")

    # near-zero variance flags (robustness), computed in chunks
    n_tracks = preds.shape[-1]
    tvar = np.zeros(n_tracks); pvar = np.zeros(n_tracks)
    for lo in range(0, preds.shape[0], 100):
        pvar += preds[lo:lo + 100].reshape(-1, n_tracks).astype(np.float64).var(0) * \
                preds[lo:lo + 100].reshape(-1, n_tracks).shape[0]
        tvar += targets[lo:lo + 100].reshape(-1, n_tracks).astype(np.float64).var(0) * \
                targets[lo:lo + 100].reshape(-1, n_tracks).shape[0]
    lowvar = (tvar < 1e-8) | (pvar < 1e-8)

    # --- filtered Borzoi best match -----------------------------------------
    borzoi = pd.read_csv(base_dir / f"borzoi_filtered_best_match_{split}.csv")
    borzoi_r = borzoi["pearson_r"].to_numpy()

    # --- per-target table ----------------------------------------------------
    d = transfer_r - borzoi_r
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(borzoi_r > 0, d / borzoi_r, np.nan)
    plaus = [classify_plausibility(borzoi.loc[i, "target_name"],
                                   borzoi.loc[i, "best_borzoi_factor"],
                                   borzoi.loc[i, "target_tissue"],
                                   borzoi.loc[i, "best_borzoi_cell_tissue"])
             for i in range(len(borzoi))]
    table = pd.DataFrame({
        "target_idx": borzoi["target_idx"],
        "target_name": borzoi["target_name"],
        "target_cell_line": borzoi["target_cell_line"],
        "target_tissue": borzoi["target_tissue"],
        "transfer_r": transfer_r,
        "borzoi_r": borzoi_r,
        "delta_transfer_minus_borzoi": d,
        "relative_improvement": rel,
        "winner": np.where(np.abs(d) <= args.tie_tol, "tie",
                           np.where(d > 0, "transfer", "borzoi")),
        "best_borzoi_track_idx": borzoi["best_borzoi_track_idx"],
        "best_borzoi_description": borzoi["best_borzoi_description"],
        "best_borzoi_cell_tissue": borzoi["best_borzoi_cell_tissue"],
        "biological_match": plaus,
        "low_variance_flag": lowvar,
    })
    table.to_csv(out / f"transfer_vs_filtered_borzoi_{split}.csv", index=False)

    # --- statistics ----------------------------------------------------------
    s_tr = per_model_summary(transfer_r); s_bo = per_model_summary(borzoi_r)
    paired = paired_comparison(transfer_r, borzoi_r, tie_tol=args.tie_tol)
    spear = float(sps.spearmanr(transfer_r, borzoi_r).correlation)

    summary_rows = []
    for name, s in (("transfer", s_tr), ("filtered_borzoi", s_bo)):
        row = {"model": name, **s}; summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out / f"summary_stats_{split}.csv", index=False)
    pd.DataFrame([paired]).to_csv(out / f"paired_stats_{split}.csv", index=False)

    print(f"\n=== {split} summary ===")
    print(f"transfer  : mean={s_tr['mean']:.3f} median={s_tr['median']:.3f} "
          f"CI95=[{s_tr['ci95_lo']:.3f},{s_tr['ci95_hi']:.3f}]")
    print(f"borzoi    : mean={s_bo['mean']:.3f} median={s_bo['median']:.3f} "
          f"CI95=[{s_bo['ci95_lo']:.3f},{s_bo['ci95_hi']:.3f}]")
    print(f"paired dbar={paired['mean_diff']:.3f} "
          f"CI95=[{paired['diff_ci95_lo']:.3f},{paired['diff_ci95_hi']:.3f}] "
          f"t_p={paired['paired_t_p']:.2e} wilcoxon_p={paired['wilcoxon_p']:.2e} "
          f"d={paired['cohens_d_paired']:.3f}")
    print(f"wins: transfer={paired['n_transfer_wins']} "
          f"borzoi={paired['n_borzoi_wins']} tie={paired['n_tie']}  "
          f"(Spearman r between models={spear:.3f})")

    # --- metadata-stratified -------------------------------------------------
    strat_rows = []
    for col in ("target_name", "target_cell_line", "target_tissue"):
        for grp, sub in table.groupby(col):
            if len(sub) < 3 or grp in ("", None):
                continue
            strat_rows.append({
                "stratum": col, "group": grp, "n": len(sub),
                "transfer_mean_r": sub["transfer_r"].mean(),
                "borzoi_mean_r": sub["borzoi_r"].mean(),
                "mean_delta": sub["delta_transfer_minus_borzoi"].mean(),
            })
    pd.DataFrame(strat_rows).to_csv(out / f"stratified_{split}.csv", index=False)

    # --- top/bottom target table --------------------------------------------
    top = pd.concat([
        table.nlargest(10, "transfer_r").assign(list="top_transfer_r"),
        table.nlargest(10, "borzoi_r").assign(list="top_borzoi_r"),
        table.nlargest(10, "delta_transfer_minus_borzoi").assign(list="top_transfer_gain"),
        table.nsmallest(10, "delta_transfer_minus_borzoi").assign(list="top_borzoi_advantage"),
    ])
    top.to_csv(out / f"top_targets_{split}.csv", index=False)

    # ============================ FIGURES ==================================
    # 1. scatter transfer vs borzoi with identity line
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(borzoi_r, transfer_r, s=18, alpha=0.6, color=TRANSFER_C, edgecolor="none")
    lim = [min(borzoi_r.min(), transfer_r.min()) - 0.05, 1.0]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Filtered Borzoi best-match Pearson r")
    ax.set_ylabel("Transfer-model Pearson r")
    ax.set_title(f"{split}: transfer vs. filtered Borzoi (n={len(table)})")
    for _, r in table.nlargest(3, "delta_transfer_minus_borzoi").iterrows():
        ax.annotate(r["target_name"], (r["borzoi_r"], r["transfer_r"]), fontsize=7)
    for _, r in table.nsmallest(3, "delta_transfer_minus_borzoi").iterrows():
        ax.annotate(r["target_name"], (r["borzoi_r"], r["transfer_r"]), fontsize=7)
    savefig(fig, out, f"scatter_transfer_vs_borzoi_{split}")

    # 2. paired dot plot (sorted by borzoi_r)
    order = np.argsort(borzoi_r)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    ax.vlines(x, borzoi_r[order], transfer_r[order], color="lightgray", lw=0.6, zorder=1)
    ax.scatter(x, borzoi_r[order], s=8, color=BORZOI_C, label="Borzoi", zorder=2)
    ax.scatter(x, transfer_r[order], s=8, color=TRANSFER_C, label="Transfer", zorder=2)
    ax.set_xlabel("Target (sorted by Borzoi r)"); ax.set_ylabel("Pearson r")
    ax.set_title(f"{split}: paired per-target performance"); ax.legend()
    savefig(fig, out, f"paired_performance_{split}")

    # 3. violin/box of per-target r
    fig, ax = plt.subplots(figsize=(4.5, 4))
    parts = ax.violinplot([borzoi_r, transfer_r], showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Filtered\nBorzoi", "Transfer"])
    ax.set_ylabel("Per-target Pearson r"); ax.set_title(f"{split}: distribution")
    savefig(fig, out, f"violin_distribution_{split}")

    # 4. histogram of paired differences
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(d, bins=30, color=TRANSFER_C, alpha=0.8)
    ax.axvline(0, color="gray", ls="--"); ax.axvline(d.mean(), color="black",
               label=f"mean Δ={d.mean():.3f}")
    ax.set_xlabel("Δ (transfer − Borzoi) Pearson r"); ax.set_ylabel("# targets")
    ax.set_title(f"{split}: paired differences"); ax.legend()
    savefig(fig, out, f"paired_difference_hist_{split}")

    # 5. empirical CDF
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r, c, lab in ((borzoi_r, BORZOI_C, "Borzoi"), (transfer_r, TRANSFER_C, "Transfer")):
        xs = np.sort(r); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, color=c, label=lab, where="post")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{split}: ECDF"); ax.legend()
    savefig(fig, out, f"ecdf_{split}")

    # 6. density of all eligible Borzoi correlations (from the full matrix)
    pmat = np.load(base_dir / f"borzoi_filtered_pearson_matrix_{split}.npy")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(pmat.ravel(), bins=80, color=BORZOI_C, alpha=0.8, density=True)
    ax.set_xlabel("Pearson r (all target × eligible-track pairs)")
    ax.set_ylabel("density"); ax.set_title(f"{split}: all eligible Borzoi correlations")
    savefig(fig, out, f"eligible_correlation_density_{split}")

    # 7. biological-plausibility bar
    fig, ax = plt.subplots(figsize=(5, 3.4))
    vc = table["biological_match"].value_counts()
    ax.bar(vc.index, vc.values, color=BORZOI_C)
    ax.set_ylabel("# targets"); ax.set_title(f"{split}: best-match biological plausibility")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, out, f"biological_plausibility_{split}")

    print(f"\nWrote per-target table, stats, and 7 figure sets (PNG+PDF) to {out}/")
    print(f"biological match breakdown:\n{table['biological_match'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
