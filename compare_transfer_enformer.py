"""
Compare the transfer-learned head against the original-Enformer best-match
baseline, per CRC target, with paper-ready statistics, robustness checks,
biological-plausibility review, and publication figures.

This is the Enformer twin of compare_transfer_borzoi.py. The one substantive
difference is resolution: Enformer predicts 896 x 128 bp over the central
114,688 bp, so the comparison is run on that grid. The transfer model's
predictions and the targets are cropped and mean-pooled the same way, which
makes the pairing honest but means `transfer_r` here is NOT numerically the same
quantity as the 32 bp `transfer_r` in the Borzoi comparison. Both are reported:
`transfer_r` (Enformer grid, used for all statistics) and
`transfer_r_native_32bp` (reference only).

Inputs (per split):
  results_full_2/{split}_preds.npy, {split}_targets.npy, {split}_per_track_metrics.csv
  results_baseline_enformer/enformer_best_match_{split}.csv
  results_baseline_enformer/enformer_pearson_matrix_{split}.npy

Outputs:
  results_comparison_enformer/transfer_vs_enformer_{split}.csv
  results_comparison_enformer/summary_stats_{split}.csv
  results_comparison_enformer/paired_stats_{split}.csv
  results_comparison_enformer/stratified_{split}.csv
  results_comparison_enformer/top_targets_{split}.csv
  results_comparison_enformer/*_{split}.{png,pdf}

Usage:
  python compare_transfer_enformer.py --split test
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
    classify_plausibility,
)
from enformer_baseline_utils import targets_to_enformer_grid

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
TRANSFER_C = "#D1495B"   # transfer model
ENFORMER_C = "#EDAE49"   # enformer baseline (distinct from the Borzoi teal)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",    default="./results_full_2")
    p.add_argument("--baseline-dir", default="./results_baseline_enformer")
    p.add_argument("--out-dir",      default="./results_comparison_enformer")
    p.add_argument("--split",        default="test", choices=["val", "test"])
    p.add_argument("--tie-tol",      type=float, default=0.01)
    p.add_argument("--chunk",        type=int, default=50)
    return p.parse_args()


def savefig(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def pooled_pearson_and_var(preds, targets, chunk: int):
    """Per-track Pearson r on the Enformer grid, plus per-track variances.

    Streams over interval chunks so only one chunk is pooled to float64 at a
    time; the full (n_intervals, 4096, n_tracks) arrays stay memory-mapped.
    """
    n_int, _, n_tracks = preds.shape
    sx = np.zeros(n_tracks); sy = np.zeros(n_tracks)
    sxy = np.zeros(n_tracks); sx2 = np.zeros(n_tracks); sy2 = np.zeros(n_tracks)
    n = 0
    for lo in range(0, n_int, chunk):
        p = targets_to_enformer_grid(preds[lo:lo + chunk]).reshape(-1, n_tracks)
        t = targets_to_enformer_grid(targets[lo:lo + chunk]).reshape(-1, n_tracks)
        n   += p.shape[0]
        sx  += p.sum(0); sy  += t.sum(0)
        sxy += (p * t).sum(0)
        sx2 += (p * p).sum(0); sy2 += (t * t).sum(0)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sx2 - sx ** 2, 0) * np.maximum(n * sy2 - sy ** 2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    pvar = sx2 / n - (sx / n) ** 2
    tvar = sy2 / n - (sy / n) ** 2
    return r, pvar, tvar


def main():
    args = parse_args()
    split = args.split
    model_dir = Path(args.model_dir); base_dir = Path(args.baseline_dir)
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    preds   = np.load(model_dir / f"{split}_preds.npy", mmap_mode="r")
    targets = np.load(model_dir / f"{split}_targets.npy", mmap_mode="r")

    # transfer model on the Enformer grid (the comparison quantity)
    transfer_r, pvar, tvar = pooled_pearson_and_var(preds, targets, args.chunk)
    lowvar = (tvar < 1e-8) | (pvar < 1e-8)

    # native 32 bp r, for cross-referencing against the Borzoi comparison
    transfer_r_native = per_track_pearson(preds, targets)
    saved = pd.read_csv(model_dir / f"{split}_per_track_metrics.csv")
    max_dev = float(np.abs(transfer_r_native - saved["pearson_r"].to_numpy()).max())
    print(f"[check] native-resolution transfer r vs saved metrics, max abs dev: {max_dev:.4f}")
    print(f"[info]  pooling to 128 bp changes mean transfer r by "
          f"{transfer_r.mean() - transfer_r_native.mean():+.4f}")

    # --- Enformer best match -------------------------------------------------
    enformer = pd.read_csv(base_dir / f"enformer_best_match_{split}.csv")
    enformer_r = enformer["pearson_r"].to_numpy()
    assert len(enformer_r) == len(transfer_r), "target count mismatch between model and baseline"

    d = transfer_r - enformer_r
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(enformer_r > 0, d / enformer_r, np.nan)
    plaus = [classify_plausibility(enformer.loc[i, "target_name"],
                                   enformer.loc[i, "best_enformer_factor"],
                                   enformer.loc[i, "target_tissue"],
                                   enformer.loc[i, "best_enformer_cell_tissue"])
             for i in range(len(enformer))]
    table = pd.DataFrame({
        "target_idx": enformer["target_idx"],
        "target_name": enformer["target_name"],
        "target_cell_line": enformer["target_cell_line"],
        "target_tissue": enformer["target_tissue"],
        "transfer_r": transfer_r,
        "transfer_r_native_32bp": transfer_r_native,
        "enformer_r": enformer_r,
        "delta_transfer_minus_enformer": d,
        "relative_improvement": rel,
        "winner": np.where(np.abs(d) <= args.tie_tol, "tie",
                           np.where(d > 0, "transfer", "enformer")),
        "best_enformer_track_idx": enformer["best_enformer_track_idx"],
        "best_enformer_description": enformer["best_enformer_description"],
        "best_enformer_cell_tissue": enformer["best_enformer_cell_tissue"],
        "biological_match": plaus,
        "low_variance_flag": lowvar,
    })
    table.to_csv(out / f"transfer_vs_enformer_{split}.csv", index=False)

    # --- statistics ----------------------------------------------------------
    s_tr = per_model_summary(transfer_r); s_en = per_model_summary(enformer_r)
    paired = paired_comparison(transfer_r, enformer_r, tie_tol=args.tie_tol)
    paired = {k.replace("borzoi", "enformer"): v for k, v in paired.items()}
    spear = float(sps.spearmanr(transfer_r, enformer_r).correlation)

    pd.DataFrame([{"model": "transfer", **s_tr},
                  {"model": "enformer", **s_en}]).to_csv(
        out / f"summary_stats_{split}.csv", index=False)
    pd.DataFrame([paired]).to_csv(out / f"paired_stats_{split}.csv", index=False)

    print(f"\n=== {split} summary (128 bp Enformer grid) ===")
    print(f"transfer  : mean={s_tr['mean']:.3f} median={s_tr['median']:.3f} "
          f"CI95=[{s_tr['ci95_lo']:.3f},{s_tr['ci95_hi']:.3f}]")
    print(f"enformer  : mean={s_en['mean']:.3f} median={s_en['median']:.3f} "
          f"CI95=[{s_en['ci95_lo']:.3f},{s_en['ci95_hi']:.3f}]")
    print(f"paired dbar={paired['mean_diff']:.3f} "
          f"CI95=[{paired['diff_ci95_lo']:.3f},{paired['diff_ci95_hi']:.3f}] "
          f"t_p={paired['paired_t_p']:.2e} wilcoxon_p={paired['wilcoxon_p']:.2e} "
          f"d={paired['cohens_d_paired']:.3f}")
    print(f"wins: transfer={paired['n_transfer_wins']} "
          f"enformer={paired['n_enformer_wins']} tie={paired['n_tie']}  "
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
                "enformer_mean_r": sub["enformer_r"].mean(),
                "mean_delta": sub["delta_transfer_minus_enformer"].mean(),
            })
    pd.DataFrame(strat_rows).to_csv(out / f"stratified_{split}.csv", index=False)

    top = pd.concat([
        table.nlargest(10, "transfer_r").assign(list="top_transfer_r"),
        table.nlargest(10, "enformer_r").assign(list="top_enformer_r"),
        table.nlargest(10, "delta_transfer_minus_enformer").assign(list="top_transfer_gain"),
        table.nsmallest(10, "delta_transfer_minus_enformer").assign(list="top_enformer_advantage"),
    ])
    top.to_csv(out / f"top_targets_{split}.csv", index=False)

    # ============================ FIGURES ==================================
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(enformer_r, transfer_r, s=18, alpha=0.6, color=TRANSFER_C, edgecolor="none")
    lim = [min(enformer_r.min(), transfer_r.min()) - 0.05, 1.0]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Enformer best-match Pearson r")
    ax.set_ylabel("Transfer-model Pearson r")
    ax.set_title(f"{split}: transfer vs. Enformer (n={len(table)})")
    for _, r in table.nlargest(3, "delta_transfer_minus_enformer").iterrows():
        ax.annotate(r["target_name"], (r["enformer_r"], r["transfer_r"]), fontsize=7)
    for _, r in table.nsmallest(3, "delta_transfer_minus_enformer").iterrows():
        ax.annotate(r["target_name"], (r["enformer_r"], r["transfer_r"]), fontsize=7)
    savefig(fig, out, f"scatter_transfer_vs_enformer_{split}")

    order = np.argsort(enformer_r)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    ax.vlines(x, enformer_r[order], transfer_r[order], color="lightgray", lw=0.6, zorder=1)
    ax.scatter(x, enformer_r[order], s=8, color=ENFORMER_C, label="Enformer", zorder=2)
    ax.scatter(x, transfer_r[order], s=8, color=TRANSFER_C, label="Transfer", zorder=2)
    ax.set_xlabel("Target (sorted by Enformer r)"); ax.set_ylabel("Pearson r")
    ax.set_title(f"{split}: paired per-target performance"); ax.legend()
    savefig(fig, out, f"paired_performance_{split}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.violinplot([enformer_r, transfer_r], showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Enformer", "Transfer"])
    ax.set_ylabel("Per-target Pearson r"); ax.set_title(f"{split}: distribution")
    savefig(fig, out, f"violin_distribution_{split}")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(d, bins=30, color=TRANSFER_C, alpha=0.8)
    ax.axvline(0, color="gray", ls="--")
    ax.axvline(d.mean(), color="black", label=f"mean Δ={d.mean():.3f}")
    ax.set_xlabel("Δ (transfer − Enformer) Pearson r"); ax.set_ylabel("# targets")
    ax.set_title(f"{split}: paired differences"); ax.legend()
    savefig(fig, out, f"paired_difference_hist_{split}")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r, c, lab in ((enformer_r, ENFORMER_C, "Enformer"), (transfer_r, TRANSFER_C, "Transfer")):
        xs = np.sort(r); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, color=c, label=lab, where="post")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{split}: ECDF"); ax.legend()
    savefig(fig, out, f"ecdf_{split}")

    pmat = np.load(base_dir / f"enformer_pearson_matrix_{split}.npy")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(pmat.ravel(), bins=80, color=ENFORMER_C, alpha=0.8, density=True)
    ax.set_xlabel("Pearson r (all target × eligible-track pairs)")
    ax.set_ylabel("density"); ax.set_title(f"{split}: all eligible Enformer correlations")
    savefig(fig, out, f"eligible_correlation_density_{split}")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    vc = table["biological_match"].value_counts()
    ax.bar(vc.index, vc.values, color=ENFORMER_C)
    ax.set_ylabel("# targets"); ax.set_title(f"{split}: best-match biological plausibility")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, out, f"biological_plausibility_{split}")

    print(f"\nWrote per-target table, stats, and 7 figure sets (PNG+PDF) to {out}/")
    print(f"biological match breakdown:\n{table['biological_match'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
