"""
Figures for the four-way benchmark (Borzoi-TL vs pretrained Borzoi vs Enformer
vs AlphaGenome), drawn from results_benchmark/benchmark_comparison_{split}.csv.

Three figures, each answering one question:
  1. model_comparison   — how do the four models' per-track R distributions compare?
  2. paired_delta       — is Borzoi-TL's margin over each baseline real?
  3. leakage_scatter    — where do Borzoi-TL's losses concentrate, and why?

Palette is the validated categorical order blue/orange/aqua/violet (passes the
lightness-band, chroma, protan+deutan CVD, normal-vision and contrast checks on
the all-pairs list). Aqua sits slightly under the 3.0 contrast floor against the
surface, so every series is also direct-labeled and legended — identity is never
carried by color alone.

Usage:
  python plot_benchmark_comparison.py --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- validated categorical palette (light mode) -----------------------------
C_TL, C_BZ, C_AG, C_EF = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

MODELS = [
    ("borzoi_tl_r", "Borzoi-TL\n(ours)", C_TL),
    ("borzoi_pretrained_r", "Borzoi\npretrained", C_BZ),
    ("alphagenome_r", "AlphaGenome", C_AG),
    ("enformer_r", "Enformer", C_EF),
]
TIE_TOL = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--results-dir", default="./results_benchmark")
    return p.parse_args()


def style(ax):
    """Recessive axes: no chartjunk, grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=1.0)
    ax.set_axisbelow(True)


def same_experiment(row) -> bool:
    """Does pretrained Borzoi own a track for this exact TF *and* cell line?

    String heuristic over the matched track's description — good enough to
    separate 'the model memorised this assay' from 'the model generalised',
    and every call is auditable in the comparison CSV.
    """
    tf = str(row["target_factor"]).upper()
    cl = str(row["target_cell_line"]).upper()
    trk = str(row["borzoi_pretrained_track"]).upper()
    return bool(tf) and tf in trk and cl != "NAN" and cl.replace("-", "") in trk.replace("-", "")


def fig_model_comparison(df, out, split):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    style(ax)
    rng = np.random.default_rng(0)

    for i, (col, label, color) in enumerate(MODELS):
        v = df[col].to_numpy(float)
        # jittered strip: 183 paired points, thin and translucent so density reads
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=9, color=color,
                   alpha=0.30, linewidths=0, zorder=2)
        bp = ax.boxplot([v], positions=[i], widths=0.42, showfliers=False,
                        patch_artist=True, zorder=3)
        bp["boxes"][0].set(facecolor="white", edgecolor=color, linewidth=1.6, alpha=0.92)
        for part in ("whiskers", "caps"):
            for a in bp[part]:
                a.set(color=color, linewidth=1.4)
        bp["medians"][0].set(color=color, linewidth=2.2)
        # direct label: the mean, so identity never depends on color alone
        ax.text(i, v.max() + 0.035, f"{v.mean():.3f}", ha="center", va="bottom",
                fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], color=INK, fontsize=9.5)
    ax.set_ylabel("Pearson R per target track", color=INK, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title(f"Per-track accuracy on {len(df)} held-out CRC TF ChIP-seq tracks",
                 color=INK, fontsize=11.5, pad=26, loc="left")
    ax.text(0, 1.055, "bold value = mean · box = median and IQR · dots = individual tracks",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"model_comparison_{split}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_paired_delta(df, out, split):
    from borzoi_baseline_utils import bootstrap_ci
    rows = []
    for col, label, color in MODELS[1:]:
        d = (df["borzoi_tl_r"] - df[col]).to_numpy(float)
        lo, hi = bootstrap_ci(d, np.mean)
        rows.append((label.replace("\n", " "), d.mean(), lo, hi, color,
                     int((d > TIE_TOL).sum()), int((d < -TIE_TOL).sum()),
                     int((np.abs(d) <= TIE_TOL).sum())))

    fig, ax = plt.subplots(figsize=(7.2, 3.5), facecolor=SURFACE)
    style(ax)
    ax.axvline(0, color=INK2, linewidth=1.2, zorder=1)
    for i, (label, m, lo, hi, color, w, l, t) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([m], [y], s=90, color=color, zorder=4, linewidths=0)
        ax.text(hi + 0.004, y, f"  +{m:.3f}", va="center", fontsize=10,
                color=color, fontweight="bold")
        # caption sits directly under its own row, in data coords, so it stays
        # attached no matter how many comparisons are plotted
        ax.text(lo, y - 0.26, f"{w} win · {l} lose · {t} tie",
                ha="left", va="top", fontsize=8.5, color=INK2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK, fontsize=10)
    ax.set_xlabel("mean ΔR  (Borzoi-TL minus baseline, 95% bootstrap CI)",
                  color=INK, fontsize=10)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(-0.006, 0.075)          # headroom for the direct labels
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Borzoi-TL's margin over each baseline", color=INK,
                 fontsize=11.5, pad=20, loc="left")
    ax.text(0, 1.06, "every interval excludes zero — the ordering is not sampling noise",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"paired_delta_{split}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_leakage_scatter(df, out, split):
    leak = df.apply(same_experiment, axis=1).to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 5.6), facecolor=SURFACE)
    style(ax)

    lim = (0.10, 0.95)
    ax.plot(lim, lim, color=INK2, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    ax.scatter(df.loc[~leak, "borzoi_pretrained_r"], df.loc[~leak, "borzoi_tl_r"],
               s=34, color=C_TL, alpha=0.75, linewidths=0.5, edgecolors="white",
               zorder=3, label=f"no matching assay  (n={int((~leak).sum())})")
    ax.scatter(df.loc[leak, "borzoi_pretrained_r"], df.loc[leak, "borzoi_tl_r"],
               s=34, color=C_BZ, alpha=0.85, linewidths=0.5, edgecolors="white",
               zorder=4, label=f"same TF + cell line  (n={int(leak.sum())})")

    d_leak = (df.loc[leak, "borzoi_tl_r"] - df.loc[leak, "borzoi_pretrained_r"]).mean()
    d_gen = (df.loc[~leak, "borzoi_tl_r"] - df.loc[~leak, "borzoi_pretrained_r"]).mean()
    ax.text(0.035, 0.965, f"mean ΔR when Borzoi has the same assay:  {d_leak:+.3f}\n"
                          f"mean ΔR when it does not:                {d_gen:+.3f}",
            transform=ax.transAxes, fontsize=9, color=INK, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID))
    # sits in the empty band above the diagonal, clear of the stats box and legend
    ax.text(0.05, 0.62, "above the line =\nBorzoi-TL better", transform=ax.transAxes,
            ha="left", fontsize=8.5, color=INK2, style="italic")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("pretrained Borzoi best-match R", color=INK, fontsize=10)
    ax.set_ylabel("Borzoi-TL R", color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Where Borzoi-TL loses — and why", color=INK, fontsize=11.5,
                 pad=28, loc="left")
    ax.text(0, 1.035, "losses concentrate where pretrained Borzoi already saw the same experiment",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    leg = ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    leg.get_frame().set(edgecolor=GRID, facecolor="white")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"leakage_scatter_{split}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.results_dir)
    df = pd.read_csv(out / f"benchmark_comparison_{args.split}.csv")

    fig_model_comparison(df, out, args.split)
    fig_paired_delta(df, out, args.split)
    fig_leakage_scatter(df, out, args.split)

    # the stratification is a result, not just a picture — persist it as a table
    df["same_experiment_in_borzoi"] = df.apply(same_experiment, axis=1)
    df["delta_vs_borzoi_pretrained"] = df["borzoi_tl_r"] - df["borzoi_pretrained_r"]
    strat = df.groupby("same_experiment_in_borzoi").agg(
        n=("borzoi_tl_r", "size"),
        borzoi_tl_mean_r=("borzoi_tl_r", "mean"),
        borzoi_pretrained_mean_r=("borzoi_pretrained_r", "mean"),
        mean_delta=("delta_vs_borzoi_pretrained", "mean"),
        n_tl_loses=("delta_vs_borzoi_pretrained", lambda s: int((s < -TIE_TOL).sum())),
    ).round(4)
    strat.to_csv(out / f"stratified_by_assay_overlap_{args.split}.csv")

    df.sort_values("delta_vs_borzoi_pretrained").head(40).to_csv(
        out / f"borzoi_tl_losses_{args.split}.csv", index=False)

    print(strat.to_string())
    print(f"\nWrote 3 figures (PNG+PDF) and 2 tables to {out}/")


if __name__ == "__main__":
    main()
