"""Fig 1: model comparison on test chr9+chr22 (183 tracks, shared 128 bp grid, Pearson r).
Sources (saved results only):
  results_phase2/exposure_stratified/exposure_stratified_stats.csv  (Borzoi-TL vs pretrained Borzoi; tile_set=all_359_tiles)
  results_phase2/e7_nonoracle/{enformer,alphagenome}/{delta_stats,model_summary}.csv
Run: ./torch-venv/bin/python figures/scripts/fig1_model_comparison.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.apply()
R = S.ROOT / "results_phase2"
SUBS = [("all_183", "All 183 tracks"), ("primary_152_non_identical", "Primary 152 (non-identical)"),
        ("identical_experiment_31", "Identical-experiment 31")]
CI_COLS = {"tileblock": ("ci_tileblock_lo", "ci_tileblock_hi"), "TFcluster": ("ci_TFcluster_lo", "ci_TFcluster_hi"),
           "cellcluster": ("ci_cellcluster_lo", "ci_cellcluster_hi")}

rows = []
b = pd.read_csv(R / "exposure_stratified/exposure_stratified_stats.csv")
b = b[(b.tile_set == "all_359_tiles") & (b.metric == "pearson_r") & b.subset.isin([s for s, _ in SUBS])]
for _, r in b.iterrows():
    rows.append(dict(model="Borzoi (pretrained)", mapping={"oracle_test": "oracle", "val_selected": "val-selected"}[r.mapping],
                     stratum=r.subset, n_tracks=r.n_tracks, mean_r_model=r.mean_pretrained, mean_r_TL=r.mean_TL,
                     delta_r_TL_minus_model=r.delta_TL_minus_pretrained,
                     **{f"ci_{k}_{e}": r[c] for k, cc in CI_COLS.items() for e, c in zip(("lo", "hi"), cc)}))
for key, name in (("enformer", "Enformer"), ("alphagenome", "AlphaGenome")):
    d = pd.read_csv(R / f"e7_nonoracle/{key}/delta_stats.csv")
    d = d[(d.metric == "pearson_r") & d.mapping.isin(["oracle", "valsel"]) & d.subset.isin([s for s, _ in SUBS])]
    ms = pd.read_csv(R / f"e7_nonoracle/{key}/model_summary.csv")
    for _, r in d.iterrows():
        rows.append(dict(model=name, mapping={"oracle": "oracle", "valsel": "val-selected"}[r.mapping], stratum=r.subset,
                         n_tracks=int(r.n_tracks), mean_r_model=r.mean_model, mean_r_TL=r.mean_TL, delta_r_TL_minus_model=r.delta,
                         **{f"ci_{k}_{e}": r[c] for k, cc in CI_COLS.items() for e, c in zip(("lo", "hi"), cc)}))
        m = ms[(ms.mapping == r.mapping) & (ms.subset == r.subset)].iloc[0]
        assert abs(m.mean_pearson_r - r.mean_model) < 1e-9, (key, r.mapping, r.subset)   # sanity: two source files agree
df = pd.DataFrame(rows)
df["n_TF_clusters_note"] = "TF-cluster CI: 77 (all), 62 (primary), 25 (identical); cell-line clusters 12/11/3"
MODELS = ["Borzoi (pretrained)", "Enformer", "AlphaGenome"]
for st, _ in SUBS:   # TL mean identical across comparisons
    assert df[df.stratum == st].mean_r_TL.round(9).nunique() == 1

fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4), sharey="row", gridspec_kw=dict(hspace=0.5, wspace=0.08))
slot = {(m, mp): i for i, (m, mp) in enumerate([(m, mp) for m in MODELS for mp in ("val-selected", "oracle")][::-1])}
# Row A: mean r
for j, (st, title) in enumerate(SUBS):
    ax = axes[0, j]
    d = df[df.stratum == st]
    tl = d.mean_r_TL.iloc[0]
    ax.axvline(tl, color=S.MODEL_COL["Borzoi-TL"], lw=1, ls="--", zorder=1)
    for (m, mp), y in slot.items():
        r = d[(d.model == m) & (d.mapping == mp)].iloc[0]
        col = S.MODEL_COL[m]
        ax.plot(r.mean_r_model, y, S.MODEL_MARK[m], ms=6, mfc=col if mp == "val-selected" else "white", mec=col, mew=1.3, zorder=3)
        # sanity: plotted value equals source value
        assert np.isclose(r.mean_r_model, df[(df.stratum == st) & (df.model == m) & (df.mapping == mp)].mean_r_model.iloc[0])
    ax.set_title(f"{title}\n(n = {int(d.n_tracks.iloc[0])})", fontsize=8)
    ax.set_xlabel("Mean per-track Pearson r")
    S.hgrid(ax, "x")
    ax.set_ylim(-0.7, len(slot) - 0.3)
    lo, hi = d.mean_r_model.min(), d.mean_r_model.max()
    ax.set_xlim(min(lo, tl) - 0.02, max(hi, tl) + 0.02)
    ax.text(tl, -0.65, "Borzoi-TL", color=S.MODEL_COL["Borzoi-TL"], fontsize=7, ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5))
axes[0, 0].set_yticks(list(slot.values()))
axes[0, 0].set_yticklabels([f"{m}\n{mp}" for (m, mp) in slot])
S.panel_label(axes[0, 0], "a", dx=-0.62)
# Row B: delta r forest with 3 CI types
offs = {"tileblock": (-0.22, 3.0, "-"), "TFcluster": (0.0, 1.8, "-"), "cellcluster": (0.22, 1.0, (0, (2, 1.5)))}
for j, (st, title) in enumerate(SUBS):
    ax = axes[1, j]
    d = df[df.stratum == st]
    ax.axvline(0, color=S.INK2, lw=0.8)
    for (m, mp), y in slot.items():
        r = d[(d.model == m) & (d.mapping == mp)].iloc[0]
        col = S.MODEL_COL[m]
        for k, (off, lw, ls) in offs.items():
            ax.plot([r[f"ci_{k}_lo"], r[f"ci_{k}_hi"]], [y + off] * 2, color=col, lw=lw, ls=ls, solid_capstyle="butt", alpha=0.9)
        ax.plot(r.delta_r_TL_minus_model, y, "o", ms=4.5, mfc=col if mp == "val-selected" else "white", mec="black", mew=0.8, zorder=4)
    ax.set_xlabel("Paired Δr (Borzoi-TL − model)")
    S.hgrid(ax, "x")
    ax.set_ylim(-0.7, len(slot) - 0.3)
axes[1, 0].set_yticks(list(slot.values()))
axes[1, 0].set_yticklabels([f"{m}\n{mp}" for (m, mp) in slot])
S.panel_label(axes[1, 0], "b", dx=-0.62)
leg = [Line2D([0], [0], color=S.INK2, lw=3, label="tile-block 95% CI (10-tile blocks)"),
       Line2D([0], [0], color=S.INK2, lw=1.8, label="TF-cluster 95% CI"),
       Line2D([0], [0], color=S.INK2, lw=1.0, ls=(0, (2, 1.5)), label="cell-line-cluster 95% CI"),
       Line2D([0], [0], marker="o", color="none", mfc="k", mec="k", label="val-selected (filled)"),
       Line2D([0], [0], marker="o", color="none", mfc="white", mec="k", label="oracle (hollow)")]
fig.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Borzoi-TL vs pretrained baselines: test chr9+chr22, 183 CRC TF ChIP tracks, shared 128 bp grid", fontsize=9, y=0.98)
S.save_figure(fig, "fig1_model_comparison", df)
