"""Fig 3: per-track Pearson r, Borzoi-TL vs pretrained Borzoi (val-selected mapping), 183 test tracks (chr9+chr22, all 359 tiles).
Source: results_phase2/exposure_stratified/per_track_metrics_by_exposure.csv (tile_set=all_359_tiles); colours: Borzoi_class / cell_line.
Run: ./torch-venv/bin/python figures/scripts/fig3_track_scatter.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.apply()
R = S.ROOT / "results_phase2/exposure_stratified"
d = pd.read_csv(R / "per_track_metrics_by_exposure.csv")
d = d[d.tile_set == "all_359_tiles"].reset_index(drop=True)
assert len(d) == 183
st = pd.read_csv(R / "exposure_stratified_stats.csv")
s0 = st[(st.tile_set == "all_359_tiles") & (st.metric == "pearson_r") & (st.mapping == "val_selected") & (st.subset == "all_183")].iloc[0]
assert np.isclose(d.tl_pearson_r.mean(), s0.mean_TL) and np.isclose(d.valsel_pearson_r.mean(), s0.mean_pretrained)
top = d.cell_line.value_counts().index[:7].tolist()
d["cell_line_group"] = np.where(d.cell_line.isin(top), d.cell_line, "other (5 lines, 15 tracks)" if False else "Other")
ccol = {c: S.CELL_COLS[i] for i, c in enumerate(top)}
ccol["Other"] = S.OTHER_COL
d["n_other"] = (d.cell_line_group == "Other").sum()

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08))
lim = (0.0, 0.95)
for ax, key, title in ((axes[0], "Borzoi_class", "Coloured by exposure class"), (axes[1], "cell_line_group", "Coloured by cell line")):
    ax.plot(lim, lim, color=S.INK2, lw=0.8, ls="--", zorder=1)
    if key == "Borzoi_class":
        for c in S.EXPO_ORDER[::-1]:
            m = d.Borzoi_class == c
            ax.scatter(d.valsel_pearson_r[m], d.tl_pearson_r[m], s=14, color=S.EXPO_COL[c], alpha=0.85, lw=0.3, ec="white", label=f"{S.EXPO_LABEL[c]} (n={m.sum()})", zorder=2)
    else:
        for c in top + ["Other"]:
            m = d.cell_line_group == c
            lab = f"{c} (n={m.sum()})" if c != "Other" else f"other cell lines/tissues (n={m.sum()})"
            ax.scatter(d.valsel_pearson_r[m], d.tl_pearson_r[m], s=14, color=ccol[c], alpha=0.85, lw=0.3, ec="white", label=lab, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel("Pretrained Borzoi per-track Pearson r (val-selected)")
    S.hgrid(ax, "both")
    ax.legend(loc="lower right", fontsize=6.3, handletextpad=0.2, labelspacing=0.25)
axes[0].set_ylabel("Borzoi-TL per-track Pearson r")
axes[0].text(0.03, 0.93, f"above diagonal: TL better\nTL better in {(d.tl_pearson_r > d.valsel_pearson_r).sum()}/183 tracks", transform=axes[0].transAxes, fontsize=6.5, color=S.INK2, va="top")
S.panel_label(axes[0], "a", dx=-0.16); S.panel_label(axes[1], "b", dx=-0.06)
out = d[["track_idx", "bw_filename", "TF", "cell_line", "cell_line_group", "Borzoi_class", "tl_pearson_r", "valsel_pearson_r", "delta_r_valsel"]]
S.save_figure(fig, "fig3_track_scatter", out)
