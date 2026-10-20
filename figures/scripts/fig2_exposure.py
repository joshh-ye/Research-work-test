"""Fig 2: exposure of pretrained Borzoi to the test targets, and tile-fold overlap.
a) per-track delta r (Borzoi-TL minus pretrained Borzoi, val-selected mapping) by exposure class (all 359 tiles).
   Class mean +- TF-cluster 95% CI are read from exposure_stratified_stats.csv (class:* rows, val_selected).
   LoVo identical vs non-identical rows are descriptive (track means, no CI).
b) fraction of tile bp in Borzoi/Enformer train/valid/test folds (exposure_audit/tile_fold_overlap.tsv).
Run: ./torch-venv/bin/python figures/scripts/fig2_exposure.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style as S

S.apply()
R = S.ROOT / "results_phase2/exposure_stratified"
pt = pd.read_csv(R / "per_track_metrics_by_exposure.csv")
pt = pt[pt.tile_set == "all_359_tiles"].copy()
st = pd.read_csv(R / "exposure_stratified_stats.csv")
st = st[(st.tile_set == "all_359_tiles") & (st.metric == "pearson_r") & (st.mapping == "val_selected")]
te = pd.read_csv(S.ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
pt = pt.merge(te[["track_idx", "Borzoi_class"]].rename(columns={"Borzoi_class": "cls_audit"}), on="track_idx")
assert (pt.cls_audit == pt.Borzoi_class).all()

cats = [(c, EXPO := S.EXPO_LABEL[c], pt[pt.Borzoi_class == c], f"class:{c}") for c in S.EXPO_ORDER]
lovo = pt[pt.cell_line == "LoVo"]
cats.insert(1, ("LoVo_identical", "  of which LoVo, identical", lovo[lovo.Borzoi_class == "identical_experiment"], None))
cats.insert(2, ("LoVo_non_identical", "  LoVo, non-identical", lovo[lovo.Borzoi_class != "identical_experiment"], None))
cats.insert(0, ("ALL", "all 183 tracks", pt, "all_183"))

fig = plt.figure(figsize=(7.4, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
rng = np.random.default_rng(1)
recs, ylabels = [], []
for i, (key, lab, d, sub) in enumerate(cats):
    y = len(cats) - 1 - i
    col = S.EXPO_COL.get(key.replace("LoVo_identical", "identical_experiment"), S.MUTED if key == "ALL" else S.OTHER_COL)
    if key == "LoVo_non_identical":
        col = S.OTHER_COL
    x = d.delta_r_valsel.values
    ax.scatter(x, y + rng.uniform(-0.22, 0.22, len(x)), s=7, color=col, alpha=0.55, lw=0, zorder=2)
    rec = dict(group=key, label=lab.strip(), n_tracks=len(d), mean_delta_r=x.mean(), ci_TFcluster_lo=np.nan, ci_TFcluster_hi=np.nan,
               n_TF_clusters=np.nan, ci_note="descriptive, no CI")
    if sub:
        s = st[st.subset == sub].iloc[0]
        assert s.n_tracks == len(d) and np.isclose(s.delta_TL_minus_pretrained, x.mean()), key   # plotted mean == source mean
        rec.update(ci_TFcluster_lo=s.ci_TFcluster_lo, ci_TFcluster_hi=s.ci_TFcluster_hi, n_TF_clusters=s.n_TF_clusters,
                   ci_note="TF-cluster bootstrap 95% CI (source file)")
        ax.plot([s.ci_TFcluster_lo, s.ci_TFcluster_hi], [y, y], color="black", lw=2, zorder=3)
    ax.plot(x.mean(), y, "D", color="white", mec="black", ms=5, mew=1.1, zorder=4)
    recs.append(rec)
    ylabels.append(f"{lab} (n={len(d)})")
ax.axvline(0, color=S.INK2, lw=0.8)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(ylabels[::-1])
ax.set_xlabel("Per-track Δr = r(Borzoi-TL) − r(pretrained Borzoi, val-selected)")
S.hgrid(ax, "x")
S.panel_label(ax, "a", dx=-0.62)
ax.set_title("Pearson r difference by exposure class", fontsize=8.5, loc="left")
ax.text(0.99, 0.01, "diamond = class mean; bar = 95% TF-cluster CI", transform=ax.transAxes, ha="right", fontsize=6.5, color=S.INK2)
dfa = pd.DataFrame(recs)
dfa["panel"] = "a"
dfa_pts = pt[["track_idx", "TF", "cell_line", "Borzoi_class", "delta_r_valsel"]].assign(panel="a_points")

# panel b
fo = pd.read_csv(S.ROOT / "exposure_audit/tile_fold_overlap.tsv", sep="\t")
fo = fo[fo.tile_region == "tile_full524288"].copy()
fo["fold_source"] = fo.fold_source.str.replace(r"\(.*\)", "", regex=True)
order = ["Borzoi_bed", "Borzoi_input", "Enformer_input", "Basenji2_bed"]
bx = fig.add_subplot(gs[1])
fcols = [("frac_train", "#E69F00", "train"), ("frac_valid", "#56B4E9", "valid"), ("frac_test", "#009E73", "test"), ("frac_uncovered", "#d0d0d0", "uncovered")]
rows = []
ytick, ylab, y = [], [], 0
for split in ["test", "val"]:
    for fs in order:
        r = fo[(fo.project_split == split) & (fo.fold_source == fs)].iloc[0]
        left = 0
        for c, colr, nm in fcols:
            bx.barh(y, r[c], left=left, color=colr, height=0.7, edgecolor="white", lw=0.8)
            if r[c] > 0.095:
                bx.text(left + r[c] / 2, y, f"{r[c]:.2f}", ha="center", va="center", fontsize=6.5)
            left += r[c]
        assert abs(left - 1) < 2e-3
        rows.append(dict(panel="b", project_split=split, fold_source=fs, n_tiles=r.n_tiles, **{c: r[c] for c, _, _ in fcols}))
        ytick.append(y); ylab.append(f"{split}: {fs.replace('_', ' ')}")
        y += 1
    y += 0.6
bx.set_yticks(ytick); bx.set_yticklabels(ylab); bx.invert_yaxis()
bx.set_xlim(0, 1); bx.set_xlabel("Fraction of tile bp (524 kb tile)")
S.panel_label(bx, "b", dx=-0.7)
bx.set_title("Tile overlap with model folds", fontsize=8.5, loc="left")
from matplotlib.patches import Patch
bx.legend(handles=[Patch(color=c, label=n) for _, c, n in fcols], ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.18), columnspacing=0.8, handlelength=1)
dfb = pd.DataFrame(rows)
out = pd.concat([dfa, dfa_pts, dfb], ignore_index=True)
S.save_figure(fig, "fig2_exposure", out)
