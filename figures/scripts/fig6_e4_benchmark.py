"""Fig 6 (manuscript 'Fig. 4' E4/E5): supervised baselines vs Borzoi-TL, Pearson r only.
Sources (saved results only; re-runnable, uses whatever fold seeds exist at run time):
  results_phase2/e4_ridge/e4_comparison.csv            orig split: pretrained val-selected, TL original ckpt, ridge, probe
  results_phase2/e5_seeds/e5_seed_comparison.csv       orig split: TL cache-recipe seeds 1-5
  results_phase2/e5_folds/e5_fold_comparison.csv       folds F2-F4: pretrained, TL seeds, ridge
  results_phase2/e4_sensitivity/A_cluster_cis.csv      orig paired deltas (TL - ridge/probe) with 3 CI types
  results_phase2/e5_folds/e5_fold_vs_ridge.csv         fold paired deltas (TL seed - ridge) with 3 CI types
Run: OMP_NUM_THREADS=2 ./torch-venv/bin/python figures/scripts/fig6_e4_benchmark.py
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
PRIM, ALL = "not_identical152", "all183"
FOLDS = ["orig", "F2", "F3", "F4"]
TEST = {"orig": "chr9,chr22 (359 tiles)", "F2": "chr6,chr21 (414)", "F3": "chr11,chr19,chr20 (490)", "F4": "chr7,chr18 (456)"}
COL = {"pretrained": S.MODEL_COL["Borzoi (pretrained)"], "TL": S.MODEL_COL["Borzoi-TL"],
       "probe": "#009E73", "ridge": "#D55E00"}
LAB = {"pretrained": "Borzoi pretrained (val-selected)", "TL": "Borzoi-TL", "probe": "linear probe (trunk 1920-d)",
       "ridge": "ridge[pre_raw:union1888]"}

# ---------------- panel a values ----------------
rows = []
def add(fold, model, kind, track_set, r):
    rows.append(dict(panel="a", fold=fold, model=model, kind=kind, track_set=track_set, mean_pearson_r=r.mean_pearson_r,
                     ci_tileblock_lo=r.r_lo, ci_tileblock_hi=r.r_hi))
e4 = pd.read_csv(R / "e4_ridge/e4_comparison.csv")
for ts in (PRIM, ALL):
    for key, model, kind in [("pretrained_Borzoi_valsel_nonoracle (REF; R2 val-calibrated)", "pretrained", "point"),
                             ("Borzoi-TL original checkpoint (Qing stats)", "TL", "point"),
                             ("ridge[probe:trunk1920]", "probe", "point"),
                             ("ridge[pre_raw:union1888]", "ridge", "point")]:
        add("orig", model, kind, ts, e4[(e4.model == key) & (e4.track_set == ts)].iloc[0])
sd = pd.read_csv(R / "e5_seeds/e5_seed_comparison.csv")
for ts in (PRIM, ALL):
    for s in range(1, 6):
        r = sd[(sd.model == f"TL seed seed{s}") & (sd.track_set == ts)]
        if len(r):
            add("orig", "TL", f"seed{s}", ts, r.iloc[0])
fc = pd.read_csv(R / "e5_folds/e5_fold_comparison.csv")
for f in FOLDS[1:]:
    for ts in (PRIM, ALL):
        d = fc[(fc.fold == f) & (fc.track_set == ts)]
        add(f, "pretrained", "point", ts, d[d.model == "pretrained_valsel (REF)"].iloc[0])
        add(f, "ridge", "point", ts, d[d.model == "ridge pre_raw union1888"].iloc[0])
        for m in sorted(x for x in d.model.unique() if x.startswith("TL seed")):
            add(f, "TL", m.split()[-1], ts, d[d.model == m].iloc[0])
A = pd.DataFrame(rows)

# ---------------- panel b values ----------------
rows = []
ci = pd.read_csv(R / "e4_sensitivity/A_cluster_cis.csv")
ci = ci[(ci.a == "TL") & ci.b.isin(["ridge_union1888_raw", "probe_trunk1920"])]
for _, r in ci.iterrows():
    comp = "ridge" if r.b.startswith("ridge") else "probe"
    rows.append(dict(panel="b", fold="orig", head="TL original ckpt", comparator=comp, track_set=r.track_set, delta_r=r.delta,
                     ci_tileblock_lo=r.tileblock_lo, ci_tileblock_hi=r.tileblock_hi, ci_TFcluster_lo=r.TFcluster_lo,
                     ci_TFcluster_hi=r.TFcluster_hi, ci_cellcluster_lo=r.cellcluster_lo, ci_cellcluster_hi=r.cellcluster_hi,
                     n_TF_clusters=r.n_TF_clusters, n_cellline_clusters=r.n_cellline_clusters))
fv = pd.read_csv(R / "e5_folds/e5_fold_vs_ridge.csv")
for _, r in fv.iterrows():
    rows.append(dict(panel="b", fold=r.fold, head=r["head"], comparator="ridge", track_set=r.track_set, delta_r=r.delta_r_TL_minus_comparator,
                     ci_tileblock_lo=r.tileblock_lo, ci_tileblock_hi=r.tileblock_hi, ci_TFcluster_lo=r.TFcluster_lo,
                     ci_TFcluster_hi=r.TFcluster_hi, ci_cellcluster_lo=r.cellcluster_lo, ci_cellcluster_hi=r.cellcluster_hi,
                     n_TF_clusters=r.n_TF_clusters, n_cellline_clusters=r.n_cellline_clusters))
B = pd.DataFrame(rows)

fig = plt.figure(figsize=(7.4, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
# ---- panel a ----
ax = fig.add_subplot(gs[0])
MODELS = ["pretrained", "TL", "probe", "ridge"]
off = {"pretrained": -0.27, "TL": -0.09, "probe": 0.09, "ridge": 0.27}
for gi, f in enumerate(FOLDS):
    for m in MODELS:
        for ts, filled in ((PRIM, True), (ALL, False)):
            d = A[(A.fold == f) & (A.model == m) & (A.track_set == ts)]
            if not len(d):
                continue
            dx = 0.035 if ts == ALL else -0.035
            main = d[(d.kind == "point")] if f == "orig" else d
            if f != "orig" and m == "TL":   # fold TL heads: each seed with its own CI
                n = len(main)
                for k, (_, r) in enumerate(main.iterrows()):
                    x = gi + off[m] + dx + (k - (n - 1) / 2) * 0.03
                    ax.plot([x, x], [r.ci_tileblock_lo, r.ci_tileblock_hi], color=COL[m], lw=1.1, alpha=.8)
                    ax.plot(x, r.mean_pearson_r, "s", ms=4.2, mfc=COL[m] if filled else "white", mec=COL[m], mew=1.0, zorder=4)
                continue
            r = main.iloc[0]
            x = gi + off[m] + dx
            ax.plot([x, x], [r.ci_tileblock_lo, r.ci_tileblock_hi], color=COL[m], lw=1.4, zorder=2)
            ax.plot(x, r.mean_pearson_r, S.MODEL_MARK.get("Borzoi-TL" if m == "TL" else "Borzoi (pretrained)", "o") if m in ("TL", "pretrained") else ("^" if m == "probe" else "D"),
                    ms=5.5, mfc=COL[m] if filled else "white", mec=COL[m], mew=1.2, zorder=4)
            if f == "orig" and m == "TL" and ts == PRIM:   # cache-recipe seeds as individual points
                sd_ = A[(A.fold == "orig") & (A.model == "TL") & (A.track_set == PRIM) & A.kind.str.startswith("seed")]
                ax.plot(np.full(len(sd_), x + 0.07) + np.linspace(-0.02, 0.02, len(sd_)), sd_.mean_pearson_r, ".", color=S.INK, ms=3.5, zorder=5)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{f}\ntest {TEST[f]}" for f in FOLDS], fontsize=7)
ax.set_ylabel("Mean per-track Pearson r")
ax.set_ylim(0.58, 0.87)
S.hgrid(ax, "y")
S.panel_label(ax, "a", dx=-0.09)
leg = [Line2D([0], [0], marker={"pretrained": "o", "TL": "s", "probe": "^", "ridge": "D"}[m], color="none", mfc=COL[m], mec=COL[m], label=LAB[m]) for m in MODELS]
leg += [Line2D([0], [0], marker="o", color="none", mfc="k", mec="k", label="primary 152 tracks (filled, left)"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec="k", label="all 183 tracks (hollow, right)"),
        Line2D([0], [0], marker=".", color="none", mfc=S.INK, mec=S.INK, label="orig: TL cache-recipe seeds 1-5 (152)"),
        Line2D([0], [0], color=S.INK2, lw=1.4, label="95% tile-block CI")]
ax.legend(handles=leg, ncol=2, loc="upper left", fontsize=6.5, columnspacing=1.0, handletextpad=0.3)
# ---- panel b ----
ax = fig.add_subplot(gs[1])
slots = []   # (fold, head, comparator)
od = B[B.fold == "orig"]
slots += [("orig", "TL original ckpt", "ridge"), ("orig", "TL original ckpt", "probe")]
for f in FOLDS[1:]:
    for h in sorted(B[(B.fold == f)]["head"].unique()):
        slots.append((f, h, "ridge"))
ypos = {s: -i for i, s in enumerate(slots)}
offs = {"tileblock": (-0.17, 3.0, "-"), "TFcluster": (0.0, 1.8, "-"), "cellcluster": (0.17, 1.0, (0, (2, 1.5)))}
ax.axvline(0, color=S.INK2, lw=0.8)
for s, y in ypos.items():
    for ts, filled in ((PRIM, True), (ALL, False)):
        r = B[(B.fold == s[0]) & (B["head"] == s[1]) & (B.comparator == s[2]) & (B.track_set == ts)].iloc[0]
        col = COL[s[2]]
        if filled:
            for k, (o, lw, ls) in offs.items():
                ax.plot([r[f"ci_{k}_lo"], r[f"ci_{k}_hi"]], [y + o] * 2, color=col, lw=lw, ls=ls, solid_capstyle="butt", alpha=.9)
        ax.plot(r.delta_r, y + (0 if filled else 0.0), "o" if filled else "o", ms=4.8 if filled else 7, mfc=col if filled else "none",
                mec="black" if filled else col, mew=0.8 if filled else 1.0, zorder=4 if filled else 3)
ax.set_yticks(list(ypos.values()))
ax.set_yticklabels([f"{f}: {h.replace('TL ', 'TL ')} − {LAB[c].split(' (')[0]}" if f != "orig" else f"orig: TL original ckpt − {LAB[c].split(' (')[0]}" for (f, h, c) in slots], fontsize=6.8)
ax.set_xlabel("Paired Δ Pearson r (Borzoi-TL − comparator), primary 152 tracks")
S.hgrid(ax, "x")
S.panel_label(ax, "b", dx=-0.09)
leg = [Line2D([0], [0], color=S.INK2, lw=3, label="tile-block 95% CI (10-tile blocks)"),
       Line2D([0], [0], color=S.INK2, lw=1.8, label="TF-cluster 95% CI (62 clusters)"),
       Line2D([0], [0], color=S.INK2, lw=1.0, ls=(0, (2, 1.5)), label="cell-line-cluster 95% CI (11 clusters)"),
       Line2D([0], [0], marker="o", color="none", mfc="k", mec="k", ms=4.8, label="primary 152 (filled)"),
       Line2D([0], [0], marker="o", color="none", mfc="none", mec="k", ms=7, label="all 183 (hollow, no CI shown)")]
ax.legend(handles=leg, loc="center right", fontsize=6.5, ncol=1)
lo = B[B.track_set == PRIM][["ci_cellcluster_lo", "ci_TFcluster_lo"]].min().min()
ax.set_xlim(lo - 0.01, 0.005)
ax.set_ylim(min(ypos.values()) - 0.6, 0.6)
fig.suptitle("E4/E5 supervised baselines vs Borzoi-TL, 183 CRC TF ChIP tracks (152 non-identical primary), hg38", fontsize=9, y=0.94)
out = pd.concat([A, B], ignore_index=True)
S.save_figure(fig, "fig6_e4_benchmark", out)
