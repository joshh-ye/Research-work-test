"""Fig 5: E2 MPRA validation, within-region P1 AUROC (stratified Mann-Whitney AUROC; not Pearson r / r^2 / R^2).
INTERIM mode (default): results_phase2/e2_validation/interim_{alphagenome,enformer}/tables/{p1,baselines,secondary_P1,p2}.tsv
FINAL mode: if results_phase2/e2_validation/final/tables/ exists, it is used instead (all models listed in its p1.tsv,
  incl. borzoi_pretrained / borzoi_tl) and the interim caveat is replaced by the table's own `status` column.
Outputs: fig5_e2_p1 (a: primary P1 + baselines, b: sensitivity analyses) and fig5S_e2_p2_caveat (supplementary P2, SD-017 caveat).
CI: 95% percentile region-bootstrap (10,000 resamples; from source tables). p-values not plotted.
Run: ./torch-venv/bin/python figures/scripts/fig5_e2_interim.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style as S

S.apply()
E2 = S.ROOT / "results_phase2/e2_validation"
FINAL = E2 / "final" / "tables"
final = FINAL.is_dir()
rd = lambda p: pd.read_csv(p, sep="\t")
if final:
    p1, base, sec, p2 = (rd(FINAL / f"{n}.tsv") for n in ("p1", "baselines", "secondary_P1", "p2"))
    srcs = [str(FINAL)]
else:
    T = {k: E2 / f"interim_{k}/tables" for k in ("alphagenome", "enformer")}
    p1 = pd.concat([rd(T[k] / "p1.tsv") for k in T], ignore_index=True)
    sec = pd.concat([rd(T[k] / "secondary_P1.tsv") for k in T], ignore_index=True)
    p2 = pd.concat([rd(T[k] / "p2.tsv") for k in T], ignore_index=True)
    b = [rd(T[k] / "baselines.tsv") for k in T]
    pd.testing.assert_frame_equal(b[0], b[1])       # baselines are model-independent -> identical in both runs
    base = b[0]
    srcs = [str(v) for v in T.values()]
def pretty(m):
    m = m.lower()
    if m.startswith("borzoi_tl") or "borzoi-tl" in m: return "Borzoi-TL"
    if m.startswith("borzoi"): return "Borzoi (pretrained)"
    if m.startswith("alphagenome"): return "AlphaGenome"
    if m.startswith("enformer"): return "Enformer"
    return m
p1["model_label"] = p1.model.map(pretty)
p1["kind"] = "model"
base["model_label"] = base.model.map({"baseline_neg_log10_dist_tss": "distance to TSS (baseline)", "baseline_chen_annot_count": "annotation count (baseline)"})
base["kind"] = "baseline"
a = pd.concat([p1, base], ignore_index=True)
status = p1.status.iloc[0] if "status" in p1 else ""
interim = (not final) or ("interim" in " ".join(p1.status.astype(str)).lower())
ttl = ("INTERIM, unadjusted, Holm deferred; Borzoi/Borzoi-TL pending" if interim else "FINAL; Holm across headline models (holm_p in source table)")
n_pos, n_neg, n_reg = int(p1.n_pos.iloc[0]), int(p1.n_neg.iloc[0]), int(p1.n_regions_informative.iloc[0])

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=1.3))
ya = a.reset_index(drop=True)
for i, r in ya.iterrows():
    y = len(ya) - 1 - i
    col = S.MODEL_COL.get(r.model_label, S.MODEL_COL["baseline"])
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2)
    ax.plot(r.auroc_stratified, y, S.MODEL_MARK.get(r.model_label, "o"), color=col, ms=6.5, mec="white", mew=0.6, zorder=3)
    ax.text(0.995, y, f"{r.auroc_stratified:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color=S.INK2)
    assert np.isclose(r.auroc_stratified, a.loc[i, "auroc_stratified"])
ax.axvline(0.5, color=S.INK2, lw=0.8, ls="--")
ax.set_yticks(range(len(ya))); ax.set_yticklabels(ya.model_label[::-1])
ax.set_xlim(0.38, 0.86); ax.set_xlabel("Within-region stratified AUROC")
S.hgrid(ax, "x"); S.panel_label(ax, "a", dx=-0.75)
ax.set_title(f"Primary P1 (MPRA pos vs neg)\n{n_pos} pos / {n_neg:,} neg; {n_reg} regions", fontsize=7.5, loc="left")

# panel b: sensitivity analyses, models only
fam = sec[sec.family.isin(["sens_labels", "sens_exclusions", "sens_clustering"]) & sec.model.isin(p1.model)].copy()
fam["model_label"] = fam.model.map(pretty)
fam["analysis"] = fam.label if False else fam.description
short = lambda d: (d.replace("S1: positives = any ST2 flag>=1", "S1 any-flag positives").replace("S2: per-cell-line label ", "S2 label ")
                   .replace("S3: non-palindromic only", "S3 non-palindromic only").replace("S4: exclude padded/N windows (AG: shifted windows)", "S4 exclude padded/shifted windows")
                   .replace("S5: signal (lead SNP) clusters", "S5 cluster by signal"))
fam["analysis"] = fam.description.map(short)
order = list(dict.fromkeys(fam.analysis))
mods = list(dict.fromkeys(fam.model_label))
off = np.linspace(-0.18, 0.18, len(mods)) if len(mods) > 1 else [0]
for k, m in enumerate(mods):
    for _, r in fam[fam.model_label == m].iterrows():
        y = len(order) - 1 - order.index(r.analysis) + off[k]
        col = S.MODEL_COL.get(m, S.MODEL_COL["baseline"])
        bx.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.6)
        bx.plot(r.auroc_stratified, y, S.MODEL_MARK.get(m, "o"), color=col, ms=5, mec="white", mew=0.5, zorder=3)
bx.axvline(0.5, color=S.INK2, lw=0.8, ls="--")
bx.set_yticks(range(len(order))); bx.set_yticklabels(order[::-1])
bx.set_xlim(0.45, 0.86); bx.set_xlabel("Within-region stratified AUROC")
S.hgrid(bx, "x"); S.panel_label(bx, "b", dx=-1.0, dy=1.02)
bx.set_title("Sensitivity analyses (exploratory; BH within family)", fontsize=8, loc="left")
from matplotlib.lines import Line2D
bx.legend(handles=[Line2D([0], [0], marker=S.MODEL_MARK.get(m, "o"), color=S.MODEL_COL.get(m), lw=1.6, label=m) for m in mods], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=6.5)
fig.suptitle(f"E2 MPRA validation, P1 AUROC (hg38; 95% region-bootstrap CI): {ttl}", fontsize=8.5, y=1.02, color=("#D55E00" if interim else S.INK))
out = pd.concat([a.assign(panel="a")[["panel", "kind", "model", "model_label", "n_scored", "auroc_stratified", "ci_lo", "ci_hi", "n_pos", "n_neg", "n_regions_informative"]],
                 fam.assign(panel="b", kind="sensitivity")[["panel", "kind", "model", "model_label", "analysis", "n_scored", "auroc_stratified", "ci_lo", "ci_hi", "n_pos", "n_neg", "n_regions_informative"]]], ignore_index=True)
out["mode"] = "final" if final else "interim"
out["source"] = ";".join(srcs)
S.save_figure(fig, "fig5_e2_p1", out)

# supplementary P2 (direction): fraction concordant, caveat
p2 = p2.copy(); p2["model_label"] = p2.model.map(pretty)
f2, ax2 = plt.subplots(figsize=(5.2, 2.5))
for i, r in p2.reset_index(drop=True).iterrows():
    y = len(p2) - 1 - i; col = S.MODEL_COL.get(r.model_label, "#999")
    ax2.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2); ax2.plot(r.frac, y, S.MODEL_MARK.get(r.model_label, "o"), color=col, ms=6.5, mec="white")
    assert np.isclose(r.frac, r.n_concordant / r.n, atol=1e-9)
ax2.axvline(0.5, color=S.INK2, ls="--", lw=0.8)
ax2.set_yticks(range(len(p2))); ax2.set_yticklabels(p2.model_label[::-1]); ax2.set_ylim(-0.6, len(p2) - 0.4)
ax2.set_xlim(0, 1); ax2.set_xlabel("Fraction of variants with model direction concordant with MPRA logFC")
S.hgrid(ax2, "x")
ax2.set_title("SUPPLEMENTARY - P2 direction: SIGN CONVENTION UNRESOLVED (SD-017); not interpretable", fontsize=7.5, loc="left", color="#D55E00")
f2.text(0.01, -0.12, "Concordance <0.5 may reflect MPRA logFC being ALT/REF vs REF/ALT; excluded from the framing rule.\n"
        f"n = {int(p2.n.iloc[0])} variants (as in source p2 table); 95% bootstrap CI. {'INTERIM, unadjusted, Holm deferred.' if interim else ''}", fontsize=6.5, color=S.INK2, va="top")
o2 = p2[["model", "model_label", "n", "n_concordant", "frac", "ci_lo", "ci_hi"]].assign(caveat="sign convention unresolved (SD-017)", mode=out["mode"].iloc[0])
S.save_figure(f2, "fig5S_e2_p2_caveat", o2)
