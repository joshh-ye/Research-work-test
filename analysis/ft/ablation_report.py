"""EXP-020 ablation summary + figures (CPU).  Reads the locked ablation evaluations (results_ft/ablation/test_eval/*/), the validation selection
(results_ft/ablation/val_selection.json), per-member inference timing and the primary/E4 records; writes
research_management/finetuning/ablation_summary.csv, ablation_per_track_test_r.csv and figures_ft_sprint/fig_{individual_vs_ensemble,ensemble_size}.{png,pdf}.
All ablation test numbers are post hoc relative to the primary lock (TEST_LOCK.csv) and descriptive."""
import sys, csv, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified")); sys.path.insert(0, str(ROOT / "figures"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ablation import MEMBERS, ARM, REPL, OUT, LOCK
from predict_ft import r_from_stats
from ftlib import masks

FIGD = ROOT / "figures_ft_sprint"; FIGD.mkdir(exist_ok=True)
RM = ROOT / "research_management/finetuning"
mk = masks(); M152 = mk["not_identical152"]
sel = json.load(open(OUT / "val_selection.json"))
lock = list(csv.DictReader(open(LOCK)))
E4_152, E4_183 = 0.740563026278276, 0.7426215288327686
sec = {r: json.load(open(OUT / "test_members" / r / "info.json"))["seconds"] for r in MEMBERS}


def row_of(cmp, comparison, ts):
    x = cmp[(cmp.comparison == comparison) & (cmp.track_set == ts)]
    return x.iloc[0]


rows, pt = [], {}
for L in lock:
    cid = L["config_id"]; od = OUT / "test_eval" / cid.replace(":", "_").replace("/", "_")
    cmp = pd.read_csv(od / "test_comparison.csv")
    w = json.loads(L["weights"]); used = [m for m in MEMBERS if w[m] > 0]
    r = r_from_stats(dict(np.load(od / "test_stats.npz"))); pt[cid] = r
    ab152, ab183 = row_of(cmp, "new (absolute)", "not_identical152"), row_of(cmp, "new (absolute)", "all183")
    d152, d183 = row_of(cmp, "new - ridge[pre_raw:union1888]", "not_identical152"), row_of(cmp, "new - ridge[pre_raw:union1888]", "all183")
    e4 = None
    rows.append(dict(config_id=cid, n_members=len(used), members="+".join(used), arms="/".join(sorted({ARM[m] for m in used})),
                     forward_passes_per_tile=2 * len(used), est_gpu_s_test=round(sum(sec[m] for m in used)),
                     val_r183=float(L["val_r_all183"]), val_r152=float(L["val_r_152"]),
                     test_r152=ab152.mean_pearson_r, test_r152_tileblock_lo=ab152.tileblock_lo, test_r152_tileblock_hi=ab152.tileblock_hi,
                     test_r183=ab183.mean_pearson_r, test_r2_152=ab152.mean_pearson_r_squared, test_predR2_152=ab152.mean_prediction_R2,
                     delta152=d152.delta, d152_TF_lo=d152.TFcluster_lo, d152_TF_hi=d152.TFcluster_hi, d152_2way_lo=d152.twoway_tileblock_x_TF_lo,
                     d152_2way_hi=d152.twoway_tileblock_x_TF_hi, d152_tile_lo=d152.tileblock_lo, d152_tile_hi=d152.tileblock_hi,
                     d152_cell_lo=d152.cellcluster_lo, d152_cell_hi=d152.cellcluster_hi,
                     delta183=d183.delta, d183_TF_lo=d183.TFcluster_lo, d183_TF_hi=d183.TFcluster_hi,
                     n_tracks_improved_152=int((r[M152] - 0 > 0).sum()) if False else None))
df = pd.DataFrame(rows)
# per-track r vs E4 ridge (from the primary per-track table, same E4 stats)
ptab = pd.read_csv(RM / "final/per_track_test_r.csv")
e4 = ptab["r_E4_ridge"].to_numpy()
df["n_tracks_improved_152"] = [int((pt[c][M152] > e4[M152]).sum()) for c in df.config_id]
df["n_tracks_improved_183"] = [int((pt[c] > e4).sum()) for c in df.config_id]
prim = ptab["r_FT-T01_PRIMARY_ENS8_rc"].to_numpy()
g8 = [c for c in df.config_id if c.startswith("ABL:greedy8")][0]
df["max_abs_track_diff_vs_locked_primary"] = [float(np.nanmax(np.abs(pt[c] - prim))) if c == g8 else np.nan for c in df.config_id]
gain8 = df.loc[df.config_id == g8, "delta152"].iloc[0]
df["fraction_of_8member_gain_152"] = df.delta152 / gain8
df.to_csv(RM / "ablation_summary.csv", index=False)
pd.DataFrame({"track_idx": np.arange(183), "name": ptab.name, "not_identical152": M152, "r_E4_ridge": e4,
              **{f"r_{c}": pt[c] for c in df.config_id}}).to_csv(RM / "ablation_per_track_test_r.csv", index=False)

# ---------------- figures
try:
    import style; style.apply()
except Exception:
    pass
C_PART, C_LORA, C_ENS, C_RIDGE = "#0072B2", "#E69F00", "#009E73", "#7a7a7a"


def save(fig, name):
    for ext, kw in [("png", dict(dpi=300)), ("pdf", {})]:
        fig.savefig(FIGD / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


# Fig A: individual vs ensembles (test r on 152 tracks, tile-block 95% CI; validation r as open markers)
order = [f"ABL:single:{m}#tta=rc" for m in MEMBERS] + [c for c in df.config_id if c.startswith("ABL:best2")] + \
        [c for c in df.config_id if c.startswith("ABL:best4")] + [g8, "ABL:weighted8#tta=rc"]
lab = MEMBERS + ["best 2 (val)", "best 4 (val)", "all 8 (primary)", "weighted 8 (val)"]
sub = df.set_index("config_id").loc[order]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
x = np.arange(len(order))
col = [C_PART if ARM[m] == "partial4" else C_LORA for m in MEMBERS] + [C_ENS] * 4
ax.errorbar(x, sub.test_r152, yerr=[sub.test_r152 - sub.test_r152_tileblock_lo, sub.test_r152_tileblock_hi - sub.test_r152], fmt="none", ecolor="#bbbbbb", lw=1, zorder=1)
ax.scatter(x, sub.test_r152, c=col, s=28, zorder=3, label="test (chr9+chr22)")
ax.scatter(x, sub.val_r152, facecolors="none", edgecolors=col, s=28, zorder=3, label="validation (chr1+chr8)")
ax.axhline(E4_152, color=C_RIDGE, ls="--", lw=0.9); ax.text(-0.4, E4_152 - 0.0008, "E4 ridge, test 0.7406", ha="left", va="top", fontsize=7, color=C_RIDGE)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=40, ha="right")
ax.set_ylabel("mean per-track Pearson r (152 tracks)")
ax.set_title("Individual fine-tuned members vs ensembles (RC test-time averaging; n = 152 tracks, 359 test tiles)\n"
             "bars: tile-block 95% CI of test r (37 blocks, 10,000 resamples); blue partial-4, orange LoRA-r16, green ensembles; ablation test values post hoc", fontsize=7.5)
ax.set_ylim(min(sub.test_r152_tileblock_lo.min(), sub.val_r152.min()) - 0.003, sub.test_r152_tileblock_hi.max() + 0.003)
ax.legend(loc="lower left", fontsize=7); save(fig, "fig_individual_vs_ensemble")

# Fig B: ensemble size (greedy forward order chosen on validation) - delta vs ridge with TF-cluster and two-way CIs
gr = df[df.config_id.str.startswith("ABL:greedy") | (df.config_id == f"ABL:single:{sel['greedy_curve'][0]['members'][0]}#tta=rc") |
        df.config_id.str.startswith("ABL:best2")].copy()
gr = gr.drop_duplicates("n_members").sort_values("n_members")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
a = axs[0]
a.plot(gr.n_members, gr.val_r152, "o--", color=C_ENS, mfc="white", label="validation")
a.plot(gr.n_members, gr.test_r152, "o-", color=C_ENS, label="test")
singles = df[df.config_id.str.startswith("ABL:single")]
a.scatter(np.ones(len(singles)) * 0.75, singles.test_r152, s=10, color="#444444", label="each single member (test)")
a.axhline(E4_152, color=C_RIDGE, ls="--", lw=0.9)
l2 = singles[singles.config_id.str.contains("FT-L2")].test_r152.iloc[0]
a.annotate(f"FT-L2 alone {l2:.3f}\n(over-prediction in chr9:20-32 Mb)", (0.75, l2), (1.3, l2 + 0.01), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
a.set_xlabel("ensemble size (greedy order selected on validation)"); a.set_ylabel("mean Pearson r (152 tracks)"); a.set_xticks(range(1, 9)); a.legend(fontsize=6.5)
a.set_title("Mean r vs ensemble size", fontsize=8)
b = axs[1]
b.errorbar(gr.n_members - 0.08, gr.delta152, yerr=[gr.delta152 - gr.d152_TF_lo, gr.d152_TF_hi - gr.delta152], fmt="o", color=C_ENS, ms=4, capsize=2, label="TF-cluster 95% CI")
b.errorbar(gr.n_members + 0.08, gr.delta152, yerr=[gr.delta152 - gr.d152_2way_lo, gr.d152_2way_hi - gr.delta152], fmt="none", ecolor="#555555", capsize=2, label="two-way tile-block x TF 95% CI")
b.axhline(0, color="k", lw=0.6); b.set_xticks(range(1, 9))
b.set_xlabel("ensemble size"); b.set_ylabel("delta r vs E4 ridge (152 tracks)"); b.legend(fontsize=6.5, loc="lower right")
b.set_title("Gain over ridge vs size (10,000 resamples)", fontsize=8)
fig.suptitle("Ensemble-size ablation (n = 152 tracks; test chr9+chr22, 359 tiles; all points post hoc; size 8 = locked primary members)", fontsize=8)
fig.tight_layout(); save(fig, "fig_ensemble_size")
print(df[["config_id", "n_members", "val_r152", "test_r152", "delta152", "d152_TF_lo", "d152_TF_hi", "d152_2way_lo", "d152_2way_hi",
          "n_tracks_improved_152", "fraction_of_8member_gain_152", "est_gpu_s_test", "max_abs_track_diff_vs_locked_primary"]].round(4).to_string())
