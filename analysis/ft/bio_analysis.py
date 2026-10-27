"""Post hoc, descriptive per-track biological analysis of the locked FT test result (EXP-019/020 lock).
FT = r_FT-T01_PRIMARY_ENS8_rc (8-member fine-tuned Borzoi, RC averaging) vs E4 ridge[pre_raw:union1888].
hg38, test chr9+chr22, 359 tiles x 896 x 128 bp; Pearson r per track pooled over all test bins (NOT r^2, NOT prediction R^2).
Primary set = 152 non-identical tracks; all-183 repeated and always labelled. CPU only. One command:
  CUDA_VISIBLE_DEVICES= ./torch-venv/bin/python analysis/ft/bio_analysis.py
Outputs: reports/ft_bio/*.csv, figures_ft_sprint/*.{png,pdf}, figures_ft_sprint/data/*.csv"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
ROOT = Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/priorityD"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from cmp_lib import load_stats, metrics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

E4 = ROOT / "results_phase2/e4_ridge"; EV = ROOT / "reports/qing_deliverables/evaluation_evidence"
PATHS = dict(TL=EV / "tl_per_tile_sufficient_stats_shared128.npz",
             probe=E4 / "orig_probe/trunk1920_test_stats.npz",
             pretrained=E4 / "pretrained_valsel_orig/valsel_test_stats.npz",
             ridge_stats=E4 / "orig_pre_raw/union1888_test_stats.npz")
TGT = Path("/tmp/yejz1_ft/pooled128/test_tgt.npy")
OUTR = ROOT / "reports/ft_bio"; OUTF = ROOT / "figures_ft_sprint"; (OUTF / "data").mkdir(parents=True, exist_ok=True); OUTR.mkdir(parents=True, exist_ok=True)
SEED, B = 20261020, 10000
FT = "r_FT-T01_PRIMARY_ENS8_rc"
ARCH = {"CTCF", "RAD21", "SMC1A", "SMC3", "NIPBL"}
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9", pink="#CC79A7", grey="#777777")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def savefig(fig, name, df):
    df.to_csv(OUTF / "data" / f"{name}.csv", index=False)
    fig.savefig(OUTF / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTF / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

def pearson_all(S):
    return metrics(S, np.ones((1, S["n"].shape[0])))["pearson_r"][0]

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

MIN_K = 8

def cluster_boot(vals, clusters, rng, B=B):
    """mean of per-track values; resample TF clusters with replacement; pooled mean over tracks in drawn clusters."""
    vals = np.asarray(vals, float); cl, inv = np.unique(np.asarray(clusters), return_inverse=True); K = len(cl)
    s = np.bincount(inv, vals, K); c = np.bincount(inv, minlength=K).astype(float)
    idx = rng.integers(0, K, (B, K)); m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, [2.5, 97.5]), K

def cat_table(df, col, dcol, setname, rng, min_n=1):
    rows = []
    for g, d in df.groupby(col, observed=True):
        if len(d) < min_n: continue
        ci, K = cluster_boot(d[dcol], d.TF, rng)
        v = d[dcol].values; ct = np.percentile(rng.choice(v, (B, len(v))).mean(1), [2.5, 97.5])   # tracks treated as independent (reference only)
        if K < MIN_K: ci = np.array([np.nan, np.nan])   # < MIN_K TF clusters: percentile cluster bootstrap not meaningful (RV-FT-04)
        rows.append(dict(set=setname, category_var=col, category=str(g), n_tracks=len(d), n_TF_clusters=K, mean_delta=d[dcol].mean(),
                         median_delta=d[dcol].median(), ci95_lo=ci[0], ci95_hi=ci[1], ci_note=('' if K >= MIN_K else f'<{MIN_K} TF clusters: no cluster CI'), ci95_track_lo=ct[0], ci95_track_hi=ct[1], n_negative=int((d[dcol] < 0).sum()),
                         mean_ridge_r=d.r_ridge.mean(), mean_FT_r=d.r_FT.mean()))
    return rows

def main():
    fin = ROOT / "research_management/finetuning/final"
    pt = pd.read_csv(fin / "per_track_test_r.csv"); ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (pt.track_idx.values == np.arange(183)).all() and (ex.track_idx.values == np.arange(183)).all() and (pt.name == ex.name).all()
    ident = (ex.Borzoi_class == "identical_experiment").values
    assert (pt.not_identical152.values == ~ident).all() and (~ident).sum() == 152
    # baselines from per-tile sufficient stats
    S = {k: load_stats(p) for k, p in PATHS.items()}
    r_stats = {k: pearson_all(v) for k, v in S.items() if k != "ridge_stats"}
    chk = np.nanmax(np.abs(pearson_all(S["ridge_stats"]) - pt.r_E4_ridge.values))
    print("max |ridge r (stats) - ridge r (csv)| =", chk); assert chk < 1e-6
    # signal metrics from TEST targets only
    T = np.load(TGT, mmap_mode="r"); assert T.shape == (359, 896, 183), T.shape
    T = np.asarray(T, np.float64).reshape(-1, 183); nb = T.shape[0]; k = int(round(0.01 * nb))
    mean = T.mean(0); sd = T.std(0); cv = np.where(mean > 0, sd / np.where(mean > 0, mean, 1), np.nan)
    fpos = (T > 0).mean(0); tot = T.sum(0)
    top = np.partition(T, nb - k, axis=0)[nb - k:].sum(0); peak = np.where(tot > 0, top / np.where(tot > 0, tot, 1), np.nan)
    print("bins/track", nb, "top1% bins", k, "neg values in targets:", int((T < 0).sum()))
    df = pd.DataFrame(dict(track_idx=np.arange(183), name=ex.name, cell_line=ex.cell_line, TF=ex.TF, GSM=ex.GSM, ENCSR=ex.ENCSR,
                           GEO_series_or_xref=ex.GEO_series_or_xref, Borzoi_class=ex.Borzoi_class,
                           Borzoi_n_same_target_ChIP_tracks=ex.Borzoi_n_same_target_ChIP_tracks, in_claim_set_152=~ident))
    df["r_ridge"] = pt.r_E4_ridge.values; df["r_TL"] = r_stats["TL"]; df["r_probe_trunk1920"] = r_stats["probe"]
    df["r_pretrained_valsel"] = r_stats["pretrained"]; df["r_FT"] = pt[FT].values
    df["delta_FT_minus_ridge"] = df.r_FT - df.r_ridge
    df["rel_delta_headroom"] = df.delta_FT_minus_ridge / (1 - df.r_ridge); df["rel_delta_over_ridge_r"] = df.delta_FT_minus_ridge / df.r_ridge
    df["FT_minus_TL"] = df.r_FT - df.r_TL; df["FT_minus_probe"] = df.r_FT - df.r_probe_trunk1920
    df["FT_minus_pretrained"] = df.r_FT - df.r_pretrained_valsel
    df["sig_mean"] = mean; df["sig_cv"] = cv; df["sig_frac_bins_pos"] = fpos; df["sig_top1pct_signal_frac"] = peak
    df["data_source"] = np.where(ex.ENCSR.notna(), "ENCSR_present", "GSM_only")
    df["minimal_class"] = np.where(df.TF == "POLR2A", "RNA polymerase (POLR2A)", np.where(df.TF == "H2AZ", "histone variant (H2AZ)", "TF/chromatin-associated (unclassified)"))
    df["TF_group3"] = np.where(df.TF == "POLR2A", "POLR2A", np.where(df.TF.isin(ARCH), "architectural / cohesin-CTCF", "other"))
    df["TF_group4"] = np.where(df.TF.isin(["POLR2A", "CTCF", "RAD21"]), df.TF, "other")
    df.to_csv(OUTR / "per_track_ft_bio.csv", index=False)
    assert abs(df[df.in_claim_set_152].delta_FT_minus_ridge.mean() - 0.0172813341286222) < 1e-9
    ci = pd.read_csv(fin / "test_ci_table.csv"); ci = ci[(ci.config == "FT-T01_PRIMARY_ENS8_rc") & (ci.track_set == "not_identical152")].iloc[0]
    rng = np.random.default_rng(SEED)
    sets = {"not_identical152": df[df.in_claim_set_152], "all183": df}
    # 2. negatives / top gains
    for sn, d in sets.items():
        cols = ["track_idx", "name", "cell_line", "TF", "GSM", "ENCSR", "Borzoi_class", "Borzoi_n_same_target_ChIP_tracks", "r_ridge", "r_FT", "delta_FT_minus_ridge", "rel_delta_headroom", "sig_mean", "sig_top1pct_signal_frac"]
        d[d.delta_FT_minus_ridge < 0].sort_values("delta_FT_minus_ridge")[cols].to_csv(OUTR / f"negative_delta_{sn}.csv", index=False)
        d.sort_values("delta_FT_minus_ridge", ascending=False).head(15)[cols].to_csv(OUTR / f"top15_gains_{sn}.csv", index=False)
    # 3. category comparisons
    rows = []; dfq = {}
    for sn, d in sets.items():
        d = d.copy(); sm = ["sig_mean", "sig_cv", "sig_frac_bins_pos", "sig_top1pct_signal_frac", "r_ridge"]
        for c in sm:
            d["Q_" + c] = pd.qcut(d[c], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        d["TF_group3"] = d.TF_group3
        rows.append(dict(set=sn, category_var="ALL", category="all", n_tracks=len(d), n_TF_clusters=d.TF.nunique(), mean_delta=d.delta_FT_minus_ridge.mean(),
                         median_delta=d.delta_FT_minus_ridge.median(), ci95_lo=(cia := cluster_boot(d.delta_FT_minus_ridge, d.TF, rng)[0])[0],
                         ci95_hi=cia[1], n_negative=int((d.delta_FT_minus_ridge < 0).sum()), mean_ridge_r=d.r_ridge.mean(), mean_FT_r=d.r_FT.mean()))
        for col in ["TF_group3", "TF_group4", "minimal_class", "cell_line", "Borzoi_class", "data_source"] + ["Q_" + c for c in sm]:
            rows += cat_table(d, col, "delta_FT_minus_ridge", sn, rng)
        dfq[sn] = d
    cat = pd.DataFrame(rows); cat.to_csv(OUTR / "category_delta_summary.csv", index=False)
    # Spearman
    srows = []
    for sn, d in sets.items():
        tfm = d.groupby("TF")[["delta_FT_minus_ridge", "rel_delta_headroom", "sig_mean", "sig_cv", "sig_frac_bins_pos", "sig_top1pct_signal_frac", "r_ridge"]].mean()
        for lvl, dd in (("track", d), ("TF_mean", tfm)):
            for y in ["delta_FT_minus_ridge", "rel_delta_headroom"]:
                for x in ["sig_mean", "sig_cv", "sig_frac_bins_pos", "sig_top1pct_signal_frac", "r_ridge"]:
                    ok = dd[[x, y]].dropna()
                    rho, p = spearmanr(ok[x], ok[y]); srows.append(dict(set=sn, level=lvl, y=y, x=x, n=len(ok), spearman_rho=rho, p=p))
    sp = pd.DataFrame(srows)
    sp["q_BH_within_set_level"] = np.nan
    for (sn, lvl), g in sp.groupby(["set", "level"]): sp.loc[g.index, "q_BH_within_set_level"] = bh(g.p.values)
    sp["q_BH_within_set_all_levels"] = np.nan
    for sn, g in sp.groupby("set"): sp.loc[g.index, "q_BH_within_set_all_levels"] = bh(g.p.values)
    sp.to_csv(OUTR / "spearman_delta_vs_metrics.csv", index=False)
    # 4. FT vs frozen-trunk baselines
    brow = []
    for sn, d in sets.items():
        for b, c in (("TL", "FT_minus_TL"), ("probe_trunk1920", "FT_minus_probe"), ("pretrained_valsel", "FT_minus_pretrained")):
            ci_, K = cluster_boot(d[c], d.TF, rng)
            brow.append(dict(set=sn, comparison=f"FT - {b}", n=len(d), mean=d[c].mean(), median=d[c].median(), ci95_TFcluster_lo=ci_[0], ci95_TFcluster_hi=ci_[1], n_FT_better=int((d[c] > 0).sum()), n_FT_worse=int((d[c] < 0).sum())))
        brow.append(dict(set=sn, comparison="FT - ridge", n=len(d), mean=d.delta_FT_minus_ridge.mean(), median=d.delta_FT_minus_ridge.median(), ci95_TFcluster_lo=(cib := cluster_boot(d.delta_FT_minus_ridge, d.TF, rng)[0])[0], ci95_TFcluster_hi=cib[1], n_FT_better=int((d.delta_FT_minus_ridge > 0).sum()), n_FT_worse=int((d.delta_FT_minus_ridge < 0).sum())))
        for b, c in (("TL", "r_TL"), ("probe_trunk1920", "r_probe_trunk1920"), ("pretrained_valsel", "r_pretrained_valsel"), ("ridge", "r_ridge"), ("FT", "r_FT")):
            brow.append(dict(set=sn, comparison=f"mean r of {b}", n=len(d), mean=d[c].mean(), median=d[c].median()))
    pd.DataFrame(brow).to_csv(OUTR / "ft_vs_frozen_baselines.csv", index=False)
    # CRC-relevant factors
    crc = ["CDX2", "HNF4A", "GATA6", "KLF5", "TCF7L2", "SOX9", "ELF3", "TCF7L1", "FOXA2", "PROX1"]
    cols = ["name", "cell_line", "TF", "Borzoi_class", "in_claim_set_152", "r_ridge", "r_FT", "delta_FT_minus_ridge", "rel_delta_headroom", "FT_minus_TL", "FT_minus_probe"]
    df[df.TF.isin(crc)].sort_values(["TF", "name"])[cols].to_csv(OUTR / "crc_relevant_factors.csv", index=False)
    df["rank_delta_152"] = df.delta_FT_minus_ridge.where(df.in_claim_set_152).rank(ascending=False)
    df[df.TF.isin(crc)].groupby("TF").agg(n=("name", "size"), n152=("in_claim_set_152", "sum"), mean_delta=("delta_FT_minus_ridge", "mean"), mean_rel=("rel_delta_headroom", "mean"), best_rank_in_152=("rank_delta_152", "min")).to_csv(OUTR / "crc_relevant_factors_by_TF.csv")
    # ---- figures
    d152 = dfq["not_identical152"]; di = df[~df.in_claim_set_152]
    f = pd.read_csv(fin / "test_ci_table.csv")
    ann = (f"{int((d152.delta_FT_minus_ridge > 0).sum())}/152 improved, mean delta {ci.delta_r_vs_E4:+.4f}, TF-cluster 95% CI [{ci.delta_TFcluster_lo:.4f}, {ci.delta_TFcluster_hi:.4f}], "
           f"two-way tile-block x TF [{ci.delta_twoway_tileXTF_lo:.4f}, {ci.delta_twoway_tileXTF_hi:.4f}]")
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(d152.r_ridge, d152.r_FT, s=14, c=OI["blue"], alpha=.8, label=f"non-identical (n=152)")
    ax.scatter(di.r_ridge, di.r_FT, s=22, marker="^", facecolors="none", edgecolors=OI["verm"], label=f"identical to Borzoi training (n=31; not in claim set)")
    lo = min(df.r_ridge.min(), df.r_FT.min()) - .02; ax.plot([lo, 1], [lo, 1], color="k", lw=.8, ls="--"); ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
    ax.set_xlabel("E4 ridge on pretrained Borzoi, test Pearson r"); ax.set_ylabel("8-member fine-tuned ensemble (RC), test Pearson r")
    ax.set_title("Per-track test r, chr9+chr22 (hg38)\n" + ann, fontsize=6.5); ax.legend(loc="lower right", fontsize=7); ax.set_aspect("equal")
    savefig(fig, "fig_paired_ridge_vs_ft", df[["name", "TF", "cell_line", "in_claim_set_152", "r_ridge", "r_FT", "delta_FT_minus_ridge"]])
    dl = d152.delta_FT_minus_ridge.values; rg = np.random.default_rng(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw=dict(width_ratios=[2, 1]))
    a1.hist(dl, bins=30, color=OI["blue"], alpha=.85); a1.axvline(0, c="k", lw=1); a1.axvline(np.median(dl), c=OI["verm"], ls="--", label=f"median {np.median(dl):+.4f}")
    a1.axvline(dl.mean(), c=OI["orange"], ls=":", label=f"mean {dl.mean():+.4f}"); a1.set_xlabel("delta r = FT - ridge (per track)"); a1.set_ylabel("tracks")
    a1.set_title(f"n=152 non-identical tracks; {int((dl < 0).sum())} negative, {int((dl > 0).sum())} positive", fontsize=9); a1.legend(fontsize=7)
    a2.scatter(rg.uniform(-.2, .2, len(dl)), dl, s=8, c=OI["blue"], alpha=.7); a2.axhline(0, c="k", lw=1); a2.set_xticks([]); a2.set_ylabel("delta r"); a2.set_title("strip (jittered)", fontsize=9)
    savefig(fig, "fig_delta_distribution", d152[["name", "TF", "cell_line", "delta_FT_minus_ridge"]])
    c1 = cat[(cat.set == "not_identical152") & (cat.category_var == "TF_group4")]; c2 = cat[(cat.set == "not_identical152") & (cat.category_var == "cell_line")].sort_values("mean_delta")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    for a, c, t in ((ax[0, 0], c1, "TF label group (n tracks)"), (ax[0, 1], c2, "cell line (n tracks; CI over TF clusters)")):
        y = np.arange(len(c)); one = c.n_TF_clusters.values < MIN_K
        lo_ = np.where(one, c.ci95_track_lo, c.ci95_lo); hi_ = np.where(one, c.ci95_track_hi, c.ci95_hi)
        for yy, m_, l_, h_, o_ in zip(y, c.mean_delta, lo_, hi_, one):
            a.errorbar(m_, yy, xerr=[[m_ - l_], [h_ - m_]], fmt="o", mfc="white" if o_ else OI["blue"], color=OI["verm"] if o_ else OI["blue"], capsize=2)
        a.text(0.99, 0.02, f"red/open = <{MIN_K} TF clusters: track-level CI only (tracks not independent)", transform=a.transAxes, ha="right", fontsize=6)
        a.axvline(0, c="k", lw=.8); a.set_yticks(y); a.set_yticklabels([f"{x} (n={n})" for x, n in zip(c.category, c.n_tracks)]); a.set_xlabel("mean delta r (FT - ridge), 95% TF-cluster bootstrap CI"); a.set_title(t, fontsize=9)
    for a, x, xl in ((ax[1, 0], "r_ridge", "baseline ridge r (headroom)"), (ax[1, 1], "sig_top1pct_signal_frac", "fraction of signal in top 1% bins (peakiness)")):
        a.scatter(d152[x], d152.delta_FT_minus_ridge, s=12, c=OI["blue"], alpha=.7); a.axhline(0, c="k", lw=.8)
        r_, p_ = spearmanr(d152[x], d152.delta_FT_minus_ridge); q_ = sp[(sp.set == "not_identical152") & (sp.level == "track") & (sp.y == "delta_FT_minus_ridge") & (sp.x == x)].q_BH_within_set_all_levels.iloc[0]
        a.set_xlabel(xl); a.set_ylabel("delta r"); a.set_title(f"n=152; Spearman rho={r_:.2f}, p={p_:.1e}, BH q={q_:.1e}", fontsize=9)
    fig.suptitle("Delta r by category (152 non-identical tracks; TF groups: other = all remaining factors)", fontsize=10); fig.tight_layout()
    savefig(fig, "fig_delta_by_category", pd.concat([c1.assign(panel="TF_group4"), c2.assign(panel="cell_line")]))
    print("done"); print(cat[cat.category_var.isin(["ALL", "TF_group3", "TF_group4", "minimal_class", "Borzoi_class", "data_source"])].round(4).to_string())

if __name__ == "__main__":
    main()
