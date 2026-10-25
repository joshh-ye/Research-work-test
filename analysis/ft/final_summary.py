"""Final FT program summary from EXISTING locked test sufficient statistics (CPU only; no predictions, no new test evaluation).
Reads per-tile stats npz (n,sx,sy,sxy,sx2,sy2,sse; tiles x 183), sums over tiles, computes per-track Pearson r (pooled over all tiles/bins),
squared Pearson r and prediction R^2 = 1 - SS_res/SS_tot.  Comparator = E4 ridge[pre_raw:union1888] (same stats file compare_ft.py uses).
Outputs (results_ft/final/): per_track_test_r.csv (183 rows x configs), test_summary.csv (mean/median/IQR/fraction improved, 183 and 152),
per_cell_line_test_r.csv, per_TF_test_r.csv, final_run_table.csv (runs: trainable, peak mem, runtime, epoch time, val r), fig_ft_test.png, fig_ft_test_values.csv.
Usage: ./torch-venv/bin/python analysis/ft/final_summary.py"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/priorityD")); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
from estats import pearson_from_sums, pearson_r2, prediction_r2
OUT = ROOT / "results_ft/final"; OUT.mkdir(parents=True, exist_ok=True)
FT = ROOT / "results_ft/ft_test_eval"
E4 = ROOT / "results_phase2/e4_ridge/orig_pre_raw/union1888_test_stats.npz"
E_ENS8 = "ENS_FT-S03+FT-R1+FT-R2+FT-R3+FT-S04+FT-L1+FT-L2+FT-L3"
CONFIGS = {   # label -> stats path ; order = primary first
    "E4_ridge": E4,
    "FT-T01_PRIMARY_ENS8_rc": FT / (E_ENS8 + "#tta=rc/test_stats.npz"),
    "FT-T02_ENS8_none": FT / (E_ENS8 + "#tta=none/test_stats.npz"),
    "FT-T03_ENS4partial_rc": FT / "ENS_FT-S03+FT-R1+FT-R2+FT-R3#tta=rc/test_stats.npz",
    "FT-T04_BLEND_ENS8_C61_w0.7_rc": FT / ("BLEND_" + E_ENS8.split("ENS_")[1].join(["ENS_", ""]) + "|FT-C61|w=0.7#tta=rc/test_stats.npz"),
    "FT-T05_FT-C61_ridge": ROOT / "results_ft/cache_exps/test_eval/FT-C61/test_stats.npz",
    "FT-T06_FT-S03_rc": FT / "FT-S03#tta=rc/test_stats.npz",
}
CONFIGS["E4_ridge_repro_pooled128"] = ROOT / "results_ft/cache_exps/test_eval/E4-ridge-repro/test_stats.npz"
KEYS = ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]


def per_track(path):
    d = np.load(path); s = {k: np.asarray(d[k], float).sum(0) for k in KEYS}
    r = pearson_from_sums(s["n"], s["sx"], s["sy"], s["sxy"], s["sx2"], s["sy2"])
    return r, pearson_r2(r), prediction_r2(s["sse"], s["sy"], s["sy2"], s["n"]), np.load(path)["gid"]


def main():
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (ex.track_idx.values == np.arange(183)).all()
    ident = (ex.Borzoi_class == "identical_experiment").values
    masks = {"all183": np.ones(183, bool), "not_identical152": ~ident}
    assert ident.sum() == 31
    R, R2, PR2, gids = {}, {}, {}, {}
    for k, p in CONFIGS.items():
        assert Path(p).exists(), p
        R[k], R2[k], PR2[k], gids[k] = per_track(p)
    for k in gids: assert (gids[k] == gids["E4_ridge"]).all(), k          # same tile order as E4
    pt = ex[["track_idx", "name", "cell_line", "TF", "Borzoi_class"]].copy(); pt["not_identical152"] = ~ident
    for k in CONFIGS: pt[f"r_{k}"] = R[k]
    for k in CONFIGS:
        if k != "E4_ridge": pt[f"delta_r_{k}_minus_E4"] = R[k] - R["E4_ridge"]
    pt.to_csv(OUT / "per_track_test_r.csv", index=False)
    rows = []
    for k in CONFIGS:
        for ms, m in masks.items():
            r = R[k][m]; d = (R[k] - R["E4_ridge"])[m]; q1, q3 = np.nanpercentile(r, [25, 75])
            rows.append(dict(config=k, track_set=ms, n_tracks=int(m.sum()), mean_pearson_r=np.nanmean(r), median_pearson_r=np.nanmedian(r), q25=q1, q75=q3, iqr=q3 - q1,
                             mean_pearson_r_squared=np.nanmean(R2[k][m]), mean_prediction_R2=np.nanmean(PR2[k][m]),
                             frac_tracks_improved_vs_E4=float((d > 1e-9).mean()) if k != "E4_ridge" else np.nan,
                             n_improved=int((d > 1e-9).sum()) if k != "E4_ridge" else np.nan,
                             mean_delta_r=float(np.nanmean(d)) if k != "E4_ridge" else np.nan, median_delta_r=float(np.nanmedian(d)) if k != "E4_ridge" else np.nan))
    S = pd.DataFrame(rows); S.to_csv(OUT / "test_summary.csv", index=False)
    # per cell line / per TF (152 non-identical and 183), mean per-track r of the primary vs E4, and delta
    P, B = "FT-T01_PRIMARY_ENS8_rc", "E4_ridge"
    for col, fn in [("cell_line", "per_cell_line_test_r.csv"), ("TF", "per_TF_test_r.csv")]:
        out = []
        for ms, m in masks.items():
            df = pd.DataFrame({"g": ex[col].values, "p": R[P], "b": R[B]})[m]
            df["d"] = df.p - df.b
            g = df.groupby("g").agg(n_tracks=("p", "size"), mean_r_primary=("p", "mean"), mean_r_E4=("b", "mean"), mean_delta_r=("d", "mean"), n_improved=("d", lambda x: int((x > 0).sum()))).reset_index()
            g.insert(0, "track_set", ms); out.append(g)
        pd.concat(out).rename(columns={"g": col}).to_csv(OUT / fn, index=False)
    ci_table()
    figure(pt, R, masks)
    run_table()
    print(S.round(4).to_string())


def ci_table():
    """collect the locked test_comparison.csv files (compare_ft.py output; 10,000 resamples, seed 20261020) into one table (test_ci_table.csv)"""
    dirs = {k: Path(p).parent for k, p in CONFIGS.items() if k != "E4_ridge"}
    rows = []
    for k, d in dirs.items():
        c = pd.read_csv(d / "test_comparison.csv")
        for ts in ["not_identical152", "all183"]:
            a = c[(c.comparison == "new (absolute)") & (c.track_set == ts)].iloc[0]
            dl = c[(c.comparison == "new - ridge[pre_raw:union1888]") & (c.track_set == ts)].iloc[0]
            tl = c[(c.comparison == "TL - new") & (c.track_set == ts)].iloc[0]
            rows.append(dict(config=k, track_set=ts, mean_pearson_r=a.mean_pearson_r, r_tileblock_lo=a.tileblock_lo, r_tileblock_hi=a.tileblock_hi,
                             mean_pearson_r_squared=a.mean_pearson_r_squared, mean_prediction_R2=a.mean_prediction_R2, mean_r_E4=dl.mean_b, delta_r_vs_E4=dl.delta,
                             delta_tileblock_lo=dl.tileblock_lo, delta_tileblock_hi=dl.tileblock_hi, delta_TFcluster_lo=dl.TFcluster_lo, delta_TFcluster_hi=dl.TFcluster_hi,
                             delta_twoway_tileXTF_lo=dl.twoway_tileblock_x_TF_lo, delta_twoway_tileXTF_hi=dl.twoway_tileblock_x_TF_hi,
                             delta_cellline_lo=dl.cellcluster_lo, delta_cellline_hi=dl.cellcluster_hi,
                             delta_twoway_tileXcell_lo=dl.twoway_tileblock_x_cell_lo, delta_twoway_tileXcell_hi=dl.twoway_tileblock_x_cell_hi,
                             n_TF_clusters=dl.n_TF_clusters, n_cellline_clusters=dl.n_cellline_clusters, new_minus_TL_delta_r=-tl.delta))
    pd.DataFrame(rows).to_csv(OUT / "test_ci_table.csv", index=False)


def figure(pt, R, masks):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P, B = "FT-T01_PRIMARY_ENS8_rc", "E4_ridge"
    x, y = R[B], R[P]; d = y - x; ni = masks["not_identical152"]
    vals = pd.DataFrame({"track_idx": pt.track_idx, "not_identical152": ni, "r_E4_ridge": x, "r_primary_ENS8_rc": y, "delta_r": d}); vals.to_csv(OUT / "fig_ft_test_values.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(x[~ni], y[~ni], s=16, c="#c0392b", marker="s", label="31 identical-experiment tracks", alpha=.8)
    ax[0].scatter(x[ni], y[ni], s=16, c="#1f4e8c", label="152 non-identical tracks (claim set)", alpha=.8)
    lo, hi = np.nanmin([x, y]) - .02, np.nanmax([x, y]) + .02; ax[0].plot([lo, hi], [lo, hi], "k--", lw=.8)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi); ax[0].set_aspect("equal")
    ax[0].set_xlabel("E4 ridge per-track test Pearson r"); ax[0].set_ylabel("Primary (8-member fine-tuned ensemble, RC TTA) per-track test Pearson r")
    ax[0].yaxis.label.set_size(8); ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_title("Per-track test r (chr9+chr22)", fontsize=10)
    bins = np.linspace(np.nanmin(d), np.nanmax(d), 40)
    ax[1].hist([d[ni], d[~ni]], bins=bins, stacked=True, color=["#1f4e8c", "#c0392b"], label=["152 non-identical", "31 identical"])
    ax[1].axvline(0, color="k", lw=.8); ax[1].axvline(np.nanmean(d[ni]), color="#1f4e8c", ls="--", lw=1.2, label=f"mean delta r (152) = {np.nanmean(d[ni]):.4f}")
    ax[1].set_xlabel("delta r = primary - E4 ridge (per track)"); ax[1].set_ylabel("number of tracks"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Delta r: {int((d[ni] > 0).sum())}/152 non-identical tracks improved", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_ft_test.png", dpi=150); plt.close(fig)


def run_table():
    """one row per training run: trainable params, peak mem, runtime (registry), mean epoch time, per-epoch full-val r (stdout.log), best val r"""
    reg = pd.read_csv(ROOT / "research_management/finetuning/FT_REGISTRY.csv").set_index("exp_id")
    rows = []
    for d in sorted((ROOT / "results_ft/runs").glob("FT-*")):
        if d.name.startswith("FT-P") and d.name != "FT-P12": continue
        log = (d / "stdout.log").read_text() if (d / "stdout.log").exists() else ""
        m = re.search(r"mode=(\w+) trainable=([\d,]+)", log)
        ep = re.findall(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] EPOCH (\d+) full val \((\d+) tiles, (\d+)s\): loss ([\d.]+) mean r183 ([\d.]+) r152 ([\d.]+)", log)
        mem = [float(x) for x in re.findall(r"mem ([\d.]+)GB", log)]
        cfg = json.load(open(d / "run_cmd.json"))["argv"]; arg = lambda f, dflt="": cfg[cfg.index(f) + 1] if f in cfg else dflt
        mt = pd.read_csv(d / "metrics.csv") if (d / "metrics.csv").exists() else None
        from datetime import datetime
        t = [datetime.strptime("2026-" + e[0], "%Y-%m-%d %H:%M:%S") for e in ep]
        r183 = [float(e[5]) for e in ep]; r152 = [float(e[6]) for e in ep]
        rows.append(dict(run=d.name, mode=m.group(1) if m else "", unfreeze_or_lora=arg("--unfreeze", arg("--lora-targets")), replicate=arg("--replicate", "0"), seed=arg("--seed"),
                         loss=arg("--loss", "poisson"), epochs=arg("--epochs"), lr_trunk=arg("--lr-trunk"), lr_head=arg("--lr-head"), head=arg("--head"),
                         trainable_params=int(m.group(2).replace(",", "")) if m else np.nan,
                         peak_logged_mem_GB=max(mem) if mem else np.nan, registry_peak_gpu_GB=reg.gpu_mem_gb.get(d.name, np.nan),
                         median_sec_per_step=float(mt.sec_per_step.median()) if mt is not None else np.nan,
                         mean_epoch_interval_min=float(np.mean(np.diff(t)).total_seconds() / 60) if len(t) > 1 else np.nan,
                         registry_runtime_s=reg.runtime_s.get(d.name, np.nan), best_val_r183=max(r183) if r183 else np.nan, best_val_r152=max(r152) if r152 else np.nan,
                         val_r183_by_epoch=";".join(map(str, r183))))
    pd.DataFrame(rows).to_csv(OUT / "final_run_table.csv", index=False)


if __name__ == "__main__":
    main()
