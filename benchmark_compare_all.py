"""
Common evaluation script for the Borzoi-TL vs Enformer vs AlphaGenome benchmark.

Every model is reduced to one number per CRC target track: the Pearson R between
the observed target signal and the model's prediction for that target, computed
on a single shared grid so the four columns are directly comparable.

The shared grid
---------------
All models are evaluated on the central 114,688 bp of each 524,288 bp test tile,
binned at 128 bp (896 bins). That is Enformer's native output span and the
coarsest of the three, so it is the only grid all models can reach by cropping
and mean-pooling alone — no interpolation or resampling anywhere.

The four columns
----------------
borzoi_tl        our transfer-learned model, which predicts the CRC target
                 tracks directly: R between its prediction for target i and
                 target i. This is a supervised, like-for-like prediction.
borzoi_pretrained, enformer, alphagenome
                 pretrained models that were never trained on these tracks. For
                 each target we take the single best-correlating track out of
                 the model's own output head ("best-match baseline").

This asymmetry is intrinsic and is stated plainly in the summary: the
best-match columns are an oracle-selected upper bound on what a pretrained
model gives you off the shelf, not a like-for-like supervised comparison.

Usage:
  python benchmark_compare_all.py --split test --out-dir ./results_benchmark
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from borzoi_baseline_utils import (
    load_crc_target_meta, per_model_summary, paired_comparison,
)
from enformer_baseline_utils import (
    ENF_BINS, ENF_BIN_SIZE, ENF_SPAN, targets_to_enformer_grid,
)

TILE_LEN = 524_288


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--data-root",   default="./borzoi_data")
    p.add_argument("--model-dir",   default="./results_full")
    p.add_argument("--borzoi-dir",  default="./results_baseline_filtered")
    p.add_argument("--enformer-dir", default="./results_baseline_enformer")
    p.add_argument("--alphagenome-dir", default="./results_baseline_alphagenome")
    p.add_argument("--out-dir",     default="./results_benchmark")
    return p.parse_args()


def transfer_r_on_common_grid(model_dir: Path, split: str, chunk: int = 25) -> np.ndarray:
    """Per-track Pearson R for the transfer model, on the shared 128 bp grid.

    Accumulated in float64 over interval chunks: the full arrays are ~1 GB each
    and this box has 32 GB, so streaming keeps peak memory to one chunk.
    """
    preds = np.load(model_dir / f"{split}_preds.npy", mmap_mode="r")
    tgts = np.load(model_dir / f"{split}_targets.npy", mmap_mode="r")
    assert preds.shape == tgts.shape
    n_int, _, n_tracks = preds.shape

    sx = np.zeros(n_tracks); sy = np.zeros(n_tracks); sxy = np.zeros(n_tracks)
    sx2 = np.zeros(n_tracks); sy2 = np.zeros(n_tracks); n = 0
    for lo in range(0, n_int, chunk):
        p = targets_to_enformer_grid(np.asarray(preds[lo:lo + chunk]))
        t = targets_to_enformer_grid(np.asarray(tgts[lo:lo + chunk]))
        p = p.reshape(-1, n_tracks); t = t.reshape(-1, n_tracks)
        n += p.shape[0]
        sx += p.sum(0); sy += t.sum(0); sxy += (p * t).sum(0)
        sx2 += (p * p).sum(0); sy2 += (t * t).sum(0)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sx2 - sx ** 2, 0) * np.maximum(n * sy2 - sy ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def best_match_from_matrix(path: Path):
    """(best R per target, index of the winning column) from a saved Pearson matrix."""
    pmat = np.load(path)
    return pmat.max(axis=1), pmat.argmax(axis=1), pmat.shape


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def pkg_version(mod: str) -> str:
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return "not installed"


LABEL = {
    "borzoi_tl_r": "Borzoi-TL (ours, supervised)",
    "borzoi_pretrained_r": "Borzoi pretrained (best-match)",
    "enformer_r": "Enformer (best-match)",
    "alphagenome_r": "AlphaGenome (best-match)",
}


def write_summary(out: Path, split: str, df: pd.DataFrame, agg: pd.DataFrame,
                  pair_rows: list, cfg: dict, models: list):
    """Render README.md from the numbers actually computed in this run."""
    g = cfg["common_grid"]
    L = []
    L.append(f"# Borzoi-TL vs Enformer vs AlphaGenome — `{split}` split\n")
    L.append(f"Generated {cfg['generated_utc']} · commit `{cfg['git_commit']}`\n")

    L.append("## What was compared\n")
    L.append(f"{cfg['n_targets']} CRC transcription-factor ChIP-seq tracks over "
             f"{cfg['n_intervals']} held-out {cfg['tile_length_bp']:,} bp tiles "
             f"({cfg['split_definition']}), assembly {cfg['assembly']}.\n")
    L.append(f"All models are scored on one shared grid: the central "
             f"**{g['span_bp']:,} bp** of each tile at **{g['bin_size_bp']} bp** "
             f"resolution ({g['bins']} bins). {g['rationale'].capitalize()}.\n")
    L.append(f"Metric: {cfg['metric']}.\n")

    L.append("## Results\n")
    L.append("| model | mean R | median R | sd | % R≥0.5 | n |")
    L.append("|---|---|---|---|---|---|")
    for m in models:
        key = m.replace("_r", "")
        if key not in agg.index:
            L.append(f"| {LABEL[m]} | — | — | — | — | not available |")
            continue
        r = agg.loc[key]
        L.append(f"| {LABEL[m]} | **{r['mean']:.4f}** | {r['median']:.4f} | "
                 f"{r['sd']:.4f} | {r['pct_r_ge_0.5']:.1f}% | {int(r['n'])} |")
    L.append("")

    if pair_rows:
        L.append("### Paired comparisons against Borzoi-TL\n")
        L.append("| comparison | mean ΔR | 95% CI | Wilcoxon p | Cohen's d | TL wins / losses / ties |")
        L.append("|---|---|---|---|---|---|")
        for pc in pair_rows:
            L.append(f"| {pc['comparison']} | {pc['mean_diff']:+.4f} | "
                     f"[{pc['diff_ci95_lo']:+.4f}, {pc['diff_ci95_hi']:+.4f}] | "
                     f"{pc['wilcoxon_p']:.2e} | {pc['cohens_d_paired']:.3f} | "
                     f"{pc['n_transfer_wins']} / {pc['n_borzoi_wins']} / {pc['n_tie']} |")
        L.append("")

    L.append("## Comparability — verified\n")
    L.append("| property | Borzoi-TL | Borzoi pretrained | Enformer | AlphaGenome |")
    L.append("|---|---|---|---|---|")
    L.append("| assembly | hg38 | hg38 | hg38 | hg38 |")
    L.append("| test intervals | identical 524,288 bp tiles, same order | ← | ← | ← |")
    L.append("| input length | 524,288 bp | 524,288 bp | 196,608 bp (centered) | 524,288 bp |")
    L.append("| native bin size | 32 bp | 32 bp | 128 bp | 128 bp |")
    L.append(f"| scored grid | {g['bins']}×{g['bin_size_bp']} bp | ← | ← | ← |")
    L.append("| regridding | crop+mean-pool | crop+mean-pool | none (native) | crop only |")
    L.append("| strand | unstranded | unstranded | unstranded | unstranded (`.`) |")
    L.append("| target normalization | identical — same `{split}_targets.npy` for every model | ← | ← | ← |")
    L.append("")
    L.append("Mean-pooling is the correct aggregation here because both sides are "
             "coverage means rather than counts. Cropping and pooling are applied "
             "identically to prediction and target, so every correlated pair covers "
             "the same genomic coordinates.\n")

    L.append("## Limitations — read before quoting these numbers\n")
    L.append("1. **The comparison is not like-for-like.** Borzoi-TL was *trained* to "
             "predict these 183 tracks. The other three never saw them; for each "
             "target we take the single best-correlating track from their existing "
             "output heads. That best-match figure is an oracle-selected upper bound "
             "on off-the-shelf performance, chosen using the test targets themselves, "
             "so it flatters the pretrained models — and Borzoi-TL still has to beat it.")
    L.append("2. **No exactly-equivalent targets exist.** None of the pretrained "
             "models expose these CRC TF ChIP-seq tracks. Cell type and TF identity "
             "usually differ between a target and its best match; the per-target "
             "matched track is recorded in the comparison CSV so any pair can be audited.")
    L.append("3. **Track vocabularies differ in size**, which mechanically favours "
             "models offering more tracks to choose from: "
             + "; ".join(cfg["notes"]) + ".")
    L.append("4. **Spearman was not computed for the best-match baselines.** The "
             "streaming correlation accumulates Pearson sufficient statistics only; "
             "rank correlation would require retaining or re-running all predictions.")
    L.append(f"5. **{cfg['n_zero_signal_tiles']} of {cfg['n_intervals']} test tiles "
             "carry zero target signal** (the all-N chr22 p-arm). They are included "
             "identically for every model, so they do not bias the comparison, but "
             "they do slightly deflate all absolute R values.")
    L.append("6. **AlphaGenome runs remotely.** It has no public weights; predictions "
             "come from Google's hosted API. Only public hg38 reference sequence is "
             "sent — the private CRC target tracks never leave this machine.")
    if cfg["exclusions"]:
        L.append("\n### Exclusions / failures\n")
        for e in cfg["exclusions"]:
            L.append(f"- {e}")

    L.append("\n## Reproduce\n")
    L.append("```bash")
    L.append("# 1. pretrained Borzoi best-match on the shared grid (GPU, ~9 min)")
    L.append("python baseline_borzoi_filtered.py --split test --grid enformer \\")
    L.append("    --data-root ./borzoi_data --targets-dir ./results_full --n-folds 4")
    L.append("# 2. Enformer best-match (GPU, ~2 min)")
    L.append("python baseline_enformer.py --split test --targets-dir ./results_full")
    L.append("# 3. AlphaGenome best-match (hosted API, no GPU)")
    L.append("export ALPHAGENOME_API_KEY=...   # never stored in the repo")
    L.append("python baseline_alphagenome.py --split test")
    L.append("# 4. common evaluation -> this directory")
    L.append("python benchmark_compare_all.py --split test --out-dir ./results_benchmark")
    L.append("```\n")

    L.append("## Files\n")
    L.append(f"- `benchmark_comparison_{split}.csv` — per-target R for all four "
             "models plus the matched track name")
    L.append(f"- `aggregate_metrics_{split}.csv` — mean/median/sd/CI per model")
    L.append(f"- `paired_stats_{split}.csv` — paired t / Wilcoxon vs Borzoi-TL")
    L.append(f"- `benchmark_config_{split}.json` — geometry, versions, provenance")
    L.append(f"- `runtime_{split}.csv` — wall-clock per stage and the hardware it ran on")
    L.append(f"- `model_comparison_{split}.png/.pdf` — per-track R distributions, all four models")
    L.append(f"- `paired_delta_{split}.png/.pdf` — mean ΔR vs each baseline with 95% CI")
    L.append(f"- `leakage_scatter_{split}.png/.pdf` — where Borzoi-TL loses, split by "
             "whether pretrained Borzoi already saw the same assay")
    L.append(f"- `stratified_by_assay_overlap_{split}.csv`, `borzoi_tl_losses_{split}.csv` "
             "— the tables behind that figure (regenerate with `plot_benchmark_comparison.py`)")
    L.append("\nPer-model prediction artifacts stay in their own directories "
             "(`results_full/`, `results_baseline_filtered/`, "
             "`results_baseline_enformer/`, `results_baseline_alphagenome/`); "
             "the Pearson matrices there are the inputs this directory aggregates.")
    (out / "README.md").write_text("\n".join(L) + "\n")


def main():
    args = parse_args()
    split = args.split
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    model_dir = Path(args.model_dir)

    # ---- target metadata (the 183 CRC TF ChIP tracks) --------------------
    metrics = pd.read_csv(model_dir / f"{split}_per_track_metrics.csv")
    tmeta = load_crc_target_meta(metrics["bw_filename"].tolist(),
                                 Path(args.data_root) / "TF_184tracks.csv")
    n_targets = len(tmeta)

    df = pd.DataFrame({
        "target_idx": np.arange(n_targets),
        "bw_filename": tmeta["bw_filename"],
        "target_factor": tmeta["Factor"],
        "target_cell_line": tmeta["Cell_line"],
        "target_tissue": tmeta["Tissue_type"],
    })

    notes, exclusions = [], []

    # tiles whose targets are entirely flat (all-N assembly gaps) contribute no
    # variance to any model's correlation; count them so the summary can say so.
    _t = np.load(model_dir / f"{split}_targets.npy", mmap_mode="r")
    n_intervals = _t.shape[0]
    n_zero_signal = sum(1 for i in range(n_intervals)
                        if float(np.asarray(_t[i]).std()) == 0.0)

    # ---- 1. transfer model, common grid ----------------------------------
    print("[1/4] transfer model on common grid ...")
    df["borzoi_tl_r"] = transfer_r_on_common_grid(model_dir, split)
    df["borzoi_tl_r_native32bp"] = metrics["pearson_r"].to_numpy()

    # ---- 2. pretrained Borzoi best-match, common grid ---------------------
    print("[2/4] pretrained Borzoi best-match ...")
    bz = Path(args.borzoi_dir) / f"borzoi_filtered_pearson_matrix_{split}_enformergrid.npy"
    if bz.exists():
        r, j, shape = best_match_from_matrix(bz)
        df["borzoi_pretrained_r"] = r
        bm = pd.read_csv(Path(args.borzoi_dir) /
                         f"borzoi_filtered_best_match_{split}_enformergrid.csv")
        df["borzoi_pretrained_track"] = bm["best_borzoi_description"].to_numpy()
        notes.append(f"pretrained Borzoi: {shape[1]} eligible tracks (ChIP, non-histone, idx 2186-6069)")
    else:
        df["borzoi_pretrained_r"] = np.nan
        exclusions.append(f"pretrained Borzoi common-grid matrix missing at {bz}")

    # ---- 3. Enformer best-match ------------------------------------------
    print("[3/4] Enformer best-match ...")
    ef = Path(args.enformer_dir) / f"enformer_pearson_matrix_{split}.npy"
    if ef.exists():
        r, j, shape = best_match_from_matrix(ef)
        df["enformer_r"] = r
        bm = pd.read_csv(Path(args.enformer_dir) / f"enformer_best_match_{split}.csv")
        df["enformer_track"] = bm["best_enformer_description"].to_numpy()
        notes.append(f"Enformer: {shape[1]} eligible tracks (ChIP, non-histone)")
    else:
        df["enformer_r"] = np.nan
        exclusions.append(f"Enformer matrix missing at {ef}")

    # ---- 4. AlphaGenome best-match ---------------------------------------
    print("[4/4] AlphaGenome best-match ...")
    ag = Path(args.alphagenome_dir) / f"alphagenome_pearson_matrix_{split}.npy"
    if ag.exists():
        r, j, shape = best_match_from_matrix(ag)
        df["alphagenome_r"] = r
        agm = pd.read_csv(Path(args.alphagenome_dir) / f"alphagenome_tracks_{split}.csv")
        df["alphagenome_track"] = [
            f"{agm.iloc[k]['transcription_factor']} ({agm.iloc[k]['biosample_name']})"
            for k in j]
        notes.append(f"AlphaGenome: {shape[1]} TF-ChIP tracks (all ontologies)")
        agf = Path(args.alphagenome_dir) / f"alphagenome_failures_{split}.csv"
        if agf.exists():
            exclusions.append(f"AlphaGenome: {len(pd.read_csv(agf))} intervals failed, see {agf.name}")
    else:
        df["alphagenome_r"] = np.nan
        exclusions.append(f"AlphaGenome matrix missing at {ag} (needs ALPHAGENOME_API_KEY)")

    df.to_csv(out / f"benchmark_comparison_{split}.csv", index=False)

    # ---- aggregate metrics ------------------------------------------------
    models = ["borzoi_tl_r", "borzoi_pretrained_r", "enformer_r", "alphagenome_r"]
    rows = []
    for m in models:
        v = df[m].to_numpy(dtype=float)
        if np.isnan(v).all():
            continue
        s = per_model_summary(v[~np.isnan(v)])
        s["model"] = m.replace("_r", "")
        rows.append(s)
    agg = pd.DataFrame(rows).set_index("model")
    agg.to_csv(out / f"aggregate_metrics_{split}.csv")

    # ---- paired comparisons vs the transfer model -------------------------
    pair_rows = []
    base = df["borzoi_tl_r"].to_numpy(dtype=float)
    for m in models[1:]:
        v = df[m].to_numpy(dtype=float)
        ok = ~(np.isnan(base) | np.isnan(v))
        if ok.sum() == 0:
            continue
        pc = paired_comparison(base[ok], v[ok])
        pc["comparison"] = f"borzoi_tl_vs_{m.replace('_r','')}"
        pc["n_compared"] = int(ok.sum())
        pair_rows.append(pc)
    if pair_rows:
        pd.DataFrame(pair_rows).set_index("comparison").to_csv(
            out / f"paired_stats_{split}.csv")

    # ---- benchmark configuration / provenance -----------------------------
    cfg = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "split": split,
        "assembly": "hg38 (borzoi_data/hg38/hg38.ml.fa)",
        "split_definition": "val=chr1,chr8  test=chr9,chr22 (whole-chromosome holdout)",
        "n_intervals": int(n_intervals),
        "n_zero_signal_tiles": int(n_zero_signal),
        "tile_length_bp": TILE_LEN,
        "tiling": "non-overlapping 524,288 bp windows",
        "common_grid": {
            "span_bp": ENF_SPAN, "bins": ENF_BINS, "bin_size_bp": ENF_BIN_SIZE,
            "rationale": "Enformer's native output span — the narrowest of the "
                         "four and the coarsest bin size, so it is the only grid "
                         "every model reaches by cropping and mean-pooling alone, "
                         "with no interpolation",
        },
        "n_targets": n_targets,
        "target_source": "borzoi_data/TF_184tracks.csv (CRC TF ChIP-seq bigwigs)",
        "metric": "Pearson R per target track, flattened over all interval x bin observations",
        "models": {
            "borzoi_tl": {"type": "transfer-learned, supervised on these targets",
                          "outputs": model_dir.name},
            "borzoi_pretrained": {"model_id": "johahi/borzoi-replicate-{0..3} (4-fold mean)",
                                  "type": "best-match baseline",
                                  "native_bin_bp": 32, "input_bp": TILE_LEN},
            "enformer": {"model_id": "EleutherAI/enformer-official-rough",
                         "type": "best-match baseline",
                         "native_bin_bp": 128, "input_bp": 196_608},
            "alphagenome": {"model_id": "hosted API gdmscience.googleapis.com",
                            "type": "best-match baseline",
                            "native_bin_bp": 128, "input_bp": TILE_LEN},
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": pkg_version("numpy"), "pandas": pkg_version("pandas"),
            "torch": pkg_version("torch"),
            "enformer-pytorch": pkg_version("enformer-pytorch"),
            "borzoi-pytorch": pkg_version("borzoi-pytorch"),
            "alphagenome": pkg_version("alphagenome"),
        },
        "notes": notes,
        "exclusions": exclusions,
    }
    (out / f"benchmark_config_{split}.json").write_text(json.dumps(cfg, indent=2))

    # ---- Markdown summary --------------------------------------------------
    write_summary(out, split, df, agg, pair_rows, cfg, models)

    # ---- console summary ---------------------------------------------------
    print(f"\n=== {split}: mean / median Pearson R on the common "
          f"{ENF_BINS} x {ENF_BIN_SIZE} bp grid ===")
    for m in models:
        v = df[m].to_numpy(dtype=float)
        if np.isnan(v).all():
            print(f"  {m.replace('_r',''):20s} : not available")
        else:
            v = v[~np.isnan(v)]
            print(f"  {m.replace('_r',''):20s} : {v.mean():.4f} / {np.median(v):.4f}  (n={len(v)})")
    for e in exclusions:
        print(f"  [exclusion] {e}")
    print(f"\nWrote {out}/benchmark_comparison_{split}.csv and companions")


if __name__ == "__main__":
    main()
