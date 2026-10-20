"""Shared primitives for E7 (non-oracle Enformer / AlphaGenome baselines). hg38, shared 896x128 bp grid.
Per-tile cross sufficient statistics (target t x candidate c) allow exact Pearson r, r^2 and validation-calibrated R^2 for any selection."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "borzoi_code"))
BR = ROOT / "reports/qing_deliverables/evaluation_evidence/baseline_borzoi_rerun"


def tile_cross_stats(y, p):
    """y (bins, T) target on shared grid; p (bins, C) predictions -> sufficient stats for one tile (float64)."""
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    assert y.shape[0] == p.shape[0] == 896
    return dict(n=float(y.shape[0]), sT=y.sum(0), sT2=(y * y).sum(0), sP=p.sum(0), sP2=(p * p).sum(0), sTP=y.T @ p)


def pearson_matrix(n, sT, sT2, sP, sP2, sTP):
    """(T,C) Pearson r from summed stats (n scalar). NaN where variance is zero."""
    cov = sTP - sT[:, None] * sP[None, :] / n
    vT = sT2 - sT ** 2 / n; vP = sP2 - sP ** 2 / n
    den = np.sqrt(np.clip(vT[:, None], 0, None) * np.clip(vP[None, :], 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, cov / den, np.nan)


def select_argmax(r, cand_ok=None):
    """Per-row argmax of r over allowed candidates (cand_ok: (C,) or (T,C) bool). NaN never wins; ties -> lowest index."""
    r = np.where(np.isnan(r), -np.inf, r)
    if cand_ok is not None:
        r = np.where(np.broadcast_to(cand_ok, r.shape), r, -np.inf)
    sel = r.argmax(1)
    assert np.isfinite(r[np.arange(len(sel)), sel]).all(), "a target has no valid candidate"
    return sel


def fit_calibration(n, sT, sP, sP2, sTP):
    """OLS obs ~ a + c*pred per pair (validation); same as reports/qing_deliverables/scripts/compute_baseline_rerun.py."""
    cov = sTP - sT * sP / n; vP = sP2 - sP ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(vP > 0, cov / vP, 0.0)
    return sT / n - c * sP / n, c


def calibrated_sse(n, sT, sT2, sP, sP2, sTP, a, c):
    return sT2 - 2 * a * sT - 2 * c * sTP + n * a ** 2 + 2 * a * c * sP + c ** 2 * sP2


def check_tiles(intervals, split):
    t = pd.read_csv(BR / f"tiles_{split}.csv")
    assert len(t) == len(intervals) and (t.chrom.values == np.array([i.chrom for i in intervals])).all() \
        and (t.start.values == np.array([i.start for i in intervals])).all(), "tile order differs from Qing tiles file"
    return t


def load_split(split, fasta=None):
    from genome_tiler import tile_genome
    fasta = fasta or str(ROOT / "borzoi_data/hg38/hg38.ml.fa")
    iv = tile_genome(fasta)[split]
    tiles = check_tiles(iv, split)
    tg = np.load(ROOT / "results_full" / f"{split}_targets.npy", mmap_mode="r")
    assert tg.shape[0] == len(iv) and tg.shape[2] == 183
    return iv, tg, tiles


def identical_candidates_enformer(vocab_included):
    """Boolean mask over included Enformer tracks that are project experiments (GSM identifier or ENCSR in file path)."""
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    gs = set(ex.GSM.dropna()); es = set(ex.ENCSR.dropna())
    full = pd.read_csv(ROOT / "enformer_data/targets_human.txt", sep="\t").set_index("index")
    files = full.loc[vocab_included.track_index.values, "file"]
    enc = files.str.extract(r"(ENCSR[0-9A-Z]{6})")[0]
    return (vocab_included.identifier.isin(gs).values | enc.isin(es).values)


AG_BIOS = {"HCT116", "Caco-2", "transverse colon", "sigmoid colon", "colonic mucosa", "mucosa of descending colon"}


def identical_mask_alphagenome(ag_tracks):
    """(183, C) bool proxy mask of excluded AlphaGenome candidates per target (see SELECTION_RULE.md item 5)."""
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    alias = {"P300": "EP300"}
    norm = lambda s: alias.get(str(s).upper(), str(s).upper())
    tf = ag_tracks.transcription_factor.map(norm).values; bios = ag_tracks.biosample_name.isin(AG_BIOS).values
    m = np.zeros((len(ex), len(ag_tracks)), bool)
    for i, r in ex.iterrows():
        if bool(r.AlphaGenome_possible_identical_ENCODE):
            m[i] = (tf == norm(r.TF)) & bios
    return m
