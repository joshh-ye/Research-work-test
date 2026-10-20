"""Metrics + tile-block bootstrap on per-tile sufficient statistics (n, sx, sy, sxy, sx2, sy2, sse; shape tiles x tracks).
Pearson r, squared Pearson r (r^2) and prediction R^2 (=1-SSE/SST) are separate outputs. Reuses estats.py."""
import numpy as np, pandas as pd
from common import *
from estats import *

KEYS = ["n", "sx", "sy", "sxy", "sx2", "sy2", "sse"]


def load_stats(path):
    d = np.load(path)
    return {k: np.asarray(d[k], float) for k in KEYS} | ({"gid": d["gid"]} if "gid" in d else {})


def metrics(S, w):
    """w (B, tiles) tile multiplicities -> pearson_r, pearson_r_squared, prediction_R2 each (B, T)."""
    g = {k: w @ S[k] for k in KEYS}
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    return dict(pearson_r=r, pearson_r_squared=pearson_r2(r), prediction_R2=prediction_r2(g["sse"], g["sy"], g["sy2"], g["n"]))


def block_w(tiles, n_boot, block=10, seed=20261020):
    """tiles: DataFrame with chrom,start (rows aligned to stats rows). Returns (n_boot+1, tiles) weights; row 0 = all ones."""
    ids = block_ids(tiles.chrom.values, tiles.start.values, block)
    w = block_weights(ids, n_boot, np.random.default_rng(seed))
    return np.vstack([np.ones((1, len(tiles))), w])


def summarize(S, w, mask=None, metric="pearson_r"):
    """mean over tracks (mask) of metric: point estimate and bootstrap percentile CI."""
    m = metrics(S, w)[metric]
    if mask is not None: m = m[:, mask]
    mm = np.nanmean(m, 1)
    lo, hi = pct_ci(mm[1:]); return float(mm[0]), lo, hi


def paired(Sa, Sb, w, mask=None, metric="pearson_r"):
    a = metrics(Sa, w)[metric]; b = metrics(Sb, w)[metric]
    d = a - b
    if mask is not None: d = d[:, mask]
    dm = np.nanmean(d, 1); lo, hi = pct_ci(dm[1:]); return float(dm[0]), lo, hi


def exposure_masks():
    e = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (e.track_idx.values == np.arange(N_TGT)).all()
    ident = (e.Borzoi_class == "identical_experiment").values
    m = dict(all183=np.ones(N_TGT, bool), not_identical152=~ident, identical31=ident)   # primary = not_identical152
    for c in ["same_cellline_same_target", "same_cellline_other_target", "same_target_other_cellline", "unexposed"]:
        m[f"class_{c}"] = (e.Borzoi_class == c).values
    return m, e


def md(df, nd=4):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    df = df.copy()
    def f(x): return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)
