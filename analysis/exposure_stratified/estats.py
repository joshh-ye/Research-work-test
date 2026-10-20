"""Metric and bootstrap primitives for the exposure-stratified TL vs pretrained-Borzoi analysis.
Pearson r, r^2 (squared Pearson) and prediction R^2 (=1-SSE/SST) are separate functions."""
import numpy as np


def pearson_from_sums(n, sx, sy, sxy, sx2, sy2):
    """Pearson r from (weighted) sufficient statistics; arrays broadcast. NaN where variance is 0."""
    cov = sxy - sx * sy / n
    vx = sx2 - sx ** 2 / n
    vy = sy2 - sy ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)


def pearson_r2(r):
    """Squared Pearson correlation (NOT prediction R^2)."""
    return np.asarray(r) ** 2


def prediction_r2(sse, sy, sy2, n):
    """Prediction R^2 = 1 - SS_res/SS_tot with SS_tot about the observed mean."""
    sst = sy2 - sy ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sst > 0, 1 - sse / sst, np.nan)


def block_ids(chrom, start, block_size):
    """Contiguous non-overlapping blocks of `block_size` tiles within each chromosome (ordered by start)."""
    chrom = np.asarray(chrom); start = np.asarray(start)
    ids = np.empty(len(chrom), dtype=int); nxt = 0
    for c in dict.fromkeys(chrom.tolist()):
        idx = np.where(chrom == c)[0]
        idx = idx[np.argsort(start[idx])]
        ids[idx] = nxt + np.arange(len(idx)) // block_size
        nxt += ids[idx].max() - nxt + 1
    return ids


def block_weights(ids, n_boot, rng):
    """(n_boot, n_tiles) integer tile multiplicities from resampling blocks with replacement."""
    ub = np.unique(ids); nb = len(ub)
    draws = rng.integers(0, nb, (n_boot, nb))
    counts = np.zeros((n_boot, nb))
    for j in range(nb):
        counts[:, j] = (draws == j).sum(1)
    return counts[:, np.searchsorted(ub, ids)]


def cluster_multiplicity(labels, n_boot, rng, member_mask=None):
    """(n_boot, n_tracks) multiplicity of each track when whole clusters (among tracks in member_mask) are
    resampled with replacement. Tracks outside member_mask get 0."""
    labels = np.asarray(labels)
    if member_mask is None:
        member_mask = np.ones(len(labels), bool)
    uniq = np.unique(labels[member_mask]); k = len(uniq)
    draws = rng.integers(0, k, (n_boot, k))
    cnt = np.zeros((n_boot, k))
    for j in range(k):
        cnt[:, j] = (draws == j).sum(1)
    pos = np.searchsorted(uniq, labels)
    pos = np.clip(pos, 0, k - 1)
    out = cnt[:, pos]
    out[:, ~np.isin(labels, uniq) | ~member_mask] = 0
    return out


def weighted_mean(D, mult):
    """Mean over tracks of D (B x T or T) with track multiplicities mult (B x T)."""
    D = np.broadcast_to(D, mult.shape)
    return (D * mult).sum(1) / mult.sum(1)


def pct_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def signflip_p(cluster_means, n_mc=100000, rng=None, exact_max=16):
    """Two-sided sign-flip test of H0: mean of cluster-level differences = 0 (exact if k<=exact_max)."""
    d = np.asarray(cluster_means, float); k = len(d); obs = abs(d.mean())
    if k <= exact_max:
        signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
        null = np.abs((signs * d).mean(1)); return float((null >= obs - 1e-15).mean()), "exact"
    rng = rng or np.random.default_rng(0)
    s = rng.integers(0, 2, (n_mc, k)) * 2 - 1
    null = np.abs((s * d).mean(1))
    return float(((null >= obs - 1e-15).sum() + 1) / (n_mc + 1)), f"monte_carlo_{n_mc}"
