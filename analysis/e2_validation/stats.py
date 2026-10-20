"""Statistics for the E2/E6 pre-specified analysis (E2_ANALYSIS_PLAN.md sec 4-5). Pure numpy/scipy/pandas; no label or score I/O.

AUROC here is the Mann-Whitney AUROC (ties = 0.5). 'Stratified AUROC' = mean of per-region AUROC weighted by n_pos*n_neg.
This is neither Pearson r, r^2 nor prediction R^2.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

SEED = 20261020
N_BOOT = 10_000
N_PERM = 10_000


# ---------------------------------------------------------------- per-region AUROC
def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC of score for label==1 vs 0 (average ranks for ties); NaN if one class is empty."""
    score = np.asarray(score, float); label = np.asarray(label).astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def region_table(score, label, cluster) -> pd.DataFrame:
    """One row per cluster (all clusters present): n_pos, n_neg, auc, w = n_pos*n_neg (0 if uninformative)."""
    df = pd.DataFrame({"s": np.asarray(score, float), "y": np.asarray(label).astype(int), "c": np.asarray(cluster)})
    rows = []
    for c, g in df.groupby("c", sort=True):
        npos, nneg = int(g.y.sum()), int((1 - g.y).sum())
        w = npos * nneg
        rows.append((c, npos, nneg, auroc(g.s.to_numpy(), g.y.to_numpy()) if w > 0 else np.nan, w))
    return pd.DataFrame(rows, columns=["cluster", "n_pos", "n_neg", "auc", "w"]).set_index("cluster")


def stratified_auroc(score, label, cluster) -> float:
    t = region_table(score, label, cluster)
    return _wmean(t["auc"].to_numpy(), t["w"].to_numpy())


def _wmean(auc, w) -> float:
    w = np.asarray(w, float); auc = np.nan_to_num(np.asarray(auc, float))
    return float((w * auc).sum() / w.sum()) if w.sum() > 0 else float("nan")


# ---------------------------------------------------------------- region-clustered bootstrap
def boot_counts(n_clusters: int, B: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(B, n_clusters) multinomial resample counts (resample clusters with replacement). Shared across models for paired comparisons
    as long as the cluster order is the same (use the full sorted cluster list)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B).astype(np.float64)


def boot_stratified(table: pd.DataFrame, counts: np.ndarray) -> np.ndarray:
    w = counts * table["w"].to_numpy()[None, :]
    num = (w * np.nan_to_num(table["auc"].to_numpy())[None, :]).sum(1)
    den = w.sum(1)
    out = np.full(len(counts), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float('nan'), float('nan')
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def p1_stratified(score, label, cluster, B: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """P1: stratified AUROC, region-bootstrap percentile CI, within-region label permutation p (one-sided 'greater' and two-sided)."""
    t = region_table(score, label, cluster)
    obs = _wmean(t["auc"].to_numpy(), t["w"].to_numpy())
    if np.isnan(obs):
        return dict(auroc_stratified=np.nan, ci_lo=np.nan, ci_hi=np.nan, boot_se=np.nan, perm_p_greater=np.nan, perm_p_two_sided=np.nan,
                    n_regions_informative=0, n_pos=0, n_neg=0)
    cnt = boot_counts(len(t), B, seed)
    bs = boot_stratified(t, cnt)
    lo, hi = percentile_ci(bs)
    p_g, p_two = permutation_p(score, label, cluster, obs, n_perm, seed + 1)
    return dict(auroc_stratified=obs, ci_lo=lo, ci_hi=hi, boot_se=float(np.nanstd(bs, ddof=1)),
                perm_p_greater=p_g, perm_p_two_sided=p_two,
                n_regions_informative=int((t["w"] > 0).sum()), n_pos=int(t["n_pos"][t["w"] > 0].sum()),
                n_neg=int(t["n_neg"][t["w"] > 0].sum()))


# ---------------------------------------------------------------- within-region permutation
def permutation_null(score, label, cluster, n_perm: int = N_PERM, seed: int = SEED, chunk: int = 1000) -> np.ndarray:
    """Null distribution of the stratified AUROC under within-region label shuffles (preserves n_pos per region)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"s": np.asarray(score, float), "y": np.asarray(label).astype(int), "c": np.asarray(cluster)})
    num = np.zeros(n_perm); den = 0.0
    for _, g in df.groupby("c"):
        k, n = int(g.y.sum()), len(g)
        if k == 0 or k == n:
            continue
        w = k * (n - k); den += w
        r = sps.rankdata(g.s.to_numpy())
        for a in range(0, n_perm, chunk):
            b = min(n_perm, a + chunk)
            u = rng.random((b - a, n))
            idx = np.argpartition(u, k - 1, axis=1)[:, :k]
            auc = (r[idx].sum(1) - k * (k + 1) / 2) / (k * (n - k))
            num[a:b] += w * auc
    return num / den if den > 0 else np.full(n_perm, np.nan)


def permutation_p(score, label, cluster, obs: float, n_perm: int = N_PERM, seed: int = SEED) -> tuple[float, float]:
    null = permutation_null(score, label, cluster, n_perm, seed)
    eps = 1e-12
    p_g = (1 + (null >= obs - eps).sum()) / (n_perm + 1)
    p_l = (1 + (null <= obs + eps).sum()) / (n_perm + 1)
    return float(p_g), float(min(1.0, 2 * min(p_g, p_l)))


# ---------------------------------------------------------------- P2 direction
def direction_test(signed_score, mpra_logfc, cluster, B: int = N_BOOT, seed: int = SEED) -> dict:
    """Among positives with signed score != 0 and logFC != 0: fraction sign agreement. Region-bootstrap CI and two-sided bootstrap p vs 0.5
    (centred-percentile: p = 2*min(P*(f<=0.5), P*(f>=0.5)) with +1 correction); exact binomial p (descriptive, ignores clustering)."""
    s = np.asarray(signed_score, float); m = np.asarray(mpra_logfc, float); c = np.asarray(cluster)
    ok = (s != 0) & (m != 0) & ~np.isnan(s) & ~np.isnan(m)
    conc = (np.sign(s[ok]) == np.sign(m[ok])).astype(float)
    cl = c[ok]
    n = int(ok.sum())
    if n == 0:
        return dict(n=0, n_concordant=0, frac=np.nan, ci_lo=np.nan, ci_hi=np.nan, boot_p_two_sided=np.nan, binom_p=np.nan)
    uc = np.unique(c)           # resample over all regions of the positives set
    kk = pd.DataFrame({"c": cl, "k": conc}).groupby("c")["k"].agg(["sum", "count"]).reindex(uc).fillna(0)
    counts = boot_counts(len(uc), B, seed)
    num = counts @ kk["sum"].to_numpy(); den = counts @ kk["count"].to_numpy()
    f = num[den > 0] / den[den > 0]
    lo, hi = percentile_ci(f)
    p_le = (1 + (f <= 0.5).sum()) / (len(f) + 1); p_ge = (1 + (f >= 0.5).sum()) / (len(f) + 1)
    return dict(n=n, n_concordant=int(conc.sum()), frac=float(conc.mean()), ci_lo=lo, ci_hi=hi,
                boot_p_two_sided=float(min(1.0, 2 * min(p_le, p_ge))), binom_p=float(sps.binomtest(int(conc.sum()), n, 0.5).pvalue))


# ---------------------------------------------------------------- paired model comparison
def paired_delta(score_a, score_b, label, cluster, B: int = N_BOOT, seed: int = SEED) -> dict:
    """Delta stratified AUROC (A - B) on the same variants; same region resamples for both models; percentile CI, two-sided bootstrap p."""
    cluster = np.asarray(cluster)
    ta, tb = region_table(score_a, label, cluster), region_table(score_b, label, cluster)
    cnt = boot_counts(len(ta), B, seed)
    ba, bb = boot_stratified(ta, cnt), boot_stratified(tb, cnt)
    d = ba - bb
    obs = _wmean(ta["auc"].to_numpy(), ta["w"].to_numpy()) - _wmean(tb["auc"].to_numpy(), tb["w"].to_numpy())
    lo, hi = percentile_ci(d)
    dd = d[~np.isnan(d)]
    p = min(1.0, 2 * min((1 + (dd <= 0).sum()) / (len(dd) + 1), (1 + (dd >= 0).sum()) / (len(dd) + 1)))
    return dict(delta_auroc=float(obs), ci_lo=lo, ci_hi=hi, boot_p_two_sided=float(p))


def minimum_detectable_auroc(label, cluster, n_sim: int = 500, B: int = 400, seed: int = SEED, z: float = 1.645 + 0.842) -> dict:
    """Label-only power approximation: simulate iid-random scores (true AUROC 0.5), take the region-bootstrap SE of the stratified AUROC,
    MDA ~ 0.5 + z*SE (z = 1.645 + 0.842: one-sided alpha 0.05, 80% power, normal approximation; ignores the Holm correction)."""
    rng = np.random.default_rng(seed)
    label = np.asarray(label).astype(int); cluster = np.asarray(cluster)
    ses = []
    for i in range(n_sim):
        t = region_table(rng.random(len(label)), label, cluster)
        cnt = boot_counts(len(t), B, seed + 7 + i)
        ses.append(np.nanstd(boot_stratified(t, cnt), ddof=1))
    se = float(np.mean(ses))
    return dict(null_boot_se=se, min_detectable_auroc=0.5 + z * se)


# ---------------------------------------------------------------- multiplicity
def holm(p) -> np.ndarray:
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p) -> np.ndarray:
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(n)
    run = 1.0
    for j, i in enumerate(order):
        rank = n - j
        run = min(run, p[i] * n / rank)
        adj[i] = run
    return adj


# ---------------------------------------------------------------- conditional logistic regression
def _log_ek(eta: np.ndarray, k: int) -> float:
    """log of the elementary symmetric polynomial e_k(exp(eta)) via log-space DP."""
    dp = np.full(k + 1, -np.inf); dp[0] = 0.0
    for e in eta:
        new = dp.copy()
        new[1:] = np.logaddexp(dp[1:], dp[:-1] + e)
        dp = new
    return dp[k]


def _prep_strata(X, y, strata):
    X = np.asarray(X, float); y = np.asarray(y).astype(int); strata = np.asarray(strata)
    out = []
    for s in pd.unique(strata):
        m = strata == s
        k = int(y[m].sum())
        if 0 < k < m.sum():
            out.append((X[m], y[m], k))
    return out


def _nll(beta, groups):
    v = 0.0
    for Xs, ys, k in groups:
        eta = Xs @ beta
        v -= eta[ys == 1].sum() - _log_ek(eta, k)
    return v


def conditional_logit(X, y, strata, names=None) -> dict:
    """Exact conditional (fixed-effects) logistic regression, strata = region; strata without both classes are dropped.
    Returns coef, se (inverse numerical Hessian), z, p, OR, loglik. (statsmodels is not installed in torch-venv; verified against brute-force
    enumeration in tests.)"""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    names = names or [f"x{i}" for i in range(X.shape[1])]
    groups = _prep_strata(X, y, strata)
    p = X.shape[1]
    if not groups:
        return dict(names=names, coef=np.full(p, np.nan), se=np.full(p, np.nan), p=np.full(p, np.nan), loglik=np.nan, n_strata=0, n=0)
    res = optimize.minimize(_nll, np.zeros(p), args=(groups,), method="BFGS", options=dict(gtol=1e-7))
    b = res.x
    H = _num_hessian(lambda z: _nll(z, groups), b)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    z = b / se
    return dict(names=names, coef=b, se=se, z=z, p=2 * sps.norm.sf(np.abs(z)), odds_ratio=np.exp(b), loglik=-res.fun,
                n_strata=len(groups), n=int(sum(len(g[1]) for g in groups)), converged=bool(res.success))


def _num_hessian(f, x, h: float = 1e-4) -> np.ndarray:
    n = len(x); H = np.zeros((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = h; ej[j] = h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def clogit_lr_test(X_full, X_reduced, y, strata) -> dict:
    """Likelihood-ratio test of the extra columns of X_full vs X_reduced (same strata/rows)."""
    f = conditional_logit(X_full, y, strata); r = conditional_logit(X_reduced, y, strata) if np.asarray(X_reduced).size else None
    ll0 = 0.0 if r is None else r["loglik"]
    # null (no covariates) loglik = -sum log C(n,k) over used strata
    if r is None:
        ll0 = -sum(_log_binom(len(g[1]), g[2]) for g in _prep_strata(X_full, y, strata))
    df = np.asarray(X_full).reshape(len(y), -1).shape[1] - (0 if r is None else np.asarray(X_reduced).reshape(len(y), -1).shape[1])
    chi2 = max(0.0, 2 * (f["loglik"] - ll0))
    return dict(chi2=chi2, df=df, p=float(sps.chi2.sf(chi2, df)))


def _log_binom(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def zscore(x) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0


# ---------------------------------------------------------------- TSS distance (GENCODE v46, hg38)
GENCODE_FEATHER = Path("/home/yejz1/experimental_research/alphagenome_variant_effects/input/annotations/gencode.v46.annotation.gtf.gz.feather")


TSS_CACHE = Path("/home/yejz1/experimental_research/results_phase2/e2_validation/tss_protein_coding_gencode_v46_hg38.tsv")


def build_tss_cache(path: Path = GENCODE_FEATHER, out: Path = TSS_CACHE) -> pd.DataFrame:
    """ag-venv (pyarrow). Transcripts with transcript_type == protein_coding. The feather uses pyranges conventions
    (Start 0-based, End exclusive) -> 1-based TSS = Start+1 (+ strand) or End (- strand). Chromosome stored without 'chr'."""
    g = pd.read_feather(path, columns=["Chromosome", "Feature", "Start", "End", "Strand", "transcript_type"])
    g = g[(g["Feature"] == "transcript") & (g["transcript_type"] == "protein_coding")]
    tss = np.where(g["Strand"].astype(str) == "+", g["Start"] + 1, g["End"])
    d = pd.DataFrame({"chrom": g["Chromosome"].astype(str).str.replace("^chr", "", regex=True), "tss_1based": tss})
    d = d.drop_duplicates().sort_values(["chrom", "tss_1based"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep="\t", index=False)
    return d


def load_tss(cache: Path = TSS_CACHE) -> dict:
    """{chrom (no 'chr'): sorted array of 1-based GENCODE v46 protein-coding-transcript TSS (hg38)}. Builds cache if pyarrow is available."""
    if not Path(cache).exists():
        build_tss_cache(out=Path(cache))
    d = pd.read_csv(cache, sep="\t", dtype={"chrom": str})
    return {c: np.sort(v["tss_1based"].to_numpy()) for c, v in d.groupby("chrom")}


def distance_to_tss(chrom, pos, tss: dict) -> np.ndarray:
    out = np.full(len(pos), np.nan)
    for i, (c, p) in enumerate(zip(chrom, pos)):
        a = tss.get(str(c).replace("chr", ""))
        if a is None:
            continue
        j = np.searchsorted(a, p)
        cand = [abs(p - a[x]) for x in (j - 1, j) if 0 <= x < len(a)]
        out[i] = min(cand)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build-tss":
        d = build_tss_cache()
        print(f"[tss] {len(d)} protein-coding transcript TSS -> {TSS_CACHE}")
