import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from estats import *


def test_pearson_matches_numpy():
    rng = np.random.default_rng(1); x = rng.normal(size=500); y = 0.5 * x + rng.normal(size=500)
    r = pearson_from_sums(len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum())
    assert np.isclose(r, np.corrcoef(x, y)[0, 1])
    assert np.isclose(pearson_r2(r), r ** 2)


def test_pearson_constant_is_nan():
    x = np.ones(10); y = np.arange(10.)
    assert np.isnan(pearson_from_sums(10, x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()))


def test_prediction_r2_differs_from_r2():
    rng = np.random.default_rng(2); y = rng.normal(size=300); p = 2 * y + 1
    sse = ((y - p) ** 2).sum()
    R2 = prediction_r2(sse, y.sum(), (y * y).sum(), len(y))
    assert np.isclose(R2, 1 - sse / ((y - y.mean()) ** 2).sum())
    assert R2 < 0 and np.isclose(pearson_r2(1.0), 1.0)
    assert np.isclose(prediction_r2(((y - y.mean()) ** 2).sum(), y.sum(), (y * y).sum(), len(y)), 0)


def test_block_ids_contiguous_and_per_chrom():
    chrom = np.array(["a"] * 7 + ["b"] * 4); start = np.concatenate([np.arange(7), np.arange(4)])[::-1].copy()
    start = np.concatenate([np.arange(7) * 10, np.arange(4) * 10])
    ids = block_ids(chrom, start, 3)
    assert list(ids[:7]) == [0, 0, 0, 1, 1, 1, 2] and list(ids[7:]) == [3, 3, 3, 4]


def test_block_weights_sum_to_blocks():
    ids = np.array([0, 0, 1, 1, 1, 2]); w = block_weights(ids, 200, np.random.default_rng(0))
    # total multiplicity of block j summed over blocks equals nblocks draws
    tot = w[:, [0, 2, 5]].sum(1)  # one tile per block
    assert (tot == 3).all()
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 2] == w[:, 4]).all()


def test_cluster_multiplicity():
    lab = np.array(list("aabbbc")); m = cluster_multiplicity(lab, 300, np.random.default_rng(0))
    assert (m[:, [0, 2, 5]].sum(1) == 3).all() and (m[:, 0] == m[:, 1]).all()
    mask = np.array([1, 1, 1, 1, 1, 0], bool)
    m2 = cluster_multiplicity(lab, 300, np.random.default_rng(0), mask)
    assert (m2[:, 5] == 0).all() and (m2[:, [0, 2]].sum(1) == 2).all()


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(3); D = rng.normal(0.02, 0.05, 60); lab = np.arange(60)
    b = weighted_mean(D, cluster_multiplicity(lab, 5000, np.random.default_rng(4)))
    lo, hi = pct_ci(b); assert lo < D.mean() < hi


def test_signflip():
    p, mode = signflip_p(np.array([1.] * 10)); assert mode == "exact" and abs(p - 2 / 1024) < 1e-12
    p2, _ = signflip_p(np.array([0.1, -0.1, 0.2, -0.2])); assert p2 == 1.0
    p3, mode3 = signflip_p(np.ones(40), n_mc=20000); assert mode3.startswith("monte") and p3 < 0.001
