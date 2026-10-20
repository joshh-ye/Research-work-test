import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmp_lib import *


def synth_stats(tiles=40, T=5, seed=0, noise=0.5):
    rs = np.random.RandomState(seed); m = 896
    P = rs.rand(tiles, m, T); Y = P + noise * rs.randn(tiles, m, T)
    return dict(n=np.full((tiles, T), float(m)), sx=P.sum(1), sy=Y.sum(1), sxy=(P * Y).sum(1), sx2=(P * P).sum(1),
                sy2=(Y * Y).sum(1), sse=((P - Y) ** 2).sum(1)), P, Y


def test_metrics_match_numpy_and_r2_distinct():
    S, P, Y = synth_stats(); w = np.ones((1, 40)); m = metrics(S, w)
    for t in range(5):
        r = np.corrcoef(P[:, :, t].ravel(), Y[:, :, t].ravel())[0, 1]
        assert abs(m["pearson_r"][0, t] - r) < 1e-9
        assert abs(m["pearson_r_squared"][0, t] - r ** 2) < 1e-9
        yy = Y[:, :, t].ravel(); R2 = 1 - ((P[:, :, t].ravel() - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        assert abs(m["prediction_R2"][0, t] - R2) < 1e-9
        assert abs(R2 - r ** 2) > 1e-6          # prediction R^2 is not squared Pearson


def test_block_bootstrap_ci_covers_and_paired_sign():
    Sa, _, _ = synth_stats(noise=0.3, seed=1); Sb, _, _ = synth_stats(noise=0.9, seed=2)
    tiles = pd.DataFrame(dict(chrom=["chr9"] * 20 + ["chr22"] * 20, start=list(range(20)) * 2))
    w = block_w(tiles, 300, block=5)
    d, lo, hi = paired(Sa, Sb, w)
    assert lo > 0 and d > 0
    pt, l, h = summarize(Sa, w); assert l <= pt <= h
