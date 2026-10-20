import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from e7lib import *

def _data(seed=0, tiles=5, T=4, C=6):
    rng = np.random.default_rng(seed)
    ys = [np.abs(rng.normal(size=(896, T))) for _ in range(tiles)]
    ps = [np.abs(rng.normal(size=(896, C))) + 0.5 * np.tile(y[:, :1], (1, C)) for y, _ in zip(ys, range(tiles)) for y in [y]]
    return ys, ps

def _sum(sts):
    return sts[0]["n"] * len(sts), {k: sum(s[k] for s in sts) for k in ["sT", "sT2", "sP", "sP2", "sTP"]}

def test_pearson_matches_numpy_flattened():
    ys, ps = _data(); n, S = _sum([tile_cross_stats(y, p) for y, p in zip(ys, ps)])
    r = pearson_matrix(n, S["sT"], S["sT2"], S["sP"], S["sP2"], S["sTP"])
    Y = np.concatenate(ys); P = np.concatenate(ps)
    for t in range(4):
        for c in range(6):
            assert abs(r[t, c] - np.corrcoef(Y[:, t], P[:, c])[0, 1]) < 1e-10

def test_zero_variance_is_nan():
    y = np.zeros((896, 2)); p = np.random.default_rng(1).normal(size=(896, 3))
    s = tile_cross_stats(y, p); r = pearson_matrix(s["n"], s["sT"], s["sT2"], s["sP"], s["sP2"], s["sTP"])
    assert np.isnan(r).all()

def test_select_argmax_nan_ties_mask():
    r = np.array([[0.5, np.nan, 0.5, 0.9], [np.nan, 0.1, 0.2, 0.2]])
    assert list(select_argmax(r)) == [3, 2]                      # ties -> lowest index, NaN never wins
    ok = np.array([True, True, True, False])
    assert list(select_argmax(r, ok)) == [0, 2]                  # excluded candidate cannot be chosen
    with pytest.raises(AssertionError):
        select_argmax(np.array([[np.nan, np.nan]]))

def test_calibration_and_sse_match_direct():
    ys, ps = _data(3); n, S = _sum([tile_cross_stats(y, p) for y, p in zip(ys, ps)])
    Y = np.concatenate(ys); P = np.concatenate(ps)
    t, c = 1, 2
    a, b = fit_calibration(n, S["sT"][t], S["sP"][c], S["sP2"][c], S["sTP"][t, c])
    slope, icpt = np.polyfit(P[:, c], Y[:, t], 1)
    assert abs(a - icpt) < 1e-9 and abs(b - slope) < 1e-9
    sse = calibrated_sse(n, S["sT"][t], S["sT2"][t], S["sP"][c], S["sP2"][c], S["sTP"][t, c], a, b)
    assert abs(sse - ((Y[:, t] - a - b * P[:, c]) ** 2).sum()) < 1e-6
    # validation-fitted (a,b) applied to different data is still computed exactly
    ys2, ps2 = _data(4); n2, S2 = _sum([tile_cross_stats(y, p) for y, p in zip(ys2, ps2)])
    Y2 = np.concatenate(ys2); P2 = np.concatenate(ps2)
    sse2 = calibrated_sse(n2, S2["sT"][t], S2["sT2"][t], S2["sP"][c], S2["sP2"][c], S2["sTP"][t, c], a, b)
    assert abs(sse2 - ((Y2[:, t] - a - b * P2[:, c]) ** 2).sum()) < 1e-6

def test_tiles_are_additive_not_averaged():
    ys, ps = _data(5, tiles=3); n, S = _sum([tile_cross_stats(y, p) for y, p in zip(ys, ps)])
    assert n == 3 * 896

def test_geometry_helpers():
    from enformer_baseline_utils import targets_to_enformer_grid, TGT_CROP_OFF
    x = np.zeros((4096, 1)); x[TGT_CROP_OFF: TGT_CROP_OFF + 4, 0] = 8.0       # first 4 kept 32-bp bins -> first 128-bp bin
    g = targets_to_enformer_grid(x); assert g.shape == (896, 1) and g[0, 0] == 8.0 and g[1:].sum() == 0
    x2 = np.zeros((4096, 1)); x2[:TGT_CROP_OFF] = 5.0                           # flank outside the 114,688 bp window is dropped
    assert targets_to_enformer_grid(x2).sum() == 0
    from baseline_alphagenome import to_common_grid
    v = np.zeros((4096, 2)); off = (4096 - 896) // 2; v[off:off + 1, 0] = 4.0    # 128-bp AG output: central 896 bins kept
    o = to_common_grid(v, 128); assert o.shape == (896, 2) and o[0, 0] == 4.0
    with pytest.raises(ValueError):
        to_common_grid(np.zeros((100, 1)), 128)

def test_identical_masks():
    inc = pd.read_csv(ROOT / "results_baseline_enformer/included_enformer_tracks.csv")
    m = identical_candidates_enformer(inc)
    assert len(m) == 1990 and 20 <= m.sum() <= 60                              # 24 LoVo + 4 colon ENCSR + 3 HCT116 expected region
    ag = pd.read_csv(ROOT / "results_baseline_alphagenome/alphagenome_tracks_test.csv")
    am = identical_mask_alphagenome(ag); assert am.shape == (183, 1617)
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (am.any(1) <= ex.ENCSR.notna().values).all()                         # only ENCSR-flagged targets get exclusions
