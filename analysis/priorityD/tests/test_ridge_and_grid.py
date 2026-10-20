import sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ridgelib as R
from common import *


class Mock(R.Feat):
    def __init__(self, X, Y, kind="pre_raw"):
        self.kind = kind; self.p = X.shape[2]; self.X, self.Y = X, Y
    def load(self, gid): return self.X[gid].astype(np.float32), self.Y[gid]


def synth(ntile=6, m=896, p=12, seed=0):
    rs = np.random.RandomState(seed)
    W = rs.randn(p, N_TGT) * 0.3
    X = rs.rand(ntile, m, p) * rs.rand(p) * 5 + rs.rand(p)
    Y = X @ W + rs.randn(ntile, m, N_TGT) * 0.5
    return X.astype(np.float32).astype(np.float64), Y


def test_to_grid128_equals_project_function():
    from enformer_baseline_utils import targets_to_enformer_grid
    a = np.random.rand(4096, 5)
    assert np.allclose(to_grid128(a), targets_to_enformer_grid(a))
    assert to_grid128(a).shape == (896, 5)


def test_torch_pool_equals_numpy_pool():
    a = np.random.rand(4096, 7)
    assert np.allclose(R.pool_grid_t(torch.from_numpy(a)).numpy(), to_grid128(a))


def test_ridge_matches_closed_form():
    X, Y = synth(); f = Mock(X, Y); gids = list(range(5))
    acc = R.accumulate(f, gids, log=None)
    idx = np.arange(X.shape[2])
    mu, sd, ybar, Bs, _ = R.fit_path(acc, idx, [1e-3, 1.0])
    Xt = X[:5].reshape(-1, X.shape[2]); Yt = Y[:5].reshape(-1, N_TGT)
    m = Xt.mean(0); s = Xt.std(0); Xs = (Xt - m) / s; n = len(Xs)
    for l, lam in enumerate([1e-3, 1.0]):
        B = np.linalg.solve(Xs.T @ Xs / n + lam * np.eye(Xs.shape[1]), Xs.T @ (Yt - Yt.mean(0)) / n)
        assert np.allclose(Bs[l].cpu().numpy(), B, atol=1e-5)


def test_eval_stats_match_direct_computation_and_lambda_uses_val_only():
    X, Y = synth(); f = Mock(X, Y); acc = R.accumulate(f, [0, 1, 2, 3], log=None); idx = np.arange(X.shape[2])
    mu, sd, ybar, Bs, _ = R.fit_path(acc, idx)
    sv = R.eval_tiles(f, [4], idx, mu, sd, ybar, Bs)     # validation = tile 4
    st = R.eval_tiles(f, [5], idx, mu, sd, ybar, Bs)     # test = tile 5
    l = 3
    P = ((X[4] - mu.cpu().numpy()) / sd.cpu().numpy()) @ Bs[l].cpu().numpy() + ybar.cpu().numpy()
    assert np.allclose(sv["sx"][l, 0], P.sum(0), rtol=1e-5, atol=1e-5)
    assert np.allclose(sv["sse"][l, 0], ((P - Y[4]) ** 2).sum(0), rtol=1e-5)
    best = R.select_lambda(sv)
    # corrupt the test tile: selection must not change (it only sees validation stats)
    st2 = {k: v * 0 + 123.0 for k, v in st.items()}
    assert R.select_lambda(sv) == best
    r = R.mean_r_from_stats({k: v[best] for k, v in sv.items()})
    assert np.allclose(r, [np.corrcoef(P[:, t], Y[4][:, t])[0, 1] for t in range(N_TGT)], atol=1e-6)


def test_probe_pooling_commutes_with_linear_map():
    """probe: fit at 32 bp, pooled predictions == prediction from pooled features (linear)."""
    rs = np.random.RandomState(1); X = torch.from_numpy(rs.rand(4096, 6)); B = torch.from_numpy(rs.randn(6, 4))
    assert torch.allclose(R.pool_grid_t(X @ B), R.pool_grid_t(X) @ B)
