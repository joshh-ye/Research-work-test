"""Tests for 07_e4_sensitivity.py helpers: removal-by-zeroed-rows equals a ridge fit on the column subset; removal rule counts;
cluster-CI function is reproducible and brackets the point estimate on synthetic data."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ridgelib
ridgelib.DEV = "cpu"
from ridgelib import fit_path
spec = importlib.util.spec_from_file_location("e4s", Path(__file__).resolve().parents[1] / "07_e4_sensitivity.py")
e4s = importlib.util.module_from_spec(spec); spec.loader.exec_module(e4s)


def synth_acc(n=3000, p=8, T=183, seed=0):
    rs = np.random.RandomState(seed); X = rs.rand(n, p) * np.arange(1, p + 1); Y = X[:, :1] @ rs.randn(1, T) + rs.randn(n, T)
    X = torch.from_numpy(X); Y = torch.from_numpy(Y)
    return dict(n=n, sx=X.sum(0), sxx=X.T @ X, sy=Y.sum(0), sxy=X.T @ Y), X, Y


def test_subset_fit_equals_direct_ridge_on_columns():
    acc, X, Y = synth_acc(); keep = np.array([0, 1, 3, 4, 7])
    mu, sd, yb, Bs, _ = fit_path(acc, keep)
    # direct closed form on standardised kept columns, lam index 3
    lam = ridgelib.LAMBDAS[3]; Xk = X[:, keep]; Xs = (Xk - Xk.mean(0)) / Xk.std(0, unbiased=False)
    R = Xs.T @ Xs / len(Xs); B = torch.linalg.solve(R + lam * torch.eye(len(keep), dtype=R.dtype), Xs.T @ (Y - Y.mean(0)) / len(Xs))
    assert torch.allclose(B, Bs[3], atol=1e-8)
    # zero-padded full-width coefficients applied to globally standardised X give identical predictions
    mu_f, sd_f, _, _, _ = fit_path(acc, np.arange(8)); Bfull = torch.zeros(8, 183, dtype=B.dtype); Bfull[keep] = Bs[3]
    P_full = ((X - mu_f) / sd_f) @ Bfull; P_sub = ((Xk - mu) / sd) @ Bs[3]
    assert torch.allclose(P_full, P_sub, atol=1e-9)


def test_delta_ci_reproducible_and_brackets_point_estimate():
    rs = np.random.RandomState(1); tiles, T, m = 60, 183, 896
    def mk(noise):
        P = rs.rand(tiles, 40, T); Y = P + noise * rs.randn(tiles, 40, T)
        return dict(n=np.full((tiles, T), 40.), sx=P.sum(1), sy=Y.sum(1), sxy=(P * Y).sum(1), sx2=(P * P).sum(1), sy2=(Y * Y).sum(1), sse=((P - Y) ** 2).sum(1))
    A, Bm = mk(0.5), mk(1.0)
    ids = np.repeat(np.arange(6), 10); B = 200
    w = np.vstack([np.ones((1, tiles)), e4s.block_weights(ids, B, np.random.default_rng(0))])
    tf = np.array([f"tf{i % 20}" for i in range(T)]); cl = np.array([f"c{i % 5}" for i in range(T)]); mask = np.ones(T, bool)
    r1 = e4s.delta_ci(A, Bm, w, mask, tf, cl, B); r2 = e4s.delta_ci(A, Bm, w, mask, tf, cl, B)
    assert r1 == r2 and r1["delta"] > 0
    for k in ["tileblock", "TFcluster", "cellcluster", "twoway_tileblock_x_TF", "twoway_tileblock_x_cell"]:
        assert r1[f"{k}_lo"] < r1["delta"] < r1[f"{k}_hi"], k
    assert r1["n_TF_clusters"] == 20 and r1["n_cellline_clusters"] == 5


def test_removal_rule_counts_match_exposure_classes():
    df, rem = e4s.removal_rule()
    assert len(rem) == 183
    ident = df.Borzoi_class == "identical_experiment"
    assert (df.n_identical_in_1888[ident] >= 1).all() and ident.sum() == 31
    assert (df.n_removed_in_1888[df.Borzoi_class == "unexposed"] == 0).all()
    assert (df.n_removed_in_1888[df.Borzoi_class == "same_target_other_cellline"] == 0).all()
