import sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ftlib as F
from common import *


def test_pool_emb_equals_central_then_grid128():
    e = (np.random.RandomState(0).randn(CBINS, 7) * 3).astype(np.float16)
    ref = to_grid128(central(e.astype(np.float32)))
    full = F.pool_emb(_wide(e)); assert full.shape == (896, 1920) and full.dtype == np.float16
    got = full.astype(np.float64)[:, :7]
    assert np.allclose(got, ref, atol=2e-3, rtol=2e-3)       # fp16 rounding of the pooled value only


def _wide(e):
    out = np.zeros((e.shape[0], EMB_DIM), np.float16); out[:, :e.shape[1]] = e; return out


def test_pool_tgt_equals_e4_path():
    t = np.random.RandomState(1).rand(CBINS, N_TGT).astype(np.float32)
    assert np.allclose(F.pool_tgt(t), to_grid128(central(t)), rtol=1e-6)
    assert F.pool_tgt(t).shape == (896, N_TGT)


def test_pool_uses_central_bins_not_margin():
    e = np.zeros((CBINS, EMB_DIM), np.float16); e[:MARGIN] = 100; e[-MARGIN:] = 100   # margin only
    assert F.pool_emb(e).max() == 0
    e = np.zeros((CBINS, EMB_DIM), np.float16); e[MARGIN + CROP: MARGIN + CROP + 4] = [[1], [2], [3], [4]]   # first pooled bin = 2.5
    p = F.pool_emb(e); assert p[0, 0] == 2.5 and p[1, 0] == 0


def test_shift_bins_semantics_and_padding():
    x = torch.arange(1, 9, dtype=torch.float64).reshape(1, 8, 1)
    assert F.shift_bins(x, 2)[0, :, 0].tolist() == [3, 4, 5, 6, 7, 8, 0, 0]       # out[t] = x[t+2]
    assert F.shift_bins(x, -1)[0, :, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]      # out[t] = x[t-1]
    assert F.shift_bins(x, 0)[0, :, 0].tolist() == list(range(1, 9))
    x2 = torch.randn(2, 8, 3, dtype=torch.float64)
    assert torch.equal(F.shift_bins(x2, 3)[1, :5], x2[1, 3:])               # no leakage across tiles
    assert F.shift_bins(x2, 3)[0, 5:].abs().sum() == 0


class FakeDS:
    def __init__(self, X, E, Y, chroms=None):
        self.pre, self.emb, self.tgt = X, E, Y; self.n = len(X)
        self.chrom = np.array(chroms or ["c1"] * self.n)


def synth(n=5, dp=6, de=4, seed=0):
    rs = np.random.RandomState(seed)
    X = (rs.rand(n, 896, dp) * 3).astype(np.float16); E = rs.randn(n, 896, de).astype(np.float16)
    W = rs.randn(dp + de, N_TGT) * 0.3
    Y = (np.concatenate([X, E], -1).astype(np.float64) @ W + rs.randn(n, 896, N_TGT) * 0.4).astype(np.float32)
    return FakeDS(X, E, Y, ["c1", "c1", "c1", "c8", "c8"][:n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")
def test_ridge_with_context_matches_closed_form():
    ds = synth(); spec = F.Spec(("pre", "emb"), "sqrt", (-2, 1), dims={"pre": 6, "emb": 4})
    tr = FakeDS(ds.pre[:3], ds.emb[:3], ds.tgt[:3]); bmu, bsd = F.base_stats(tr, spec)
    acc = F.accumulate(tr, spec, bmu, bsd, ("raw", "log1p"), bs=2, log=None); eig = F.RidgeEig(acc)
    # reference in numpy
    base = np.concatenate([np.sqrt(np.maximum(tr.pre.astype(np.float32), 0)), tr.emb.astype(np.float32)], -1).astype(np.float64)
    base = (base - bmu.cpu().numpy()) / bsd.cpu().numpy()
    def sh(a, k):
        o = np.zeros_like(a)
        if k > 0: o[:, :-k] = a[:, k:]
        else: o[:, -k:] = a[:, :k]
        return o
    Xf = np.concatenate([base, sh(base, -2), sh(base, 1)], -1).reshape(-1, 30)
    Yf = np.log1p(np.maximum(tr.tgt.reshape(-1, N_TGT).astype(np.float64), 0))
    m = Xf.mean(0); s = Xf.std(0); Xs = (Xf - m) / s; n = len(Xs)
    for lam in (1e-3, 1.0):
        B = np.linalg.solve(Xs.T @ Xs / n + lam * np.eye(30), Xs.T @ (Yf - Yf.mean(0)) / n)
        got = eig.coefs("log1p", [lam])[0].cpu().numpy()
        assert np.allclose(got, B, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")
def test_eval_stream_matches_direct_and_halves_sum():
    ds = synth(); spec = F.Spec(("pre",), "raw", (1,), dims={"pre": 6, "emb": 4})
    tr = FakeDS(ds.pre[:3], ds.emb[:3], ds.tgt[:3]); va = FakeDS(ds.pre[3:], ds.emb[3:], ds.tgt[3:], ["c1", "c8"])
    bmu, bsd = F.base_stats(tr, spec); acc = F.accumulate(tr, spec, bmu, bsd, ("raw",), bs=2, log=None); eig = F.RidgeEig(acc)
    lams = np.array([1e-2, 1.0]); sm, ys, _ = F.eval_stream(va, spec, bmu, bsd, eig, [("raw", False)], lams, bs=1)
    r, r2, pr2 = F.sums_to_metrics(sm[("raw", False)], ys)
    # direct
    B = eig.coefs("raw", lams).cpu().numpy(); mu, sd, yb = eig.mu.cpu().numpy(), eig.sd.cpu().numpy(), eig.ybar["raw"].cpu().numpy()
    base = (va.pre.astype(np.float32).astype(np.float64) - bmu.cpu().numpy()) / bsd.cpu().numpy()
    o = np.zeros_like(base); o[:, :-1] = base[:, 1:]
    Xf = np.concatenate([base, o], -1).reshape(-1, 12); P = ((Xf - mu) / sd) @ B[1] + yb; Y = va.tgt.reshape(-1, N_TGT).astype(np.float64)
    rr = np.array([np.corrcoef(P[:, t], Y[:, t])[0, 1] for t in range(N_TGT)])
    assert np.allclose(r[1], rr, atol=1e-7)
    assert np.allclose(r2[1], rr ** 2, atol=1e-7)
    assert np.allclose(pr2[1], 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0), atol=1e-7)


def test_target_transform_roundtrip():
    y = torch.rand(10, 3, dtype=torch.float64) * 50
    for t in ("log1p", "sqrt"):
        assert torch.allclose(F.y_back(F.y_transform(y, t), t), y)


def test_ctx_only_on_prefix_sources_and_dims():
    sp = F.Spec(("pre", "emb"), "raw", (-1, 1), dims={"pre": 6, "emb": 4}, ctx=("pre",))
    assert sp.dim() == 10 + 6 * 2
    c = {"pre": torch.rand(1, 896, 6), "emb": torch.randn(1, 896, 4)}
    f = F.build_features(c, sp, torch.zeros(10, dtype=torch.float64), torch.ones(10, dtype=torch.float64)).reshape(1, 896, 22)
    assert torch.allclose(f[0, 5, 10:16], f[0, 4, :6].double() if False else F.shift_bins(f[:, :, :6], -1)[0, 5])
    assert torch.allclose(f[0, 5, 10:16], f[0, 4, :6])         # offset -1 -> value of bin t-1 (pre part only)


def test_test_eval_refuses_unlocked_config(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "test_eval_cache.py"), "--config-id", "NOT-LOCKED-XYZ", "--stage", "predict"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "not in TEST_LOCK.csv" in (r.stderr + r.stdout)
    import shutil; shutil.rmtree(Path("/panfs/accrepfs.vampire/home/yejz1/experimental_research/results_ft/cache_exps/test_eval/NOT-LOCKED-XYZ"), ignore_errors=True)
