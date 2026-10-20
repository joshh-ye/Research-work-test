"""Streaming ridge regression (E4). Sufficient statistics X'X, X'Y, sums are accumulated on the GPU in float64 from the
cache, so training outputs are never stored in RAM. Penalised problem on STANDARDISED features (train mean/sd) with
intercept; lambda is on the correlation-matrix scale: B = (R + lam I)^-1 Xs'Y/n, R = Xs'Xs/n. lambda is selected ONLY on the
validation chromosomes (mean per-track Pearson r over the 183 targets, shared 128 bp grid)."""
import numpy as np, torch
from common import *
from headlib import DEV, prefetch

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
D = torch.float64


def pool_grid_t(a):
    """torch (4096, C) -> (896, C) shared grid (crop 256/side, mean-pool x4), same as common.to_grid128."""
    return a[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)


class Feat:
    """Feature source. kind: 'pre_raw' / 'pre_log1p' (pretrained Borzoi outputs restricted to pre_channels(): 1,888 channels, shared grid) or
    'probe' (frozen-trunk embedding 1920 channels at 32 bp; fit at 32 bp, predictions pooled to the shared grid)."""
    def __init__(self, kind): self.kind = kind; self.p = len(np.load(CACHE / "pre_channels.npy")) if kind.startswith("pre") else EMB_DIM

    def load(self, gid):
        """returns X (m, p) float32 np (training resolution), Y (m, 183) float64 np (training resolution),
        and a function mapping an (m,.) torch tensor to the shared grid (identity for pre_*)."""
        t = np.load(tile_path(gid, "tgt"), mmap_mode="r")
        yc = central(np.asarray(t))
        if self.kind.startswith("pre"):
            x = np.load(tile_path(gid, "pre128")).astype(np.float32)
            if self.kind == "pre_log1p": x = np.log1p(np.maximum(x, 0))
            return x, to_grid128(yc)
        x = np.array(np.load(tile_path(gid, "emb_fwd"), mmap_mode="r")[MARGIN:MARGIN + CENTER_BINS])
        return x.astype(np.float32), yc.astype(np.float64)

    def to_grid(self, a): return a if self.kind.startswith("pre") else pool_grid_t(a)


def _stream(feat, gids):
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(8); futs = [ex.submit(feat.load, int(g)) for g in gids[:16]]; nxt = 16
    for i in range(len(gids)):
        x, y = futs[i].result(); futs[i] = None
        if nxt < len(gids): futs.append(ex.submit(feat.load, int(gids[nxt]))); nxt += 1
        yield x, y
    ex.shutdown()


@torch.no_grad()
def accumulate(feat, gids, log=print):
    p = feat.p; n = 0
    sx = torch.zeros(p, dtype=D, device=DEV); sxx = torch.zeros(p, p, dtype=D, device=DEV)
    sy = torch.zeros(N_TGT, dtype=D, device=DEV); sxy = torch.zeros(p, N_TGT, dtype=D, device=DEV)
    for k, (x, y) in enumerate(_stream(feat, gids)):
        X = torch.from_numpy(x).to(DEV).to(D); Y = torch.from_numpy(y).to(DEV).to(D)
        n += X.shape[0]; sx += X.sum(0); sxx += X.T @ X; sy += Y.sum(0); sxy += X.T @ Y
        if log and (k + 1) % 500 == 0: log(f"  accumulated {k+1}/{len(gids)} tiles")
    return dict(n=n, sx=sx, sxx=sxx, sy=sy, sxy=sxy)


@torch.no_grad()
def fit_path(acc, idx, lambdas=LAMBDAS):
    """Ridge path for feature subset idx. Returns (mu, sd, ybar, Bs (L,p',T)); pred = ((X-mu)/sd) @ Bs[l] + ybar."""
    idx = torch.as_tensor(idx, device=DEV); n = acc["n"]
    sx = acc["sx"][idx]; sxx = acc["sxx"][idx][:, idx]; sxy = acc["sxy"][idx]
    mu = sx / n; cov = sxx / n - torch.outer(mu, mu)
    sd = cov.diagonal().clamp_min(0).sqrt(); live = sd > 1e-12
    sd = torch.where(live, sd, torch.ones_like(sd))
    R = cov / torch.outer(sd, sd); R[~live] = 0; R[:, ~live] = 0
    ybar = acc["sy"] / n
    cxy = (sxy / n - torch.outer(mu, ybar)) / sd[:, None]; cxy[~live] = 0
    ev, V = torch.linalg.eigh(R); ev = ev.clamp_min(0)
    VtC = V.T @ cxy
    Bs = torch.stack([V @ (VtC / (ev[:, None] + lam)) for lam in lambdas])
    return mu, sd, ybar, Bs, int(live.sum())


@torch.no_grad()
def eval_tiles(feat, gids, idx, mu, sd, ybar, Bs):
    """Per-tile sufficient statistics on the shared grid for every lambda. Returns dict of arrays (L, tiles, T)."""
    idx = torch.as_tensor(idx, device=DEV); L = Bs.shape[0]
    Bcat = Bs.permute(1, 0, 2).reshape(Bs.shape[1], -1)         # (p', L*T)
    out = {k: [] for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    for x, y in _stream(feat, gids):
        X = torch.from_numpy(x).to(DEV).to(D)[:, idx]
        P = (((X - mu) / sd) @ Bcat).reshape(X.shape[0], L, N_TGT) + ybar   # (m, L, T)
        Y = torch.from_numpy(y).to(DEV).to(D)
        P = feat.to_grid(P.reshape(X.shape[0], L * N_TGT)).reshape(G_BINS, L, N_TGT)
        Yg = feat.to_grid(Y)                                       # (896, T)
        out["sx"].append(P.sum(0)); out["sx2"].append((P * P).sum(0)); out["sy"].append(Yg.sum(0).expand(L, -1))
        out["sy2"].append((Yg * Yg).sum(0).expand(L, -1)); out["sxy"].append((P * Yg[:, None, :]).sum(0))
        out["sse"].append(((P - Yg[:, None, :]) ** 2).sum(0))
    st = {k: torch.stack(v, 1).cpu().numpy() for k, v in out.items()}   # (L, tiles, T)
    st["n"] = np.full(st["sx"].shape, float(G_BINS))
    return st


def mean_r_from_stats(st):
    """st arrays (..., tiles, T) -> per-track pooled Pearson r (..., T)."""
    from estats import pearson_from_sums
    g = {k: st[k].sum(-2) for k in ["n", "sx", "sy", "sxy", "sx2", "sy2"]}
    return pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])


def select_lambda(val_stats, lambdas=LAMBDAS):
    """Index of lambda maximising mean per-track pooled Pearson r on VALIDATION stats (L, tiles, T) only."""
    return int(np.argmax(np.nanmean(mean_r_from_stats(val_stats), 1)))
