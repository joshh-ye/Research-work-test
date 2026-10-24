"""Heads on the pooled-128 features (same feature builder as the ridge search). Losses: 'mse' on per-track standardised (optionally
sqrt-transformed) targets, or 'poisson' (Poisson NLL on raw targets + Softplus output = the original Borzoi-TL head loss)."""
import sys, json, time, threading, queue
from pathlib import Path
import numpy as np, torch
from torch import nn
sys.path.insert(0, str(Path(__file__).parent))
from ftlib import *


def make_head(kind, p, hidden=1024, dropout=0.2, poisson=False):
    if kind == "linear": layers = [nn.Linear(p, N_TGT)]
    else: layers = [nn.Linear(p, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, N_TGT)]
    if poisson: layers.append(nn.Softplus())
    return nn.Sequential(*layers)


def target_norm(ds, ytf):
    """train per-track mean/sd of (transformed) targets for MSE heads"""
    s = torch.zeros(N_TGT, dtype=D, device=DEV); s2 = torch.zeros_like(s); n = 0
    for lo, hi, c in chunks(ds, 16, need=(), tgt=True):
        y = y_transform(c["tgt"].reshape(-1, N_TGT), ytf); n += y.shape[0]; s += y.sum(0); s2 += (y * y).sum(0)
    m = s / n; sd = (s2 / n - m * m).clamp_min(1e-12).sqrt()
    return m.float(), sd.float()


def raw_prediction(out, cfg, ymu, ysd):
    """network output -> prediction on the RAW target scale"""
    if cfg["loss"] == "poisson": return out
    p = out * ysd + ymu
    return y_back(p, cfg["ytf"]) if cfg["backtransform"] else p


class RamData:
    """training tiles in RAM (only the needed sources)"""
    def __init__(self, ds, src):
        self.n = ds.n; self.src = src
        self.arr = {k: np.ascontiguousarray(getattr(ds, k)) for k in tuple(src) + ("tgt",)}
    def batch(self, idx):
        idx = np.sort(idx)
        return {k: torch.from_numpy(v[idx]) for k, v in self.arr.items()}


@torch.no_grad()
def predict_stats(head, ds, spec, bmu, bsd, cfg, ymu, ysd, bs=8, collect_tiles=False):
    """streams ds; returns per-tile sums (tiles, T) on RAW scale: n, sx, sy, sxy, sx2, sy2, sse"""
    head.eval(); out = {k: [] for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    for lo, hi, c in chunks(ds, bs, need=spec.src):
        F_ = build_features(c, spec, bmu, bsd, dtype=torch.float32)
        P = raw_prediction(head(F_), cfg, ymu, ysd).double().reshape(hi - lo, G_BINS, N_TGT); Y = c["tgt"]
        out["sx"].append(P.sum(1)); out["sx2"].append((P * P).sum(1)); out["sy"].append(Y.sum(1)); out["sy2"].append((Y * Y).sum(1))
        out["sxy"].append((P * Y).sum(1)); out["sse"].append(((P - Y) ** 2).sum(1))
    st = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}; st["n"] = np.full(st["sx"].shape, float(G_BINS))
    return st


def r_from_tile_stats(st):
    g = {k: st[k].sum(0) for k in ["n", "sx", "sy", "sxy", "sx2", "sy2"]}
    return pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])


def train_head(cfg, spec, tr_ram, tr_ds, va_ds, bmu, bsd, seed, log=print, max_epochs=15, patience=3, tiles_per_chunk=16, mb=2048):
    torch.manual_seed(seed); np.random.seed(seed); rs = np.random.RandomState(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    ymu, ysd = target_norm(tr_ds, cfg["ytf"]) if cfg["loss"] == "mse" else (None, None)
    head = make_head(cfg["kind"], spec.dim(), cfg.get("hidden", 1024), cfg.get("dropout", 0.2), cfg["loss"] == "poisson").to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    best, best_ep, best_state, bad, hist = -1, 0, None, 0, []
    for ep in range(max_epochs):
        t0 = time.time(); head.train(); perm = rs.permutation(tr_ram.n); losses = []
        groups = [perm[i:i + tiles_per_chunk] for i in range(0, len(perm), tiles_per_chunk)]
        q = queue.Queue(maxsize=3)
        def producer():
            for g in groups: q.put(tr_ram.batch(g))
            q.put(None)
        th = threading.Thread(target=producer, daemon=True); th.start()
        while True:
            b = q.get()
            if b is None: break
            c = {k: v.to(DEV) for k, v in b.items()}
            X = build_features(c, spec, bmu, bsd, dtype=torch.float32); Y = c["tgt"].reshape(-1, N_TGT).float()
            o = torch.randperm(X.shape[0], device=DEV)
            for i in range(0, X.shape[0], mb):
                j = o[i:i + mb]; out = head(X[j])
                if cfg["loss"] == "mse": loss = ((out - (y_transform(Y[j], cfg["ytf"]) - ymu) / ysd) ** 2).mean()
                else: loss = torch.nn.functional.poisson_nll_loss(out, Y[j], log_input=False)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        th.join()
        st = predict_stats(head, va_ds, spec, bmu, bsd, cfg, ymu, ysd)
        r = r_from_tile_stats(st); m = float(np.nanmean(r)); sched.step(m)
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), val_r_all183=m, secs=round(time.time() - t0)))
        log(f"  ep {ep+1} loss {np.mean(losses):.4f} val r {m:.4f} ({time.time()-t0:.0f}s)")
        if m > best: best, best_ep, bad = m, ep + 1, 0; best_state = {k: v.clone() for k, v in head.state_dict().items()}; best_r = r
        else:
            bad += 1
            if bad >= patience: break
    head.load_state_dict(best_state); head.eval()
    return head, dict(best_epoch=best_ep, best_val_r=best, hist=hist), (ymu, ysd), best_r
