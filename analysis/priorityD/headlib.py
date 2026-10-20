"""Head training/evaluation on the cached frozen-trunk embeddings. Mirrors main.py train_resumable defaults:
MLP 1920-1024-(n_tracks), GELU, dropout 0.2, Softplus; Adam lr 1e-4 wd 1e-4; Poisson NLL (log_input=False) mean over
bins x tracks; batch = 1 tile; ReduceLROnPlateau(factor .5, patience 1) on val loss; epochs 8; early stop patience 3;
best checkpoint on val loss; train aug = random whole-bin shift in [-4,4] bins (+-128 bp) and RC with p=.5.
Deviation (documented in DESIGN.md): shift is emulated by re-slicing the cached 4104-bin embedding of the UNSHIFTED window
(the original recomputed the trunk on the shifted window) and embeddings/targets are fp16/fp32 cache values."""
import json, queue, threading, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from common import *

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_head(n_out, hidden=1024, dropout=0.2):
    return nn.Sequential(nn.Linear(EMB_DIM, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out), nn.Softplus())


def load_tile(gid, rc=False, shift=0, need_tgt=True):
    """Return (emb (4096,1920) fp16 np, tgt (4096,183) fp32 np) for the window shifted by `shift` bins (genomic coordinates)
    and optionally reverse-complemented. RC: emb row j <-> genomic bin 4103-j, target reversed (as dataset.py)."""
    lo = MARGIN + shift
    t = np.load(tile_path(gid, "tgt"), mmap_mode="r")[lo:lo + CENTER_BINS] if need_tgt else None
    if not rc:
        e = np.load(tile_path(gid, "emb_fwd"), mmap_mode="r")[lo:lo + CENTER_BINS]
        return np.array(e), (np.array(t) if t is not None else None)
    e = np.load(tile_path(gid, "emb_rc"), mmap_mode="r")[MARGIN - shift:MARGIN - shift + CENTER_BINS]
    return np.array(e), (np.array(t[::-1]) if t is not None else None)


def prefetch(jobs, n_threads=8, depth=24):
    """jobs: list of (gid, rc, shift). Yields (job, emb, tgt) in order using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(n_threads)
    futs = []
    it = iter(jobs)
    for _ in range(depth):
        j = next(it, None)
        if j is None: break
        futs.append((j, ex.submit(load_tile, *j)))
    i = 0
    while futs:
        j, f = futs.pop(0)
        nj = next(it, None)
        if nj is not None: futs.append((nj, ex.submit(load_tile, *nj)))
        e, t = f.result(); yield j, e, t
    ex.shutdown()


def poisson_nll(pred, tgt):
    return torch.nn.functional.poisson_nll_loss(pred, tgt, log_input=False)


def epoch_jobs(gids, epoch, seed, rc_prob=0.5, max_shift_bins=4):
    rs = np.random.RandomState((seed * 1000003 + epoch * 7919) % (2 ** 31 - 1))
    order = rs.permutation(len(gids))
    shifts = rs.randint(-max_shift_bins, max_shift_bins + 1, len(gids)) if max_shift_bins > 0 else np.zeros(len(gids), int)
    rcs = rs.rand(len(gids)) < rc_prob
    return [(int(gids[o]), bool(rcs[k]), int(shifts[k])) for k, o in enumerate(order)]


@torch.no_grad()
def predict_tiles(head, gids, cols=None, with_stats=True, log_every=0):
    """Eval (fwd, unshifted, no RC, like the original eval). Returns per-tile sufficient statistics on the shared 128 bp
    grid (pred and target pooled identically, float64): n, sx, sy, sxy, sx2, sy2, sse each (tiles, tracks), plus val loss."""
    head.eval(); S = {k: [] for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}; losses = []
    for (g, _, _), e, t in prefetch([(int(g), False, 0) for g in gids]):
        tt = torch.from_numpy(t if cols is None else t[:, cols]).to(DEV)
        p = head(torch.from_numpy(e).to(DEV).float())
        losses.append(poisson_nll(p, tt).item())
        P = p.double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
        Y = tt.double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
        S["sx"].append(P.sum(0)); S["sy"].append(Y.sum(0)); S["sxy"].append((P * Y).sum(0))
        S["sx2"].append((P * P).sum(0)); S["sy2"].append((Y * Y).sum(0)); S["sse"].append(((P - Y) ** 2).sum(0))
    out = {k: torch.stack(v).cpu().numpy() for k, v in S.items()}
    out["n"] = np.full(out["sx"].shape, float(G_BINS))
    return out, float(np.mean(losses))


def mean_r(stats):
    from estats import pearson_from_sums
    g = {k: stats[k].sum(0) for k in ["n", "sx", "sy", "sxy", "sx2", "sy2"]}
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    return r


def train_head(train_gids, val_gids, out_dir, seed, cols=None, rc_prob=0.5, max_shift_bins=4, epochs=8, patience=3,
               lr=1e-4, wd=1e-4, hidden=1024, dropout=0.2, log=print):
    """Resumable (epoch granularity; per-epoch data order/shift/RC derive from seed+epoch). Returns best head (eval mode)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_out = EMB_DIM and (N_TGT if cols is None else len(cols))
    torch.manual_seed(seed); np.random.seed(seed)
    head = make_head(n_out, hidden, dropout).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    hist = dict(train_loss=[], val_loss=[], best_epoch=None); best = float("inf"); no_imp = 0; start = 0
    ck = out_dir / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=DEV)
        head.load_state_dict(s["head"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        hist, best, no_imp, start = s["hist"], s["best"], s["no_imp"], s["epoch"]
        torch.set_rng_state(s["rng"].cpu()); log(f"resumed at epoch {start}")
    for ep in range(start, epochs):
        t0 = time.time(); head.train(); ls = []
        for (g, rc, sh), e, t in prefetch(epoch_jobs(train_gids, ep, seed, rc_prob, max_shift_bins)):
            x = torch.from_numpy(e).to(DEV).float(); y = torch.from_numpy(t if cols is None else t[:, cols]).to(DEV)
            loss = poisson_nll(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        _, vl = predict_tiles(head, val_gids, cols)
        hist["train_loss"].append(float(np.mean(ls))); hist["val_loss"].append(vl); sched.step(vl)
        if vl < best:
            best, no_imp = vl, 0; hist["best_epoch"] = ep + 1
            torch.save(head.state_dict(), out_dir / "head_best.pt")
        else:
            no_imp += 1
        torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, best=best,
                        no_imp=no_imp, epoch=ep + 1, rng=torch.get_rng_state()), ck)
        (out_dir / "history.json").write_text(json.dumps(hist))
        log(f"epoch {ep+1}/{epochs} train {hist['train_loss'][-1]:.4f} val {vl:.4f} best_ep {hist['best_epoch']} ({time.time()-t0:.0f}s)")
        if patience > 0 and no_imp >= patience:
            log("early stop"); break
    head.load_state_dict(torch.load(out_dir / "head_best.pt", map_location=DEV)); head.eval()
    return head, hist
