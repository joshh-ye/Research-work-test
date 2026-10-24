"""Cache-based ridge / head search on the pooled-128 copy (see README.md). All selection on validation only.
Pearson r is always per-track r pooled over all bins of the split, averaged over tracks (same as E4 / estats.pearson_from_sums).
Ridge: standardised features (train mean/sd), intercept, penalty lambda on the correlation-matrix scale
(B = (R + lam I)^-1 Xs'Y/n), identical to priorityD/ridgelib.py; fp64 on GPU."""
import sys, json, time, csv, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from poolcache import *          # also puts priorityD + exposure_stratified on sys.path (common)
from estats import pearson_from_sums, pearson_r2, prediction_r2

DEV = "cuda"
D = torch.float64
OUTDIR = ROOT / "results_ft/cache_exps"
REGISTRY = ROOT / "research_management/finetuning/FT_REGISTRY.csv"
REG_COLS = "exp_id,method,config,train,val,seed,start,end,runtime_s,gpu_mem_gb,checkpoint,val_r_all183,val_r_152,test_r_all183,test_r_152,status,notes".split(",")
LAMBDAS = 10.0 ** np.arange(-7, 2.01, 0.5)          # 19 values; superset of the E4 grid 1e-6..1e2 (decades)
E4_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def cap_gpu(gb=29.0):
    torch.cuda.set_per_process_memory_fraction(min(0.99, gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)))


def gpu_peak_gb(): return torch.cuda.max_memory_allocated() / 2 ** 30


def masks():
    import pandas as pd
    e = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (e.track_idx.values == np.arange(N_TGT)).all()
    ident = (e.Borzoi_class == "identical_experiment").values
    return dict(all183=np.ones(N_TGT, bool), not_identical152=~ident)


def mean_r(r, mk=None):
    return {k: float(np.nanmean(r[..., m], -1)) if r.ndim == 1 else np.nanmean(r[..., m], -1) for k, m in (mk or masks()).items()}


# ------------------------------------------------------------------ feature construction
def shift_bins(x, k):
    """x (B, 896, p): out[:, t] = x[:, t+k] (neighbouring bin at offset k, in 128 bp bins); zero where t+k is outside the tile."""
    out = torch.zeros_like(x)
    if k > 0: out[:, :-k] = x[:, k:]
    elif k < 0: out[:, -k:] = x[:, :k]
    else: out = x.clone()
    return out


def pre_transform(x, tf):
    if tf == "raw": return x
    x = x.clamp_min(0)
    if tf == "log1p": return torch.log1p(x)
    if tf == "sqrt": return torch.sqrt(x)
    raise ValueError(tf)


def y_transform(y, tf):
    if tf == "raw": return y
    y = y.clamp_min(0)
    return torch.log1p(y) if tf == "log1p" else torch.sqrt(y)


def y_back(p, tf):
    if tf == "raw": return p
    p = p.clamp_min(0) if tf == "sqrt" else p
    return torch.expm1(p).clamp_min(0) if tf == "log1p" else p * p


class Spec:
    """Feature set: src subset of {'pre','emb'}, pre_tf in raw/log1p/sqrt, context offsets (bins of 128 bp, excluding 0)."""
    def __init__(self, src=("pre",), pre_tf="raw", offsets=(), dims=None, ctx=None):
        self.src = tuple(src); self.pre_tf = pre_tf; self.offsets = tuple(sorted(offsets))
        self.ctx = tuple(ctx) if ctx is not None else self.src      # sources that receive the neighbouring-bin copies
        self.dims = dims or {"pre": 1888, "emb": EMB_DIM}      # override only in unit tests
        assert 0 not in self.offsets
    def as_dict(self): return dict(src=list(self.src), pre_tf=self.pre_tf, offsets=list(self.offsets), ctx=list(self.ctx))
    @classmethod
    def from_dict(cls, d): return cls(d["src"], d["pre_tf"], d["offsets"], ctx=d.get("ctx"))
    def base_dim(self): return sum(self.dims[s] for s in self.src)
    def ctx_dim(self): return sum(self.dims[s] for s in self.src if s in self.ctx)
    def dim(self): return self.base_dim() + self.ctx_dim() * len(self.offsets)
    def key(self): return f"{'+'.join(self.src)}|{self.pre_tf if 'pre' in self.src else '-'}"


def raw_base(chunk, spec):
    """chunk: dict of torch tensors on GPU {'pre': (B,896,1888), 'emb': (B,896,1920)} -> untransformed-standardisation base (B,896,p0) float32"""
    parts = []
    if "pre" in spec.src: parts.append(pre_transform(chunk["pre"].float(), spec.pre_tf))
    if "emb" in spec.src: parts.append(chunk["emb"].float())
    return torch.cat(parts, -1)


def build_features(chunk, spec, bmu, bsd, dtype=D):
    """Standardise base columns with TRAIN mean/sd (bmu, bsd), then append shifted copies (zero-padded in the standardised
    space = train-mean padding at tile edges). Returns (B*896, p) in `dtype`."""
    base = (raw_base(chunk, spec).to(dtype) - bmu.to(dtype)) / bsd.to(dtype)
    cb = base[..., :spec.ctx_dim()] if tuple(s_ for s_ in spec.src if s_ in spec.ctx) == spec.src[:len([1 for s_ in spec.src if s_ in spec.ctx])] else None
    assert cb is not None, "context sources must be a prefix of src order (pre, emb)"
    outs = [base] + [shift_bins(cb, k) for k in spec.offsets]
    f = torch.cat(outs, -1) if len(outs) > 1 else base
    return f.reshape(-1, f.shape[-1])


def load_chunk(ds, lo, hi, need=("pre", "emb"), tgt=True):
    c = {k: torch.from_numpy(np.ascontiguousarray(getattr(ds, k)[lo:hi])).to(DEV, non_blocking=True) for k in need}
    if tgt: c["tgt"] = torch.from_numpy(np.ascontiguousarray(ds.tgt[lo:hi])).to(DEV).double()
    return c


def chunks(ds, bs=8, need=("pre", "emb"), tgt=True):
    """prefetching iterator of (lo, hi, chunk) over consecutive tiles"""
    from concurrent.futures import ThreadPoolExecutor
    def rd(lo):
        hi = min(ds.n, lo + bs)
        d = {k: np.ascontiguousarray(getattr(ds, k)[lo:hi]) for k in need}
        if tgt: d["tgt"] = np.ascontiguousarray(ds.tgt[lo:hi])
        return lo, hi, d
    los = list(range(0, ds.n, bs))
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(rd, l) for l in los[:6]]; nxt = 6
        for i in range(len(los)):
            lo, hi, d = futs[i].result(); futs[i] = None
            if nxt < len(los): futs.append(ex.submit(rd, los[nxt])); nxt += 1
            yield lo, hi, {k: (torch.from_numpy(v).to(DEV).double() if k == "tgt" else torch.from_numpy(v).to(DEV)) for k, v in d.items()}


@torch.no_grad()
def base_stats(ds, spec, bs=16):
    """train mean/sd of the (transformed) base columns, fp64 (population sd)."""
    p0 = spec.base_dim(); s = torch.zeros(p0, dtype=D, device=DEV); s2 = torch.zeros_like(s); n = 0
    sp0 = Spec(spec.src, spec.pre_tf, (), spec.dims)
    for lo, hi, c in chunks(ds, bs, need=spec.src, tgt=False):
        x = raw_base(c, sp0).double().reshape(-1, p0); n += x.shape[0]; s += x.sum(0); s2 += (x * x).sum(0)
    mu = s / n; sd = (s2 / n - mu ** 2).clamp_min(0).sqrt(); sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    return mu, sd


# ------------------------------------------------------------------ ridge
YTF = ["raw", "log1p", "sqrt"]


@torch.no_grad()
def accumulate(ds, spec, bmu, bsd, ytfs=("raw",), bs=8, log=print):
    """Train Gram on features of `spec`; X'Y and Y sums for each target transform in ytfs (targets pooled to 128 bp first, then transformed)."""
    p = spec.dim(); n = 0
    sx = torch.zeros(p, dtype=D, device=DEV); sxx = torch.zeros(p, p, dtype=D, device=DEV)
    sy = {t: torch.zeros(N_TGT, dtype=D, device=DEV) for t in ytfs}; sxy = {t: torch.zeros(p, N_TGT, dtype=D, device=DEV) for t in ytfs}
    k = 0
    for lo, hi, c in chunks(ds, bs, need=spec.src):
        X = build_features(c, spec, bmu, bsd); Y = c["tgt"].reshape(-1, N_TGT)
        n += X.shape[0]; sx += X.sum(0); sxx += X.T @ X
        for t in ytfs:
            Yt = y_transform(Y, t); sy[t] += Yt.sum(0); sxy[t] += X.T @ Yt
        k += 1
        if log and k % 100 == 0: log(f"  gram {hi}/{ds.n} tiles")
    return dict(n=n, sx=sx, sxx=sxx, sy=sy, sxy=sxy, p=p)


class RidgeEig:
    """Eigendecomposition of the train correlation matrix; coefficients for any (target-transform, lambda)."""
    @torch.no_grad()
    def __init__(self, acc):
        n = acc["n"]; self.n = n; self.p = acc["p"]
        mu = acc["sx"] / n; sxx = acc["sxx"]; acc["sxx"] = None
        sxx.div_(n); sxx.sub_(torch.outer(mu, mu))                 # covariance (in place)
        sd = sxx.diagonal().clamp_min(0).sqrt(); live = sd > 1e-12; sd = torch.where(live, sd, torch.ones_like(sd))
        sxx.div_(torch.outer(sd, sd)); dead = ~live
        sxx[dead] = 0; sxx[:, dead] = 0
        self.mu, self.sd, self.live = mu, sd, live
        self.ev, self.V = torch.linalg.eigh(sxx); del sxx; torch.cuda.empty_cache()
        self.ev.clamp_min_(0)
        self.ybar = {t: v / n for t, v in acc["sy"].items()}
        self.VtC = {}
        for t, sxy in acc["sxy"].items():
            c = (sxy / n - torch.outer(mu, self.ybar[t])) / sd[:, None]; c[dead] = 0
            self.VtC[t] = self.V.T @ c
        acc["sxy"] = None

    @torch.no_grad()
    def coefs(self, ytf, lams):
        """(L, p, T) standardised-space coefficients"""
        return torch.stack([self.V @ (self.VtC[ytf] / (self.ev[:, None] + lam)) for lam in lams])


@torch.no_grad()
def eval_stream(ds, spec, bmu, bsd, eig, modes, lams, half_by_chrom=True, bs=8, collect_tiles=False, Bfixed=None):
    """Stream split `ds`; for each mode (ytf, backtransform: bool) compute pooled sums per (lambda, track) vs RAW targets.
    Returns {mode: dict(sums per half h in {0,1} (halves = first / second chromosome of the split))}. If collect_tiles, per-tile sums.
    Prediction in standardised-space: ((F - eig.mu)/eig.sd) @ B + ybar."""
    L = len(lams); T = N_TGT
    if Bfixed is not None:       # explicit (p, T) standardised coefficients per mode (e.g. per-track lambda); then L must be 1
        assert L == 1; Bc = {m: Bfixed[m] for m in modes}
    else:
        Bc = {m: eig.coefs(m[0], lams).permute(1, 0, 2).reshape(eig.p, L * T) for m in modes}
    chroms = list(dict.fromkeys(ds.chrom.tolist()))
    half = np.array([min(chroms.index(c), 1) for c in ds.chrom]) if half_by_chrom else np.zeros(ds.n, int)
    keys = ["sx", "sx2", "sxy", "sse"]
    acc = {m: {h: {k: torch.zeros(L, T, dtype=D, device=DEV) for k in keys} for h in (0, 1)} for m in modes}
    ysum = {h: dict(sy=torch.zeros(T, dtype=D, device=DEV), sy2=torch.zeros(T, dtype=D, device=DEV), n=0) for h in (0, 1)}
    tiles = {m: [] for m in modes} if collect_tiles else None
    ytile = [] if collect_tiles else None
    for lo, hi, c in chunks(ds, bs, need=spec.src):
        Xs = (build_features(c, spec, bmu, bsd) - eig.mu) / eig.sd; Xs[:, ~eig.live] = 0
        Y = c["tgt"].reshape(-1, T); nt = hi - lo
        for m in modes:
            P = (Xs @ Bc[m]).reshape(nt, G_BINS, L, T) + eig.ybar[m[0]]
            if m[1]: P = y_back(P, m[0])
            for t in range(nt):
                h = half[lo + t]; Pt = P[t]; Yt = Y[t * G_BINS:(t + 1) * G_BINS][:, None, :]
                s = dict(sx=Pt.sum(0), sx2=(Pt * Pt).sum(0), sxy=(Pt * Yt).sum(0), sse=((Pt - Yt) ** 2).sum(0))
                for k in keys: acc[m][h][k] += s[k]
                if collect_tiles: tiles[m].append({k: v.cpu().numpy() for k, v in s.items()})
        for t in range(nt):
            h = half[lo + t]; Yt = Y[t * G_BINS:(t + 1) * G_BINS]
            ysum[h]["sy"] += Yt.sum(0); ysum[h]["sy2"] += (Yt * Yt).sum(0); ysum[h]["n"] += G_BINS
            if collect_tiles: ytile.append(dict(sy=Yt.sum(0).cpu().numpy(), sy2=(Yt * Yt).sum(0).cpu().numpy()))
    out = {}
    for m in modes:
        out[m] = {h: {k: v.cpu().numpy() for k, v in acc[m][h].items()} for h in (0, 1)}
    return out, {h: dict(sy=ysum[h]["sy"].cpu().numpy(), sy2=ysum[h]["sy2"].cpu().numpy(), n=ysum[h]["n"]) for h in (0, 1)}, (tiles, ytile)


def sums_to_metrics(sm, ys, hs=(0, 1)):
    """combine halves -> per (L,T): r, r2, predR2"""
    n = sum(ys[h]["n"] for h in hs); sy = sum(ys[h]["sy"] for h in hs); sy2 = sum(ys[h]["sy2"] for h in hs)
    g = {k: sum(sm[h][k] for h in hs) for k in ["sx", "sx2", "sxy", "sse"]}
    r = pearson_from_sums(n, g["sx"], sy, g["sxy"], g["sx2"], sy2)
    return r, pearson_r2(r), prediction_r2(g["sse"], sy, sy2, n)


def lambda_choice(sm, ys, lams):
    """shared-lambda and per-track-lambda results on validation; plus chromosome-cross-fit estimate of the per-track optimism.
    Returns dict."""
    mk = masks()
    r, r2, pr2 = sums_to_metrics(sm, ys)                      # (L,T)
    mean_all = np.nanmean(r, 1); ls = int(np.argmax(mean_all))
    rt = np.where(np.isnan(r), -np.inf, r); lt = rt.argmax(0)       # per-track lambda index
    r_pt = r[lt, np.arange(N_TGT)]
    res = dict(shared_idx=ls, shared_lam=float(lams[ls]), per_track_idx=lt.tolist(),
               shared_r=mean_r(r[ls], mk), per_track_r=mean_r(r_pt, mk), shared_r2=float(np.nanmean(r2[ls])), shared_predR2=float(np.nanmean(pr2[ls])),
               grid_edge=bool(ls in (0, len(lams) - 1)), r_by_lambda_all183=mean_all.tolist(),
               per_track_r_vec=r_pt, shared_r_vec=r[ls])
    # cross-fit over validation chromosomes: choose on half a, evaluate on half b
    cf_s, cf_p = [], []
    for a, b in ((0, 1), (1, 0)):
        ra = sums_to_metrics(sm, ys, (a,))[0]; rb = sums_to_metrics(sm, ys, (b,))[0]
        la = int(np.argmax(np.nanmean(ra, 1))); lta = np.where(np.isnan(ra), -np.inf, ra).argmax(0)
        cf_s.append(float(np.nanmean(rb[la]))); cf_p.append(float(np.nanmean(rb[lta, np.arange(N_TGT)])))
    res["crossfit_shared_r"] = float(np.mean(cf_s)); res["crossfit_per_track_r"] = float(np.mean(cf_p))
    return res


def final_coefs(eig, ytf, lams, lam_idx):
    """lam_idx: int (shared) or array (T,) -> (p, T) standardised coefficients"""
    Bs = eig.coefs(ytf, lams)
    if np.isscalar(lam_idx): return Bs[int(lam_idx)]
    li = torch.as_tensor(np.asarray(lam_idx), device=DEV)
    return Bs[li, :, torch.arange(N_TGT, device=DEV)].T


# ------------------------------------------------------------------ registry
def next_id():
    """Next FT-Cnn id, reserved atomically (O_EXCL marker file) so concurrent processes never share an id."""
    n = 0
    if REGISTRY.exists():
        for r in csv.DictReader(open(REGISTRY)):
            if r["exp_id"].startswith("FT-C"): n = max(n, int(r["exp_id"][4:]))
    rd = OUTDIR / "reserved"; rd.mkdir(parents=True, exist_ok=True)
    for f in rd.glob("FT-C*"): n = max(n, int(f.name[4:]))
    while True:
        n += 1; eid = f"FT-C{n:02d}"
        try: os.close(os.open(rd / eid, os.O_CREAT | os.O_EXCL)); return eid
        except FileExistsError: continue


def log_registry(**kw):
    new = not REGISTRY.exists()
    row = {c: kw.get(c, "NA") for c in REG_COLS}
    with open(REGISTRY, "a", newline="") as f:
        w = csv.DictWriter(f, REG_COLS)
        if new: w.writeheader()
        w.writerow(row)
    return row


def now(): return time.strftime("%Y-%m-%dT%H:%M:%S")
