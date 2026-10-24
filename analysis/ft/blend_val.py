"""VALIDATION-ONLY ensembles / blends of fine-tuned (FT) predictions with cache-model (ridge) predictions on the shared 896 x 128 bp grid.
hg38, val = chr1 + chr8 (750 tiles).  Nothing here touches test tiles.
Candidate syntax (--cands, ';'-separated):
   FT-S05                      single FT run (w irrelevant)
   ENS:FT-S01,FT-S05           mean of the runs' val predictions
   FT-S05+E4                   blend  w*FT + (1-w)*ridge   (w = weight on the FT/ENS prediction; w=1 -> FT alone, w=0 -> ridge alone)
   ENS:FT-S01,FT-S05+FT-C61    blend of an ensemble with the FT-C61 ridge
Cache models: E4 -> config FT-C02 (ridge[pre_raw:union1888], lambda 1e-4; logged val r 0.7337), FT-C61 (logged val r 0.7392).  Ridge val predictions are regenerated
on the pooled-128 copy (/tmp/yejz1_ft/pooled128, same 896 x 128 grid / val gid order as the FT predictions) by refitting the ridge on TRAIN tiles
(deterministic Gram+eigh, as test_eval_cache.py) and cached as results_ft/preds/<ID>/val_pred.npy (fp16).  They must reproduce the logged val r (1e-3).
Weight grid w in {0,0.1,...,1}; w* = argmax mean Pearson r (all183) on all validation tiles ("selected on validation": optimistic).  Honest estimate =
2-fold cross-fit over validation chromosomes: choose w on chr1 -> score chr8, choose w on chr8 -> score chr1, report the mean of the two fold scores
(same convention as ftlib.lambda_choice); components are scored on the same folds for comparison.  Blend sums are computed exactly from per-tile
sx, sx2, sxy, sse and the cross term sum(P_A*P_B) (no per-w pass over the data).
Metrics: Pearson r (mean over tracks of per-track r pooled over all bins of the tile set), r^2 = squared Pearson r, predR2 = 1 - SSE/SST (labelled).
Outputs: results_ft/blends/blend_val_results.csv (one row per candidate; upserted), blend_val_grid.csv (all w).
Usage: blend_val.py --cands "FT-S05;FT-S05+E4;FT-S05+FT-C61"   [--fit-ridge-max-gb 18]"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
from estats import pearson_from_sums, pearson_r2, prediction_r2

PREDS = ROOT / "results_ft/preds"
BLENDS = ROOT / "results_ft/blends"
CACHE_MODELS = {"E4": ("FT-C02", "E4_ridge"), "FT-C61": ("FT-C61", "FT-C61")}   # alias -> (config id, preds dir)
LOGGED_RIDGE_R = {"E4": 0.7337, "FT-C61": 0.7392}
W_GRID = np.round(np.arange(0, 1.0001, 0.1), 2)
REPRO_TOL = 1e-3
G, T = 896, 183


# ---------------------------------------------------------------------------------------------------------------- pure stats helpers
def pair_stats(PA, PB, Y):
    """per-tile sums (tiles x T) for two prediction arrays PA, PB and targets Y, each (tiles, G, T) any float dtype; fp64 accumulation per chunk"""
    n = PA.shape[0]
    keys = ["sa", "saa", "say", "sb", "sbb", "sby", "sab", "sy", "syy"]
    S = {k: np.zeros((n, T)) for k in keys}
    for i in range(0, n, 25):
        a = np.asarray(PA[i:i + 25], np.float64); b = np.asarray(PB[i:i + 25], np.float64); y = np.asarray(Y[i:i + 25], np.float64)
        S["sa"][i:i + 25] = a.sum(1); S["saa"][i:i + 25] = (a * a).sum(1); S["say"][i:i + 25] = (a * y).sum(1)
        S["sb"][i:i + 25] = b.sum(1); S["sbb"][i:i + 25] = (b * b).sum(1); S["sby"][i:i + 25] = (b * y).sum(1)
        S["sab"][i:i + 25] = (a * b).sum(1); S["sy"][i:i + 25] = y.sum(1); S["syy"][i:i + 25] = (y * y).sum(1)
    S["n"] = np.full((n, T), float(G))
    return S


def blend_sums(S, w, sel=slice(None)):
    """pooled (over the tiles `sel`) sums of the blend w*A + (1-w)*B -> dict n,sx,sy,sxy,sx2,sy2,sse (each (T,))"""
    g = {k: v[sel].sum(0) for k, v in S.items()}
    v = 1.0 - w
    sx = w * g["sa"] + v * g["sb"]; sxy = w * g["say"] + v * g["sby"]
    sx2 = w * w * g["saa"] + 2 * w * v * g["sab"] + v * v * g["sbb"]
    sse = sx2 - 2 * sxy + g["syy"]
    return dict(n=g["n"], sx=sx, sy=g["sy"], sxy=sxy, sx2=sx2, sy2=g["syy"], sse=sse)


def metrics_from(g):
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    return r, pearson_r2(r), prediction_r2(g["sse"], g["sy"], g["sy2"], g["n"])


def mean_over(v, mask):
    return float(np.nanmean(v[mask]))


def evaluate_grid(S, masks, sel=slice(None), ws=W_GRID):
    """-> dict w -> dict(r183, r152, r2_183, predR2_183)"""
    out = {}
    for w in ws:
        r, r2, pr2 = metrics_from(blend_sums(S, float(w), sel))
        out[float(w)] = dict(r_all183=mean_over(r, masks["all183"]), r_152=mean_over(r, masks["not_identical152"]),
                             r2_sqpearson_all183=mean_over(r2, masks["all183"]), predR2_all183=mean_over(pr2, masks["all183"]))
    return out


def select_w(grid, key="r_all183"):
    ws = sorted(grid)
    return max(ws, key=lambda w: grid[w][key])      # ties -> smallest w (first max in ascending order)


def crossfit(S, masks, chrom, ws=W_GRID):
    """choose w on chromosome a, score on chromosome b (and vice versa); returns dict of per-fold w / r and their means"""
    chs = list(dict.fromkeys(chrom.tolist())); assert len(chs) == 2, chs
    res = {}
    for k, (a, b) in enumerate(((chs[0], chs[1]), (chs[1], chs[0]))):
        ia, ib = np.where(chrom == a)[0], np.where(chrom == b)[0]
        ga = evaluate_grid(S, masks, ia, ws); w = select_w(ga)
        gb = evaluate_grid(S, masks, ib, ws)[w]
        res[f"fold{k}"] = dict(train_chrom=a, test_chrom=b, w=w, r_all183=gb["r_all183"], r_152=gb["r_152"])
    res["mean_r_all183"] = float(np.mean([res["fold0"]["r_all183"], res["fold1"]["r_all183"]]))
    res["mean_r_152"] = float(np.mean([res["fold0"]["r_152"], res["fold1"]["r_152"]]))
    return res


# ---------------------------------------------------------------------------------------------------------------- prediction I/O
def val_tables():
    import pandas as pd
    tt = pd.read_csv(ROOT / "results_phase2/trunk_cache/tiles.csv")
    t = tt[tt.split == "val"].sort_values("split_idx")
    return t.gid.to_numpy(), t.chrom.to_numpy()


def split_tta(run):
    """'FT-S05@rc' -> ('FT-S05', '_rc'): RC-averaged TTA predictions written by predict_ft.py --tta rc"""
    name, _, tta = run.partition("@")
    return name, ("_" + tta if tta else "")


def load_ft_pred(run, gid):
    name, sfx = split_tta(run)
    d = PREDS / name
    assert (np.load(d / "val_gid.npy") == gid).all(), f"{run}: gid order differs from pooled val order"
    return np.load(d / f"val_pred{sfx}.npy", mmap_mode="r")


class Ens:
    """lazy mean of several fp16 prediction memmaps -> array-like supporting slicing on the first axis"""
    def __init__(self, arrs): self.arrs = arrs; self.shape = arrs[0].shape
    def __len__(self): return self.shape[0]
    def __getitem__(self, sl): return np.mean([np.asarray(a[sl], np.float32) for a in self.arrs], 0)


def ridge_val_pred(alias, max_gb=18.0, force=False, log=print):
    """regenerate (or load cached) ridge val predictions on the pooled grid; verifies the logged val r"""
    cid, dname = CACHE_MODELS[alias]
    od = PREDS / dname; p = od / "val_pred.npy"
    if p.exists() and not force:
        return np.load(p, mmap_mode="r")
    import torch
    from ftlib import (Spec, Pooled, base_stats, accumulate, RidgeEig, final_coefs, build_features, chunks, cap_gpu, LAMBDAS, DEV)
    cfg = json.load(open(ROOT / "results_ft/cache_exps/configs" / f"{cid}.json"))
    assert cfg["family"] == "ridge" and cfg["lam_mode"] == "shared"
    cap_gpu(max_gb); t0 = time.time()
    spec = Spec.from_dict(cfg); tr, va = Pooled("train"), Pooled("val")
    bmu, bsd = base_stats(tr, spec)
    acc = accumulate(tr, spec, bmu, bsd, (cfg["ytf"],)); eig = RidgeEig(acc)
    li = int(cfg["lam_idx"]); assert abs(np.log10(LAMBDAS[li]) - np.log10(cfg["lam"])) < 1e-6
    B = final_coefs(eig, cfg["ytf"], LAMBDAS, li)
    from ftlib import y_back
    od.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(str(p) + ".tmp.npy", "w+", np.float16, (va.n, G, T))
    with torch.no_grad():
        for lo, hi, c in chunks(va, 8, need=spec.src, tgt=False):
            Xs = (build_features(c, spec, bmu, bsd) - eig.mu) / eig.sd; Xs[:, ~eig.live] = 0
            P = (Xs @ B).reshape(hi - lo, G, T) + eig.ybar[cfg["ytf"]]
            if cfg["backtransform"]: P = y_back(P, cfg["ytf"])
            out[lo:hi] = P.float().cpu().numpy().astype(np.float16)
    out.flush(); del out
    import os; os.replace(str(p) + ".tmp.npy", p)
    np.save(od / "val_gid.npy", va.gid)
    json.dump(dict(alias=alias, config=cid, cfg=cfg, seconds=round(time.time() - t0), gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)),
              open(od / "info.json", "w"), indent=1)
    log(f"ridge {alias} ({cid}) val preds written {p} ({time.time() - t0:.0f}s)")
    del eig, acc; torch.cuda.empty_cache()
    return np.load(p, mmap_mode="r")


def blend_tile_stats(A, B, Y, w):
    """per-tile sufficient stats (n,sx,sy,sxy,sx2,sy2,sse; tiles x T) of the fixed-weight blend w*A + (1-w)*B vs Y (exact, from pair_stats)"""
    S = pair_stats(A, B, Y); v = 1.0 - w
    sx = w * S["sa"] + v * S["sb"]; sxy = w * S["say"] + v * S["sby"]
    sx2 = w * w * S["saa"] + 2 * w * v * S["sab"] + v * v * S["sbb"]
    return dict(n=S["n"], sx=sx, sy=S["sy"], sxy=sxy, sx2=sx2, sy2=S["syy"], sse=sx2 - 2 * sxy + S["syy"])


def ridge_split_pred(cfg_name, split, out_path, max_gb=18.0, log=print):
    """Refit a ridge config on TRAIN tiles (deterministic Gram+eigh, as test_eval_cache.py) and write its predictions for `split` (fp16, tiles x 896 x 183).
    cfg_name: 'E4' (= FT-C02) or an id with results_ft/cache_exps/configs/<id>.json.  Used by the LOCKED test-blend path (split='test') - never call for test outside it."""
    import torch, os
    from ftlib import (Spec, Pooled, base_stats, accumulate, RidgeEig, final_coefs, build_features, chunks, cap_gpu, LAMBDAS, y_back)
    cid = CACHE_MODELS[cfg_name][0] if cfg_name in CACHE_MODELS else cfg_name
    cfg = json.load(open(ROOT / "results_ft/cache_exps/configs" / f"{cid}.json"))
    assert cfg["family"] == "ridge" and cfg["lam_mode"] == "shared"
    cap_gpu(max_gb); t0 = time.time()
    spec = Spec.from_dict(cfg); tr, te = Pooled("train"), Pooled(split)
    bmu, bsd = base_stats(tr, spec)
    acc = accumulate(tr, spec, bmu, bsd, (cfg["ytf"],)); eig = RidgeEig(acc)
    li = int(cfg["lam_idx"]); assert abs(np.log10(LAMBDAS[li]) - np.log10(cfg["lam"])) < 1e-6
    B = final_coefs(eig, cfg["ytf"], LAMBDAS, li)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(str(out_path) + ".tmp.npy", "w+", np.float16, (te.n, G, T))
    with torch.no_grad():
        for lo, hi, c in chunks(te, 8, need=spec.src, tgt=False):
            Xs = (build_features(c, spec, bmu, bsd) - eig.mu) / eig.sd; Xs[:, ~eig.live] = 0
            P = (Xs @ B).reshape(hi - lo, G, T) + eig.ybar[cfg["ytf"]]
            if cfg["backtransform"]: P = y_back(P, cfg["ytf"])
            out[lo:hi] = P.float().cpu().numpy().astype(np.float16)
    out.flush(); del out
    os.replace(str(out_path) + ".tmp.npy", out_path)
    np.save(out_path.parent / f"{split}_gid.npy", te.gid)
    log(f"ridge {cfg_name} ({cid}) {split} preds written {out_path} ({time.time() - t0:.0f}s)")
    del eig, acc; torch.cuda.empty_cache()
    return cid


def parse_cand(s):
    """'ENS:a,b+E4' -> (ft_runs list, cache alias or None)"""
    parts = s.split("+")
    assert len(parts) <= 2, f"bad candidate {s}"
    ft, cache = parts[0], (parts[1] if len(parts) == 2 else "")
    runs = ft[4:].split(",") if ft.startswith("ENS:") else [ft]
    cache = cache or None
    assert cache is None or cache in CACHE_MODELS, f"unknown cache model {cache}; known: {list(CACHE_MODELS)}"
    return runs, cache


# ---------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True); ap.add_argument("--fit-ridge-max-gb", type=float, default=18.0); ap.add_argument("--force-ridge", action="store_true")
    ap.add_argument("--out-dir", default=str(BLENDS))
    a = ap.parse_args(argv)
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    from ftlib import masks as _masks, Pooled
    masks = _masks()
    gid, chrom = val_tables()
    va = Pooled("val"); assert (va.gid == gid).all()
    Y = va.tgt
    rows, grids = [], []
    ridge_cache = {}
    for cand in [c.strip() for c in a.cands.split(";") if c.strip()]:
        runs, cache = parse_cand(cand)
        ft = [load_ft_pred(r, gid) for r in runs]
        for r in runs:       # target identity: FT-loader targets (stats sy / sy2) == pooled-128 targets used for the ridge
            st = np.load(PREDS / split_tta(r)[0] / "val_stats.npz")
            sy = np.asarray(Y, np.float64).sum(1); assert np.allclose(st["sy"], sy, rtol=1e-5, atol=1e-5), f"{r}: sy differs from pooled val targets"
        A = ft[0] if len(ft) == 1 else Ens(ft)
        if cache:
            if cache not in ridge_cache:
                ridge_cache[cache] = ridge_val_pred(cache, a.fit_ridge_max_gb, a.force_ridge)
                assert (np.load(PREDS / CACHE_MODELS[cache][1] / "val_gid.npy") == gid).all()
            B = ridge_cache[cache]
        else:
            B = A
        S = pair_stats(A, B, Y)
        grid = evaluate_grid(S, masks)
        cf = crossfit(S, masks, chrom) if cache else None
        ws = select_w(grid) if cache else 1.0
        # component references
        ft_only, cache_only = grid[1.0], (grid[0.0] if cache else None)
        if cache:
            logged = LOGGED_RIDGE_R[cache]
            assert abs(cache_only["r_all183"] - logged) <= REPRO_TOL, f"ridge {cache} val r {cache_only['r_all183']:.4f} != logged {logged}"
        # component scores on the cross-fit folds (chr-wise, mean of two chromosome scores) for an apples-to-apples comparison
        def chromwise(w):
            v = [evaluate_grid(S, masks, np.where(chrom == c)[0], [w])[w] for c in dict.fromkeys(chrom.tolist())]
            return float(np.mean([x["r_all183"] for x in v])), float(np.mean([x["r_152"] for x in v]))
        row = dict(candidate=cand, ft_runs="+".join(runs), cache_model=cache or "", n_tiles=len(gid), w_grid="0:0.1:1", w_ft_selected_on_val=ws,
                   val_r_all183_at_w=grid[ws]["r_all183"], val_r_152_at_w=grid[ws]["r_152"],
                   val_r2_sqpearson_all183_at_w=grid[ws]["r2_sqpearson_all183"], val_predR2_all183_at_w=grid[ws]["predR2_all183"],
                   val_r_all183_ft_only=ft_only["r_all183"], val_r_152_ft_only=ft_only["r_152"],
                   val_r_all183_cache_only=cache_only["r_all183"] if cache else "", val_r_152_cache_only=cache_only["r_152"] if cache else "")
        if cf:
            row.update(crossfit_w_chr_fold0=cf["fold0"]["w"], crossfit_w_chr_fold1=cf["fold1"]["w"], crossfit_train_chroms=f"{cf['fold0']['train_chrom']}|{cf['fold1']['train_chrom']}",
                       crossfit_r_all183=cf["mean_r_all183"], crossfit_r_152=cf["mean_r_152"])
            f1, f0 = chromwise(1.0), chromwise(0.0)
            row.update(chromwise_mean_r_all183_ft_only=f1[0], chromwise_mean_r_all183_cache_only=f0[0])
        else:
            f1 = chromwise(1.0)
            row.update(chromwise_mean_r_all183_ft_only=f1[0])
        rows.append(row)
        for w, v in grid.items():
            grids.append(dict(candidate=cand, w_ft=w, **v))
        print(f"{cand}: w*={ws} val r183 {grid[ws]['r_all183']:.4f} r152 {grid[ws]['r_152']:.4f}"
              + (f" | crossfit r183 {cf['mean_r_all183']:.4f} (w folds {cf['fold0']['w']},{cf['fold1']['w']}) | FT-only {ft_only['r_all183']:.4f} cache-only {cache_only['r_all183']:.4f}" if cf else ""), flush=True)

    def upsert(path, new, key):
        df = pd.DataFrame(new)
        if Path(path).exists():
            old = pd.read_csv(path); old = old[~old[key].isin(df[key])]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False)
    upsert(outd / "blend_val_results.csv", rows, "candidate"); upsert(outd / "blend_val_grid.csv", grids, "candidate")
    print("wrote", outd / "blend_val_results.csv")


if __name__ == "__main__":
    main()
