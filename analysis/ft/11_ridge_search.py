"""Ridge optimisation on the pooled-128 copy; validation-only selection; test is never touched here.
Stages: s1 = feature transform x target transform (+shared vs per-track lambda) on pretrained outputs;
        s2 = feature sets (outputs / trunk / combined) with the s1 winner; s3 = spatial context offsets on A/T/C.
Usage: 11_ridge_search.py --stage s1|s2|s3 [--pre-tf raw --mode log1p:1]"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ftlib import *
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True)
ap.add_argument("--pre-tf", default="raw"); ap.add_argument("--mode", default="raw:0", help="ytf:backtransform(0/1) winner of s1, used in s2/s3")
ap.add_argument("--offsets", default="1;1,2;1,2,4", help="s3 symmetric offset sets, ';' separated")
ap.add_argument("--sets", default="A,T,C", help="s3 feature sets; 'Cp' = outputs+trunk with context only on outputs (ctx=pre)"); ap.add_argument("--max-gb", type=float, default=29.0)
a = ap.parse_args()
CFG = OUTDIR / "configs"; CFG.mkdir(exist_ok=True)
RES = OUTDIR / "ridge_val_results.csv"; PT = OUTDIR / "per_track_val_r"; PT.mkdir(exist_ok=True)
cap_gpu(a.max_gb)
tr, va = Pooled("train"), Pooled("val")
SETS = dict(A=("pre",), T=("emb",), C=("pre", "emb"), Cp=("pre", "emb"))
CTX = dict(Cp=("pre",))
ytf0, bt0 = a.mode.split(":"); MODE0 = (ytf0, bool(int(bt0)))


def run_group(spec, modes, tag):
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); st = now()
    ytfs = sorted({m[0] for m in modes}, key=YTF.index)
    bmu, bsd = base_stats(tr, spec)
    acc = accumulate(tr, spec, bmu, bsd, ytfs); eig = RidgeEig(acc); t1 = time.time()
    sm, ys, _ = eval_stream(va, spec, bmu, bsd, eig, modes, LAMBDAS)
    rows = []
    for m in modes:
        res = lambda_choice(sm[m], ys, LAMBDAS)
        for lm in ("shared", "per_track"):
            r = res["shared_r"] if lm == "shared" else res["per_track_r"]
            eid = next_id()
            cfg = dict(family="ridge", **spec.as_dict(), ytf=m[0], backtransform=m[1], lam_mode=lm, grid="logspace(-7,2,0.5)",
                       lam=res["shared_lam"] if lm == "shared" else None, lam_idx=res["shared_idx"] if lm == "shared" else res["per_track_idx"],
                       p=spec.dim(), tag=tag)
            (CFG / f"{eid}.json").write_text(json.dumps(cfg))
            rv = res["shared_r_vec"] if lm == "shared" else res["per_track_r_vec"]
            np.savez(PT / f"{eid}.npz", r=rv)
            note = f"{tag}; p={spec.dim()}; shared_lam={res['shared_lam']:.2g}{' GRID-EDGE' if res['grid_edge'] else ''}; val r2(sq pearson)={res['shared_r2']:.4f} predR2={res['shared_predR2']:.4f} (shared lam; predR2 not meaningful if no back-transform); " \
                   f"crossfit(chr1<->chr8) shared={res['crossfit_shared_r']:.4f} per_track={res['crossfit_per_track_r']:.4f}"
            log_registry(exp_id=eid, method="ridge", config=json.dumps({k: cfg[k] for k in ("src", "pre_tf", "offsets", "ytf", "backtransform", "lam_mode", "lam")}),
                         train="train(4659 tiles)", val="chr1+chr8", seed="NA", start=st, end=now(), runtime_s=round(time.time() - t0),
                         gpu_mem_gb=round(gpu_peak_gb(), 1), checkpoint=f"configs/{eid}.json",
                         val_r_all183=round(r["all183"], 4), val_r_152=round(r["not_identical152"], 4), status="done", notes=note)
            rows.append(dict(exp_id=eid, tag=tag, src="+".join(spec.src), pre_tf=spec.pre_tf, offsets=",".join(map(str, spec.offsets)), p=spec.dim(),
                             ytf=m[0], backtransform=int(m[1]), lam_mode=lm, lam=cfg["lam"], val_r_all183=r["all183"], val_r_152=r["not_identical152"],
                             val_r2_sqpearson_shared=res["shared_r2"], val_predR2_shared=res["shared_predR2"], crossfit_shared=res["crossfit_shared_r"],
                             crossfit_per_track=res["crossfit_per_track_r"], grid_edge=res["grid_edge"], seconds=round(time.time() - t0), gpu_gb=round(gpu_peak_gb(), 1)))
            print(f"{eid} {tag} {m} {lm}: val all183 {r['all183']:.4f} 152 {r['not_identical152']:.4f} lam {cfg['lam']} (grp {time.time()-t0:.0f}s, gram+eig {t1-t0:.0f}s, gpu {gpu_peak_gb():.1f}GB)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES, mode="a", header=not RES.exists(), index=False)
    del eig, acc; torch.cuda.empty_cache()
    return df


if a.stage == "s1":
    modes = [("raw", False), ("log1p", True), ("log1p", False), ("sqrt", True), ("sqrt", False)]
    for tf in ("raw", "log1p", "sqrt"):
        run_group(Spec(("pre",), tf, ()), modes, f"s1 A pre_tf={tf}")
elif a.stage == "s2":
    for k in ("A", "T", "C"):
        spec = Spec(SETS[k], a.pre_tf, ())
        run_group(spec, [MODE0] + ([("raw", False)] if MODE0 != ("raw", False) else []), f"s2 {k}")
elif a.stage == "s3":
    for k in a.sets.split(","):
        for o in a.offsets.split(";"):
            ks = [int(x) for x in o.split(",")]; offs = sorted(ks + [-x for x in ks])
            try:
                run_group(Spec(SETS[k], a.pre_tf, offs, ctx=CTX.get(k)), [MODE0], f"s3 {k} offsets={offs}")
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM {k} {offs}: {e}", flush=True); log_registry(exp_id=next_id(), method="ridge", config=json.dumps(dict(set=k, offsets=offs)), status="failed", notes="CUDA OOM under 29 GB cap"); torch.cuda.empty_cache()
