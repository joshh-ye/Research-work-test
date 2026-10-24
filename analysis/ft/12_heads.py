"""Phase-4 controlled head comparison on identical pooled-128 features: linear/MSE, MLP/MSE, MLP/Poisson+Softplus.
Early stopping on VALIDATION mean r (all 183 tracks); seed explicit; test never computed here.
Usage: 12_heads.py --set A|T|C [--pre-tf sqrt] [--offsets 1,2,4] [--families linear_mse,mlp_mse,mlp_poisson] [--only i,j]"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ftheads import *
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--pre-tf", default="sqrt")
ap.add_argument("--offsets", default=""); ap.add_argument("--families", default="linear_mse,mlp_mse,mlp_poisson")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-epochs", type=int, default=15); ap.add_argument("--only", default="")
ap.add_argument("--max-gb", type=float, default=29.0)
a = ap.parse_args()
cap_gpu(a.max_gb)
SETS = dict(A=("pre",), T=("emb",), C=("pre", "emb"), Cp=("pre", "emb")); CTX = dict(Cp=("pre",))
ks = [int(x) for x in a.offsets.split(",") if x]; offs = sorted(ks + [-x for x in ks])
spec = Spec(SETS[a.set], a.pre_tf, offs, ctx=CTX.get(a.set))
CFGS = {
    "linear_mse": [dict(kind="linear", loss="mse", ytf="raw", backtransform=False, lr=lr, wd=1e-4) for lr in (3e-4, 1e-3, 3e-3)],
    "mlp_mse": [dict(kind="mlp", loss="mse", ytf="raw", backtransform=False, lr=3e-4, wd=1e-4, hidden=1024, dropout=0.2),
                dict(kind="mlp", loss="mse", ytf="raw", backtransform=False, lr=1e-3, wd=1e-4, hidden=1024, dropout=0.2),
                dict(kind="mlp", loss="mse", ytf="sqrt", backtransform=True, lr=3e-4, wd=1e-4, hidden=1024, dropout=0.2)],
    "mlp_poisson": [dict(kind="mlp", loss="poisson", ytf="raw", backtransform=False, lr=lr, wd=1e-4, hidden=1024, dropout=0.2) for lr in (1e-4, 3e-4, 1e-3)],
}
todo = [(f, c) for f in a.families.split(",") for c in CFGS[f]]
if a.only: todo = [todo[int(i)] for i in a.only.split(",")]
tr_ds, va_ds = Pooled("train"), Pooled("val")
t0 = time.time(); ram = RamData(tr_ds, spec.src); print(f"RAM load {time.time()-t0:.0f}s", flush=True)
bmu, bsd = base_stats(tr_ds, spec)
CK = Path("/tmp/yejz1_ft/ckpt"); CK.mkdir(parents=True, exist_ok=True); CFG = OUTDIR / "configs"; CFG.mkdir(exist_ok=True)
PT = OUTDIR / "per_track_val_r"; PT.mkdir(exist_ok=True); RES = OUTDIR / "heads_val_results.csv"
for fam, cfg in todo:
    torch.cuda.reset_peak_memory_stats(); st = now(); t1 = time.time(); eid = next_id()
    print(f"{eid} {fam} set={a.set} p={spec.dim()} {cfg}", flush=True)
    try:
        head, info, (ymu, ysd), r = train_head(cfg, spec, ram, tr_ds, va_ds, bmu, bsd, a.seed, max_epochs=a.max_epochs, log=lambda s: print(s, flush=True))
    except torch.cuda.OutOfMemoryError as e:
        log_registry(exp_id=eid, method=fam, config=json.dumps(dict(cfg, set=a.set)), status="failed", notes="CUDA OOM"); torch.cuda.empty_cache(); continue
    mk = masks(); rr = {k: float(np.nanmean(r[m])) for k, m in mk.items()}
    full = dict(family=fam, **spec.as_dict(), p=spec.dim(), seed=a.seed, set=a.set, **cfg, best_epoch=info["best_epoch"], max_epochs=a.max_epochs,
                checkpoint=str(CK / f"{eid}.pt"), ymu=None if ymu is None else ymu.cpu().tolist(), ysd=None if ysd is None else ysd.cpu().tolist())
    torch.save(head.state_dict(), CK / f"{eid}.pt"); (CFG / f"{eid}.json").write_text(json.dumps(full)); np.savez(PT / f"{eid}.npz", r=r)
    (OUTDIR / f"hist_{eid}.json").write_text(json.dumps(info))
    log_registry(exp_id=eid, method=fam, config=json.dumps({k: full[k] for k in ("set", "src", "pre_tf", "offsets", "kind", "loss", "ytf", "backtransform", "lr", "wd", "dropout") if k in full}),
                 train="train(4659 tiles)", val="chr1+chr8", seed=a.seed, start=st, end=now(), runtime_s=round(time.time() - t1), gpu_mem_gb=round(gpu_peak_gb(), 1),
                 checkpoint=f"configs/{eid}.json (weights /tmp/yejz1_ft/ckpt/{eid}.pt, not persistent unless copied)", val_r_all183=round(rr["all183"], 4), val_r_152=round(rr["not_identical152"], 4),
                 status="done", notes=f"best_epoch {info['best_epoch']} (max_epochs {a.max_epochs}); early-stop on val mean r (all183); 128 bp pooled features/targets (original head: 32 bp)")
    pd.DataFrame([dict(exp_id=eid, family=fam, set=a.set, p=spec.dim(), offsets=",".join(map(str, offs)), pre_tf=spec.pre_tf, **{k: cfg[k] for k in cfg},
                       best_epoch=info["best_epoch"], val_r_all183=rr["all183"], val_r_152=rr["not_identical152"], seconds=round(time.time() - t1), gpu_gb=round(gpu_peak_gb(), 1))]
                 ).to_csv(RES, mode="a", header=not RES.exists(), index=False)
    print(f"{eid} done val all183 {rr['all183']:.4f} 152 {rr['not_identical152']:.4f} ({time.time()-t1:.0f}s)", flush=True)
