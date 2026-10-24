"""Gate (FT-C01): reproduce the E4 ridge[pre_raw:union1888] (shared lambda on the E4 grid, val-selected) from the pooled copy.
Test is computed ONLY here, for this already-public E4 config. Also checks pooled targets == E4 code path on all tiles."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ftlib import *
import pandas as pd

cap_gpu(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); st = now()
tr, va, te = Pooled("train"), Pooled("val"), Pooled("test")
# 1. targets identical to the E4 path
mx = 0.0; mxrel = 0.0
for ds in (tr, va, te):
    for i, g in enumerate(ds.gid):
        e4 = to_grid128(central(np.load(tile_path(int(g), "tgt"), mmap_mode="r")))
        d = np.abs(ds.tgt[i].astype(np.float64) - e4); mx = max(mx, float(d.max())); mxrel = max(mxrel, float((d / np.maximum(np.abs(e4), 1e-12)).max()))
print("pooled targets vs E4 float64 path: max abs diff", mx, "max rel diff", mxrel, flush=True)
# 2. ridge
spec = Spec(("pre",), "raw", ()); bmu, bsd = base_stats(tr, spec)
acc = accumulate(tr, spec, bmu, bsd, ("raw",)); eig = RidgeEig(acc)
lams = np.array(E4_LAMBDAS)
sm, ys, _ = eval_stream(va, spec, bmu, bsd, eig, [("raw", False)], lams)
res = lambda_choice(sm[("raw", False)], ys, lams)
print("val mean r by lambda", np.round(res["r_by_lambda_all183"], 4).tolist(), "-> lam", res["shared_lam"], res["shared_r"], flush=True)
li = res["shared_idx"]
smt, yst, _ = eval_stream(te, spec, bmu, bsd, eig, [("raw", False)], lams[li:li + 1])
rt = sums_to_metrics(smt[("raw", False)], yst)[0][0]; test = mean_r(rt)
ref = json.load(open(ROOT / "results_phase2/e4_ridge/orig_pre_raw/summary.json"))[0]
out = dict(pooled_target_max_abs_diff=mx, pooled_target_max_rel_diff=mxrel, lam=res["shared_lam"], val=res["shared_r"], test=test,
           e4_lam=ref["lam"], e4_val=ref["val_mean_pearson_r"], e4_test=ref["test_mean_pearson_r"],
           diff_val=res["shared_r"]["all183"] - ref["val_mean_pearson_r"], diff_test=test["all183"] - ref["test_mean_pearson_r"],
           seconds=round(time.time() - t0), gpu_peak_gb=gpu_peak_gb())
print(json.dumps(out, indent=1))
(OUTDIR / "gate_E4_repro.json").write_text(json.dumps(out, indent=1))
id_ = next_id()
log_registry(exp_id=id_, method="ridge", config=json.dumps(dict(spec.as_dict(), ytf="raw", lam_grid="E4", lam_mode="shared", lam=res["shared_lam"])),
             train="train(4659 tiles)", val="chr1+chr8", seed="NA", start=st, end=now(), runtime_s=round(time.time() - t0), gpu_mem_gb=round(gpu_peak_gb(), 1),
             checkpoint="NA", val_r_all183=round(res["shared_r"]["all183"], 4), val_r_152=round(res["shared_r"]["not_identical152"], 4),
             test_r_all183=round(test["all183"], 4), test_r_152=round(test["not_identical152"], 4), status="done",
             notes=f"GATE: E4 ridge[pre_raw:union1888] from pooled copy (public config; test computed only here). E4 val {ref['val_mean_pearson_r']:.4f} test {ref['test_mean_pearson_r']:.4f}")
