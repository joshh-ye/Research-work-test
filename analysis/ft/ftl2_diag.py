"""Post hoc diagnostic (EXP-020) of the FT-L2 test failure: tile-level prediction-level ratio FT-L2/FT-L1 on test and validation, and
per-track r with / without chr9:20-32 Mb (hg38). Descriptive only; never used for a claim. Writes research_management/finetuning/ftl2_diag.json"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "analysis/ft"))
from poolcache import Pooled
from ftlib import masks
mk = masks()["not_identical152"]
P = Pooled("test"); Y = np.asarray(P.tgt)
ld = lambda r: np.asarray(np.load(ROOT / f"results_ft/ablation/test_members/{r}/test_pred_rc.npy", mmap_mode="r"), dtype=np.float32)
A, B = ld("FT-L2"), ld("FT-L1")
reg = (P.chrom == "chr9") & (P.start >= 20_000_000) & (P.start < 32_000_000)
def r(x, keep):
    x = x[keep].reshape(-1, 183).astype(np.float64); y = Y[keep].reshape(-1, 183).astype(np.float64)
    x -= x.mean(0); y -= y.mean(0); return (x * y).sum(0) / np.sqrt((x * x).sum(0) * (y * y).sum(0))
rat_t = A.reshape(359, -1).mean(1) / B.reshape(359, -1).mean(1)
V2 = np.load(ROOT / "results_ft/preds/FT-L2/val_pred_rc.npy", mmap_mode="r"); V1 = np.load(ROOT / "results_ft/preds/FT-L1/val_pred_rc.npy", mmap_mode="r")
rat_v = np.array([float(np.asarray(V2[i], np.float32).mean() / np.asarray(V1[i], np.float32).mean()) for i in range(V2.shape[0])])
tr = A[reg].mean((0, 1)) / B[reg].mean((0, 1))
others = {m: float(np.median(ld(m)[reg].mean((0, 1)) / B[reg].mean((0, 1)))) for m in ["FT-S03", "FT-R1", "FT-R2", "FT-R3", "FT-S04", "FT-L3"]}
tgt_ratio = float(np.median(Y[reg].mean((0, 1)) / B[reg].mean((0, 1))))
gap = (Y.reshape(359, -1).max(1) == 0)
out = dict(region="chr9:20,000,000-32,000,000 hg38", n_region_tiles=int(reg.sum()),
           test_tile_level_ratio_quantiles_0_1_50_99_100=np.quantile(rat_t, [0, .01, .5, .99, 1]).round(3).tolist(),
           val_tile_level_ratio_quantiles_0_1_50_99_100=np.quantile(rat_v, [0, .01, .5, .99, 1]).round(3).tolist(),
           region_per_track_ratio_quantiles_0_10_50_90_100=np.quantile(tr, [0, .1, .5, .9, 1]).round(2).tolist(),
           region_per_track_median_ratio_targets_over_FTL1=tgt_ratio, region_per_track_median_ratio_member_over_FTL1=others, n_test_tiles_all_zero_target=int(gap.sum()), ratio_on_zero_target_tiles=np.unique(rat_t[gap].round(3)).tolist(), FT_L2_r152_all=float(np.nanmean(r(A, np.ones(359, bool))[mk])), FT_L2_r152_excl_region=float(np.nanmean(r(A, ~reg)[mk])),
           FT_L1_r152_all=float(np.nanmean(r(B, np.ones(359, bool))[mk])), FT_L1_r152_excl_region=float(np.nanmean(r(B, ~reg)[mk])))
json.dump(out, open(ROOT / "research_management/finetuning/ftl2_diag.json", "w"), indent=1); print(json.dumps(out, indent=1))
