"""Pretrained-Borzoi non-oracle baseline per fold from the cache (no new inference): for each of the 183 targets pick the
eligible pretrained track (1882 ChIP non-histone, borzoi_baseline_utils.filter_borzoi_tracks) with the highest VALIDATION
pooled Pearson r on the shared 128 bp grid, then score on the fold's test tiles (raw Pearson r; prediction R^2 with
validation-calibrated linear map T ~ a + c P, as in reports/qing_deliverables). Also saves an oracle (test-selected,
optimistic upper bound) for reference. Output per-tile stats npz in estats format.
Usage: 04_pretrained_valsel.py --fold orig"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ridgelib import *
from ridgelib import _stream
from folds import split_gids
from borzoi_baseline_utils import load_borzoi_targets, filter_borzoi_tracks
from estats import pearson_from_sums

ap = argparse.ArgumentParser(); ap.add_argument("--fold", default="orig"); ap.add_argument("--out", default=None)
a = ap.parse_args()
out = Path(a.out or ROOT / f"results_phase2/e4_ridge/pretrained_valsel_{a.fold}"); out.mkdir(parents=True, exist_ok=True)
tr, va, te = split_gids(a.fold)
elig = filter_borzoi_tracks(load_borzoi_targets(ROOT / "borzoi_data/targets_human.csv"))[0]["track_index"].to_numpy()
elig_t = torch.as_tensor(elig, device=DEV)
feat = Feat("pre_raw")


@torch.no_grad()
def tile_stats(gids):
    keys = ["sT", "sT2", "sP", "sP2", "sTP"]; S = {k: [] for k in keys}
    for x, y in _stream(feat, gids):
        P = torch.from_numpy(x).to(DEV).to(D)[:, elig_t]; Y = torch.from_numpy(y).to(DEV).to(D)
        S["sT"].append(Y.sum(0)); S["sT2"].append((Y * Y).sum(0)); S["sP"].append(P.sum(0)); S["sP2"].append((P * P).sum(0))
        S["sTP"].append(Y.T @ P)
    return {k: torch.stack(v).cpu().numpy() for k, v in S.items()}


def pooled_r(S, n):
    sT, sT2, sP, sP2, sTP = [S[k].sum(0) for k in ["sT", "sT2", "sP", "sP2", "sTP"]]
    cov = sTP - sT[:, None] * sP[None] / n; vt = sT2 - sT ** 2 / n; vp = sP2 - sP ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vt[:, None] * vp[None])
    return np.where(np.isfinite(r), r, -np.inf), cov, vp


Sv = tile_stats(va); nv = len(va) * G_BINS
rv, covv, vpv = pooled_r(Sv, nv)
sel = rv.argmax(1)                                             # validation-selected (non-oracle)
St = tile_stats(te); nt = len(te) * G_BINS
rt, _, _ = pooled_r(St, nt); orc = rt.argmax(1)
# validation calibration for the selected tracks: T ~ a + c P
ar = np.arange(N_TGT)
c = covv[ar, sel] / vpv[sel]
a0 = Sv["sT"].sum(0) / nv - c * Sv["sP"].sum(0)[sel] / nv


def pick(S, s, cal=None):
    n = np.full((len(te), N_TGT), float(G_BINS))
    sT, sT2 = S["sT"], S["sT2"]; sP, sP2, sTP = S["sP"][:, s], S["sP2"][:, s], S["sTP"][:, ar, s]
    d = dict(n=n, sx=sP, sy=sT, sxy=sTP, sx2=sP2, sy2=sT2, sse_raw=sT2 - 2 * sTP + sP2)
    if cal is None: d["sse"] = d["sse_raw"]
    else:
        a_, c_ = cal; d["sse"] = sT2 - 2 * a_ * sT - 2 * c_ * sTP + n * a_ ** 2 + 2 * a_ * c_ * sP + c_ ** 2 * sP2
    return d

res = {}
for nm, s, cal in [("valsel", sel, (a0, c)), ("oracle", orc, None)]:
    d = pick(St, s, cal); d["gid"] = np.asarray(te); d["selected_track_index"] = elig[s]
    np.savez(out / f"{nm}_test_stats.npz", **d)
    g = {k: d[k].sum(0) for k in ["n", "sx", "sy", "sxy", "sx2", "sy2"]}
    r = pearson_from_sums(g["n"], g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    r2 = 1 - d["sse"].sum(0) / (g["sy2"] - g["sy"] ** 2 / g["n"])
    key = "mean_prediction_R2_validation_calibrated" if cal is not None else "mean_prediction_R2_uncalibrated_oracle_selection"
    res[nm] = {"mean_pearson_r": float(np.nanmean(r)), "n_tiles": len(te), key: float(np.nanmean(r2))}
pd_ = __import__("pandas")
pd_.DataFrame(dict(track_idx=ar, val_selected=elig[sel], oracle=elig[orc], val_r=rv[ar, sel], calib_a=a0, calib_c=c)).to_csv(out / "mapping.csv", index=False)
(out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
if a.fold == "orig":
    ref = json.loads((ROOT / "reports/qing_deliverables/evaluation_evidence/baseline_borzoi_rerun/summary.json").read_text())
    print("reference (Qing rerun summary.json keys):", list(ref)[:12])
