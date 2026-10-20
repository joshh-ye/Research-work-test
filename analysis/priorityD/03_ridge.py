"""E4 supervised baseline: ridge from pretrained-Borzoi outputs (or a linear probe on trunk embeddings) to the 183 CRC targets.
Same training chromosomes as Borzoi-TL, lambda chosen on validation chromosomes only, per-tile test stats saved in the same
format as the TL per-tile stats (reusable with analysis/exposure_stratified/estats.py).
Usage: 03_ridge.py --fold orig --feat pre_raw [--out DIR]"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from ridgelib import *
from folds import split_gids

ap = argparse.ArgumentParser()
ap.add_argument("--fold", default="orig"); ap.add_argument("--feat", required=True, choices=["pre_raw", "pre_log1p", "probe"])
ap.add_argument("--out", default=None); ap.add_argument("--subsets", default=None)
a = ap.parse_args()
out = Path(a.out or ROOT / f"results_phase2/e4_ridge/{a.fold}_{a.feat}"); out.mkdir(parents=True, exist_ok=True)
lf = open(out / "run.log", "a")
def log(*x):
    s = " ".join(map(str, x)); print(s, flush=True); lf.write(s + "\n"); lf.flush()

tr, va, te = split_gids(a.fold)
feat = Feat(a.feat)
# feature subsets (positions in the cached pre128 columns; slim cache stores 1,888 of the 7,611 human tracks)
subsets = {}
if a.feat == "probe":
    subsets["trunk1920"] = np.arange(EMB_DIM)
else:
    from borzoi_baseline_utils import load_borzoi_targets, filter_borzoi_tracks
    df = load_borzoi_targets(ROOT / "borzoi_data/targets_human.csv")
    elig = filter_borzoi_tracks(df)[0]["track_index"].to_numpy(); assert len(elig) == 1882
    crc = pd.read_csv(ROOT / "variant_scoring/track_lists/borzoi_pretrained_crc_tracks.tsv", sep="\t").track_index.to_numpy()
    U = np.load(CACHE / "pre_channels.npy")      # slim cache: pre128 columns = U (human-track indices)
    subsets = {"union1888": np.arange(len(U)), "eligible1882_chip": np.searchsorted(U, elig), "crc132": np.searchsorted(U, np.sort(crc))}
    assert (U[subsets["eligible1882_chip"]] == elig).all() and (U[subsets["crc132"]] == np.sort(crc)).all()
if a.subsets: subsets = {k: v for k, v in subsets.items() if k in a.subsets.split(',')}; assert subsets, 'no such subset'
log(f"fold {a.fold} feat {a.feat}: train {len(tr)} val {len(va)} test {len(te)} tiles; subsets { {k: len(v) for k, v in subsets.items()} }")
t0 = time.time(); acc = accumulate(feat, tr, log); log(f"gram done {time.time()-t0:.0f}s, n_samples={acc['n']}")
summ = []
for name, idx in subsets.items():
    t1 = time.time(); mu, sd, ybar, Bs, nlive = fit_path(acc, idx)
    sv = eval_tiles(feat, va, idx, mu, sd, ybar, Bs)
    rv = mean_r_from_stats(sv)                                   # (L, T)
    val_mean = np.nanmean(rv, 1); best = select_lambda(sv)
    pd.DataFrame(dict(lam=LAMBDAS, val_mean_pearson_r=val_mean)).to_csv(out / f"{name}_val_lambda_curve.csv", index=False)
    edge = best in (0, len(LAMBDAS) - 1)
    log(f"[{name}] p={len(idx)} live={nlive} val mean r by lambda: {np.round(val_mean, 4).tolist()} -> lam*={LAMBDAS[best]}"
        + ("  WARNING: optimum at grid edge" if edge else ""))
    st = eval_tiles(feat, te, idx, mu, sd, ybar, Bs[best:best + 1])
    st = {k: v[0] for k, v in st.items()}; st["gid"] = np.asarray(te)
    np.savez(out / f"{name}_test_stats.npz", **st)
    r = mean_r_from_stats(st)
    summ.append(dict(subset=name, p=len(idx), lam=LAMBDAS[best], grid_edge=edge, val_mean_pearson_r=float(val_mean[best]),
                     test_mean_pearson_r=float(np.nanmean(r)), n_test_tiles=len(te), seconds=round(time.time() - t1)))
    log(summ[-1])
(out / "summary.json").write_text(json.dumps(summ, indent=1))
