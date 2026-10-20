"""Train + evaluate one Borzoi-TL head variant on the trunk cache. See headlib.py for the recipe.
--split-name: orig (val chr1,8; test chr9,22) or fold names defined in folds.py.  --exclude-tracks: E8 (list of track idx
dropped from the head/loss). Outputs: OUT/test_stats.npz (per-tile sufficient stats, shared128), val_stats.npz,
metrics.json, history.json, head_best.pt."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from headlib import *
from folds import FOLDS, split_gids

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split-name", default="orig"); ap.add_argument("--rc", action="store_true", help="RC augmentation (needs emb_rc; NOT in the slim cache)")
ap.add_argument("--exclude-tracks", default="", help="comma-separated track idx dropped (E8)")
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--eval-ckpt", default=None,
    help="skip training; evaluate this original head checkpoint (183 outputs) on the cache")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
logf = open(out / "train.log", "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
tr, va, te = split_gids(a.split_name)
cols = None
if a.exclude_tracks:
    ex = [int(i) for i in a.exclude_tracks.split(",")]; cols = [i for i in range(N_TGT) if i not in ex]
log(f"split {a.split_name}: train {len(tr)} val {len(va)} test {len(te)} tiles; seed {a.seed}; rc={a.rc}; n_out {N_TGT if cols is None else len(cols)}")
if a.eval_ckpt:
    head = make_head(N_TGT).to(DEV); head.load_state_dict(torch.load(a.eval_ckpt, map_location=DEV)); head.eval(); hist = None
else:
    head, hist = train_head(tr, va, out / "ckpt", a.seed, cols=cols, rc_prob=0.5 if a.rc else 0.0, epochs=a.epochs, log=log)
    torch.save(head.state_dict(), out / "head_best.pt")
res = {}
for nm, g in [("val", va), ("test", te)]:
    st, loss = predict_tiles(head, g, cols)
    st["gid"] = np.asarray(g); np.savez(out / f"{nm}_stats.npz", **st)
    r = mean_r(st); res[nm] = dict(mean_pearson_r=float(np.nanmean(r)), loss=loss, n_tiles=len(g), n_tracks=int(r.size))
    log(nm, res[nm])
res.update(seed=a.seed, split=a.split_name, rc=a.rc, best_epoch=None if hist is None else hist["best_epoch"],
           columns=cols)
(out / "metrics.json").write_text(json.dumps(res, indent=1))
