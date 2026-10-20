"""E8 held-out-vocabulary generalisation, labelled throughout as VALIDATION-SELECTED SAME-TF TRANSFER (4 tracks use a fallback: validation-selected any-TL-track). Head trained WITHOUT the 17 tracks of cell lines absent from the Borzoi/Enformer
vocabularies (GP5d, LS174T, LS180, DLD-1; exposure_audit/track_exposure.tsv). Each held-out track is predicted by the output of
a trained TL track chosen by a rule fixed in advance using VALIDATION chromosomes (chr1, chr8) only:
  (1) candidates = trained tracks with the same TF (all in other cell lines); pick highest validation Pearson r vs the
      held-out track's target; (2) if no same-TF trained track exists, pick the highest-validation-r track among all 166.
Comparator: pretrained Borzoi non-oracle validation-selected mapping (04_pretrained_valsel.py, same tiles/grid/selection data).
Leaky reference: original TL head (trained on all 183 incl. the held-out tracks).
Inputs: e8 head from 02_train_head.py (--exclude-tracks), orig-fold pretrained valsel stats. Pearson r primary."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from headlib import *
from folds import split_gids
from cmp_lib import *
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--head-dir", default=str(ROOT / "results_phase2/e8_generalisation/head_seed1"))
ap.add_argument("--pre-dir", default=str(ROOT / "results_phase2/e4_ridge/pretrained_valsel_orig"))
ap.add_argument("--out", default=str(ROOT / "results_phase2/e8_generalisation"))
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
exp = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
HELD = exp[exp.cell_line.isin(["GP5d", "LS174T", "LS180", "DLD-1"])].track_idx.values; assert len(HELD) == 17
cols = [i for i in range(N_TGT) if i not in set(HELD)]
assert json.loads((Path(a.head_dir) / "metrics.json").read_text())["columns"] == cols, "head was not trained on the expected 166 tracks"
tr, va, te = split_gids("orig"); tt = tile_table().set_index("gid")
head = make_head(len(cols)).to(DEV); head.load_state_dict(torch.load(Path(a.head_dir) / "head_best.pt", map_location=DEV)); head.eval()


@torch.no_grad()
def cross(gids):
    S = {k: [] for k in ["sH", "sH2", "sP", "sP2", "sHP"]}
    for (g, _, _), e, t in prefetch([(int(g), False, 0) for g in gids]):
        p = head(torch.from_numpy(e).to(DEV).float()); P = pool_grid_t(p.double())
        H = pool_grid_t(torch.from_numpy(t[:, HELD]).to(DEV).double())
        S["sH"].append(H.sum(0)); S["sH2"].append((H * H).sum(0)); S["sP"].append(P.sum(0)); S["sP2"].append((P * P).sum(0)); S["sHP"].append(H.T @ P)
    return {k: torch.stack(v).cpu().numpy() for k, v in S.items()}

from ridgelib import pool_grid_t
Sv, St = cross(va), cross(te); nv, nt = len(va) * G_BINS, len(te) * G_BINS


def pooled_r(S, n):
    sH, sH2, sP, sP2, sHP = [S[k].sum(0) for k in ["sH", "sH2", "sP", "sP2", "sHP"]]
    cov = sHP - sH[:, None] * sP[None] / n
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt((sH2 - sH ** 2 / n)[:, None] * (sP2 - sP ** 2 / n)[None])
    return np.where(np.isfinite(r), r, -np.inf), cov, sP2 - sP ** 2 / n


rv, covv, vpv = pooled_r(Sv, nv)
tf_cols = exp.TF.values[cols]
rows = []; sel = np.zeros(17, int); rule = []
for h, tr_i in enumerate(HELD):
    cand = np.where(tf_cols == exp.TF.values[tr_i])[0]
    if len(cand): j = cand[np.argmax(rv[h, cand])]; rule.append("same_TF")
    else: j = int(np.argmax(rv[h])); rule.append("fallback_any_TL_track")
    sel[h] = j
c = covv[np.arange(17), sel] / vpv[sel]; a0 = Sv["sH"].sum(0) / nv - c * Sv["sP"].sum(0)[sel] / nv
n = np.full((len(te), 17), float(G_BINS)); ar = np.arange(17)
sT, sT2, sP, sP2, sTP = St["sH"], St["sH2"], St["sP"][:, sel], St["sP2"][:, sel], St["sHP"][:, ar, sel]
TLx = dict(n=n, sx=sP, sy=sT, sxy=sTP, sx2=sP2, sy2=sT2,
           sse=sT2 - 2 * a0 * sT - 2 * c * sTP + n * a0 ** 2 + 2 * a0 * c * sP + c ** 2 * sP2)
# pretrained valsel + leaky TL reference restricted to the held-out tracks
pre = load_stats(Path(a.pre_dir) / "valsel_test_stats.npz"); assert (pre["gid"] == np.asarray(te)).all()
PRE = {k: pre[k][:, HELD] for k in KEYS}
tlq = np.load(ROOT / "reports/qing_deliverables/evaluation_evidence/tl_per_tile_sufficient_stats_shared128.npz")
LEAK = {k: tlq[k][:, HELD] for k in KEYS}
assert np.allclose(LEAK["sy"], TLx["sy"], rtol=1e-4, atol=1e-3), "target alignment mismatch between cache and Qing TL stats"
tiles = tt.loc[te].reset_index()
w = block_w(tiles, a.n_boot)
M = {"valsel_sameTF_TL_transfer": TLx, "pretrained_valsel_nonoracle": PRE, "TL_head_saw_track_LEAKY_reference": LEAK}
per = metrics_pt = {k: metrics(S, w[:1]) for k, S in M.items()}
df = pd.DataFrame(dict(track_idx=HELD, TF=exp.TF.values[HELD], cell_line=exp.cell_line.values[HELD],
                       Borzoi_class=exp.Borzoi_class.values[HELD], Enformer_class=exp.Enformer_class.values[HELD], rule=rule,
                       selected_TL_track_idx=np.asarray(cols)[sel], selected_TL_TF=tf_cols[sel], selected_TL_cell_line=exp.cell_line.values[cols][sel],
                       val_r_selected=rv[ar, sel]))
for k in M:
    df[f"r_{k}"] = per[k]["pearson_r"][0]; df[f"R2_{k}"] = per[k]["prediction_R2"][0]
df.to_csv(out / "e8_per_track.csv", index=False)
agg = []
for sub, m in [("all17", np.ones(17, bool)), ("same_TF_rule_13", np.array([r == "same_TF" for r in rule])),
               ("fallback_4", np.array([r != "same_TF" for r in rule]))]:
    for k in M:
        pt, lo, hi = summarize(M[k], w, m); agg.append(dict(subset=sub, n_tracks=int(m.sum()), model=k, mean_pearson_r=pt, ci_lo=lo, ci_hi=hi, ci_type="tile-block bootstrap (10-tile blocks), tracks fixed"))
    for k in ["valsel_sameTF_TL_transfer"]:
        pt, lo, hi = paired(M[k], PRE, w, m); agg.append(dict(subset=sub, n_tracks=int(m.sum()), model="DIFF TL_transfer - pretrained_valsel", mean_pearson_r=pt, ci_lo=lo, ci_hi=hi, ci_type="tile-block bootstrap, paired"))
pd.DataFrame(agg).to_csv(out / "e8_aggregate.csv", index=False)
# descriptive cell-line cluster table (4 clusters only)
cl = df.groupby("cell_line").agg(n=("track_idx", "size"), r_TL=("r_valsel_sameTF_TL_transfer", "mean"), r_pre=("r_pretrained_valsel_nonoracle", "mean")).reset_index()
cl["diff"] = cl.r_TL - cl.r_pre; cl.to_csv(out / "e8_cellline_clusters.csv", index=False)
p, kind = signflip_p(cl["diff"].values)
lines = ["# E8 held-out-vocabulary generalisation: validation-selected same-TF transfer (SUMMARY, auto-generated by 05_e8.py)", "",
         f"Head: {a.head_dir} (trained on 166 tracks; 17 tracks of GP5d/LS174T/LS180/DLD-1 excluded). Test = chr9+chr22 (359 tiles), shared 896x128 bp grid, hg38. Pearson r.",
         "", "## Aggregate (mean Pearson r over tracks)", "", md(pd.DataFrame(agg)), "",
         "## Cell-line clusters (descriptive; 4 clusters)", "", md(cl),
         f"\nSign-flip test over {len(cl)} clusters: p={p:.3f} ({kind}); minimum attainable two-sided p with 4 clusters is 0.125, so this is descriptive only.", "",
         "## Per-track", "", md(df), "",
         "Caveats: only 4 cell-line clusters; same-TF candidate tracks (LoVo/HCT-116/Caco-2) are themselves TL training tracks, and 'TL_head_saw_track_LEAKY' is shown only as an upper reference. Pretrained comparator uses 1,882 ChIP-TF outputs with validation selection (non-oracle). Chromosomes chr9/22 remain largely inside Borzoi/Enformer training folds (exposure_audit), so this tests cell-line/vocabulary transfer, not sequence-held-out generalisation. Tile-block bootstrap does not capture track/cell-line variance (hence cluster table)."]
(out / "SUMMARY.md").write_text("\n".join(lines)); print("\n".join(lines[:12]))
