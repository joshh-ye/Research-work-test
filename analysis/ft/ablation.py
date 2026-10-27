"""Ensemble ablation of the locked 8-member fine-tuned ensemble (EXP-020).  hg38; CPU except `test-preds`.
Selection uses VALIDATION only (chr1+chr8, 750 tiles); the test split (chr9+chr22, 359 tiles) is touched only after ABLATION_LOCK.csv is
committed to Git.  All ablation test numbers are POST HOC relative to the primary lock (TEST_LOCK.csv) and are descriptive.

Members: per-member RC-TTA predictions on the shared 896 x 128 bp grid (predict_ft.run_prediction, tta=rc), fp16.
  val : results_ft/preds/<RUN>/val_pred_rc.npy (existing)          test: results_ft/ablation/test_members/<RUN>/test_pred_rc.npy (new)
Ensembles are means of member predictions (equal weight) or a weighted mean with weights on the simplex.  Per-track Pearson r pooled over all
bins of the split (same as predict_ft.r_from_stats).  Because every candidate ensemble is a linear combination of members, per-track r of any
weighting follows exactly from the per-track member Gram matrix  G[t] = sum_bins x x^T  (x = member preds), cross terms c[t] = sum x*y,
and the target sums; selection is therefore exhaustive over subsets.

Subcommands
  select      : val Gram -> individual, best-2 / best-4 equal-weight subsets (exhaustive, by val mean r over all 183 tracks; 152 reported),
                greedy forward curve (sizes 1..8), weighted-8 (SLSQP on the simplex, maximise val mean r183; chromosome cross-fit reported),
                diversity; writes results_ft/ablation/val_selection.json + research_management/finetuning/ABLATION_LOCK.csv
  test-preds  : GPU; refuses unless ABLATION_LOCK.csv is committed and unchanged; writes per-member test predictions (one forward per member)
  test-eval   : CPU; refuses unless lock committed; per locked config -> tile stats npz + compare_ft.run_compare (same CIs as the primary)
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, itertools, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
MEMBERS = ["FT-S03", "FT-R1", "FT-R2", "FT-R3", "FT-S04", "FT-L1", "FT-L2", "FT-L3"]
ARM = {"FT-S03": "partial4", "FT-R1": "partial4", "FT-R2": "partial4", "FT-R3": "partial4",
       "FT-S04": "lora16", "FT-L1": "lora16", "FT-L2": "lora16", "FT-L3": "lora16"}
REPL = {"FT-S03": 0, "FT-R1": 1, "FT-R2": 2, "FT-R3": 3, "FT-S04": 0, "FT-L1": 1, "FT-L2": 2, "FT-L3": 3}
OUT = ROOT / "results_ft/ablation"
LOCK = ROOT / "research_management/finetuning/ABLATION_LOCK.csv"
PRIMARY_ID = "ENS:" + "+".join(MEMBERS) + "#tta=rc"


def member_pred_path(run, split):
    return (ROOT / "results_ft/preds" / run / "val_pred_rc.npy") if split == "val" else (OUT / "test_members" / run / "test_pred_rc.npy")


def load_members(split):
    from poolcache import Pooled
    P = Pooled(split)
    for r in MEMBERS:
        g = np.load(member_pred_path(r, split).parent / f"{split}_gid.npy") if split == "test" else np.load(ROOT / "results_ft/preds" / r / "val_gid.npy")
        assert (g == P.gid).all(), f"{r}: tile order != pooled order"
    X = [np.load(member_pred_path(r, split), mmap_mode="r") for r in MEMBERS]
    return X, P


def gram_stats(split, tile_mask=None):
    """per-track sums over bins (optionally a tile subset): n, s (T,M), G (T,M,M), c (T,M), sy, syy"""
    X, P = load_members(split)
    Y = np.asarray(P.tgt, dtype=np.float32)
    sel = np.arange(P.n) if tile_mask is None else np.flatnonzero(tile_mask)
    M, T = len(MEMBERS), Y.shape[-1]
    Xs = np.stack([np.asarray(x[sel], dtype=np.float32) for x in X], -1)       # (n, 896, T, M)
    Ys = Y[sel]
    Xs = Xs.reshape(-1, T, M).astype(np.float64); Ys = Ys.reshape(-1, T).astype(np.float64)
    out = dict(n=float(Xs.shape[0]), s=Xs.sum(0), sy=Ys.sum(0), syy=(Ys * Ys).sum(0),
               G=np.einsum("btm,btk->tmk", Xs, Xs, optimize=True), c=np.einsum("btm,bt->tm", Xs, Ys, optimize=True))
    return out


def r_of_weights(st, w):
    """per-track Pearson r of the ensemble sum_m w_m x_m (w (M,) or (K,M)); exact from Gram stats"""
    w = np.atleast_2d(w)
    n = st["n"]
    sx = w @ st["s"].T                                                     # (K,T)
    sxx = np.einsum("km,tmj,kj->kt", w, st["G"], w)
    sxy = w @ st["c"].T
    cov = sxy - sx * st["sy"] / n; vx = sxx - sx ** 2 / n; vy = st["syy"] - st["sy"] ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)


def subset_w(idx, M=len(MEMBERS)):
    w = np.zeros(M); w[list(idx)] = 1.0 / len(idx); return w


def member_pair_corr(st):
    """per-track Pearson correlation between member predictions (diversity), (T,M,M)"""
    n = st["n"]; s = st["s"]
    cov = st["G"] - s[:, :, None] * s[:, None, :] / n
    d = np.sqrt(np.einsum("tmm->tm", cov))
    return cov / (d[:, :, None] * d[:, None, :])


def fit_simplex(st, mask):
    from scipy.optimize import minimize
    M = len(MEMBERS)
    f = lambda w: -np.nanmean(r_of_weights(st, w)[0][mask])
    res = minimize(f, np.full(M, 1 / M), method="SLSQP", bounds=[(0, 1)] * M,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options=dict(maxiter=500, ftol=1e-12))
    w = np.clip(res.x, 0, None); return w / w.sum()


def cmd_select(a):
    from ftlib import masks
    from poolcache import Pooled
    mk = masks(); m183, m152 = mk["all183"], mk["not_identical152"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = gram_stats("val")
    P = Pooled("val")
    folds = {c: gram_stats("val", P.chrom == c) for c in ["chr1", "chr8"]}
    np.savez(OUT / "val_gram.npz", **{k: v for k, v in st.items()})
    mean = lambda w, S=st, m=m183: float(np.nanmean(r_of_weights(S, w)[0][m]))
    M = len(MEMBERS)
    res = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), selection_metric="validation mean per-track Pearson r, all 183 tracks (152 reported)",
               members=MEMBERS, arm=ARM, replicate=REPL, seconds_gram=round(time.time() - t0))
    res["individual"] = {r: dict(val_r183=mean(subset_w([i])), val_r152=mean(subset_w([i]), m=m152)) for i, r in enumerate(MEMBERS)}
    res["primary8_check"] = dict(val_r183=mean(subset_w(range(M))), val_r152=mean(subset_w(range(M)), m=m152),
                                 locked_val_r183=0.7482, locked_val_r152=0.7476)
    for k in [2, 4]:
        sc = sorted(((mean(subset_w(c)), c) for c in itertools.combinations(range(M), k)), reverse=True)
        res[f"best{k}"] = dict(members=[MEMBERS[i] for i in sc[0][1]], val_r183=sc[0][0], val_r152=mean(subset_w(sc[0][1]), m=m152),
                              runner_up=[[MEMBERS[i] for i in c] + [round(v, 5)] for v, c in sc[1:6]], n_candidates=len(sc),
                              worst=dict(members=[MEMBERS[i] for i in sc[-1][1]], val_r183=sc[-1][0]))
    cur, curve = [], []
    for _ in range(M):
        best = max((mean(subset_w(cur + [j])), j) for j in range(M) if j not in cur)
        cur.append(best[1]); curve.append(dict(size=len(cur), members=[MEMBERS[i] for i in cur], val_r183=best[0], val_r152=mean(subset_w(cur), m=m152)))
    res["greedy_curve"] = curve
    w = fit_simplex(st, m183)
    cross = {}
    for fit_c, ev_c in [("chr1", "chr8"), ("chr8", "chr1")]:
        wf = fit_simplex(folds[fit_c], m183)
        cross[f"fit_{fit_c}_eval_{ev_c}"] = dict(w=wf.round(4).tolist(), weighted=mean(wf, folds[ev_c]), equal8=mean(subset_w(range(M)), folds[ev_c]))
    res["weighted8"] = dict(w=dict(zip(MEMBERS, w.round(4).tolist())), val_r183=mean(w), val_r152=mean(w, m=m152), crossfit=cross)
    C = member_pair_corr(st)
    res["diversity"] = dict(note="per-track Pearson r between member predictions on validation, mean over 183 tracks",
                            matrix=np.nanmean(C, 0).round(4).tolist(),
                            mean_offdiag=float(np.nanmean(C[:, ~np.eye(M, dtype=bool)])),
                            within_arm_mean=float(np.nanmean([np.nanmean(C[:, i, j]) for i in range(M) for j in range(M) if i < j and ARM[MEMBERS[i]] == ARM[MEMBERS[j]]])),
                            between_arm_same_replicate_mean=float(np.nanmean([np.nanmean(C[:, i, j]) for i in range(M) for j in range(M) if i < j and ARM[MEMBERS[i]] != ARM[MEMBERS[j]] and REPL[MEMBERS[i]] == REPL[MEMBERS[j]]])),
                            between_arm_diff_replicate_mean=float(np.nanmean([np.nanmean(C[:, i, j]) for i in range(M) for j in range(M) if i < j and ARM[MEMBERS[i]] != ARM[MEMBERS[j]] and REPL[MEMBERS[i]] != REPL[MEMBERS[j]]])))
    json.dump(res, open(OUT / "val_selection.json", "w"), indent=1)
    # ---- lock rows (config ids; member order as in MEMBERS; weights fixed here)
    rows = []
    def add(cid, rule, w_, r183, r152):
        rows.append(dict(config_id=cid, weights=json.dumps(dict(zip(MEMBERS, np.round(w_, 6).tolist()))), selection_rule=rule,
                         val_r_all183=f"{r183:.4f}", val_r_152=f"{r152:.4f}", locked_at_cdt=time.strftime("%Y-%m-%dT%H:%M"), notes="post hoc vs primary lock; descriptive; tta=rc"))
    for i, r in enumerate(MEMBERS):
        add(f"ABL:single:{r}#tta=rc", "individual member (all 8 reported; FT-S03 is also TEST_LOCK row 6)", subset_w([i]), **{"r183": res["individual"][r]["val_r183"], "r152": res["individual"][r]["val_r152"]})
    for k in [2, 4]:
        b = res[f"best{k}"]; add(f"ABL:best{k}:" + "+".join(b["members"]) + "#tta=rc", f"best equal-weight {k}-subset of the 8 by val r183 (exhaustive, {b['n_candidates']} candidates)",
                                subset_w([MEMBERS.index(x) for x in b["members"]]), b["val_r183"], b["val_r152"])
    for c in curve:
        if c["size"] in (1, 2):                          # size 1 = a single row, size 2 = best2 (identical); greedy4 differs from best4 and is kept (RV-FT-04 #4)
            continue
        add(f"ABL:greedy{c['size']}:" + "+".join(c["members"]) + "#tta=rc", "greedy forward selection on val r183, equal weight", subset_w([MEMBERS.index(x) for x in c["members"]]), c["val_r183"], c["val_r152"])
    add("ABL:weighted8#tta=rc", "8 members, weights on the simplex maximising val r183 (SLSQP)", w, res["weighted8"]["val_r183"], res["weighted8"]["val_r152"])
    with open(LOCK, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(json.dumps({k: v for k, v in res.items() if k not in ("diversity",)}, indent=1)[:6000])
    print("diversity", {k: v for k, v in res["diversity"].items() if k != "matrix"})
    print(f"wrote {LOCK} ({len(rows)} rows)")


def check_lock_committed():
    g = lambda *x: subprocess.run(["git", "-C", str(ROOT)] + list(x), capture_output=True, text=True)
    rel = str(LOCK.relative_to(ROOT))
    if g("ls-files", "--error-unmatch", rel).returncode != 0 or g("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit("ABLATION_LOCK.csv is not committed / has local changes; refusing to touch test")
    return g("rev-parse", "HEAD").stdout.strip()


def cmd_test_preds(a):
    commit = check_lock_committed()
    import predict_ft as PF
    for r in MEMBERS:
        p = member_pred_path(r, "test")
        if p.exists():
            print(f"{p} exists; skipping"); continue
        p.parent.mkdir(parents=True, exist_ok=True)
        st, _, metas, secs = PF.run_prediction([r], "test", "rc", p, None, a.max_gb)
        np.save(p.parent / "test_gid.npy", st["gid"]); np.savez(p.parent / "test_stats_rc.npz", **st)
        json.dump(dict(run=r, tta="rc", seconds=round(secs), n_tiles=len(st["gid"]), forward_passes_per_tile=2, git_commit=commit,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(p.parent / "info.json", "w"), indent=1)
        print(f"{r}: {secs:.0f}s", flush=True)


def tile_stats_from(Xw, Y):
    P = Xw.astype(np.float64); Yd = Y.astype(np.float64)
    return dict(sx=P.sum(1), sy=Yd.sum(1), sxy=(P * Yd).sum(1), sx2=(P * P).sum(1), sy2=(Yd * Yd).sum(1), sse=((P - Yd) ** 2).sum(1))


def cmd_test_eval(a):
    commit = check_lock_committed()
    import compare_ft as CF
    X, P = load_members("test")
    Y = np.asarray(P.tgt, dtype=np.float32)
    rows = list(csv.DictReader(open(LOCK)))
    if a.only:
        rows = [r for r in rows if r["config_id"] in a.only.split(",")]
    for row in rows:
        od = OUT / "test_eval" / row["config_id"].replace(":", "_").replace("/", "_")
        if (od / "test_stats.npz").exists():
            print(f"{od} exists; skipping (one evaluation per config)"); continue
        od.mkdir(parents=True, exist_ok=True)
        w = json.loads(row["weights"]); wv = np.array([w[m] for m in MEMBERS])
        st = {k: np.zeros((P.n, 183)) for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
        for i0 in range(0, P.n, 40):
            sl = slice(i0, min(P.n, i0 + 40))
            E = sum(wv[j] * np.asarray(X[j][sl], dtype=np.float32) for j in range(len(MEMBERS)) if wv[j] > 0)
            for k, v in tile_stats_from(E, Y[sl]).items():
                st[k][sl] = v
        st["n"] = np.full((P.n, 183), 896.0); st["gid"] = P.gid
        np.savez(od / "test_stats.npz", **st)
        json.dump(dict(config_id=row["config_id"], weights=w, lock_commit=commit, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       posthoc=True), open(od / "run_info.json", "w"), indent=1)
        CF.run_compare(od / "test_stats.npz", od, a.n_boot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    t = sub.add_parser("test-preds"); t.add_argument("--max-gb", type=float, default=20.0)
    e = sub.add_parser("test-eval"); e.add_argument("--n-boot", type=int, default=10000); e.add_argument("--only", default=None)
    a = ap.parse_args()
    {"select": cmd_select, "test-preds": cmd_test_preds, "test-eval": cmd_test_eval}[a.cmd](a)
