"""E4 sensitivity / reporting analyses (SD-023). hg38, test chr9+chr22 (359 tiles), 183 targets, shared 896x128bp grid.
Parts (all CPU; no GPU is used):
  A  cluster CIs (tile-block, TF-cluster, cell-line-cluster, two-way) for paired deltas, EXP-006 seeds/resample conventions
  B  parity checks (cache targets vs results_full/test_targets.npy; tile order; worst pre-specified ridge)
  C  exposure-controlled ridge (per-track removal of identical-experiment / same-cell-line+same-TF Borzoi channels)
  D  stratification of test tiles by Borzoi training/held-out fold overlap
  S  write SUMMARY.md from the CSVs
Usage: 07_e4_sensitivity.py --part A|B|C|D|S [--n-boot 10000]. Outputs: results_phase2/e4_sensitivity/."""
import os, sys, re, json, argparse, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import torch
torch.set_num_threads(3)
from cmp_lib import *            # common (ROOT, CACHE, ...), estats, load_stats, metrics, block_w, exposure_masks, md, KEYS
from folds import split_gids

OUT = ROOT / "results_phase2/e4_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
E4 = ROOT / "results_phase2/e4_ridge"; EV = ROOT / "reports/qing_deliverables/evaluation_evidence"
SEED = 20261020          # = EXP-006 run_exposure_stratified.py default; TF draws SEED+1, cell-line draws SEED+8
TL_PATHS = dict(
    TL=EV / "tl_per_tile_sufficient_stats_shared128.npz",
    ridge_union1888_raw=E4 / "orig_pre_raw/union1888_test_stats.npz",
    probe_trunk1920=E4 / "orig_probe/trunk1920_test_stats.npz",
    pretrained_valsel=E4 / "pretrained_valsel_orig/valsel_test_stats.npz",
    ridge_crc132_log1p_WORST_pre=E4 / "orig_pre_log1p/crc132_test_stats.npz",
    ridge_eligible1882_raw=E4 / "orig_pre_raw/eligible1882_chip_test_stats.npz",
)


def load_models(names):
    out = {}
    for n in names:
        d = np.load(TL_PATHS[n]); out[n] = {k: np.asarray(d[k], float) for k in KEYS} | ({"gid": d["gid"]} if "gid" in d else {})
    return out


def masks_and_labels():
    masks, exp = exposure_masks()
    return {k: masks[k] for k in ["all183", "not_identical152"]}, exp.TF.values, exp.cell_line.values


def delta_ci(Sa, Sb, w, mask, tf, cl, B, metric="pearson_r"):
    """Paired delta (a - b) of mean per-track metric over `mask` tracks with the EXP-006 CI schemes.
    w: (B+1, tiles) tile weights (row 0 = point estimate). Same construction as analysis/exposure_stratified/run_exposure_stratified.py."""
    ma = metrics(Sa, w)[metric]; mb = metrics(Sb, w)[metric]
    dpt = ma[0] - mb[0]; db = ma[1:] - mb[1:]
    ok_t = mask & np.isfinite(dpt); dd = np.where(np.isfinite(dpt), dpt, 0.0); dbn = np.nan_to_num(db)
    T = len(dpt)
    row = dict(n_tracks=int(ok_t.sum()), n_TF_clusters=len(np.unique(tf[ok_t])), n_cellline_clusters=len(np.unique(cl[ok_t])),
               mean_a=float(np.nanmean(ma[0][ok_t])), mean_b=float(np.nanmean(mb[0][ok_t])), delta=float(dpt[ok_t].mean()))
    one = np.broadcast_to(ok_t.astype(float), (B, T))
    row["tileblock_lo"], row["tileblock_hi"] = pct_ci(weighted_mean(dbn, one))
    for sch, lab, sd_off in [("TF", tf, 1), ("cell", cl, 8)]:
        mult = cluster_multiplicity(lab, B, np.random.default_rng(SEED + sd_off), mask) * ok_t[None]
        okb = mult.sum(1) > 0
        row[f"{sch}cluster_lo"], row[f"{sch}cluster_hi"] = pct_ci(weighted_mean(dd, mult)[okb])
        row[f"twoway_tileblock_x_{sch}_lo"], row[f"twoway_tileblock_x_{sch}_hi"] = pct_ci(weighted_mean(dbn, mult)[okb])
    return row


# ------------------------------------------------------------------ A
def part_A(B):
    masks, tf, cl = masks_and_labels()
    M = load_models(["TL", "ridge_union1888_raw", "probe_trunk1920", "pretrained_valsel", "ridge_crc132_log1p_WORST_pre", "ridge_eligible1882_raw"])
    tl_tiles = pd.read_csv(EV / "baseline_borzoi_rerun/tiles_test.csv")
    tt = tile_table(); te = split_gids("orig")[2]; tiles = tt.set_index("gid").loc[te].reset_index()
    assert (tiles.chrom.values == tl_tiles.chrom.values).all() and (tiles.start.values == tl_tiles.start.values).all()
    for k, S in M.items():
        if "gid" in S: assert (S["gid"] == te).all(), k
    w = block_w(tiles, B)               # seed 20261020, block 10 tiles, same as EXP-006 / 06_summaries
    pairs = [("TL", "ridge_union1888_raw"), ("TL", "probe_trunk1920"), ("ridge_union1888_raw", "pretrained_valsel"),
             ("probe_trunk1920", "pretrained_valsel"), ("TL", "pretrained_valsel"), ("TL", "ridge_crc132_log1p_WORST_pre"),
             ("TL", "ridge_eligible1882_raw")]
    rows = []
    for a, b in pairs:
        for mk, m in masks.items():
            rows.append(dict(a=a, b=b, track_set=mk, **delta_ci(M[a], M[b], w, m, tf, cl, B)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "A_cluster_cis.csv", index=False)
    # gate against 06_summaries tile-block numbers
    ref = pd.read_csv(E4 / "e4_TL_minus_ridge.csv"); g = ref[(ref.ridge_model == "ridge[pre_raw:union1888]") & (ref.track_set == "all183")].iloc[0]
    r = df[(df.a == "TL") & (df.b == "ridge_union1888_raw") & (df.track_set == "all183")].iloc[0]
    print("gate", g.delta_r_TL_minus_ridge, g.lo, g.hi, "vs", r.delta, r.tileblock_lo, r.tileblock_hi)
    assert abs(g.delta_r_TL_minus_ridge - r.delta) < 5e-5 and abs(g.lo - r.tileblock_lo) < 5e-5 and abs(g.hi - r.tileblock_hi) < 5e-5
    print(df.round(4).to_string())


# ------------------------------------------------------------------ B
def part_B():
    te = split_gids("orig")[2]; rep = {}
    ref = np.load(ROOT / "results_full/test_targets.npy", mmap_mode="r")            # (359, 4096, 183) float32, original loader
    assert ref.shape == (359, 4096, 183)
    mx = 0.0; mxrel = 0.0
    for i, g in enumerate(te):
        c = central(np.load(tile_path(int(g), "tgt"), mmap_mode="r"))
        d = np.abs(np.asarray(c, np.float64) - np.asarray(ref[i], np.float64)); mx = max(mx, float(d.max()))
    rep["max_abs_diff_cache_tgt_vs_results_full_test_targets"] = mx
    rep["n_tiles_compared"] = len(te)
    tt = tile_table(); tiles = tt.set_index("gid").loc[te]
    qt = pd.read_csv(EV / "baseline_borzoi_rerun/tiles_test.csv")
    rep["test_tile_order_equals_Qing_tiles_test_csv"] = bool((tiles.chrom.values == qt.chrom.values).all() and (tiles.start.values == qt.start.values).all())
    rep["test_chromosomes"] = sorted(set(tiles.chrom))
    M = load_models(["TL", "ridge_union1888_raw", "probe_trunk1920", "pretrained_valsel"])
    rep["stats_gid_equal_test_gids"] = {k: bool((S["gid"] == te).all()) for k, S in M.items() if "gid" in S}
    rep["n_per_tile_per_track_equal_896"] = {k: bool((S["n"] == 896).all()) for k, S in M.items()}
    # same data pooled per track: TL sy (target sums) equals ridge sy (same targets, same tiles)
    rep["target_sums_TL_vs_ridge_max_rel_diff"] = float(np.abs(M["TL"]["sy"] - M["ridge_union1888_raw"]["sy"]).max() / np.abs(M["TL"]["sy"]).max())
    rep["target_sumsq_TL_vs_ridge_max_rel_diff"] = float(np.abs(M["TL"]["sy2"] - M["ridge_union1888_raw"]["sy2"]).max() / np.abs(M["TL"]["sy2"]).max())
    # worst pre-specified ridge variant (all 7 E4 variants, by all183 test mean r from e4_comparison.csv)
    cmpdf = pd.read_csv(E4 / "e4_comparison.csv"); c = cmpdf[(cmpdf.track_set == "all183") & cmpdf.model.str.startswith("ridge[")]
    rep["ridge_variants_all183_mean_r"] = dict(zip(c.model, c.mean_pearson_r.round(4)))
    rep["worst_ridge_variant_overall"] = c.sort_values("mean_pearson_r").iloc[0].model
    rep["worst_ridge_variant_among_pretrained_output_features"] = c[c.model.str.contains("pre_")].sort_values("mean_pearson_r").iloc[0].model
    (OUT / "B_parity.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ D
def borzoi_tile_overlap():
    import gzip, collections
    F = ROOT / "external_data/folds"
    rows = [l.split("\t") for l in gzip.open(F / "borzoi_sequences_human.bed.gz", "rt")]
    lab = {"fold3": "test", "fold4": "valid"}          # Borzoi README: test=fold3, valid=fold4, rest train
    ivs = [(r[0], int(r[1]), int(r[2]), lab.get(r[3].strip(), "train")) for r in rows]
    def merge(iv):
        iv = sorted(iv); o = []
        for s, e in iv:
            if o and s <= o[-1][1]: o[-1][1] = max(o[-1][1], e)
            else: o.append([s, e])
        return o
    def build(groups):
        d = collections.defaultdict(lambda: collections.defaultdict(list))
        for c, s, e, l in ivs:
            for g, members in groups.items():
                if l in members: d[g][c].append((s, e))
        return {g: {c: merge(v) for c, v in dd.items()} for g, dd in d.items()}
    def ov(m, ch, s, e):
        t = 0
        for a, b in m.get(ch, []):
            if b <= s: continue
            if a >= e: break
            t += min(b, e) - max(a, s)
        return t
    G = build({"train": {"train"}, "heldout": {"valid", "test"}, "valid": {"valid"}, "test": {"test"}})
    tt = tile_table(); te = tt[tt.split == "test"]
    out = []
    for _, r in te.iterrows():
        row = dict(gid=int(r.gid), chrom=r.chrom, start=int(r.start), end=int(r.end))
        for reg, (off, w) in {"centre131072": ((524288 - 131072) // 2, 131072), "full524288": (0, 524288)}.items():
            a = r.start + off; b = a + w
            for g in ["train", "heldout", "valid", "test"]: row[f"{reg}_frac_{g}"] = ov(G[g], r.chrom, a, b) / w
        out.append(row)
    return pd.DataFrame(out)


def part_D(B):
    masks, tf, cl = masks_and_labels()
    tile_df = borzoi_tile_overlap(); te = split_gids("orig")[2]; assert (tile_df.gid.values == te).all()
    # primary definition: held-out = >=50% of the 131,072-bp evaluated (target) region in Borzoi valid(fold4)+test(fold3) windows
    tile_df["group_primary_centre"] = np.where(tile_df.centre131072_frac_heldout >= 0.5, "heldout", "train_or_mixed")
    tile_df["group_alt_full524288"] = np.where(tile_df.full524288_frac_heldout >= 0.5, "heldout", "train_or_mixed")
    tile_df["group_strict_train_centre"] = np.where(tile_df.centre131072_frac_train >= 0.5, "train>=50%", np.where(tile_df.centre131072_frac_heldout >= 0.5, "heldout>=50%", "neither"))
    tile_df.to_csv(OUT / "D_tile_fold_classification.csv", index=False)
    M = load_models(["TL", "ridge_union1888_raw", "probe_trunk1920", "pretrained_valsel"])
    tiles = tile_table().set_index("gid").loc[te].reset_index()
    wall = block_w(tiles, B); rows = []
    defs = {"primary_centre_heldout50": ("group_primary_centre", ["heldout", "train_or_mixed"]),
            "alt_full524288_heldout50": ("group_alt_full524288", ["heldout", "train_or_mixed"]),
            "strict_centre": ("group_strict_train_centre", ["train>=50%", "heldout>=50%", "neither"])}
    for dn, (col, lv) in defs.items():
        for g in lv:
            keep = (tile_df[col] == g).values
            if keep.sum() < 5: continue
            w = wall * keep[None].astype(float); w[0] = keep.astype(float)
            for a, b in [("TL", "ridge_union1888_raw"), ("TL", "probe_trunk1920"), ("TL", "pretrained_valsel"), ("ridge_union1888_raw", "pretrained_valsel")]:
                for mk, m in masks.items():
                    r = delta_ci(M[a], M[b], w, m, tf, cl, B)
                    rows.append(dict(definition=dn, group=g, n_tiles=int(keep.sum()), n_bp_frac_of_test=float(keep.mean()), a=a, b=b, track_set=mk,
                                     **{k: v for k, v in r.items() if k in ("n_tracks", "mean_a", "mean_b", "delta", "tileblock_lo", "tileblock_hi")}))
    df = pd.DataFrame(rows); df.to_csv(OUT / "D_fold_stratified.csv", index=False); print(df.round(4).to_string())
    print(tile_df.group_primary_centre.value_counts(), tile_df.group_alt_full524288.value_counts(), tile_df.group_strict_train_centre.value_counts())


# ------------------------------------------------------------------ C
def removal_rule():
    """For each of the 183 targets: Borzoi human-track indices to remove from the feature set.
    Rule (a) identical experiment: Borzoi vocabulary row whose GSM identifier equals the target's GSM or whose file ENCSR equals the target's ENCSR
    (exposure_audit/03_track_exposure.py logic; ENCSR for LoVo/HCT116 GSM targets via ENCODE dbxref as in track_exposure.tsv).
    Rule (b) same cell line + same TF: CHIP row in the Borzoi vocabulary with TF (P300=EP300 alias) equal to the target's TF and description matching the target's
    cell line regex (colon tissue: 'transverse colon'; primary_colon_tumor has no vocabulary match). Only channels present in the 1,888 feature set matter."""
    exp = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    v = pd.read_csv(ROOT / "borzoi_data/targets_human.csv")
    v["acc_gsm"] = np.where(v.identifier.astype(str).str.startswith("GSM"), v.identifier, None)
    v["encsr"] = v.file.str.extract(r"(ENCSR[0-9A-Z]{6})")[0]
    v["assay"] = v.description.str.split(":").str[0]; v["target"] = np.where(v.assay == "CHIP", v.description.str.split(":").str[1], None)
    CELLS = {"HCT-116": r"HCT[- ]?116", "LoVo": r"lovo", "Caco-2": r"caco-?2", "GP5d": r"gp5d", "LS180": r"ls ?180", "LS174T": r"ls ?174", "SW480": r"sw ?480",
             "RKO": r"\bRKO\b", "DLD-1": r"dld-?1\b", "HT-29": r"ht-?29", "transverse_colon_tissue": r"transverse colon"}
    norm = lambda s: {"P300": "EP300"}.get(str(s).upper(), str(s).upper())
    U = np.load(CACHE / "pre_channels.npy"); posU = {int(u): i for i, u in enumerate(U)}
    rows = []; rem = []
    for _, t in exp.iterrows():
        ident = np.zeros(len(v), bool)
        if isinstance(t.GSM, str): ident |= (v.acc_gsm == t.GSM).values
        if isinstance(t.ENCSR, str): ident |= (v.encsr == t.ENCSR).values
        same = np.zeros(len(v), bool)
        if t.cell_line in CELLS:
            same = ((v.assay == "CHIP") & (v.target.map(lambda x: norm(x) == norm(t.TF) if x is not None else False)) & v.description.str.contains(CELLS[t.cell_line], case=False, regex=True)).values
        both = ident | same
        idx_all = v.track_index.values[both]; idx_ident = v.track_index.values[ident]; idx_same = v.track_index.values[same & ~ident]
        inU = [posU[int(i)] for i in idx_all if int(i) in posU]
        rows.append(dict(track_idx=t.track_idx, TF=t.TF, cell_line=t.cell_line, Borzoi_class=t.Borzoi_class, n_removed_identical_expt_all7611=int(ident.sum()),
                         n_removed_sameCL_sameTF_extra_all7611=int((same & ~ident).sum()), n_removed_total_all7611=int(both.sum()),
                         n_identical_in_1888=int(sum(int(i) in posU for i in idx_ident)), n_removed_in_1888=len(inU)))
        rem.append(np.array(sorted(inU), int))
    return pd.DataFrame(rows), rem


def gram_path(): return OUT / "gram_pre_raw_union1888_train.pt"


def part_C(B, force=False):
    import ridgelib
    ridgelib.DEV = "cpu"
    from ridgelib import Feat, accumulate, fit_path, _stream, LAMBDAS, N_TGT
    tr, va, te = split_gids("orig")
    feat = Feat("pre_raw"); p = feat.p; assert p == 1888
    rdf, rem = removal_rule(); rdf.to_csv(OUT / "C_removal_per_track.csv", index=False)
    print(rdf.groupby("Borzoi_class")[["n_removed_in_1888", "n_removed_total_all7611"]].describe().T.round(1).to_string())
    # train-only Gram (identical to 03_ridge.py accumulate; on CPU fp64), cached for reuse
    if gram_path().exists() and not force:
        acc = torch.load(gram_path())
    else:
        t0 = time.time(); acc = accumulate(feat, tr, lambda s: print(s, time.time() - t0, flush=True)); torch.save(acc, gram_path()); print("gram", time.time() - t0)
    assert acc["n"] == len(tr) * 896
    # per-track ridge path: group tracks sharing a removal set
    groups = {}
    for t, r in enumerate(rem): groups.setdefault(tuple(r.tolist()), []).append(t)
    print(len(groups), "distinct removal sets")
    Ball = np.zeros((N_TGT, len(LAMBDAS), p)); gstat = {}
    mu_all = sd_all = ybar = None
    for gi, (rs, tracks) in enumerate(groups.items()):
        keep = np.setdiff1d(np.arange(p), np.array(rs, int))
        mu, sd, yb, Bs, nlive = fit_path(acc, keep)
        for t in tracks: Ball[t][:, keep] = Bs[:, :, t].cpu().numpy()
        if gi == 0: pass
        gstat[rs] = nlive
    mu_f, sd_f, yb_f, _, _ = fit_path(acc, np.arange(p))     # full-set mu/sd are identical column-wise to any subset's
    mu_f = mu_f.cpu().numpy(); sd_f = sd_f.cpu().numpy(); yb = yb_f.cpu().numpy()
    # validation pass: pooled per-track Pearson r for each lambda (same quantity as select_lambda)
    BV = torch.from_numpy(Ball.transpose(2, 0, 1).reshape(p, N_TGT * len(LAMBDAS)))      # (p, T*L), fp64
    acc_s = np.zeros((6, N_TGT, len(LAMBDAS))); nrow = 0
    def chunk_sums(gids):
        nonlocal acc_s, nrow
        for x, y in _stream(feat, gids):
            X = (torch.from_numpy(x).double() - torch.from_numpy(mu_f)) / torch.from_numpy(sd_f)
            P = (X @ BV).reshape(X.shape[0], N_TGT, len(LAMBDAS)) + torch.from_numpy(yb)[None, :, None]
            Y = torch.from_numpy(y).double()[:, :, None]
            yield P, Y
    sx = sy = sxy = sx2 = sy2 = 0.0; n = 0
    t0 = time.time()
    for P, Y in chunk_sums(va):
        n += P.shape[0]; sx = sx + P.sum(0); sx2 = sx2 + (P * P).sum(0); sy = sy + Y.sum(0); sy2 = sy2 + (Y * Y).sum(0); sxy = sxy + (P * Y).sum(0)
    print("val pass", time.time() - t0)
    rv = pearson_from_sums(n, sx.numpy(), sy.expand(-1, len(LAMBDAS)).numpy(), sxy.numpy(), sx2.numpy(), sy2.expand(-1, len(LAMBDAS)).numpy())   # (T, L)
    best = np.nanargmax(np.where(np.isfinite(rv), rv, -np.inf), 1)                  # per-track lambda on validation r only
    # NOTE: the primary E4 ridge picks ONE lambda for all tracks (max mean val r). Per-track lambda here would differ; record both and use global.
    glob = int(np.argmax(np.nanmean(rv, 0)))
    lam_idx = np.full(N_TGT, glob)                                                  # same rule as 03_ridge.py: one lambda, mean val r over tracks
    pd.DataFrame(dict(lam=LAMBDAS, val_mean_pearson_r=np.nanmean(rv, 0))).to_csv(OUT / "C_val_lambda_curve.csv", index=False)
    Bbest = torch.from_numpy(np.stack([Ball[t, glob] for t in range(N_TGT)], 1))    # (p, T)
    # test pass: per-tile sufficient statistics
    st = {k: [] for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    for x, y in _stream(feat, te):
        X = (torch.from_numpy(x).double() - torch.from_numpy(mu_f)) / torch.from_numpy(sd_f)
        P = X @ Bbest + torch.from_numpy(yb); Y = torch.from_numpy(y).double()
        for k, v_ in dict(sx=P.sum(0), sy=Y.sum(0), sxy=(P * Y).sum(0), sx2=(P * P).sum(0), sy2=(Y * Y).sum(0), sse=((P - Y) ** 2).sum(0)).items(): st[k].append(v_.numpy())
    S = {k: np.stack(v) for k, v in st.items()}; S["n"] = np.full(S["sx"].shape, 896.0); S["gid"] = np.asarray(te)
    np.savez(OUT / "C_exposure_controlled_ridge_test_stats.npz", **S)
    json.dump(dict(lam_global=LAMBDAS[glob], lam_pertrack_valmax=[LAMBDAS[i] for i in best]), open(OUT / "C_lambda.json", "w"))
    print("lambda (global, val mean r)", LAMBDAS[glob], "mean val r", float(np.nanmean(rv[:, glob])))


def part_C_report(B):
    masks, tf, cl = masks_and_labels(); M = load_models(["TL", "ridge_union1888_raw", "pretrained_valsel"])
    d = np.load(OUT / "C_exposure_controlled_ridge_test_stats.npz"); te = split_gids("orig")[2]; assert (d["gid"] == te).all()
    M["ridge_exposure_controlled"] = {k: np.asarray(d[k], float) for k in KEYS}
    tiles = tile_table().set_index("gid").loc[te].reset_index(); w = block_w(tiles, B); rows = []
    for a, b in [("TL", "ridge_exposure_controlled"), ("ridge_exposure_controlled", "ridge_union1888_raw"), ("ridge_exposure_controlled", "pretrained_valsel"), ("TL", "ridge_union1888_raw")]:
        for mk, m in masks.items(): rows.append(dict(a=a, b=b, track_set=mk, **delta_ci(M[a], M[b], w, m, tf, cl, B)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "C_comparison.csv", index=False)
    # per-track: r with and without removal, by Borzoi class
    rc = metrics(M["ridge_exposure_controlled"], w[:1])["pearson_r"][0]; r0 = metrics(M["ridge_union1888_raw"], w[:1])["pearson_r"][0]
    rdf = pd.read_csv(OUT / "C_removal_per_track.csv"); rdf["r_ridge_full"] = r0; rdf["r_ridge_exposure_controlled"] = rc; rdf["r_TL"] = metrics(M["TL"], w[:1])["pearson_r"][0]
    rdf.to_csv(OUT / "C_per_track.csv", index=False)
    print(df.round(4).to_string()); print(rdf.groupby("Borzoi_class")[["n_removed_in_1888", "r_ridge_full", "r_ridge_exposure_controlled", "r_TL"]].mean().round(4))


def part_S():
    A = pd.read_csv(OUT / "A_cluster_cis.csv"); B_ = json.loads((OUT / "B_parity.json").read_text()); C = pd.read_csv(OUT / "C_comparison.csv")
    R = pd.read_csv(OUT / "C_removal_per_track.csv"); P = pd.read_csv(OUT / "C_per_track.csv"); D = pd.read_csv(OUT / "D_fold_stratified.csv")
    T = pd.read_csv(OUT / "D_tile_fold_classification.csv"); lam = json.loads((OUT / "C_lambda.json").read_text())
    def ci(df):
        d = df.copy()
        for k in ["tileblock", "TFcluster", "cellcluster", "twoway_tileblock_x_TF", "twoway_tileblock_x_cell"]:
            d[k] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(d[f"{k}_lo"], d[f"{k}_hi"])]
        return d[["a", "b", "track_set", "mean_a", "mean_b", "delta", "tileblock", "TFcluster", "cellcluster", "twoway_tileblock_x_TF", "twoway_tileblock_x_cell"]]
    rem = R.groupby("Borzoi_class").n_removed_in_1888.agg(n_tracks="count", n_tracks_with_removal=lambda x: int((x > 0).sum()), min="min", median="median", max="max", total_removed="sum").reset_index()
    cls = P.groupby("Borzoi_class").apply(lambda d: pd.Series(dict(n=len(d), n_with_removal=int((d.n_removed_in_1888 > 0).sum()), mean_r_ridge_full=d.r_ridge_full.mean(),
                                                               mean_r_ridge_exposure_controlled=d.r_ridge_exposure_controlled.mean(), mean_r_TL=d.r_TL.mean()))).reset_index()
    Dp = D[D.definition == "primary_centre_heldout50"]
    L = ["# E4 sensitivity / reporting analyses (SD-023) - generated by analysis/priorityD/07_e4_sensitivity.py --part S", "",
         "All analyses are pre-declared sensitivity/reporting analyses (SD-023); they do not change the pre-registered primary E4 results (SD-019). hg38; test = chr9+chr22 (359 tiles); 183 targets; shared 896 x 128 bp grid; metric = mean per-track Pearson r (NOT r^2, NOT prediction R^2). Delta = a - b. CPU only; no GPU used.", "",
         "## A. Cluster CIs (EXP-006 conventions: 10,000 resamples; tile blocks of 10 tiles within chromosome, seed 20261020; TF-cluster draws seed+1; cell-line-cluster draws seed+8; two-way = tile-block replicate b paired with cluster draw b, per-track deltas weighted by cluster multiplicity)",
         "Gate: tile-block CI of TL - ridge[pre_raw:union1888] reproduces e4_TL_minus_ridge.csv to 5e-5; TL - pretrained valsel CIs reproduce results_phase2/exposure_stratified/exposure_stratified_stats.csv (all_359_tiles, val_selected, pearson_r). 2.5/97.5 percentiles.",
         "Clusters: 77 TF clusters / 12 cell-line clusters (all183); 62 / 11 (not_identical152). Cell-line clusters include `transverse_colon_tissue` and `primary_colon_tumor`. Region uncertainty spans only 2 chromosomes (SD-006).", "", md(ci(A)), "",
         "## B. Parity checks", "",
         f"1. Cache targets vs results_full/test_targets.npy (central 4096 bins, 359 tiles x 4096 x 183, float32): max abs diff = {B_['max_abs_diff_cache_tgt_vs_results_full_test_targets']}. Test tile order (chrom,start) equals Qing tiles_test.csv: {B_['test_tile_order_equals_Qing_tiles_test_csv']}; stored gids of ridge/probe/valsel stats equal split_gids('orig')[2]: {B_['stats_gid_equal_test_gids']}.",
         "2. Train-only standardisation / val-only lambda / no refit (code read, files as of 2026-10-20): `03_ridge.py` line 21 `tr, va, te = split_gids(a.fold)`; line 37 `acc = accumulate(feat, tr, log)` accumulates X'X, X'Y and sums over TRAIN tiles only (ridgelib.py `accumulate`, lines 49-58); line 40 `fit_path(acc, idx)` (ridgelib.py lines 61-76) takes feature mean/sd and the ridge path from that train-only accumulator (ridgelib.py lines 65-67: mu = sx/n, sd from the train covariance); line 41 evaluates all lambdas on VALIDATION tiles `va`; line 43 `best = select_lambda(sv)` (ridgelib.py line 104: argmax over lambdas of mean per-track pooled Pearson r on validation stats only); line 48 evaluates the TEST tiles once, with only the already-selected `Bs[best:best+1]` from the train-only fit. `accumulate` is called once, with `tr`; no train+val refit exists in 03_ridge.py or ridgelib.py. The same train-only/val-only structure is used in part C of 07_e4_sensitivity.py (Gram from `tr`, lambda from validation r, test touched once).",
         f"3. Same tiles and per-track flattening as TL: TL (Qing shared128 stats), ridge, probe and pretrained valsel are all per-tile sufficient statistics (n=896 bins per tile per track: {B_['n_per_tile_per_track_equal_896']}) on the same 359 tiles in the same order; r per track is pooled over all tiles x 896 bins (sums over tiles then `pearson_from_sums`), identical code path (estats/cmp_lib.metrics). Target sums and sums of squares in TL and ridge stats agree: max relative diff of sy = {B_['target_sums_TL_vs_ridge_max_rel_diff']}, of sy2 = {B_['target_sumsq_TL_vs_ridge_max_rel_diff']}.",
         f"4. Worst pre-specified ridge variant (all183 test mean r, e4_comparison.csv): {B_['worst_ridge_variant_overall']} (overall, among 7 E4 variants incl. the trunk probe); among pretrained-output features: {B_['worst_ridge_variant_among_pretrained_output_features']}. All variants: {B_['ridge_variants_all183_mean_r']}. Cluster CIs for TL - worst variant are in table A (a=TL, b=ridge_crc132_log1p_WORST_pre).", "",
         "## C. Exposure-controlled ridge (sensitivity)", "",
         "Rule: for each target track, remove from the 1,888 feature channels every Borzoi vocabulary track that is (a) the identical experiment (GSM identifier equal to target GSM, or file ENCSR equal to target ENCSR; ENCSR for GSM targets from ENCODE dbxref via track_exposure.tsv), or (b) a ChIP track of the same TF (P300=EP300) in the same cell line (regex as exposure_audit/03_track_exposure.py; colon tissue targets: 'transverse colon'). Targets with no match (cell line without Borzoi same-TF track, primary tumour) keep all channels. Everything else is unchanged: train-only Gram X'X/X'Y (recomputed on CPU fp64), same standardisation, same lambda grid, ONE lambda for all tracks chosen as max validation mean r " + f"(selected lambda = {lam['lam_global']:g}, same rule as 03_ridge.py; the per-track-validation-max lambdas are stored in C_lambda.json for information only and not used). Per-track removal is implemented by zero-padded coefficients on the full standardised feature vector; unit-tested equal to a ridge fit on the kept columns (tests/test_e4_sensitivity.py).",
         "Gate: the 130 tracks with no removed channel reproduce the existing ridge[pre_raw:union1888] per-track test r to max abs diff " + f"{float((P[P.n_removed_in_1888==0].r_ridge_full - P[P.n_removed_in_1888==0].r_ridge_exposure_controlled).abs().max()):.1e}.", "",
         "Channels removed per track (of 1,888; all matched channels are inside the 1,888 set, identical counts over all 7,611 are in C_removal_per_track.csv):", "", md(rem), "",
         f"Distribution of removed-channel counts over the 183 tracks: {R.n_removed_in_1888.value_counts().sort_index().to_dict()}; total {int(R.n_removed_in_1888.sum())} channel removals, {int((R.n_removed_in_1888>0).sum())} tracks affected. Per track: C_removal_per_track.csv.", "",
         "Per-class means (only classes with removals shown):", "", md(cls[cls.n_with_removal > 0]), "",
         "Comparison (a - b), tile-block / TF-cluster / cell-line-cluster / two-way CIs:", "", md(ci(C)), "",
         "Caveat: the exposure control removes only the identical experiment and same-cell-line same-TF channels. The ridge still uses channels of other TFs/assays in the same cell line, which carry cell-line-specific information that Borzoi saw in training (class same_cellline_other_target, 112 tracks); this is why removal changes mean r by < 0.001. This is NOT a fully unexposed ridge.", "",
         "## D. Stratification of test tiles by Borzoi fold overlap", "",
         "Borzoi folds: sequences_human.bed.gz (external_data/folds; README: test=fold3, valid=fold4, rest train; windows 196,608 bp). Held-out tile := >= 50% of the 131,072-bp evaluated (central target) region lies in Borzoi valid+test windows (merged union); everything else = 'train_or_mixed'. Alternative (full 524,288-bp input window) gives identical group membership (0 tiles differ). Strict three-way: train>=50% / heldout>=50% / neither (tiles where neither train nor held-out reaches 50% of the central region, i.e. substantially uncovered by Borzoi bed windows; coverage not separately tabulated). Per-tile fractions: D_tile_fold_classification.csv. Per-track r recomputed on the tile subset; tile-block bootstrap = EXP-006 block weights x subset indicator (10,000 resamples).",
         f"Group sizes (primary): heldout {int((T.group_primary_centre=='heldout').sum())} tiles ({int(((T.group_primary_centre=='heldout')&(T.chrom=='chr9')).sum())} on chr9, {int(((T.group_primary_centre=='heldout')&(T.chrom=='chr22')).sum())} on chr22), train_or_mixed {int((T.group_primary_centre=='train_or_mixed').sum())}. Strict: {T.group_strict_train_centre.value_counts().to_dict()}.",
         "Tiles are nearly all-or-nothing (held-out group median held-out fraction ~1.0; other group 0.0). The groups are confounded with chromosome/region (held-out tiles are 96% chr9), with signal level, and TL/ridge/pretrained all have lower r on the held-out group; group differences in absolute r are therefore not attributable to exposure alone. Only Borzoi folds are used (Enformer fold exposure not assessed here).", "",
         "### Primary definition", "", md(Dp.drop(columns=["definition"])), "",
         "### All definitions", "", md(D), "",
         "## Deviations / notes", "",
         "- Part C was run on CPU (Gram accumulation in fp64 on 3 threads, ~12.5 min) because no GPU use was authorised; features are the same fp16-cached pre128 arrays as the primary run.",
         "- TL - valsel rows in A reproduce EXP-006 exactly (same seeds); ridge/probe rows use the same resampling indices.",
         "- Cell-line-cluster CIs have only 12 clusters (11 for not_identical152) and are wide; TF-cluster CIs 77/62 clusters.", "",
         "## Commands", "", "```", "cd /home/yejz1/experimental_research",
         *[f"./torch-venv/bin/python analysis/priorityD/07_e4_sensitivity.py --part {x}" for x in ["A", "B", "D", "C", "Cr", "S"]],
         "./torch-venv/bin/python -m pytest -q analysis/priorityD/tests/test_e4_sensitivity.py", "```"]
    (OUT / "SUMMARY.md").write_text("\n".join(L)); print("wrote", OUT / "SUMMARY.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--force-gram", action="store_true"); a = ap.parse_args()
    if a.part == "A": part_A(a.n_boot)
    elif a.part == "B": part_B()
    elif a.part == "D": part_D(a.n_boot)
    elif a.part == "C": part_C(a.n_boot, a.force_gram)
    elif a.part == "Cr": part_C_report(a.n_boot)
    elif a.part == "S": part_S()
    else: raise SystemExit("unknown part")
