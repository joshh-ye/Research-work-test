"""Diff clean-room CPU re-run outputs (scripts/reproduce_cpu.sh) against the recorded outputs, with tolerances.

Usage: ./torch-venv/bin/python scripts/compare_outputs.py --scratch DIR [--skip-e2] [--with-provenance]
Part A: every numeric cell of each listed table/JSON is compared (rtol/atol below); non-numeric cells must match exactly.
Part B: named key numbers (those quoted in PROJECT_STATE / registry) vs recorded value AND vs the rounded literal quoted in the state files.
Exit status 1 if any check fails.  Same seeds => expected to be (near-)bit-identical; tolerances only absorb BLAS/platform float noise.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yejz1/experimental_research")
RP2 = "results_phase2"
RTOL, ATOL = 1e-6, 1e-9           # same-seed numerical reproduction
fails, rows = [], []


def cmp_table(rel, label=None, sep=None, rtol=RTOL, atol=ATOL, note="", ignore_cols=()):
    a, b = REPO / rel, None
    b = SCR / rel
    label = label or rel
    if not b.exists():
        fails.append(label); rows.append((label, "MISSING in scratch", "", "", "FAIL")); return
    sep = sep or ("\t" if rel.endswith(".tsv") else ",")
    if a.stat().st_size <= 1 or b.stat().st_size <= 1:     # empty table (e.g. no rows for a family)
        ok = a.stat().st_size == b.stat().st_size
        if not ok: fails.append(label)
        rows.append((label, "empty table", "", "", "PASS" if ok else "FAIL")); return
    ra, rb = pd.read_csv(a, sep=sep, dtype=object), pd.read_csv(b, sep=sep, dtype=object)
    ra, rb = ra.drop(columns=[c for c in ignore_cols if c in ra.columns]), rb.drop(columns=[c for c in ignore_cols if c in rb.columns])  # e.g. figure CSV `source` = absolute input path
    if ra.shape != rb.shape or list(ra.columns) != list(rb.columns):
        fails.append(label); rows.append((label, f"shape/cols differ {ra.shape} vs {rb.shape}", "", "", "FAIL")); return
    n = nfail = 0; mx = 0.0
    for c in ra.columns:
        xa, xb = pd.to_numeric(ra[c], errors="coerce"), pd.to_numeric(rb[c], errors="coerce")
        isnum = xa.notna().any() or xb.notna().any()
        if isnum:
            both_nan = xa.isna() & xb.isna()
            d = (xa - xb).abs(); tol = atol + rtol * xb.abs()
            bad = ~both_nan & ~(d <= tol)
            n += int((~both_nan).sum()); nfail += int(bad.sum()); mx = max(mx, float(np.nanmax(d.values)) if d.notna().any() else 0.0)
        else:
            diff = (ra[c].fillna("NA") != rb[c].fillna("NA")); n += len(ra); nfail += int(diff.sum())
    ok = nfail == 0
    if not ok: fails.append(label)
    rows.append((label, f"{n} cells", f"max|diff|={mx:.2e}", f"fail={nfail}" + (f" ({note})" if note and not ok else ""), "PASS" if ok else "FAIL"))


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    else: yield p, o


def cmp_json(rel, skip=()):
    a, b = json.load(open(REPO / rel)), json.load(open(SCR / rel))
    fa, fb = dict(flat(a)), dict(flat(b)); n = nfail = 0; mx = 0.0
    for k, v in fa.items():
        if any(s in k for s in skip): continue
        n += 1; w = fb.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(w, (int, float)):
            d = abs(v - w); mx = max(mx, d); nfail += int(not d <= ATOL + RTOL * abs(w))
        else: nfail += int(v != w)
    nfail += len(set(fb) - set(fa)); ok = nfail == 0
    if not ok: fails.append(rel)
    rows.append((rel, f"{n} values", f"max|diff|={mx:.2e}", f"fail={nfail}", "PASS" if ok else "FAIL"))


def cmp_npz(rel, keys=None):
    za, zb = np.load(REPO / rel, allow_pickle=True), np.load(SCR / rel, allow_pickle=True); n = nfail = 0; mx = 0.0
    for k in (keys or za.files):
        x, y = za[k], zb[k]
        if x.dtype.kind in "fiu": d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0; mx = max(mx, d); nfail += int(not np.allclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True))
        else: nfail += int(not (x == y).all())
        n += int(x.size)
    ok = nfail == 0
    if not ok: fails.append(rel)
    rows.append((rel, f"{len(keys or za.files)} arrays/{n} elems", f"max|diff|={mx:.2e}", f"fail={nfail}", "PASS" if ok else "FAIL"))


def val(root, rel, sep, query, col):
    df = pd.read_csv(root / rel, sep=sep)
    for k, v in query.items(): df = df[df[k] == v]
    assert len(df) == 1, (rel, query, len(df)); return float(df.iloc[0][col])


KEYS = []   # (name, file, sep, query, col, literal quoted in state files, literal tol)
def key(name, rel, query, col, lit, tol, sep=None): KEYS.append((name, rel, sep or ("\t" if rel.endswith(".tsv") else ","), query, col, lit, tol))

ES = f"{RP2}/exposure_stratified/exposure_stratified_stats.csv"
base = dict(tile_set="all_359_tiles", mapping="oracle_test", metric="pearson_r")
key("TL mean per-track Pearson r (shared128, all_183)", ES, {**base, "subset": "all_183"}, "mean_TL", 0.682, 5e-4)
key("pretrained Borzoi oracle mean r", ES, {**base, "subset": "all_183"}, "mean_pretrained", 0.664, 5e-4)
key("delta r TL-oracle all_183", ES, {**base, "subset": "all_183"}, "delta_TL_minus_pretrained", 0.018, 5e-4)
key("delta r CI tracks-only lo", ES, {**base, "subset": "all_183"}, "ci_tracks_only_lo", 0.009, 5e-4)
key("delta r CI tracks-only hi", ES, {**base, "subset": "all_183"}, "ci_tracks_only_hi", 0.026, 5e-4)
key("delta r TF-cluster CI lo (PROJECT_STATE quotes +0.004; recorded 0.00457 rounds to +0.005)", ES, {**base, "subset": "all_183"}, "ci_TFcluster_lo", 0.004, 5e-4)
key("delta r TF-cluster CI hi", ES, {**base, "subset": "all_183"}, "ci_TFcluster_hi", 0.028, 5e-4)
key("delta r cell-line-cluster-only CI lo (not the PROJECT_STATE interval)", ES, {**base, "subset": "all_183"}, "ci_cellcluster_lo", -0.0197, 5e-4)
key("delta r two-way tileblock x cell-line CI lo (PROJECT_STATE -0.021)", ES, {**base, "subset": "all_183"}, "ci_twoway_tileblock_x_cell_lo", -0.021, 5e-4)
key("delta r two-way tileblock x cell-line CI hi (PROJECT_STATE +0.038)", ES, {**base, "subset": "all_183"}, "ci_twoway_tileblock_x_cell_hi", 0.038, 5e-4)
key("delta r cell-line-cluster-only CI hi", ES, {**base, "subset": "all_183"}, "ci_cellcluster_hi", 0.0367, 5e-4)
key("TL r^2 (squared Pearson, shared128)", ES, {**{**base, "metric": "pearson_r_squared"}, "subset": "all_183"}, "mean_TL", 0.4866, 5e-4)
key("TL prediction R2 (shared128)", ES, {**{**base, "metric": "prediction_R2"}, "subset": "all_183"}, "mean_TL", 0.4810, 5e-4)
key("pretrained validation-calibrated R2 (val_selected mapping)", ES, {**{**base, "metric": "prediction_R2", "mapping": "val_selected"}, "subset": "all_183"}, "mean_pretrained", 0.444, 5e-4)
key("delta r oracle primary-152 (EXP-006 +0.030)", ES, {**base, "subset": "primary_152_non_identical"}, "delta_TL_minus_pretrained", 0.030, 5e-4)
key("delta r oracle identical-31 (EXP-006 -0.043)", ES, {**base, "subset": "identical_experiment_31"}, "delta_TL_minus_pretrained", -0.043, 5e-4)
key("pretrained valsel r", ES, {**base, "mapping": "val_selected", "subset": "all_183"}, "mean_pretrained", 0.661, 5e-4)
for m, name in (("enformer", "Enformer"), ("alphagenome", "AlphaGenome")):
    MS = f"{RP2}/e7_nonoracle/{m}/model_summary.csv"
    lit_or, lit_vs = {"enformer": (0.6443, 0.6406), "alphagenome": (0.6612, 0.6569)}[m]
    key(f"{name} oracle mean r", MS, {"mapping": "oracle", "subset": "all_183"}, "mean_pearson_r", lit_or, 5e-4)
    key(f"{name} validation-selected mean r", MS, {"mapping": "valsel", "subset": "all_183"}, "mean_pearson_r", lit_vs, 5e-4)
LB = f"{RP2}/e2_validation/labels/label_counts.tsv"
key("E2 MPRA primary n_pos", LB, {"label": "mpra_primary"}, "n_pos", 134, 0)
key("E2 MPRA primary n_neg", LB, {"label": "mpra_primary"}, "n_neg", 4098, 0)
key("E2 min detectable AUROC (label-only power)", f"{RP2}/e2_validation/labels/power_statement.tsv", {"n_pos": 134}, "min_detectable_auroc", 0.5834, 5e-4)


def main():
    global SCR
    ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True); ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--with-provenance", action="store_true")
    ap.add_argument("--extended", action="store_true", help="also compare E2 final, E4/E5 summaries, E4 sensitivity part A, concordance, figure data + supp tables")
    a = ap.parse_args(); SCR = Path(a.scratch)
    # ---- Part A
    for f in ["exposure_stratified_stats.csv", "per_track_metrics_by_exposure.csv", "unexposed_cellline_per_cluster.csv",
              "signal_variance_sensitivity.csv", "tl_vs_enformer_alphagenome_by_class.csv"]:
        cmp_table(f"{RP2}/exposure_stratified/{f}")
    cmp_json(f"{RP2}/exposure_stratified/summary.json"); cmp_npz(f"{RP2}/exposure_stratified/per_tile_selected_pair_stats.npz")
    for m in ("enformer", "alphagenome"):
        for f in ("model_summary.csv", "delta_stats.csv", "per_track_metrics.csv"): cmp_table(f"{RP2}/e7_nonoracle/{m}/{f}")
        cmp_json(f"{RP2}/e7_nonoracle/{m}/summary.json"); cmp_npz(f"{RP2}/e7_nonoracle/{m}/per_tile_selected_pair_stats.npz")
    for f in ("variant_labels.tsv", "label_counts.tsv", "power_statement.tsv", "st4_resolution.tsv"): cmp_table(f"{RP2}/e2_validation/labels/{f}")
    if not a.skip_e2:
        for p in sorted(REPO.glob(f"{RP2}/e2_validation/interim_*/tables/*.tsv")): cmp_table(str(p.relative_to(REPO)))
    if a.extended:
        if not a.skip_e2:
            for p in sorted((REPO / f"{RP2}/e2_validation/final/tables").glob("*.tsv")): cmp_table(str(p.relative_to(REPO)))
        for f in ("e4_ridge/e4_comparison.csv", "e4_ridge/e4_TL_minus_ridge.csv", "e5_seeds/e5_seed_comparison.csv", "e5_seeds/e5_seed_spread.csv",
                  "e5_seeds/e5_noRC_ablation.csv", "e5_seeds/e5_seed_vs_ridge.csv", "e5_folds/e5_fold_comparison.csv", "e5_folds/e5_fold_vs_ridge.csv",
                  "e4_sensitivity/A_cluster_cis.csv"):
            if (REPO / RP2 / f).exists(): cmp_table(f"{RP2}/{f}")
        for f in ("score_distributions.tsv", "pairwise_concordance.tsv", "sign_agreement.tsv"): cmp_table(f"{RP2}/exploratory/{f}")
        for pth in sorted((REPO / "figures/data").glob("*.csv")): cmp_table(str(pth.relative_to(REPO)), ignore_cols=("source",))
        for pth in sorted((REPO / "figures/tables").glob("*.tsv")): cmp_table(str(pth.relative_to(REPO)), ignore_cols=("source",))
        for pth in sorted((REPO / "figures/out").glob("*.png")):   # rendered files are not bit-compared (matplotlib/PDF metadata); must exist and be non-empty
            q = SCR / pth.relative_to(REPO); ok = q.exists() and q.stat().st_size > 0
            if not ok: fails.append(str(pth.relative_to(REPO)))
            rows.append((str(pth.relative_to(REPO)), "exists/non-empty only", "", "", "PASS" if ok else "FAIL"))
    if a.with_provenance:
        for f in ("allele_matched_variants.tsv", "unmatched_or_excluded.tsv", "exploratory_all_rows.tsv"): cmp_table(f"variant_provenance/{f}")
    print(f"{'file':<95}{'size':<28}{'diff':<22}{'detail':<12}status")
    for r in rows: print(f"{r[0]:<95}{r[1]:<28}{r[2]:<22}{r[3]:<12}{r[4]}")
    # ---- Part B
    print("\nKey numbers: scratch vs recorded (tol rtol 1e-6) and vs the rounded value quoted in state files")
    print(f"{'number':<70}{'scratch':>12}{'recorded':>12}{'quoted':>9}{'tol':>8}  status")
    for name, rel, sep, q, col, lit, tol in KEYS:
        try:
            s, r = val(SCR, rel, sep, q, col), val(REPO, rel, sep, q, col)
        except Exception as e:
            fails.append(name); print(f"{name:<70} ERROR {e}"); continue
        repro = abs(s - r) <= ATOL + RTOL * abs(r); quoted = abs(s - lit) <= tol
        # known documentation-only mismatch (PROJECT_STATE truncates 0.00457 to +0.004): reproduction is exact, so it is reported, not failed
        doc_only = repro and not quoted and "PROJECT_STATE quotes +0.004" in name
        ok = repro and quoted
        if not ok and not doc_only: fails.append(name)
        print(f"{name:<70}{s:>12.5f}{r:>12.5f}{lit:>9.4f}{tol:>8.0e}  {'PASS' if ok else ('PASS (reproduced exactly; quoted literal truncated, doc-only)' if doc_only else 'FAIL')}")
    print(f"\nTOTAL: {len(rows)} file checks + {len(KEYS)} key numbers; failures: {len(fails)}")
    for f in fails: print("  FAIL:", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
