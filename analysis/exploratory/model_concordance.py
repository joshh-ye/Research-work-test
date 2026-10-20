"""Label-free model concordance on the EXPLORATORY set (all 10,148 rows). No validation label is read or joined.

Inputs : variant_provenance/exploratory_all_rows.tsv (key 'chr:pos:ref:alt', chen_region = locus cluster)
         variant_provenance/allele_matched_variants.tsv (ONLY column S6_CCV_input_key is read -> 5,729-row allele-matched subset)
         finalised summary.tsv of Enformer / Borzoi (never partial chunks) via analysis/e2_validation/scores.py loaders
         AlphaGenome per-track parquet cache rebuilt into results_phase2/exploratory/cache/ (ag-venv; does NOT read results_phase2/e2_validation)
Outputs: results_phase2/exploratory/{score_distributions,pairwise_concordance,sign_agreement}.tsv, README.md is hand-written from these.

Steps:
  ./ag-venv/bin/python analysis/exploratory/model_concordance.py --build-ag-cache      (CPU, parquet scan)
  ./torch-venv/bin/python analysis/exploratory/model_concordance.py [--models enformer borzoi_pretrained borzoi_tl ...]
Spearman = rank correlation (not Pearson r / R^2). Bootstrap: resample chen_region clusters with replacement, re-rank each resample.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis" / "e2_validation"))
import scores as S  # noqa: E402  (score loaders only; no label code is imported)

OUT = ROOT / "results_phase2" / "exploratory"
CACHE = OUT / "cache"
EXPL = ROOT / "variant_provenance" / "exploratory_all_rows.tsv"
MATCHED = ROOT / "variant_provenance" / "allele_matched_variants.tsv"
KEY = "chr:pos:ref:alt"
SEED, NBOOT = 20261020, 2000
N_ALL, N_MATCHED = 10148, 5729
AVAILABLE = ["alphagenome_quantile", "alphagenome_raw", "enformer", "borzoi_pretrained", "borzoi_tl", "borzoi_tl_all183"]
DEFAULT = ["alphagenome_quantile", "alphagenome_raw", "enformer"]


def load_frames(models: list[str]) -> dict:
    out = {}
    for m in models:
        if m.startswith("alphagenome"):
            out[m] = S.load_alphagenome("quantile" if m.endswith("quantile") else "raw", CACHE)
        else:
            f = S.SCORES / S.SUMMARY_MODELS[m][0] / "summary.tsv"
            if not f.exists():
                raise FileNotFoundError(f"{m}: {f} missing (scoring not finalised); partial chunks are never used")
            out[m] = S.load_summary(m, S.SCORES)
    return out


def build_table(models: list[str]):
    ex = pd.read_csv(EXPL, sep="\t", usecols=[KEY, "chen_region"])
    assert len(ex) == N_ALL and ex[KEY].is_unique, "exploratory key not unique / wrong row count"
    matched = set(pd.read_csv(MATCHED, sep="\t", usecols=["S6_CCV_input_key"])["S6_CCV_input_key"])
    assert len(matched) == N_MATCHED and matched <= set(ex[KEY]), "matched keys not a subset of exploratory keys"
    ex["allele_matched"] = ex[KEY].isin(matched)
    ex = ex.set_index(KEY)
    for m, df in load_frames(models).items():
        assert df.index.isin(ex.index).all() and ex.index.isin(df.index).all(), f"{m}: key join incomplete"
        ex[f"{m}__primary"] = df["primary"].reindex(ex.index)
        ex[f"{m}__signed"] = df["signed"].reindex(ex.index)
    assert ex.drop(columns=["chen_region", "allele_matched"]).notna().all().all(), "NaN scores present"
    return ex


def distributions(ex: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    qs = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    for setname, sub in (("all", ex), ("allele_matched", ex[ex.allele_matched])):
        for m in models:
            for kind in ("primary", "signed"):
                x = sub[f"{m}__{kind}"].to_numpy(float)
                vc = pd.Series(x).value_counts()
                r = dict(set=setname, model=m, score=kind, n=len(x), frac_exactly_0=float((x == 0).mean()),
                         tie_fraction=1 - len(vc) / len(x),  # 1 - n_unique/n
                         n_unique=len(vc), modal_value=float(vc.index[0]), modal_value_fraction=float(vc.iloc[0] / len(x)),
                         frac_negative=float((x < 0).mean()), frac_positive=float((x > 0).mean()))
                r.update({f"q{int(q*100):03d}": float(np.quantile(x, q)) for q in qs})
                rows.append(r)
    return pd.DataFrame(rows)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def sign_agree(a, b):
    nz = (a != 0) & (b != 0)
    return float((np.sign(a[nz]) == np.sign(b[nz])).mean()) if nz.any() else np.nan, int(nz.sum())


def boot_indices(clusters: np.ndarray, nboot: int, seed: int):
    codes, uniq = pd.factorize(clusters)
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(nboot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def concordance(ex: pd.DataFrame, models: list[str]):
    pairs = list(itertools.combinations(models, 2))
    conc, sign = [], []
    for setname, sub in (("allele_matched", ex[ex.allele_matched]), ("all", ex)):
        cl = sub["chen_region"].to_numpy()
        idx_list = list(boot_indices(cl, NBOOT, SEED))
        for m1, m2 in pairs:
            for kind in ("primary", "signed"):
                a, b = sub[f"{m1}__{kind}"].to_numpy(float), sub[f"{m2}__{kind}"].to_numpy(float)
                est = spearman(a, b)
                bs = np.array([spearman(a[i], b[i]) for i in idx_list])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                conc.append(dict(set=setname, model_a=m1, model_b=m2, score=kind, n=len(a), n_clusters=int(pd.Series(cl).nunique()),
                                 spearman=est, ci95_lo=lo, ci95_hi=hi, n_boot=NBOOT, seed=SEED, n_boot_nan=int(np.isnan(bs).sum())))
            a, b = sub[f"{m1}__signed"].to_numpy(float), sub[f"{m2}__signed"].to_numpy(float)
            est, nnz = sign_agree(a, b)
            bs = np.array([sign_agree(a[i], b[i])[0] for i in idx_list])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            sign.append(dict(set=setname, model_a=m1, model_b=m2, n_rows=len(a), n_both_nonzero=nnz, sign_agreement=est,
                             ci95_lo=lo, ci95_hi=hi, n_boot=NBOOT, seed=SEED))
    return pd.DataFrame(conc), pd.DataFrame(sign)


def self_check(ex, models, conc, sign):
    assert len(ex) == N_ALL and int(ex.allele_matched.sum()) == N_MATCHED
    assert ex["chen_region"].notna().all()
    n_pairs = len(models) * (len(models) - 1) // 2
    assert len(conc) == n_pairs * 2 * 2 and len(sign) == n_pairs * 2
    assert conc["spearman"].between(-1, 1).all() and (conc["ci95_lo"] <= conc["ci95_hi"]).all()
    # primary scores are max |.| so must be >= |signed| at argmax, and non-negative
    for m in models:
        assert (ex[f"{m}__primary"] >= 0).all()
        assert np.allclose(ex[f"{m}__primary"], ex[f"{m}__signed"].abs())
    # spearman helper sanity
    x = np.arange(10.0)
    assert abs(spearman(x, x) - 1) < 1e-12 and abs(spearman(x, -x) + 1) < 1e-12
    print("[self-check] OK: rows", len(ex), "matched", int(ex.allele_matched.sum()), "models", models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT, choices=AVAILABLE)
    ap.add_argument("--build-ag-cache", action="store_true", help="ag-venv only; writes results_phase2/exploratory/cache/alphagenome_scores.tsv")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.build_ag_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        S.CACHE = CACHE
        S.build_ag_cache()
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ex = build_table(a.models)
    dist = distributions(ex, a.models)
    conc, sign = concordance(ex, a.models)
    self_check(ex, a.models, conc, sign)
    dist.to_csv(out / "score_distributions.tsv", sep="\t", index=False)
    conc.to_csv(out / "pairwise_concordance.tsv", sep="\t", index=False)
    sign.to_csv(out / "sign_agreement.tsv", sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
