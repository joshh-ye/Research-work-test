"""Synthetic-data tests (known answers). No real scores or labels are used."""
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import labels as L
import run_e2
import scores as S
import stats as ST

ROOT = Path("/home/yejz1/experimental_research")


def make_regions(n_reg=30, n_per=20, n_pos=4, seed=0):
    rng = np.random.default_rng(seed)
    reg = np.repeat(np.arange(n_reg), n_per)
    y = np.zeros(len(reg), int)
    for r in range(n_reg):
        y[r * n_per + rng.choice(n_per, n_pos, replace=False)] = 1
    return reg, y, rng


def test_perfect_and_random_auroc():
    reg, y, rng = make_regions()
    assert ST.stratified_auroc(y + rng.random(len(y)) * 0.1, y, reg) == 1.0
    assert ST.stratified_auroc(-y + rng.random(len(y)) * 0.1, y, reg) == 0.0
    big_reg, big_y, rng = make_regions(200, 20, 4, 1)
    assert abs(ST.stratified_auroc(rng.random(len(big_y)), big_y, big_reg) - 0.5) < 0.03


def test_auroc_ties_and_weights():
    assert ST.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5
    # region A: auc 1 (1 pos x 1 neg, w=1); region B: auc 0 (2 pos x 2 neg, w=4) -> (1*1+4*0)/5
    s = np.array([2, 1, 0, 0, 1, 1.5, 1.4, 1.6][:8], float)
    y = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    reg = np.array(["A", "A", "B", "B", "B", "B", "B", "B"])
    t = ST.region_table(np.array([2, 1, 0.1, 0.2, 1, 2, 3, 4.0]), y, reg)
    assert t.loc["A", "auc"] == 1.0 and t.loc["B", "auc"] == 0.0 and t.loc["B", "w"] == 2 * 4
    assert ST.stratified_auroc(np.array([2, 1, 0.1, 0.2, 1, 2, 3, 4.0]), y, reg) == pytest.approx(1 / 9)
    # uninformative region (no positives) has weight 0
    t2 = ST.region_table([1, 2, 3], [0, 0, 0], ["Z"] * 3)
    assert t2["w"].iloc[0] == 0


def test_bootstrap_ci_covers_and_perfect_has_tight_ci():
    reg, y, rng = make_regions(40, 15, 3, 2)
    r = ST.p1_stratified(y + 0.01 * rng.random(len(y)), y, reg, B=500, n_perm=500)
    assert r["auroc_stratified"] == 1.0 and r["ci_lo"] == 1.0 and r["perm_p_greater"] < 0.01
    s = rng.random(len(y))
    r = ST.p1_stratified(s, y, reg, B=1000, n_perm=500)
    assert r["ci_lo"] < r["auroc_stratified"] < r["ci_hi"]


def test_permutation_p_calibration_under_null():
    ps = []
    for i in range(150):
        reg, y, rng = make_regions(12, 10, 2, 100 + i)
        s = rng.random(len(y))
        obs = ST.stratified_auroc(s, y, reg)
        ps.append(ST.permutation_p(s, y, reg, obs, n_perm=199, seed=i)[0])
    ps = np.array(ps)
    assert 0.02 <= (ps < 0.05).mean() <= 0.10          # nominal 0.05 (binomial sd ~0.018)
    assert 0.35 <= (ps < 0.5).mean() <= 0.65
    # power: informative score gives small p
    reg, y, rng = make_regions(30, 15, 3, 7)
    s = y * 1.0 + rng.normal(size=len(y))
    assert ST.permutation_p(s, y, reg, ST.stratified_auroc(s, y, reg), n_perm=499)[0] < 0.01


def test_permutation_preserves_marginals():
    reg, y, rng = make_regions(5, 10, 3, 3)
    null = ST.permutation_null(rng.random(len(y)), y, reg, n_perm=300, seed=1)
    assert abs(null.mean() - 0.5) < 0.03


def test_holm_and_bh_known_values():
    np.testing.assert_allclose(ST.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(ST.holm([0.5, 0.9]), [1.0, 1.0])
    np.testing.assert_allclose(ST.bh([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    a = ST.holm([0.001, 0.2, 0.0001])
    assert (a >= np.array([0.001, 0.2, 0.0001])).all() and a.max() <= 1


def test_orientation_keep_flip_exclude():
    assert L.orientation("A", "G", "A", "G") == "keep"
    assert L.orientation("A", "G", "G", "A") == "flip"
    assert L.orientation("A", "G", "T", "C") == "exclude"      # complement is not treated as a match
    assert L.orientation("A", "G", "A", "C") == "exclude"


def _toy_label_inputs():
    m = pd.DataFrame(dict(
        S6_CCV_input_key=["1:10:A:G", "1:20:C:T", "1:30:G:A", "1:40:A:C", "1:50:T:C", "2:60:A:G"],
        rsid=["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"], chrom="1", pos=[10, 20, 30, 40, 50, 60],
        ref=["A", "C", "G", "A", "T", "A"], alt=["G", "T", "A", "C", "C", "G"], chen_region="r1", chen_lead_snp="s1",
        palindromic=False))
    st2 = pd.DataFrame(dict(rsid=["rs1", "rs2", "rs3", "rs4", "rsX"], Chr=["1"] * 4 + ["2"], **{"bp (b38)": [10, 20, 30, 40, 60]},
                            **{"MPRA_HT29": [2, 0, 2, 1, 0], "MPRA_SW403": [0, 0, 0, 0, 0], "MPRA_HCEC-1CT": [0, 0, 0, 0, 0]}))
    st4 = pd.DataFrame(dict(rsid=["rs1", "rs3", "rsX"], REF=["A", "A", "T"], ALT=["G", "G", "G"],
                            HT29_FDR=[1e-9, 1e-9, 1e-9], HT29_logFC=[0.5, 0.4, 0.3], SW403_FDR=[0.5] * 3, SW403_logFC=[0.0] * 3,
                            HCEC_FDR=[0.5] * 3, HCEC_logFC=[0.0] * 3))
    return m, st2, st4


def test_label_builder_orientation_flags_and_position_fallback():
    m, st2, st4 = _toy_label_inputs()
    t = L.build_mpra_labels(m, st2, st4).set_index("rsid")
    assert t.loc["rs1", "mpra_primary"] == "pos" and t.loc["rs1", "orientation"] == "keep" and t.loc["rs1", "logFC_alt_vs_ref"] == 0.5
    # rs3: matched G>A, ST4 A>G -> swapped -> logFC sign flipped
    assert t.loc["rs3", "orientation"] == "flip" and t.loc["rs3", "logFC_alt_vs_ref"] == -0.4 and t.loc["rs3", "mpra_primary"] == "pos"
    assert t.loc["rs2", "mpra_primary"] == "neg"
    assert t.loc["rs4", "mpra_primary"] == "excluded" and t.loc["rs4", "exclusion_reason"] == "st2_flag1_not_in_ST4"
    assert t.loc["rs5", "mpra_primary"] == "untested"
    # rs6 not in ST2 by rsid -> position fallback finds rsX at chr2? matched chrom is '1' so key '1:60' is absent -> untested
    assert t.loc["rs6", "st2_join"] == "none"


def test_label_builder_pos_fallback_and_orientation_exclusion():
    m, st2, st4 = _toy_label_inputs()
    m.loc[5, ["chrom", "pos"]] = ["2", 60]
    st4.loc[2, ["REF", "ALT"]] = ["C", "T"]       # rsX ST4 alleles unrelated to matched A/G -> excluded
    t = L.build_mpra_labels(m, st2, st4).set_index("rsid")
    assert t.loc["rs6", "st2_join"] == "pos" and t.loc["rs6", "st4_join"] == "st2_rsid"
    assert t.loc["rs6", "orientation"] == "exclude" and t.loc["rs6", "mpra_primary"] == "excluded"
    assert t.loc["rs6", "exclusion_reason"] == "st4_allele_orientation_mismatch"


def test_eqtl_harmonise_swap():
    mk = {"1:100:A:G": 1}
    k, s = L.harmonise(pd.Series(["chr1_100_A_G_b38", "chr1_100_G_A_b38", "chr1_100_A_T_b38"]), mk)
    assert list(k) == ["1:100:A:G", "1:100:A:G", ""] and list(s) == [1, -1, 0]


def test_clogit_matches_bruteforce_and_recovers_effect():
    rng = np.random.default_rng(5)
    strata, X, y = [], [], []
    for s in range(60):
        n, k = 6, 2
        x = rng.normal(size=(n, 2))
        eta = x @ np.array([1.0, -0.5])
        subsets = list(itertools.combinations(range(n), k))
        w = np.array([np.exp(eta[list(c)].sum()) for c in subsets]); w /= w.sum()
        chosen = subsets[rng.choice(len(subsets), p=w)]
        yy = np.zeros(n, int); yy[list(chosen)] = 1
        strata += [s] * n; X.append(x); y.append(yy)
    X = np.vstack(X); y = np.concatenate(y); strata = np.array(strata)

    def nll(b):  # brute-force enumeration
        v = 0.0
        for s in np.unique(strata):
            m = strata == s; xs, ys = X[m], y[m]; k = ys.sum(); eta = xs @ b
            den = np.log(sum(np.exp(eta[list(c)].sum()) for c in itertools.combinations(range(len(ys)), k)))
            v -= eta[ys == 1].sum() - den
        return v
    from scipy.optimize import minimize
    ref = minimize(nll, np.zeros(2), method="BFGS").x
    f = ST.conditional_logit(X, y, strata)
    np.testing.assert_allclose(f["coef"], ref, atol=2e-3)
    assert abs(f["coef"][0] - 1.0) < 0.4 and abs(f["coef"][1] + 0.5) < 0.4 and f["p"][0] < 1e-4 and f["n_strata"] == 60
    lr = ST.clogit_lr_test(X, X[:, [1]], y, strata)
    assert lr["df"] == 1 and lr["p"] < 1e-4


def test_direction_test():
    reg = np.repeat(np.arange(20), 5)
    m = np.ones(100)
    r = ST.direction_test(np.ones(100), m, reg, B=500)
    assert r["frac"] == 1.0 and r["ci_lo"] == 1.0 and r["boot_p_two_sided"] < 0.01 and r["binom_p"] < 1e-20
    r = ST.direction_test(-np.ones(100), m, reg, B=500)
    assert r["frac"] == 0.0
    rng = np.random.default_rng(0)          # null: random signs -> 95% CI should cover 0.5 in most replicates
    cover = [(lambda r: r["ci_lo"] <= 0.5 <= r["ci_hi"])(ST.direction_test(rng.choice([-1.0, 1.0], 400), m.repeat(4), np.repeat(np.arange(100), 4), B=300))
             for _ in range(60)]
    assert np.mean(cover) > 0.85
    assert ST.direction_test([0.0, 1.0], [1.0, 1.0], [0, 1], B=50)["n"] == 1      # zero scores dropped


def test_paired_delta_sign():
    reg, y, rng = make_regions(40, 15, 3, 9)
    good = y + 0.3 * rng.normal(size=len(y)); bad = rng.random(len(y))
    r = ST.paired_delta(good, bad, y, reg, B=500)
    assert r["delta_auroc"] > 0.2 and r["ci_lo"] > 0
    assert ST.paired_delta(good, good, y, reg, B=200)["delta_auroc"] == 0


def test_minimum_detectable_auroc_sane():
    reg, y, rng = make_regions(40, 15, 3, 4)
    r = ST.minimum_detectable_auroc(y, reg, n_sim=5, B=100)
    assert 0.5 < r["min_detectable_auroc"] < 1.0


def test_distance_to_tss_one_based():
    tss = {"1": np.array([100, 500, 1000])}
    d = ST.distance_to_tss(["chr1", "1", "2"], [100, 520, 5], tss)
    assert d[0] == 0 and d[1] == 20 and np.isnan(d[2])


# ---------------- loaders ----------------
def test_summarize_matrix_and_log2fc():
    x = np.array([[0.1, -0.5, 0.2], [0.0, 0.0, 0.0]])
    t = S.summarize_matrix(x, np.array([10, 11, 12]))
    assert t["primary"][0] == 0.5 and t["signed"][0] == -0.5 and t["argmax_track"][0] == 11
    assert t["mean_abs"][0] == pytest.approx(0.8 / 3) and t["mean_signed"][0] == pytest.approx(-0.2 / 3)
    assert S.log2fc(np.array([0.0, 3.0]), np.array([3.0, 3.0])).tolist() == [2.0, 0.0]


def test_chunk_loader_matches_hand_computation(tmp_path):
    keys = ["1:10:A:G", "1:10:A:T", "1:20:C:T"]
    s6 = tmp_path / "s6.tsv"; s6.write_text("k\trs\n" + "\n".join(f"{k}\trs{i}" for i, k in enumerate(keys)) + "\n")
    mask = S.track_mask("borzoi_pretrained")
    T = len(mask); tracks = np.flatnonzero(mask)
    ref = np.full((2, T), 7.0, np.float32)           # 2 positions
    alt = np.full((3, T), 7.0, np.float32)
    alt[0, tracks[3]] = 15.0                         # lfc = log2(16/8) = 1 at selected track
    alt[1, tracks[5]] = 0.0                          # lfc = log2(1/8) = -3
    alt[1, 5] = 100.0                                # non-selected track must not matter (unless in mask)
    assert not mask[5]
    d = tmp_path / "borzoi" / "chunks"; d.mkdir(parents=True)
    np.savez(d / "chunk_00000.npz", row_ids=np.array([0, 1, 2]), pos_keys=np.array(["chr1:10", "chr1:20"]), pos_of_row=np.array([0, 0, 1]),
             n_pad_left=np.array([0, 0, 5]), n_pad_right=np.zeros(3, int), n_N_in_agg=np.zeros(3, int),
             ref_pretrained=ref, alt_pretrained=alt)
    t = S.load_from_chunks("borzoi_pretrained", tmp_path, s6)
    assert t.loc["1:10:A:G", "primary"] == pytest.approx(1.0) and t.loc["1:10:A:G", "signed"] == pytest.approx(1.0)
    assert t.loc["1:10:A:T", "signed"] == pytest.approx(-3.0) and t.loc["1:10:A:T", "argmax_track"] == tracks[5]
    assert t.loc["1:20:C:T", "primary"] == 0 and bool(t.loc["1:20:C:T", "excl_window"]) and not bool(t.loc["1:10:A:G", "excl_window"])
    # load_model falls back to chunks when summary.tsv is absent
    assert S.load_model("borzoi_pretrained", tmp_path).shape[0] == 3


def test_summary_loader_on_documented_format(tmp_path):
    d = tmp_path / "enformer"; d.mkdir()
    p = "pretrained__crc_dnase_tfchip__"
    pd.DataFrame({"row_id": [0, 1], "input_variant": ["1:1:A:G", "1:2:C:T"], "window_padded": [False, True], "n_N_in_agg_window": [0, 0],
                  p + "primary_max_abs_log2fc": [0.3, 0.1], p + "signed_at_argmax": [-0.3, 0.1], p + "argmax_track_pos": [4, 9],
                  p + "mean_signed_log2fc": [0.0, 0.0], p + "mean_abs_log2fc": [0.1, 0.05]}).to_csv(d / "summary.tsv", sep="\t", index=False)
    t = S.load_summary("enformer", tmp_path)
    assert t.loc["1:1:A:G", "signed"] == -0.3 and bool(t.loc["1:2:C:T", "excl_window"]) and t.index.name == "S6_CCV_input_key"


def test_summary_loader_on_real_smoke_run_columns():
    sm = ROOT / "results_phase2/variant_scores"
    if not (sm / "borzoi_smoke/summary.tsv").exists():
        pytest.skip("smoke run absent")
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "borzoi").mkdir(); shutil.copy(sm / "borzoi_smoke/summary.tsv", Path(td) / "borzoi/summary.tsv")
        for m in ("borzoi_pretrained", "borzoi_tl", "borzoi_tl_all183"):
            t = S.load_summary(m, Path(td))
            assert len(t) > 0 and (t["primary"] >= 0).all() and (t["primary"] >= t["signed"].abs() - 1e-6).all()


def test_ag_loader_flags_shifted_windows():
    c = S.CACHE
    if not (c / "alphagenome_scores.tsv").exists():
        pytest.skip("AG cache not built")
    q = S.load_alphagenome("quantile")
    assert len(q) == 10148 and int(q["excl_window"].sum()) == 242 and q.index.is_unique


# ---------------- CLI + end-to-end plumbing on synthetic data ----------------
def test_cli_refuses_without_unblind():
    r = subprocess.run([sys.executable, str(ROOT / "analysis/e2_validation/run_e2.py")], capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSING" in (r.stderr + r.stdout)


def test_run_analysis_end_to_end_synthetic(tmp_path):
    rng = np.random.default_rng(11)
    n_reg, n_per = 25, 20
    reg = np.repeat([f"region_{i}" for i in range(n_reg)], n_per)
    y = np.zeros(len(reg), int)
    for r in range(n_reg):
        y[r * n_per + rng.choice(n_per, 3, replace=False)] = 1
    keys = [f"1:{i + 1}:A:G" for i in range(len(reg))]
    lab = pd.DataFrame(dict(S6_CCV_input_key=keys, rsid=[f"rs{i}" for i in range(len(reg))], chrom="1", pos=np.arange(len(reg)) + 1,
                            chen_region=reg, chen_lead_snp=[f"sig{i // 10}" for i in range(len(reg))], palindromic=False,
                            mpra_primary=np.where(y == 1, "pos", "neg"), mpra_S1=np.where(y == 1, "pos", "neg"),
                            orientation="keep", logFC_alt_vs_ref=np.where(y == 1, rng.choice([-0.5, 0.5], len(reg)), np.nan),
                            chen_annot_count=rng.integers(0, 5, len(reg)), dist_tss=rng.integers(1, 100000, len(reg)),
                            eqtl_gtex_v10_colon_any=rng.integers(0, 2, len(reg)), eqtl_cedar_credible=0))
    for L_ in ("HT29", "SW403", "HCEC"):
        lab[f"mpra_S2_{L_}"] = lab["mpra_primary"]
    sign = np.sign(lab["logFC_alt_vs_ref"].fillna(1).to_numpy())

    def tab(strength):
        p = np.abs(rng.normal(size=len(y))) + strength * y
        return pd.DataFrame(dict(primary=p, signed=p * sign, argmax_track=0, mean_abs=p / 2, mean_signed=0.0, excl_window=rng.random(len(y)) < 0.02), index=keys)
    models = {"alphagenome_quantile": tab(2.0), "borzoi_pretrained": tab(0.0), "borzoi_tl": tab(1.0), "enformer": tab(0.0), "alphagenome_raw": tab(2.0)}
    res = run_e2.run_analysis(lab, models, tmp_path, B=200, n_perm=199)
    p1 = res["p1"].set_index("model")
    assert p1.loc["alphagenome_quantile", "auroc_stratified"] > 0.8 and p1.loc["alphagenome_quantile", "holm_p"] < 0.05
    assert 0.3 < p1.loc["enformer", "auroc_stratified"] < 0.7
    assert res["p2"].set_index("model").loc["alphagenome_quantile", "frac"] == 1.0     # signed = sign(logFC) by construction
    assert (tmp_path / "SUMMARY.md").exists() and (tmp_path / "tables/p1.tsv").exists()
    assert not res["clogit"].empty and res["framing"]["framing_B_criteria_met"].dtype == bool


def test_track_matched_columns_verified_against_targets():
    cols = S.track_matched_borzoi_cols()
    tg = pd.read_csv(ROOT / "borzoi_data/targets_human.csv")
    m = pd.read_csv(S.QING_MAP)
    tl = pd.read_csv(S.TL_DIR / "borzoi_tl_tracks.tsv", sep="\t")
    sel = m[tl["primary"].to_numpy(bool)]
    assert len(cols) == 73 and (tg.loc[cols, "track_index"].to_numpy() == cols).all()
    assert set(tg.loc[cols, "identifier"]) == set(sel["val_selected_identifier"])


def test_cellline_groups_and_chunk_lfc_full_vocabulary(tmp_path):
    assert [S._group(x) for x in ("HCT-116", "HCT116", "LoVo", "Caco-2")] == ["HCT116", "HCT116", "LoVo", "other"]
    s6 = tmp_path / "s6.tsv"; s6.write_text("k\trs\n1:10:A:G\trs0\n")
    T = 7611
    d = tmp_path / "borzoi" / "chunks"; d.mkdir(parents=True)
    np.savez(d / "chunk_00000.npz", row_ids=np.array([0]), pos_keys=np.array(["chr1:10"]), pos_of_row=np.array([0]), n_pad_left=np.zeros(1, int),
             n_pad_right=np.zeros(1, int), n_N_in_agg=np.zeros(1, int), ref_pretrained=np.full((1, T), 7.0, np.float32), alt_pretrained=np.full((1, T), 15.0, np.float32))
    x = S.lfc_columns("borzoi_pretrained", np.array([0, 7610]), tmp_path, s6)
    assert x.shape == (1, 2) and x.iloc[0, 0] == pytest.approx(1.0)


def test_p2_excludes_swapped_rows():
    reg = np.repeat(np.arange(10), 4)
    lab = pd.DataFrame(dict(S6_CCV_input_key=[f"1:{i}:A:G" for i in range(40)], chen_region=reg, orientation=["keep"] * 38 + ["flip"] * 2))
    # emulate the filter used in run_analysis
    assert (lab[lab.orientation == "keep"].shape[0]) == 38
