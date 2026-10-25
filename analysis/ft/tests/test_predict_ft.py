"""Tests for predict_ft.py (lock/refusal, val reproduction tolerance) and blend_val.py (blend algebra, cross-fit).  No test evaluation is run."""
import csv, subprocess, sys
from pathlib import Path
import numpy as np, pytest
ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/ft")); sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
import predict_ft as PF
import blend_val as BV

PY = ROOT / "torch-venv/bin/python"


def _lock(tmp_path, ids):
    p = tmp_path / "LOCK.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config_id", "selection_rule"]); [w.writerow([i, "x"]) for i in ids]
    return p


def test_refusal_without_lock(tmp_path):
    with pytest.raises(SystemExit, match="not in TEST_LOCK"):
        PF.check_test_allowed("FT-XX", _lock(tmp_path, ["FT-S05"]), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refusal_on_second_evaluation(tmp_path):
    lock = _lock(tmp_path, ["FT-S05", "ENS:FT-S05+FT-S06"])
    out = PF.check_test_allowed("FT-S05", lock, tmp_path / "out")                  # first time allowed
    assert out == tmp_path / "out" / "FT-S05"
    out.mkdir(parents=True); np.savez(out / "test_stats.npz", x=np.zeros(1))
    with pytest.raises(SystemExit, match="already exists"):
        PF.check_test_allowed("FT-S05", lock, tmp_path / "out")
    assert PF.check_test_allowed("ENS:FT-S05+FT-S06", lock, tmp_path / "out").name == "ENS_FT-S05+FT-S06"   # other id unaffected


def test_cli_refuses_unlocked_test_before_loading_anything():
    r = subprocess.run([str(PY), str(ROOT / "analysis/ft/predict_ft.py"), "--run", "NOT-LOCKED-XYZ", "--split", "test", "--config-id", "NOT-LOCKED-XYZ#tta=none"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "not in TEST_LOCK.csv" in r.stderr
    assert not any(PF.TEST_OUT.glob("NOT-LOCKED-XYZ*"))
    assert "loaded" not in r.stdout        # no model was loaded


def test_default_config_id():
    assert PF.ens_config_id(["FT-S05"]) == "FT-S05"
    assert PF.ens_config_id(["FT-S05", "FT-S06"]) == "ENS:FT-S05+FT-S06"


def test_val_reproduction_tolerance():
    assert PF.check_repro(0.7338, 0.73380) and PF.check_repro(0.7338 + 9e-4, 0.7338)
    assert not PF.check_repro(0.7338 + 2e-3, 0.7338) and not PF.check_repro(0.7338 - 2e-3, 0.7338)


def test_saved_val_predictions_reproduce_logged_r():
    """artifact check (skipped if predictions not yet generated): recomputed r from saved per-tile stats == logged best full-val r within 1e-3"""
    import json
    found = False
    for run in ["FT-S01", "FT-S05"]:
        f = PF.PREDS / run / "val_repro.json"
        if f.exists():
            d = json.load(open(f)); found = True
            assert d["reproduces"] and d["abs_diff"] <= PF.REPRO_TOL and d["max_abs_target_diff_vs_pooled"] < 1e-4
            st = np.load(PF.PREDS / run / "val_stats.npz")
            assert abs(float(np.nanmean(PF.r_from_stats(st))) - d["val_mean_pearson_r_all183"]) < 1e-12
    if not found:
        pytest.skip("no val predictions yet")


def test_split_order_matches_pooled_val_gid():
    p = Path("/tmp/yejz1_ft/pooled128/val_gid.npy")
    if not p.exists():
        pytest.skip("pooled copy absent")
    assert (PF.split_gids("val")[1] == np.load(p)).all()


# ---------------------------------------------------------------- blend algebra
def _toy(n=6, seed=0):
    rs = np.random.RandomState(seed)
    Y = rs.rand(n, BV.G, BV.T).astype(np.float32) * 3
    A = (Y + rs.randn(*Y.shape) * 0.5).astype(np.float32); B = (0.7 * Y + rs.randn(*Y.shape)).astype(np.float32)
    return A, B, Y


def _direct_r(P, Y):
    from estats import pearson_from_sums
    p = P.reshape(-1, BV.T).astype(np.float64); y = Y.reshape(-1, BV.T).astype(np.float64); n = np.full(BV.T, float(len(p)))
    return pearson_from_sums(n, p.sum(0), y.sum(0), (p * y).sum(0), (p * p).sum(0), (y * y).sum(0))


def test_blend_w1_w0_equal_components_and_direct_blend():
    A, B, Y = _toy(); S = BV.pair_stats(A, B, Y)
    r1 = BV.metrics_from(BV.blend_sums(S, 1.0))[0]; r0 = BV.metrics_from(BV.blend_sums(S, 0.0))[0]
    assert np.allclose(r1, _direct_r(A, Y), atol=1e-10) and np.allclose(r0, _direct_r(B, Y), atol=1e-10)
    for w in (0.3, 0.8):                                   # exact analytic blend == explicit blend
        assert np.allclose(BV.metrics_from(BV.blend_sums(S, w))[0], _direct_r(w * A + (1 - w) * B, Y), atol=1e-9)
    g1, g0 = BV.blend_sums(S, 1.0), BV.blend_sums(S, 0.0)
    assert np.allclose(g1["sse"], ((A.astype(float) - Y) ** 2).sum((0, 1)), rtol=1e-9) and np.allclose(g0["sse"], ((B.astype(float) - Y) ** 2).sum((0, 1)), rtol=1e-9)


def test_grid_selection_and_crossfit_pick_endpoint_when_component_dominates():
    A, B, Y = _toy(8); masks = dict(all183=np.ones(BV.T, bool), not_identical152=np.r_[np.ones(152, bool), np.zeros(31, bool)])
    S = BV.pair_stats(A, B, Y); grid = BV.evaluate_grid(S, masks)
    assert set(grid) == set(BV.W_GRID.tolist()) and len(BV.W_GRID) == 11
    chrom = np.array(["chr1"] * 4 + ["chr8"] * 4)
    cf = BV.crossfit(S, masks, chrom)
    assert cf["fold0"]["test_chrom"] == "chr8" and cf["fold1"]["test_chrom"] == "chr1"
    assert BV.select_w(grid) == max(grid, key=lambda w: grid[w]["r_all183"])
    # identical components: every w gives the same r
    S2 = BV.pair_stats(A, A, Y); g2 = BV.evaluate_grid(S2, masks)
    assert max(v["r_all183"] for v in g2.values()) - min(v["r_all183"] for v in g2.values()) < 1e-12


def test_parse_cand():
    assert BV.parse_cand("FT-S05") == (["FT-S05"], None)
    assert BV.parse_cand("FT-S05+E4") == (["FT-S05"], "E4")
    assert BV.parse_cand("ENS:FT-S01,FT-S05+FT-C61") == (["FT-S01", "FT-S05"], "FT-C61")
    with pytest.raises(AssertionError):
        BV.parse_cand("FT-S05+NOPE")


# ---------------------------------------------------------------- ids, binding, locked blend path
def test_parse_id():
    assert PF.parse_id("FT-S03#tta=rc") == dict(kind="ft", runs=["FT-S03"], ridge=None, w=None, tta="rc", id="FT-S03#tta=rc")
    p = PF.parse_id("ENS:FT-S03+FT-R1+FT-R2#tta=none"); assert p["runs"] == ["FT-S03", "FT-R1", "FT-R2"] and p["tta"] == "none" and p["kind"] == "ft"
    p = PF.parse_id("BLEND:ENS:FT-S05+FT-S06|FT-C61|w=0.7#tta=rc"); assert (p["kind"], p["runs"], p["ridge"], p["w"], p["tta"]) == ("blend", ["FT-S05", "FT-S06"], "FT-C61", 0.7, "rc")
    assert PF.parse_blend_id("BLEND:FT-S05|E4|w=1#tta=none") == (["FT-S05"], "E4", 1.0)
    for bad in ["FT-S05", "FT-S05#tta=maybe", "FT-S05#tta=", "#tta=rc", "BLEND:FT-S05|FT-C61#tta=rc", "BLEND:FT-S05|FT-C61|0.4#tta=rc",
                "BLEND:FT-S05|FT-C61|w=1.5#tta=rc", "ENS:A+A#tta=rc", "ENS:A++B#tta=rc"]:
        with pytest.raises(SystemExit):
            PF.parse_id(bad)


def test_binding_runs_and_tta():
    cid = "ENS:FT-S03+FT-R1#tta=rc"
    assert PF.bind_checks(cid, ["FT-S03", "FT-R1"], "rc")["tta"] == "rc"
    assert PF.bind_checks(cid, None, None)["runs"] == ["FT-S03", "FT-R1"]
    with pytest.raises(SystemExit, match="--runs"):
        PF.bind_checks(cid, ["FT-R1", "FT-S03"], None)                 # order matters
    with pytest.raises(SystemExit, match="--runs"):
        PF.bind_checks(cid, ["FT-S03"], None)
    with pytest.raises(SystemExit, match="--tta"):
        PF.bind_checks(cid, None, "none")
    with pytest.raises(SystemExit, match="--runs"):
        PF.bind_checks("BLEND:FT-S03|E4|w=0.5#tta=rc", ["FT-S04"], None)


def test_do_test_refusals_before_any_loading(tmp_path):
    cid = "ENS:FT-S05+FT-S06#tta=rc"
    lock = _lock(tmp_path, [cid]); root = tmp_path / "t"
    rows = list(csv.DictReader(open(lock)))
    with pytest.raises(SystemExit, match="--runs"):
        PF.do_test(cid, ["FT-S06", "FT-S05"], None, 20, 10, lock_path=lock, test_root=root)
    with pytest.raises(SystemExit, match="--tta"):
        PF.do_test(cid, None, "none", 20, 10, lock_path=lock, test_root=root)
    with pytest.raises(SystemExit, match="not in TEST_LOCK"):          # same members, other tta variant => other id => not locked
        PF.do_test("ENS:FT-S05+FT-S06#tta=none", None, None, 20, 10, lock_path=lock, test_root=root)
    with pytest.raises(SystemExit, match="do not declare"):             # lock row without notes declaring tta
        PF.do_test(cid, None, None, 20, 10, lock_path=lock, test_root=root, require_committed=False)
    assert not root.exists()


def test_blend_refusal_unlocked_and_second_eval_create_nothing(tmp_path):
    cid = "BLEND:FT-S05|FT-C61|w=0.4#tta=none"
    lock = _lock(tmp_path, [cid]); root = tmp_path / "t"
    with pytest.raises(SystemExit, match="not in TEST_LOCK"):          # different w => different id
        PF.do_test_blend("BLEND:FT-S05|FT-C61|w=0.5#tta=none", None, None, 20, 10, lock_path=lock, test_root=root)
    with pytest.raises(SystemExit, match="--runs"):
        PF.do_test_blend(cid, ["FT-S06"], None, 20, 10, lock_path=lock, test_root=root)
    assert not root.exists()
    out = root / PF.safe_name(cid); out.mkdir(parents=True); np.savez(out / "test_stats.npz", x=np.zeros(1))
    with pytest.raises(SystemExit, match="already exists"):
        PF.do_test_blend(cid, None, None, 20, 10, lock_path=lock, test_root=root)
    assert not (root / "_components").exists()
    assert PF.safe_name("BLEND:X#tta=rc") != PF.safe_name("BLEND:X#tta=none")   # variants never collide


def test_cli_blend_refused_when_unlocked():
    r = subprocess.run([str(PY), str(ROOT / "analysis/ft/predict_ft.py"), "--split", "test", "--config-id", "BLEND:FT-S05|FT-C61|w=0.123#tta=none"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "not in TEST_LOCK.csv" in r.stderr
    assert not any(PF.TEST_OUT.glob("BLEND_FT-S05*w=0.123*"))


def test_cli_missing_tta_suffix_and_tta_mismatch_refused():
    base = [str(PY), str(ROOT / "analysis/ft/predict_ft.py"), "--split", "test"]
    r = subprocess.run(base + ["--config-id", "FT-S05"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "#tta=" in r.stderr
    r = subprocess.run(base + ["--config-id", "FT-S05#tta=rc", "--tta", "none"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "--tta" in r.stderr


def test_lock_must_be_committed_unchanged(tmp_path, monkeypatch):
    def git(*a): subprocess.run(["git", "-C", str(tmp_path)] + list(a), check=True, capture_output=True,
                                env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})
    git("init", "-q"); lk = tmp_path / "TEST_LOCK.csv"; lk.write_text("config_id,notes\nA#tta=rc,tta=rc\n")
    monkeypatch.setattr(PF, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="not tracked"):
        PF.check_lock_committed(lk)
    git("add", "TEST_LOCK.csv"); git("commit", "-qm", "lock")
    assert len(PF.check_lock_committed(lk)) == 40
    lk.write_text("config_id,notes\nA#tta=rc,tta=rc\nB#tta=rc,tta=rc\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        PF.check_lock_committed(lk)


def test_blend_tile_stats_algebra_w0_w1_and_direct():
    A, B, Y = _toy(5, 3)
    for w, ref in [(1.0, A), (0.0, B), (0.35, 0.35 * A.astype(np.float64) + 0.65 * B.astype(np.float64))]:
        st = BV.blend_tile_stats(A, B, Y, w)
        d = ref.astype(np.float64); y = Y.astype(np.float64)
        assert np.allclose(st["sx"], d.sum(1), rtol=1e-9, atol=1e-9) and np.allclose(st["sx2"], (d * d).sum(1), rtol=1e-8)
        assert np.allclose(st["sxy"], (d * y).sum(1), rtol=1e-8) and np.allclose(st["sse"], ((d - y) ** 2).sum(1), rtol=1e-7, atol=1e-6)
        assert np.allclose(st["sy"], y.sum(1)) and (st["n"] == BV.G).all()
        g = {k: v.sum(0) for k, v in st.items()}                       # pooled == blend_sums
        S = BV.pair_stats(A, B, Y); g2 = BV.blend_sums(S, w)
        assert all(np.allclose(g[k], g2[k], rtol=1e-9, atol=1e-9) for k in g2)


def test_lock_must_declare_tta():
    PF.check_lock_tta({"config_id": "X", "notes": "primary; tta=rc"}, "rc")
    with pytest.raises(SystemExit, match="tta=none"):
        PF.check_lock_tta({"config_id": "X", "notes": "primary; tta=rc"}, "none")
    with pytest.raises(SystemExit, match="tta=rc"):
        PF.check_lock_tta({"config_id": "X", "notes": ""}, "rc")
