"""CPU tests for SD-005 variant scoring: coordinates, alleles, centering, RC, REF==ALT, chromosome ends, resume."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "borzoi_code"))

import common as C  # noqa: E402
import runner  # noqa: E402

FA = C.Fasta()
VARS = C.load_variants()


def test_variant_table_shape_and_refs_match_fasta():
    assert len(VARS) == 10148 and VARS[["chrom", "pos1"]].drop_duplicates().shape[0] == 5739
    bad = [(c, p) for c, p, r in zip(VARS.chrom, VARS.pos1, VARS.ref) if FA.base(c, p) != r]
    assert bad == []                                       # 1-based POS -> 0-based index POS-1


@pytest.mark.parametrize("geom", [C.BORZOI, C.ENFORMER])
@pytest.mark.parametrize("rsid", ["rs6983267", "rs480419"])
def test_ref_alt_substitution_index_and_centering(geom, rsid):
    r = VARS[VARS.rsid == rsid].iloc[0]
    oh, pl, pr, nN = C.ref_window_onehot(FA, r.chrom, r.pos1, r.ref, geom)
    c = geom.seq_len // 2
    p0 = r.pos1 - 1
    assert (pl, pr) == (0, 0) and oh.shape == (geom.seq_len, 4)
    # window index c corresponds to genome 0-based index p0; check 21 bases around it against the FASTA slice
    seq = "ACGT"
    got = "".join(seq[i] for i in oh[c - 10:c + 11].argmax(1))
    assert got == str(FA.f[r.chrom][p0 - 10:p0 + 11]).upper()
    assert seq[oh[c].argmax()] == r.ref
    alt = C.alt_onehot(oh, r.alt, geom, ref=r.ref)
    diff = np.flatnonzero((alt != oh).any(1))
    assert diff.tolist() == [c] and seq[alt[c].argmax()] == r.alt and alt[c].sum() == 1


def test_wrong_ref_raises():
    r = VARS.iloc[0]
    wrong = next(b for b in "ACGT" if b != r.ref)
    with pytest.raises(AssertionError):
        C.ref_window_onehot(FA, r.chrom, r.pos1, wrong, C.BORZOI)


def test_geometry_agg_slices_and_variant_bin():
    assert C.BORZOI.agg_slice() == slice(3064, 3080) and C.ENFORMER.agg_slice() == slice(446, 450)
    for g, vb in [(C.BORZOI, 3072), (C.ENFORMER, 448)]:
        # the output bin starting at window index L//2 (variant base is its first base) is the centre bin
        assert g.out_span_start + vb * g.bin_bp == g.seq_len // 2
        sl = g.agg_slice()
        assert sl.start + sl.stop == g.n_bins            # symmetric about the output centre -> RC bins map to same bins
        assert (sl.stop - sl.start) * g.bin_bp == 512


def test_onehot_matches_project_encoder_and_rc():
    from sequence_utils import one_hot_encode, reverse_complement
    s = str(FA.f["chr8"][127_400_000:127_402_000]).upper() + "NNNN"
    assert np.array_equal(C.one_hot(s), one_hot_encode(s))
    assert np.array_equal(C.rc_onehot(C.one_hot(s)), one_hot_encode(reverse_complement(s)))


def _asym_model(batch):
    """Deterministic, deliberately strand-asymmetric mock: 16 bins x 3 tracks from the central 512 bp."""
    L = batch.shape[1]
    c = L // 2
    out = []
    w = np.arange(512, dtype=np.float64) + 1
    for x in batch:
        seg = x[c - 256:c + 256].astype(np.float64)
        bins = np.stack([(seg[:, t % 4] * w).reshape(16, 32).sum(1) + t for t in range(3)], 1)
        out.append(bins)
    return np.stack(out).astype(np.float32)


def test_rc_average_is_strand_invariant():
    r = VARS[VARS.rsid == "rs6983267"].iloc[0]
    g = C.Geometry("mock", 1024, 64, 32)           # small mock geometry, 512 bp window = 16 bins
    seq, _, _ = FA.window(r.chrom, r.pos1, g.seq_len)
    oh = C.one_hot(seq)
    a = C.rc_average(_asym_model, oh, slice(None))
    b = C.rc_average(_asym_model, C.rc_onehot(oh), slice(None))
    assert np.allclose(a, b[::-1], atol=1e-5)
    f = _asym_model(oh[None])[0]
    assert not np.allclose(f, f[::-1])             # the mock really is asymmetric


def _mock_predict(batch):
    return {"h": _asym_model(batch)}


MOCK_GEOM = C.Geometry("mock", 1024, 64, 32)


def _small_variants(n=6):
    v = VARS.iloc[[0, 1, 2, 4283, 5000, 9000]].copy().reset_index(drop=True)
    v["row_id"] = np.arange(len(v))
    return v


def test_ref_equals_alt_zero_effect_and_alt_nonzero(tmp_path):
    v = _small_variants()
    runner.run(v, MOCK_GEOM, _mock_predict, {"h": 3}, tmp_path, positions_per_chunk=3, control_ref_eq_alt=True)
    s = runner.finalize(v, {"h": 3}, tmp_path, {"h": {"all": np.ones(3, bool)}})
    assert (s["h__ctrl_ref_eq_alt_max_abs_log2fc"] == 0).all()
    lfc = np.load(tmp_path / "log2fc_h.npy")
    assert lfc.shape == (6, 3) and (np.abs(lfc).max(1) > 0).all()
    # independent expectation for track 0 (counts channel A weighted by position): recompute directly
    for i, r in v.iterrows():
        seq, _, _ = FA.window(r.chrom, r.pos1, MOCK_GEOM.seq_len)
        alts = seq[:512] + r.alt + seq[513:]
        fr = C.rc_average(_asym_model, C.one_hot(seq), slice(None)).sum(0)
        fa = C.rc_average(_asym_model, C.one_hot(alts), slice(None)).sum(0)
        assert np.allclose(lfc[i], np.log2(fa + 1) - np.log2(fr + 1), atol=1e-5)


def test_chromosome_end_rule_pad_not_shift():
    # right end: chr17 position whose 524,288-bp window overruns the chromosome
    clen = FA.chrom_len("chr17")
    p = 83_094_142
    ref = FA.base("chr17", p)
    oh, pl, pr, _ = C.ref_window_onehot(FA, "chr17", p, ref, C.BORZOI)
    expect = (p - 1 - 262144 + 524288) - clen
    assert pl == 0 and pr == expect > 0
    assert (oh[-pr:].sum(1) == 0).all() and oh[C.BORZOI.seq_len // 2].argmax() == "ACGT".index(ref)
    # left end
    p = next(i for i in range(10_000, 200_000) if FA.base("chr1", i) in "ACGT")   # first non-N bases of chr1
    ref = FA.base("chr1", p)
    oh, pl, pr, _ = C.ref_window_onehot(FA, "chr1", p, ref, C.BORZOI)
    assert pl == 262144 - (p - 1) and pr == 0 and (oh[:pl].sum(1) == 0).all()
    assert oh[262144].argmax() == "ACGT".index(ref)          # variant still at window centre


def test_padded_row_counts():
    v = VARS
    L = v.chrom.map(FA.chrom_len)
    n_b = int(((v.pos1 - 1 - 262144 < 0) | (v.pos1 - 1 + 262144 > L)).sum())
    n_e = int(((v.pos1 - 1 - 98304 < 0) | (v.pos1 - 1 + 98304 > L)).sum())
    assert (n_b, n_e) == (149, 30)


def test_summarize():
    x = np.array([[0.1, -0.5, 0.3], [0.2, 0.1, -0.05]], np.float32)
    s = C.summarize(x, np.array([True, True, False]))
    assert np.allclose(s["primary_max_abs_log2fc"], [0.5, 0.2])
    assert np.allclose(s["signed_at_argmax"], [-0.5, 0.2])
    assert s["argmax_track_pos"].tolist() == [1, 0]
    assert np.allclose(s["mean_signed_log2fc"], [-0.2, 0.15]) and np.allclose(s["mean_abs_log2fc"], [0.3, 0.15])


def test_resume_recomputes_only_missing_chunk(tmp_path):
    v = _small_variants()
    runner.run(v, MOCK_GEOM, _mock_predict, {"h": 3}, tmp_path, positions_per_chunk=2)
    runner.finalize(v, {"h": 3}, tmp_path, {"h": {"all": np.ones(3, bool)}})
    first = np.load(tmp_path / "log2fc_h.npy")
    files = sorted((tmp_path / "chunks").glob("chunk_*.npz"))
    assert len(files) == 3
    files[1].unlink()
    calls = {"n": 0}

    def counting(b):
        calls["n"] += 1
        return _mock_predict(b)
    runner.run(v, MOCK_GEOM, counting, {"h": 3}, tmp_path, positions_per_chunk=2)
    assert 0 < calls["n"] < 2 * 3 * 4 and len(list((tmp_path / "chunks").glob("chunk_*.npz"))) == 3
    runner.finalize(v, {"h": 3}, tmp_path, {"h": {"all": np.ones(3, bool)}})
    assert np.array_equal(first, np.load(tmp_path / "log2fc_h.npy"))


def test_track_lists_frozen():
    t = pd.read_csv(HERE / "track_lists" / "borzoi_tl_tracks.tsv", sep="\t")
    assert len(t) == 183 and t.primary.sum() == 152 and t.contaminated.sum() == 31
    assert t.loc[t.contam_lovo_gsm, "tl_head_index"].tolist() == list(range(13, 37))
    assert t.loc[t.contam_encsr_145_148, "tl_head_index"].tolist() == [145, 146, 147, 148]
    assert t.loc[t.contaminated, "tl_head_index"].tolist() == list(range(13, 37)) + [78, 81, 93, 145, 146, 147, 148]
    for f, n in [("borzoi_pretrained_crc_tracks.tsv", 132), ("enformer_pretrained_crc_tracks.tsv", 132),
                 ("alphagenome_crc_tracks.tsv", 33)]:
        assert len(pd.read_csv(HERE / "track_lists" / f, sep="\t")) == n
