"""Alignment tests for the trunk cache vs existing project artifacts (use cached test tiles 123-126; skip if absent)."""
import sys, json
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

TEST0 = 5409  # gid of first test tile (4659 train + 750 val)
IDX = (123, 124, 125, 126)
GIDS = [TEST0 + i for i in IDX]
pytestmark = pytest.mark.skipif(not all(tile_path(g, "emb_fwd").exists() for g in GIDS), reason="cache tiles absent")


def test_tile_table_alignment():
    tt = tile_table()
    assert (tt.split == "train").sum() == 4659 and (tt.split == "val").sum() == 750 and (tt.split == "test").sum() == 359
    assert tt.gid[tt.split == "test"].min() == TEST0
    assert set(tt.chrom[tt.split == "test"]) == {"chr9", "chr22"} and set(tt.chrom[tt.split == "val"]) == {"chr1", "chr8"}


def test_targets_match_results_full():
    T = np.load(ROOT / "results_full/test_targets.npy", mmap_mode="r")
    for i, g in zip(IDX, GIDS):
        c = np.load(tile_path(g, "tgt"))
        assert c.shape == (CBINS, N_TGT)
        assert np.allclose(central(c), T[i], atol=1e-5)


def test_original_head_on_cached_embedding_reproduces_test_preds():
    from torch import nn
    head = nn.Sequential(nn.Linear(1920, 1024), nn.GELU(), nn.Dropout(.2), nn.Linear(1024, 183), nn.Softplus())
    head.load_state_dict(torch.load(ROOT / "borzoi_ckpt_full/model_head_best.pt", map_location="cpu")); head.eval()
    P = np.load(ROOT / "results_full/test_preds.npy", mmap_mode="r")
    for i, g in zip(IDX, GIDS):
        e = torch.from_numpy(np.load(tile_path(g, "emb_fwd"))[MARGIN:MARGIN + CENTER_BINS].astype(np.float32))
        with torch.no_grad(): out = head(e).numpy()
        r = np.corrcoef(out.ravel(), np.asarray(P[i]).ravel())[0, 1]
        assert r > 0.999, r
        assert np.abs(out - P[i]).max() < 0.05 * max(1.0, np.abs(P[i]).max())


def test_pretrained_pool_matches_qing_rerun():
    pr = ROOT / "reports/qing_deliverables/evaluation_evidence/baseline_borzoi_rerun"
    P = np.load(pr / "preds_test_shared128.npy", mmap_mode="r")
    elig = np.load(pr / "per_tile_stats_test_shared128.npz")["eligible_track_index"]
    for i, g in zip(IDX, GIDS):
        a = np.load(tile_path(g, "pre128"))[:, elig].astype(np.float32)
        assert a.shape == (G_BINS, len(elig))
        assert np.allclose(a, P[i], rtol=2e-3, atol=1e-3 * max(1, np.abs(P[i]).max()) * 0 + 1e-3)


def test_rc_embedding_orientation():
    """RC embedding row j corresponds to genomic bin 4103-j: per-channel profiles over bins should correlate better
    after flipping (use the 50 highest-variance channels of the fwd embedding)."""
    f = np.load(tile_path(GIDS[0], "emb_fwd")).astype(np.float32); r = np.load(tile_path(GIDS[0], "emb_rc")).astype(np.float32)
    ch = np.argsort(f.var(0))[-50:]
    def mc(a, b): return np.mean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in ch])
    assert mc(f, r[::-1]) > mc(f, r) + 0.2, (mc(f, r[::-1]), mc(f, r))


def test_fast_bigwig_binning_equals_stats():
    import pyBigWig
    from bigwig_loader import BigWigLoader
    paths = json.loads((ROOT / "borzoi_ckpt_full/bw_files.json").read_text())
    tt = tile_table(); sub = tt[tt.split == "train"]
    ld = BigWigLoader(paths[::20], 32)
    for k in (5, 900, 2500, 4000):                       # dense + sparse regions
        r = sub.iloc[k]; c = (r.start + r.end) // 2; st, en = c - 65536 - 128, c + 65536 + 128
        ref = ld.load(r.chrom, st, en)
        fast = np.stack([bw_bin_means(pyBigWig.open(p), r.chrom, st, en) for p in paths[::20]], 1)
        assert np.allclose(fast, ref, rtol=1e-5, atol=1e-6), np.abs(fast - ref).max()


def test_shift_emulation_sign_against_real_shifted_forward():
    """Real trunk forward on the window re-centred by k bins (k=+3) vs the cached-slice emulation, for fwd and RC.
    Emulation must correlate far better with the correct sign/direction than with the opposite shift."""
    if not torch.cuda.is_available(): pytest.skip("no GPU")
    import os, importlib.util
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from borzoi_pytorch import Borzoi
    from fasta_reader import FastaReader
    spec = importlib.util.spec_from_file_location("bc", str(Path(__file__).resolve().parents[1] / "01_build_cache.py")); bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    from headlib import load_tile
    g = GIDS[0]; r = tile_table().set_index("gid").loc[g]; k = 3
    fa = FastaReader(str(ROOT / "borzoi_data/hg38/hg38.ml.fa")); c = (r.start + r.end) // 2 + 32 * k
    s = torch.from_numpy(bc.seq_idx(fa.fetch(r.chrom, c - SEQ_LEN // 2, c + SEQ_LEN // 2))).cuda()
    buf = {}; ms = [Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}").eval().cuda() for i in range(4)]
    for i, m in enumerate(ms): m.human_head.register_forward_pre_hook(lambda mod, inp, i=i: buf.__setitem__(i, inp[0]))
    def emb(seq):
        with torch.no_grad():
            es = []
            for i, m in enumerate(ms): m(bc.onehot_t(seq)); es.append(buf[i][0].T)
            return torch.stack(es).mean(0)[1024:1024 + CENTER_BINS].cpu().numpy()
    def corr(a, b):
        ch = np.argsort(a.var(0))[-50:]; return np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in ch])
    for rc in (False, True):
        real = emb(bc.rc_idx(s) if rc else s)
        good, _ = load_tile(g, rc=rc, shift=k, need_tgt=False)
        bad, _ = load_tile(g, rc=rc, shift=-k, need_tgt=False)
        cg, cb = corr(real, good.astype(np.float32)), corr(real, bad.astype(np.float32))
        assert cg > 0.9 and cg > cb + 0.2, (rc, cg, cb)
