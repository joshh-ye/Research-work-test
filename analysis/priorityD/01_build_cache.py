"""Build the frozen-trunk cache for E4/E5/E8 (one GPU pass, resumable, per-tile files).
Per tile (gid), written atomically (tmp+rename) under results_phase2/trunk_cache/:
  emb_fwd/tGID.npy  float16 (4104, 1920)  4-replicate-mean trunk embedding (the tap = input of human_head, as in
                                           borzoi_code/model.py), central 4104 of 6144 bins (4096 + 4 margin each side)
  emb_rc/tGID.npy   float16 (4104, 1920)  same from the reverse-complemented 524,288 bp window (train tiles only);
                                           row j <-> genomic bin (4103-j)
  pre128/tGID.npy   float16 (896, 7611)   4-replicate-mean PRETRAINED human-head outputs (all 7611 tracks), central
                                           4096 bins -> shared 128 bp grid (crop 256/side, mean-pool x4)
  tgt/tGID.npy      float32 (4104, 183)   BigWig targets at 32 bp, same 4104-bin window (bin grid aligned to 32 bp)
fp16 is a storage deviation (embedding rel. error ~5e-4); no shift/RC augmentation randomness here -- it is emulated at
head-training time from the margin bins and the RC cache (see DESIGN.md).
Usage: 01_build_cache.py [--splits test val train] [--limit N] [--workers 6]"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from torch.utils.data import Dataset, DataLoader
from fasta_reader import FastaReader
from bigwig_loader import BigWigLoader

_LUT = np.full(256, 255, dtype=np.uint8)
for k, ch in enumerate(b"ACGT"):
    _LUT[ch] = k; _LUT[ch + 32] = k


def seq_idx(seq: str):
    """string -> uint8 indices 0..3 (255 = N / non-ACGT, encoded as all-zero one-hot like sequence_utils)."""
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def onehot_t(idx_t):
    """(L,) uint8 tensor -> (1, 4, L) float32 one-hot on its device (N -> zeros)."""
    oh = torch.zeros(4, idx_t.numel(), device=idx_t.device)
    ok = idx_t < 4
    oh[idx_t[ok].long(), torch.nonzero(ok).squeeze(1)] = 1.0
    return oh[None]


def rc_idx(idx_t):
    """reverse complement of an index tensor (A<->T, C<->G, N stays N)."""
    comp = torch.where(idx_t < 4, 3 - idx_t, idx_t)
    return comp.flip(0)


class TileDS(Dataset):
    def __init__(self, rows):
        self.rows = rows; self.fa = None; self.bw = None; self.paths = None

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        if self.fa is None:
            import json
            self.fa = FastaReader(str(ROOT / "borzoi_data/hg38/hg38.ml.fa"))
            self.bw = BigWigLoader(json.loads((ROOT / "borzoi_ckpt_full/bw_files.json").read_text()), bin_size=32)
        r = self.rows[i]
        c = (r["start"] + r["end"]) // 2
        s = seq_idx(self.fa.fetch(r["chrom"], c - SEQ_LEN // 2, c + SEQ_LEN // 2))
        half = CENTER_BINS * 32 // 2 + MARGIN * 32
        t = np.stack([bw_bin_means(h, r["chrom"], c - half, c + half) for h in self.bw.handles], 1) \
            if self.bw.handles is not None else None
        if t is None:
            self.bw._open()
            t = np.stack([bw_bin_means(h, r["chrom"], c - half, c + half) for h in self.bw.handles], 1)
        assert s.shape[0] == SEQ_LEN and t.shape == (CBINS, N_TGT)
        return r["gid"], torch.from_numpy(s), torch.from_numpy(t)


def save_atomic(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{threading.get_ident()}.npy")
    np.save(tmp, arr); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-rc-for", nargs="*", default=[], help="splits where RC embeddings are skipped (default none: folds need RC for every chromosome)")
    a = ap.parse_args()
    from borzoi_pytorch import Borzoi
    tt = tile_table(); CACHE.mkdir(parents=True, exist_ok=True)
    tt.to_csv(CACHE / "tiles.csv", index=False)
    dev = "cuda"
    models = [Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}").eval().to(dev) for i in range(4)]
    buf = {}
    for k, m in enumerate(models):
        for p in m.parameters(): p.requires_grad = False
        m.human_head.register_forward_pre_hook(lambda mod, inp, k=k: buf.__setitem__(k, inp[0]))
    pool = ThreadPoolExecutor(4); futs = []
    t_start = time.time(); n_done = 0; maxv = 0.0
    for split in a.splits:
        sub = tt[tt.split == split]
        if a.limit: sub = sub.iloc[a.start:a.start + a.limit]
        need_rc = split not in a.no_rc_for
        rows = []
        for r in sub.to_dict("records"):
            g = r["gid"]
            done = all(tile_path(g, k).exists() for k in (["emb_fwd", "pre128", "tgt"] + (["emb_rc"] if need_rc else [])))
            if not done: rows.append(r)
        print(f"[{split}] {len(sub)} tiles, {len(rows)} to do (rc={need_rc})", flush=True)
        if not rows: continue
        dl = DataLoader(TileDS(rows), batch_size=None, num_workers=a.workers, prefetch_factor=2)
        for gid, s, t in dl:
            s = s.to(dev)
            with torch.no_grad():
                outs, embs = [], []
                x = onehot_t(s)
                for k, m in enumerate(models):
                    y = m(x)[0]; outs.append(y); embs.append(buf[k][0].T)   # (7611,6144), (6144,1920)
                avg = torch.stack(outs).mean(0); emb = torch.stack(embs).mean(0)
                del outs, embs
                off = (emb.shape[0] - CBINS) // 2
                assert emb.shape[0] == 6144 and off == 1020
                e_f = emb[off:off + CBINS].half().cpu().numpy()
                pre = avg[:, 1024:1024 + CENTER_BINS].T.cpu().numpy()      # (4096, 7611) float32
                maxv = max(maxv, float(pre.max()))
                p128 = to_grid128(pre).astype(np.float16)
                e_r = None
                if need_rc:
                    xr = onehot_t(rc_idx(s)); embs = []
                    for k, m in enumerate(models):
                        m(xr); embs.append(buf[k][0].T)
                    e_r = torch.stack(embs).mean(0)[off:off + CBINS].half().cpu().numpy()
            g = int(gid)
            futs.append(pool.submit(save_atomic, tile_path(g, "emb_fwd"), e_f))
            futs.append(pool.submit(save_atomic, tile_path(g, "pre128"), p128))
            futs.append(pool.submit(save_atomic, tile_path(g, "tgt"), t.numpy()))
            if e_r is not None: futs.append(pool.submit(save_atomic, tile_path(g, "emb_rc"), e_r))
            n_done += 1
            if len(futs) > 64:
                [f.result() for f in futs]; futs.clear()
            if n_done % 25 == 0:
                print(f"[{split}] done {n_done} wall {time.time()-t_start:.0f}s  max_pre={maxv:.1f} (fp16 max 65504)", flush=True)
    [f.result() for f in futs]
    (CACHE / "build_info.json").write_text(json.dumps(dict(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tiles_done_this_run=n_done, max_pretrained_output=maxv, torch=torch.__version__), indent=2))
    print("finished", flush=True)

if __name__ == "__main__":
    main()
