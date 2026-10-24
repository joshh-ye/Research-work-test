"""Pooled-128 bp copy of the Phase-2 trunk cache (node-local, regenerable). Same crop/pool as common.to_grid128:
central 4096 x 32 bp bins -> crop 256 bins/side -> mean-pool x4 -> 896 bins x 128 bp. emb_fwd has MARGIN=4 extra bins/side,
so central 4096 = [4:4100]. hg38; tile order = trunk_cache/tiles.csv gid order within each split."""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "priorityD"))
from common import *

POOLED = Path("/tmp/yejz1_ft/pooled128")


def pool_emb(e):
    """(4104, 1920) fp16 (32 bp, 4 margin bins/side) -> (896, 1920) fp16 on the shared 128 bp grid (mean in fp32)."""
    c = np.asarray(e)[MARGIN + CROP: MARGIN + CROP + G_BINS * POOL].astype(np.float32)
    assert c.shape[0] == G_BINS * POOL
    return c.reshape(G_BINS, POOL, -1).mean(1).astype(np.float16)


def pool_tgt(t):
    """(4104, 183) fp32 -> (896, 183) fp32, via the E4 code path (central + to_grid128, float64 mean) then cast to fp32."""
    return to_grid128(central(np.asarray(t))).astype(np.float32)


def split_table():
    import pandas as pd
    tt = pd.read_csv(CACHE / "tiles.csv")
    return tt


def build(root=POOLED, n_threads=16, log=print):
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    tt = split_table(); t0 = time.time(); info = {}
    U = len(np.load(CACHE / "pre_channels.npy"))
    for sp in ["train", "val", "test"]:
        g = tt[tt.split == sp].sort_values("split_idx").gid.values; n = len(g)
        np.save(root / f"{sp}_gid.npy", g)
        E = np.lib.format.open_memmap(root / f"{sp}_emb.npy", "w+", np.float16, (n, G_BINS, EMB_DIM))
        P = np.lib.format.open_memmap(root / f"{sp}_pre.npy", "w+", np.float16, (n, G_BINS, U))
        T = np.lib.format.open_memmap(root / f"{sp}_tgt.npy", "w+", np.float32, (n, G_BINS, N_TGT))

        def work(i):
            gid = int(g[i])
            E[i] = pool_emb(np.load(tile_path(gid, "emb_fwd"), mmap_mode="r"))
            p = np.load(tile_path(gid, "pre128")); assert p.shape == (G_BINS, U) and p.dtype == np.float16, (p.shape, p.dtype)
            P[i] = p
            T[i] = pool_tgt(np.load(tile_path(gid, "tgt")))
        with ThreadPoolExecutor(n_threads) as ex:
            for k, _ in enumerate(ex.map(work, range(n))):
                if (k + 1) % 500 == 0: log(f"{sp}: {k+1}/{n} tiles ({time.time()-t0:.0f}s)")
        E.flush(); P.flush(); T.flush(); info[sp] = n
    size = sum(f.stat().st_size for f in root.glob("*.npy"))
    info.update(build_seconds=round(time.time() - t0), size_gb=round(size / 1e9, 2), pre_channels=U)
    (root / "build_info.json").write_text(json.dumps(info))
    return info


class Pooled:
    """memmapped pooled arrays for one split"""
    def __init__(self, split, root=POOLED):
        root = Path(root); self.split = split
        self.gid = np.load(root / f"{split}_gid.npy")
        self.emb = np.load(root / f"{split}_emb.npy", mmap_mode="r")
        self.pre = np.load(root / f"{split}_pre.npy", mmap_mode="r")
        self.tgt = np.load(root / f"{split}_tgt.npy", mmap_mode="r")
        self.n = len(self.gid)
        tt = split_table().set_index("gid").loc[self.gid]
        self.chrom = tt.chrom.values; self.start = tt.start.values
