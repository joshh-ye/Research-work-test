"""One-off (2026-10-20 quota redesign): convert already-cached pre128 files from 7611 to the 1,888 pre_channels() columns
(lossless column selection, atomic replace) and delete emb_rc files except the 4 test tiles used by unit tests (gids 5532-5535).
Also removes stale tmp files. Idempotent."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
U = pre_channels()[0]; np.save(CACHE / "pre_channels.npy", U)
keep_rc = {5532, 5533, 5534, 5535}; nconv = nrc = 0
for p in sorted((CACHE / "pre128").glob("t*.npy")):
    if ".tmp" in p.name: p.unlink(); continue
    a = np.load(p)
    if a.shape[1] == N_HUMAN:
        tmp = p.with_suffix(".tmpslim.npy"); np.save(tmp, a[:, U]); os.replace(tmp, p); nconv += 1
for k in ["emb_fwd", "emb_rc", "tgt"]:
    for p in (CACHE / k).glob("*tmp*"): p.unlink()
for p in (CACHE / "emb_rc").glob("t*.npy"):
    if int(p.stem[1:]) not in keep_rc: p.unlink(); nrc += 1
print(f"converted {nconv} pre128 files to {len(U)} channels; deleted {nrc} emb_rc files")
