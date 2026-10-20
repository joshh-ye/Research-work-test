"""E7: pretrained Enformer per-tile cross sufficient statistics (183 CRC targets x 1,990 eligible CHIP tracks) on the shared grid.
Same tiling/cropping as baseline_enformer.py (central 196,608 bp of each 524,288 bp tile -> 896x128 bp; targets cropped+pooled identically).
Resumable (checkpoint every 50 tiles). Usage: run_enformer.py --split val|test [--limit N]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e7lib import *
from enformer_baseline_utils import ENF_SEQ_LEN, ENF_BINS, targets_to_enformer_grid, load_enformer_targets, filter_enformer_tracks

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["val", "test"]); ap.add_argument("--limit", type=int)
args = ap.parse_args()
OUT = ROOT / "results_phase2/e7_nonoracle/enformer"; OUT.mkdir(parents=True, exist_ok=True)
part = OUT / f"tile_stats_{args.split}.partial.npz"; final = OUT / f"tile_stats_{args.split}.npz"

import torch
from fasta_reader import FastaReader
from sequence_utils import one_hot_encode
from enformer_pytorch import Enformer
dev = "cuda"
inc, _ = filter_enformer_tracks(load_enformer_targets(ROOT / "enformer_data/targets_human.txt"))
elig = inc.track_index.values; assert len(elig) == 1990
elig_t = torch.as_tensor(elig, dtype=torch.long, device=dev)
iv, tg, tiles = load_split(args.split)
N = args.limit or len(iv); C = len(elig)
S = dict(n=np.zeros(N), sT=np.zeros((N, 183)), sT2=np.zeros((N, 183)), sP=np.zeros((N, C)), sP2=np.zeros((N, C)), sTP=np.zeros((N, 183, C)))
done = 0
if part.exists():
    z = np.load(part); done = int(z["done"]); S = {k: z[k] for k in S}; print("resume from", done, flush=True)
model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").eval().to(dev)
fa = FastaReader(str(ROOT / "borzoi_data/hg38/hg38.ml.fa")); half = ENF_SEQ_LEN // 2; t0 = time.time()
for i in range(done, N):
    c = (iv[i].start + iv[i].end) // 2
    seq = fa.fetch(iv[i].chrom, c - half, c + half)
    with torch.no_grad():
        pred = model(torch.from_numpy(one_hot_encode(seq))[None].to(dev), head="human").squeeze(0)
    assert pred.shape[0] == ENF_BINS
    p = pred.index_select(1, elig_t).float().cpu().numpy()
    st = tile_cross_stats(targets_to_enformer_grid(tg[i]), p)
    for k in S: S[k][i] = st[k]
    if (i + 1) % 50 == 0 or i + 1 == N:
        np.savez(part, done=i + 1, **S); print(f"{i+1}/{N} {time.time()-t0:.0f}s", flush=True)
np.savez(final if not args.limit else OUT / f"tile_stats_{args.split}_limit{N}.npz", **S, track_index=elig, chrom=tiles.chrom.values[:N], start=tiles.start.values[:N],
         tile_zero_signal=tiles.zero_signal_all_tracks.values[:N] if "zero_signal_all_tracks" in tiles else np.zeros(N, bool))
print("saved", flush=True)
