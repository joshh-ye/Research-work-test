"""
Original-Enformer best-match baseline, the Enformer counterpart of
baseline_borzoi_filtered.py.

For every CRC target track we run pretrained Enformer over the split's
intervals, correlate each target against every eligible (non-histone ChIP)
Enformer track, and keep the best-correlating track as that target's baseline.

Unlike the Borzoi baseline there is no --from-matrix shortcut: no Enformer
correlation matrix exists yet, so this always runs inference (GPU strongly
recommended).

Alignment
---------
Intervals are the same 524,288 bp tiles the transfer model was trained on, so
`{split}_targets.npy` row i still corresponds to interval i. Enformer sees the
central 196,608 bp of each tile and predicts the central 114,688 bp of that, at
128 bp. The targets are cropped and mean-pooled onto that same grid by
enformer_baseline_utils.targets_to_enformer_grid, so both sides of every
correlation cover identical genomic coordinates.

Outputs:
  results_baseline_enformer/enformer_pearson_matrix_{split}.npy   (n_targets, n_elig)
  results_baseline_enformer/enformer_best_match_{split}.csv

Usage (ACCRE):
  python baseline_enformer.py --split test --targets-dir ./results_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from borzoi_baseline_utils import (
    load_crc_target_meta, select_best_match, StreamingPearson,
)
from enformer_baseline_utils import (
    ENF_SEQ_LEN, ENF_BINS, load_enformer_targets, filter_enformer_tracks,
    targets_to_enformer_grid,
)

MODEL_ID = "EleutherAI/enformer-official-rough"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",     default="./borzoi_data",
                   help="holds hg38/ and TF_184tracks.csv (shared with the Borzoi run)")
    p.add_argument("--enformer-root", default="./enformer_data",
                   help="holds targets_human.txt")
    p.add_argument("--targets-dir",   default="./results_full",
                   help="dir with {split}_targets.npy and {split}_per_track_metrics.csv")
    p.add_argument("--results-dir",   default="./results_baseline_enformer")
    p.add_argument("--split",         default="test", choices=["val", "test"])
    p.add_argument("--model-id",      default=MODEL_ID)
    p.add_argument("--limit",         type=int, default=None,
                   help="debug: only process the first N intervals")
    return p.parse_args()


def eligible_tracks(enformer_root: Path):
    df = load_enformer_targets(enformer_root / "targets_human.txt")
    included, _ = filter_enformer_tracks(df)
    meta = included.set_index("track_index")
    return meta, included["track_index"].tolist()


def run_inference(args):
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))
    from fasta_reader import FastaReader
    from sequence_utils import one_hot_encode
    from genome_tiler import tile_genome
    from enformer_pytorch import Enformer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    enf_meta, elig_idx = eligible_tracks(Path(args.enformer_root))
    elig_t = torch.as_tensor(elig_idx, dtype=torch.long, device=device)

    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")
    intervals  = tile_genome(fasta_path)[args.split]
    targets    = np.load(Path(args.targets_dir) / f"{args.split}_targets.npy",
                         mmap_mode="r")
    assert targets.shape[0] == len(intervals), (
        f"{targets.shape[0]} target rows vs {len(intervals)} intervals — "
        "the tiling used for the transfer model and here must match")
    if args.limit:
        intervals = intervals[:args.limit]
    n_targets = targets.shape[2]

    model = Enformer.from_pretrained(args.model_id).eval().to(device)

    pearson = StreamingPearson(n_targets, len(elig_idx))
    fasta = FastaReader(fasta_path)
    half = ENF_SEQ_LEN // 2
    for idx, iv in enumerate(tqdm(intervals, desc="Enformer inference")):
        center = (iv.start + iv.end) // 2
        seq = fasta.fetch(iv.chrom, center - half, center + half)
        # batch dim is explicit: a bare (seq_len, 4) tensor is ambiguous with the
        # integer-token input form enformer-pytorch also accepts.
        enc = torch.from_numpy(one_hot_encode(seq))[None].to(device)
        with torch.no_grad():
            pred = model(enc, head="human")           # (1, 896, 5313)
        pred = pred.squeeze(0)
        assert pred.shape[0] == ENF_BINS, f"unexpected Enformer output {tuple(pred.shape)}"
        pred = pred.index_select(1, elig_t).float().cpu().numpy()
        y = targets_to_enformer_grid(targets[idx])    # (896, n_targets)
        pearson.update(y, pred)
    fasta.close()
    return pearson.result(), enf_meta, elig_idx


def main():
    args = parse_args()
    out = Path(args.results_dir); out.mkdir(exist_ok=True)
    split = args.split

    print(f"[enformer] {args.model_id} — running inference over {split} intervals")
    pmat, enf_meta, elig_idx = run_inference(args)
    np.save(out / f"enformer_pearson_matrix_{split}.npy", pmat)

    metrics = pd.read_csv(Path(args.targets_dir) / f"{split}_per_track_metrics.csv")
    target_meta = load_crc_target_meta(metrics["bw_filename"].tolist(),
                                       Path(args.data_root) / "TF_184tracks.csv")
    best = select_best_match(pmat, elig_idx, enf_meta, target_meta)
    best = best.rename(columns=lambda c: c.replace("borzoi", "enformer"))
    best.to_csv(out / f"enformer_best_match_{split}.csv", index=False)

    print(f"\n[{split}] eligible tracks: {len(elig_idx)}  targets: {pmat.shape[0]}")
    print(f"  mean best-match Pearson R : {best['pearson_r'].mean():.4f}")
    print(f"  median                    : {best['pearson_r'].median():.4f}")
    print(f"  min / max                 : {best['pearson_r'].min():.4f} / {best['pearson_r'].max():.4f}")
    print(f"Wrote {out}/enformer_pearson_matrix_{split}.npy  {pmat.shape}")
    print(f"Wrote {out}/enformer_best_match_{split}.csv")


if __name__ == "__main__":
    main()
