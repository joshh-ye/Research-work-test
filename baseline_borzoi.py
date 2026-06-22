"""
Baseline: find the best-matching original Borzoi track for each of the 183
CRC TF targets by Pearson correlation, then save those predictions as a baseline
to compare against the transfer-learned model.

Search space: Borzoi human tracks 2186-6069 (ChIP), minus H3 histone marks.

Pearson R is accumulated in a streaming fashion (one interval at a time) so the
full ~43 GB Borzoi prediction array is never held in memory.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))

from fasta_reader import FastaReader
from sequence_utils import one_hot_encode
from genome_tiler import tile_genome

from borzoi_pytorch import Borzoi
from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="./borzoi_data",
                   help="Directory containing hg38/ and targets_human.csv")
    p.add_argument("--targets-dir", default="./results_full",
                   help="Directory containing val_targets.npy / test_targets.npy")
    p.add_argument("--results-dir", default="./results_baseline",
                   help="Output directory for baseline predictions and CSVs")
    p.add_argument("--n-folds",     type=int, default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split",       default="val", choices=["val", "test"])
    return p.parse_args()


CENTER_BINS = 4096
SEQ_LEN     = 524288


def load_filtered_track_indices(targets_csv: str):
    df = pd.read_csv(targets_csv)
    chip = df[(df.index >= 2186) & (df.index <= 6069)]
    chip = chip[~chip["description"].str.contains("H3", case=True)]
    return chip.index.tolist(), df


class StreamingPearson:
    """
    Accumulates the five running sums needed to compute Pearson R between
    every (target_track, borzoi_track) pair without storing all predictions.

    Shapes:
      n  : scalar — number of data points seen so far
      sx : (n_targets,)        — sum of target values
      sy : (n_borzoi,)         — sum of borzoi values
      sxy: (n_targets, n_borzoi) — sum of products
      sx2: (n_targets,)        — sum of squared target values
      sy2: (n_borzoi,)         — sum of squared borzoi values
    """
    def __init__(self, n_targets: int, n_borzoi: int):
        self.n   = 0
        self.sx  = np.zeros(n_targets, dtype=np.float64)
        self.sy  = np.zeros(n_borzoi,  dtype=np.float64)
        self.sxy = np.zeros((n_targets, n_borzoi), dtype=np.float64)
        self.sx2 = np.zeros(n_targets, dtype=np.float64)
        self.sy2 = np.zeros(n_borzoi,  dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray):
        # x: (N, n_targets)   y: (N, n_borzoi)
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n   += x.shape[0]
        self.sx  += x.sum(axis=0)
        self.sy  += y.sum(axis=0)
        self.sxy += x.T @ y          # (n_targets, n_borzoi)
        self.sx2 += (x ** 2).sum(axis=0)
        self.sy2 += (y ** 2).sum(axis=0)

    def result(self) -> np.ndarray:
        n = self.n
        num   = n * self.sxy - np.outer(self.sx, self.sy)
        denom = np.sqrt(
            np.maximum(n * self.sx2 - self.sx ** 2, 0)[:, None] *
            np.maximum(n * self.sy2 - self.sy ** 2, 0)[None, :]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom > 0, num / denom, 0.0)
        return r.astype(np.float32)   # (n_targets, n_borzoi)


def infer_interval(backbones, seq_str, keep_indices):
    seq_enc = one_hot_encode(seq_str)
    seq_t   = torch.from_numpy(seq_enc).permute(1, 0).to(next(backbones[0].parameters()).device)   # (4, seq_len)
    with torch.no_grad():
        pred = predict_tracks(backbones, seq_t, keep_indices)
    avg    = pred.mean(axis=1).squeeze(0)                # (n_bins, n_borzoi)
    n_bins = avg.shape[0]
    offset = (n_bins - CENTER_BINS) // 2
    return avg[offset: offset + CENTER_BINS]             # (CENTER_BINS, n_borzoi)


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    results   = Path(args.results_dir)
    results.mkdir(exist_ok=True)

    # 1. Filter track indices
    keep_indices, df_meta = load_filtered_track_indices(str(data_root / "targets_human.csv"))
    print(f"Filtered ChIP (no H3) tracks: {len(keep_indices)}")

    # 2. Val/test intervals
    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")
    intervals  = tile_genome(fasta_path)[args.split]
    print(f"Intervals ({args.split}): {len(intervals)}")

    # 3. Load targets
    targets = np.load(Path(args.targets_dir) / f"{args.split}_targets.npy")
    # targets: (n_intervals, CENTER_BINS, n_targets)
    assert targets.shape[0] == len(intervals)
    n_targets = targets.shape[2]
    print(f"Targets shape: {targets.shape}")

    # 4. Load Borzoi backbones
    print(f"Loading {args.n_folds} Borzoi folds ...")
    backbones = []
    for i in range(args.n_folds):
        b = Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}")
        b.eval().to(args.device)
        backbones.append(b)

    # 5. Stream inference + correlation accumulation
    # Also keep best-matched Borzoi preds in memory: (n_intervals, CENTER_BINS, n_targets)
    # This is only 2.2 GB — fine.
    pearson = StreamingPearson(n_targets, len(keep_indices))
    fasta   = FastaReader(fasta_path)

    # We need per-interval Borzoi preds to save baseline_preds at the end,
    # but we don't know best_local_idx until after all intervals are processed.
    # Store all preds in a memory-mapped temp file to avoid RAM pressure.
    mmap_path = results / f"_tmp_borzoi_{args.split}.npy"
    mmap_shape = (len(intervals), CENTER_BINS, len(keep_indices))
    borzoi_mmap = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float32, shape=mmap_shape
    )

    for idx, iv in enumerate(tqdm(intervals, desc="Borzoi inference")):
        center  = (iv.start + iv.end) // 2
        half    = SEQ_LEN // 2
        seq_str = fasta.fetch(iv.chrom, center - half, center + half)
        avg     = infer_interval(backbones, seq_str, keep_indices)  # (CENTER_BINS, n_borzoi)
        borzoi_mmap[idx] = avg

        x = targets[idx].reshape(CENTER_BINS, n_targets)   # (CENTER_BINS, n_targets)
        pearson.update(x, avg)

    fasta.close()

    # 6. Compute Pearson matrix
    print("Finalizing Pearson correlation matrix ...")
    pmat = pearson.result()   # (n_targets, n_borzoi)
    np.save(results / f"baseline_pearson_matrix_{args.split}.npy", pmat)
    print(f"Pearson matrix shape: {pmat.shape}")

    # 7. Best match per target
    best_local_idx  = pmat.argmax(axis=1)
    best_borzoi_ids = [keep_indices[i] for i in best_local_idx]
    best_r          = pmat.max(axis=1)

    pd.DataFrame({
        "target_track_idx":        range(n_targets),
        "best_borzoi_track_idx":   best_borzoi_ids,
        "best_borzoi_description": [df_meta.loc[i, "description"] for i in best_borzoi_ids],
        "pearson_r":               best_r,
    }).to_csv(results / f"baseline_best_match_{args.split}.csv", index=False)

    # 8. Save baseline predictions (slice best Borzoi track per target from mmap)
    print("Saving baseline predictions ...")
    baseline_preds = np.stack(
        [borzoi_mmap[:, :, i] for i in best_local_idx], axis=2
    )  # (n_intervals, CENTER_BINS, n_targets)
    np.save(results / f"baseline_preds_{args.split}.npy", baseline_preds)

    mmap_path.unlink()   # clean up temp mmap file

    print(f"\nMean baseline Pearson R : {best_r.mean():.4f}")
    print(f"Best  baseline Pearson R: {best_r.max():.4f}")
    print(f"Worst baseline Pearson R: {best_r.min():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
