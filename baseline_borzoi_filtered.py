"""
Filtered-Borzoi best-match baseline.

Two modes:

  --from-matrix PATH   Reuse an existing (n_targets, n_old_eligible) Pearson matrix
                       (e.g. results_baseline/baseline_pearson_matrix_val.npy) and
                       re-select the best match over the *refined* eligible track
                       set. Valid because the correlations are identical (same
                       Borzoi model, intervals and targets) — only which tracks
                       are eligible for the argmax changes. Requires that every
                       refined-eligible track is present in the old matrix's track
                       set (checked). No torch / GPU needed.

  (default)            Full Borzoi inference over the split's intervals, restricted
                       to the refined eligible track set. Needs torch + GPU
                       (ACCRE A100). Produces the filtered Pearson matrix directly.

Outputs (both modes):
  results_baseline_filtered/borzoi_filtered_pearson_matrix_{split}.npy   (n_targets, n_elig)
  results_baseline_filtered/borzoi_filtered_best_match_{split}.csv

Usage (laptop, reuse):
  python baseline_borzoi_filtered.py --split val \
      --from-matrix results_baseline/baseline_pearson_matrix_val.npy

Usage (ACCRE, full inference):
  python baseline_borzoi_filtered.py --split test --data-root ./borzoi_data \
      --targets-dir ./results_full --n-folds 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from borzoi_baseline_utils import (
    CENTER_BINS, SEQ_LEN, INDEX_LO, INDEX_HI,
    load_borzoi_targets, filter_borzoi_tracks, load_crc_target_meta,
    select_best_match, StreamingPearson,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="./borzoi_data")
    p.add_argument("--targets-dir", default="./results_full",
                   help="dir with {split}_targets.npy and {split}_per_track_metrics.csv")
    p.add_argument("--results-dir", default="./results_baseline_filtered")
    p.add_argument("--split",       default="val", choices=["val", "test"])
    p.add_argument("--n-folds",     type=int, default=4)
    p.add_argument("--from-matrix", default=None,
                   help="reuse an existing pearson matrix instead of running Borzoi")
    p.add_argument("--old-range-filter", default="H3",
                   help="case-SENSITIVE histone substring used by the ORIGINAL baseline "
                        "to build --from-matrix (default 'H3', matching baseline_borzoi.py)")
    return p.parse_args()


def refined_eligible(data_root: Path) -> tuple[pd.DataFrame, list[int]]:
    df = load_borzoi_targets(data_root / "targets_human.csv")
    included, _ = filter_borzoi_tracks(df)
    df_meta = df.copy()
    # attach parsed factor/cell_tissue for the full frame (needed by select_best_match)
    from borzoi_baseline_utils import parse_borzoi_description
    parsed = df_meta["description"].apply(parse_borzoi_description)
    df_meta["factor"]      = [p["factor"] for p in parsed]
    df_meta["cell_tissue"] = [p["cell_tissue"] for p in parsed]
    df_meta = df_meta.set_index("track_index")
    return df_meta, included["track_index"].tolist()


def old_matrix_track_indices(data_root: Path, case_sensitive_term: str) -> list[int]:
    """Reconstruct the ordered track list the ORIGINAL baseline used to build its
    Pearson matrix: range [2186,6069] minus rows whose description contains the
    given case-SENSITIVE term. Order = ascending track_index (CSV order)."""
    df = load_borzoi_targets(data_root / "targets_human.csv")
    sub = df[(df["track_index"] >= INDEX_LO) & (df["track_index"] <= INDEX_HI)]
    keep = sub[~sub["description"].str.contains(case_sensitive_term, case=True, na=False)]
    return keep["track_index"].tolist()


def load_target_meta(targets_dir: Path, data_root: Path, split: str) -> pd.DataFrame:
    metrics = pd.read_csv(targets_dir / f"{split}_per_track_metrics.csv")
    return load_crc_target_meta(metrics["bw_filename"].tolist(),
                                data_root / "TF_184tracks.csv")


def run_from_matrix(args):
    data_root = Path(args.data_root)
    df_meta, elig_idx = refined_eligible(data_root)
    old_idx = old_matrix_track_indices(data_root, args.old_range_filter)

    pmat_old = np.load(args.from_matrix)   # (n_targets, len(old_idx))
    assert pmat_old.shape[1] == len(old_idx), (
        f"matrix cols {pmat_old.shape[1]} != reconstructed old track list {len(old_idx)}; "
        f"the --old-range-filter may not match how the matrix was built")

    pos = {t: j for j, t in enumerate(old_idx)}
    missing = [t for t in elig_idx if t not in pos]
    if missing:
        raise ValueError(f"{len(missing)} refined-eligible tracks absent from the old "
                         f"matrix — cannot reuse; run full inference instead. e.g. {missing[:5]}")
    cols = [pos[t] for t in elig_idx]
    pmat = pmat_old[:, cols]               # (n_targets, n_elig) refined
    return pmat, df_meta, elig_idx


def run_full_inference(args):
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))
    from fasta_reader import FastaReader
    from sequence_utils import one_hot_encode
    from genome_tiler import tile_genome
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks

    data_root = Path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_meta, elig_idx = refined_eligible(data_root)

    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")
    intervals  = tile_genome(fasta_path)[args.split]
    targets    = np.load(Path(args.targets_dir) / f"{args.split}_targets.npy")
    assert targets.shape[0] == len(intervals)
    n_targets = targets.shape[2]

    backbones = []
    for i in range(args.n_folds):
        b = Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}").eval().to(device)
        backbones.append(b)

    pearson = StreamingPearson(n_targets, len(elig_idx))
    fasta = FastaReader(fasta_path)
    from tqdm import tqdm
    for idx, iv in enumerate(tqdm(intervals, desc="Borzoi inference")):
        center = (iv.start + iv.end) // 2
        half = SEQ_LEN // 2
        seq = fasta.fetch(iv.chrom, center - half, center + half)
        enc = torch.from_numpy(one_hot_encode(seq)).permute(1, 0).to(device)
        with torch.no_grad():
            pred = predict_tracks(backbones, enc, elig_idx)
        avg = pred.mean(axis=1).squeeze(0)
        n_bins = avg.shape[0]
        off = (n_bins - CENTER_BINS) // 2
        avg = avg[off: off + CENTER_BINS]
        pearson.update(targets[idx].reshape(CENTER_BINS, n_targets), np.asarray(avg))
    fasta.close()
    return pearson.result(), df_meta, elig_idx


def main():
    args = parse_args()
    out = Path(args.results_dir); out.mkdir(exist_ok=True)

    if args.from_matrix:
        print(f"[from-matrix] reusing {args.from_matrix}")
        pmat, df_meta, elig_idx = run_from_matrix(args)
    else:
        print("[full-inference] running Borzoi over intervals")
        pmat, df_meta, elig_idx = run_full_inference(args)

    np.save(out / f"borzoi_filtered_pearson_matrix_{args.split}.npy", pmat)

    target_meta = load_target_meta(Path(args.targets_dir), Path(args.data_root), args.split)
    best = select_best_match(pmat, elig_idx, df_meta, target_meta)
    best.to_csv(out / f"borzoi_filtered_best_match_{args.split}.csv", index=False)

    # variance guard: flag any selected best-match column that is ~constant
    print(f"\n[{args.split}] eligible tracks: {len(elig_idx)}  targets: {pmat.shape[0]}")
    print(f"  mean best-match Pearson R : {best['pearson_r'].mean():.4f}")
    print(f"  median                    : {best['pearson_r'].median():.4f}")
    print(f"  min / max                 : {best['pearson_r'].min():.4f} / {best['pearson_r'].max():.4f}")
    print(f"Wrote {out}/borzoi_filtered_pearson_matrix_{args.split}.npy  {pmat.shape}")
    print(f"Wrote {out}/borzoi_filtered_best_match_{args.split}.csv")


if __name__ == "__main__":
    main()
