"""
AlphaGenome best-match baseline — the AlphaGenome counterpart of
baseline_enformer.py / baseline_borzoi_filtered.py.

For every CRC target track we run AlphaGenome over the split's intervals,
correlate each target against every returned TF-ChIP track, and keep the
best-correlating track as that target's baseline.

Model access
------------
AlphaGenome has no public weights: DeepMind serves it as a hosted gRPC API
(gdmscience.googleapis.com), and the `alphagenome` PyPI package is a thin
client. Running this therefore requires an API key from
https://deepmind.google.com/science/alphagenome, supplied via the
ALPHAGENOME_API_KEY environment variable. The key is never printed or written
to any output file.

Privacy note: this sends *reference genome sequence only* (public hg38) to the
Google endpoint. The CRC target tracks — the private data — never leave this
machine; they are only used locally, to correlate against returned predictions.

Alignment
---------
Intervals are the same 524,288 bp tiles used everywhere else, and AlphaGenome
accepts exactly that length (SEQUENCE_LENGTH_500KB). To stay directly
comparable with the Enformer baseline we reduce AlphaGenome's output to the
same common grid: the central 114,688 bp at 128 bp = 896 bins. The targets are
cropped and mean-pooled onto that identical grid, so both sides of every
correlation cover the same genomic coordinates at the same resolution.

Outputs:
  results_baseline_alphagenome/alphagenome_pearson_matrix_{split}.npy
  results_baseline_alphagenome/alphagenome_best_match_{split}.csv
  results_baseline_alphagenome/alphagenome_tracks_{split}.csv

Usage:
  export ALPHAGENOME_API_KEY=...        # not stored in the repo
  python baseline_alphagenome.py --split test --limit 2     # smoke test
  python baseline_alphagenome.py --split test               # full run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from borzoi_baseline_utils import (
    load_crc_target_meta, select_best_match, StreamingPearson,
)
from enformer_baseline_utils import (
    ENF_BINS, ENF_BIN_SIZE, ENF_SPAN, targets_to_enformer_grid,
)

TILE_LEN = 524_288          # our tiling, and AlphaGenome's SEQUENCE_LENGTH_500KB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="./borzoi_data",
                   help="holds hg38/ and TF_184tracks.csv (shared with the other runs)")
    p.add_argument("--targets-dir", default="./results_full",
                   help="dir with {split}_targets.npy and {split}_per_track_metrics.csv")
    p.add_argument("--results-dir", default="./results_baseline_alphagenome")
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--limit",       type=int, default=None,
                   help="debug/smoke: only process the first N intervals")
    p.add_argument("--max-retries", type=int, default=4,
                   help="per-interval retries on transient API errors")
    return p.parse_args()


def get_client():
    """Build the official AlphaGenome client, or fail with an actionable message."""
    api_key = os.environ.get("ALPHAGENOME_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ALPHAGENOME_API_KEY is not set.\n"
            "AlphaGenome has no public weights — it is only reachable as a hosted\n"
            "API. Obtain a key at https://deepmind.google.com/science/alphagenome\n"
            "then re-run with:  export ALPHAGENOME_API_KEY=<your key>"
        )
    try:
        from alphagenome.models import dna_client
    except ImportError as e:
        raise SystemExit(
            "The `alphagenome` package is not installed in this interpreter.\n"
            "Install it with:  python -m pip install alphagenome"
        ) from e
    return dna_client.create(api_key), dna_client


def to_common_grid(values: np.ndarray, resolution: int) -> np.ndarray:
    """Crop+pool an AlphaGenome output (n_bins, n_tracks) covering the whole
    524,288 bp tile onto the common 896 x 128 bp grid.

    Cropping happens at the model's own resolution first, then mean-pooling
    reaches 128 bp — the same order used for the targets, so the two stay on
    identical coordinates.
    """
    n_bins = values.shape[0]
    span = n_bins * resolution
    if span != TILE_LEN:
        raise ValueError(
            f"AlphaGenome returned {n_bins} bins x {resolution} bp = {span} bp, "
            f"expected the full {TILE_LEN} bp tile")
    if ENF_BIN_SIZE % resolution:
        raise ValueError(
            f"resolution {resolution} bp does not divide the {ENF_BIN_SIZE} bp "
            "common grid; cannot pool without resampling")

    keep = ENF_SPAN // resolution           # bins covering the central 114,688 bp
    off = (n_bins - keep) // 2
    cropped = values[off: off + keep]

    pool = ENF_BIN_SIZE // resolution
    out = cropped.reshape(ENF_BINS, pool, cropped.shape[-1]).astype(np.float64).mean(axis=1)
    assert out.shape[0] == ENF_BINS
    return out


def run_inference(args):
    sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))
    from genome_tiler import tile_genome
    from tqdm import tqdm
    from alphagenome.data import genome

    client, dna_client = get_client()

    data_root = Path(args.data_root)
    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")
    intervals = tile_genome(fasta_path)[args.split]
    targets = np.load(Path(args.targets_dir) / f"{args.split}_targets.npy", mmap_mode="r")
    assert targets.shape[0] == len(intervals), (
        f"{targets.shape[0]} target rows vs {len(intervals)} intervals — "
        "the tiling used for the transfer model and here must match")
    if args.limit:
        intervals = intervals[:args.limit]
    n_targets = targets.shape[2]

    pearson = None
    track_meta = None
    n_failed = 0
    failures = []

    for idx, iv in enumerate(tqdm(intervals, desc="AlphaGenome inference")):
        assert iv.end - iv.start == TILE_LEN, f"tile {iv.name} is not {TILE_LEN} bp"
        interval = genome.Interval(chromosome=iv.chrom, start=iv.start, end=iv.end)

        out = None
        for attempt in range(args.max_retries):
            try:
                out = client.predict_interval(
                    interval=interval,
                    organism=dna_client.Organism.HOMO_SAPIENS,
                    requested_outputs=[dna_client.OutputType.CHIP_TF],
                    ontology_terms=None,          # None = every available track
                )
                break
            except Exception as e:                      # transient API/network error
                if attempt == args.max_retries - 1:
                    n_failed += 1
                    msg = f"{type(e).__name__}: {e}"
                    failures.append({"interval": iv.name, "error": msg})
                    # surface the first failure immediately; a silent skip loop
                    # would otherwise look like progress
                    if n_failed == 1:
                        print(f"\n[error] {iv.name}: {msg[:500]}", file=sys.stderr)
                else:
                    time.sleep(2 ** attempt)
        if out is None:
            continue

        td = out.chip_tf
        if track_meta is None:
            track_meta = td.metadata.reset_index(drop=True)
            pearson = StreamingPearson(n_targets, td.values.shape[-1])
            print(f"[alphagenome] CHIP_TF tracks={td.values.shape[-1]} "
                  f"resolution={td.resolution} bp bins={td.values.shape[0]}")

        prd = to_common_grid(np.asarray(td.values), int(td.resolution))
        tgt = targets_to_enformer_grid(targets[idx])
        pearson.update(tgt, prd)

    if pearson is None:
        raise SystemExit("every AlphaGenome request failed — see errors above")
    if n_failed:
        print(f"[warn] {n_failed}/{len(intervals)} intervals failed and were skipped")
    return pearson.result(), track_meta, failures


def main():
    args = parse_args()
    out = Path(args.results_dir); out.mkdir(exist_ok=True)
    split = args.split

    print(f"[alphagenome] hosted API — running inference over {split} intervals")
    t0 = time.time()
    pmat, track_meta, failures = run_inference(args)
    elapsed = time.time() - t0

    np.save(out / f"alphagenome_pearson_matrix_{split}.npy", pmat)
    track_meta.to_csv(out / f"alphagenome_tracks_{split}.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / f"alphagenome_failures_{split}.csv", index=False)

    # select_best_match wants a frame indexed by track index, with factor/cell
    # columns; AlphaGenome's metadata uses different column names, so map them.
    meta = track_meta.copy()
    meta["track_index"] = np.arange(len(meta))
    meta["identifier"] = meta.get("name", meta.get("track_name", ""))
    meta["description"] = meta.get("name", "")
    meta["factor"] = meta.get("transcription_factor", meta.get("name", ""))
    meta["cell_tissue"] = meta.get("biosample_name", meta.get("ontology_curie", ""))
    meta = meta.set_index("track_index")
    elig_idx = list(range(len(meta)))

    metrics = pd.read_csv(Path(args.targets_dir) / f"{split}_per_track_metrics.csv")
    target_meta = load_crc_target_meta(metrics["bw_filename"].tolist(),
                                       Path(args.data_root) / "TF_184tracks.csv")
    best = select_best_match(pmat, elig_idx, meta, target_meta)
    best = best.rename(columns=lambda c: c.replace("borzoi", "alphagenome"))
    best.to_csv(out / f"alphagenome_best_match_{split}.csv", index=False)

    print(f"\n[{split}] tracks: {len(elig_idx)}  targets: {pmat.shape[0]}  "
          f"elapsed: {elapsed/60:.1f} min")
    print(f"  mean best-match Pearson R : {best['pearson_r'].mean():.4f}")
    print(f"  median                    : {best['pearson_r'].median():.4f}")
    print(f"  min / max                 : {best['pearson_r'].min():.4f} / {best['pearson_r'].max():.4f}")
    print(f"Wrote {out}/alphagenome_pearson_matrix_{split}.npy  {pmat.shape}")
    print(f"Wrote {out}/alphagenome_best_match_{split}.csv")


if __name__ == "__main__":
    main()
