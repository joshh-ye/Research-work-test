"""
Filter Borzoi output tracks down to CRC-relevant ChIP (TF / DNA-binding-protein)
tracks and emit reproducibility artifacts.

Strategy:
  1. Keep track_index in [2186, 6069] inclusive (Borzoi output channel index,
     0-based, == row position in targets_human.csv).
  2. Keep descriptions containing 'chip' (case-insensitive).
  3. Exclude histone-mark tracks (H2A/H2B/H3/H4 terms), case-insensitive, with the
     matched term logged as the exclusion reason.

Outputs:
  results_baseline_filtered/included_borzoi_tracks.csv
  results_baseline_filtered/excluded_borzoi_tracks.csv

Usage:
  python filter_borzoi_tracks.py [--data-root ./borzoi_data]
                                 [--out-dir ./results_baseline_filtered]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from borzoi_baseline_utils import (
    INDEX_LO, INDEX_HI, load_borzoi_targets, filter_borzoi_tracks, is_histone,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./borzoi_data")
    p.add_argument("--out-dir",   default="./results_baseline_filtered")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_borzoi_targets(Path(args.data_root) / "targets_human.csv")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    included, excluded = filter_borzoi_tracks(df)

    cols = ["track_index", "identifier", "description", "assay", "factor",
            "cell_tissue", "reason"]
    included[cols].to_csv(out / "included_borzoi_tracks.csv", index=False)
    excluded[cols].to_csv(out / "excluded_borzoi_tracks.csv", index=False)

    # --- pre-run report -----------------------------------------------------
    in_range = df[(df["track_index"] >= INDEX_LO) & (df["track_index"] <= INDEX_HI)]
    n_chip = in_range["description"].str.contains("chip", case=False, na=False).sum()
    n_hist = in_range["description"].apply(lambda d: is_histone(d) is not None).sum()

    print("=" * 64)
    print(f"Borzoi track filtering report  (range {INDEX_LO}-{INDEX_HI} inclusive)")
    print("=" * 64)
    print(f"  tracks in range 2186-6069 : {len(in_range)}")
    print(f"  containing 'CHIP'         : {n_chip}")
    print(f"  histone (H2A/H2B/H3/H4)   : {n_hist}")
    print(f"  eligible (ChIP, non-hist) : {len(included)}")
    print(f"  excluded                  : {len(excluded)}")
    print("-" * 64)
    print("Sample tracks at range boundaries (index | identifier | description):")
    for idx in (2186, 2187, 6068, 6069):
        row = df[df["track_index"] == idx].iloc[0]
        print(f"  {idx} | {row['identifier']} | {row['description'][:70]}")
    print("-" * 64)
    print(f"Wrote {out/'included_borzoi_tracks.csv'} ({len(included)} rows)")
    print(f"Wrote {out/'excluded_borzoi_tracks.csv'} ({len(excluded)} rows)")

    # assertion: all eligible indices are valid output channels
    assert included["track_index"].max() < len(df)
    assert (included["track_index"] >= INDEX_LO).all()
    print("Validation: all eligible indices within Borzoi output dim — OK")


if __name__ == "__main__":
    main()
