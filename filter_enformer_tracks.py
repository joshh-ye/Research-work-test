"""
Filter Enformer human-head output tracks down to CRC-relevant ChIP (TF /
DNA-binding-protein) tracks and emit reproducibility artifacts.

Strategy:
  1. Keep tracks whose description assay prefix is 'CHIP'.
  2. Exclude histone-mark tracks (H2A/H2B/H3/H4 terms, shared term list with the
     Borzoi baseline), with the matched term logged as the exclusion reason.

Outputs:
  results_baseline_enformer/included_enformer_tracks.csv
  results_baseline_enformer/excluded_enformer_tracks.csv

Usage:
  python filter_enformer_tracks.py [--data-root ./enformer_data]
                                   [--out-dir ./results_baseline_enformer]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from enformer_baseline_utils import (
    load_enformer_targets, filter_enformer_tracks, ENF_BINS, ENF_SEQ_LEN,
)
from borzoi_baseline_utils import is_histone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./enformer_data")
    p.add_argument("--out-dir",   default="./results_baseline_enformer")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_enformer_targets(Path(args.data_root) / "targets_human.txt")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    included, excluded = filter_enformer_tracks(df)

    cols = ["track_index", "identifier", "description", "assay", "factor",
            "cell_tissue", "reason"]
    included[cols].to_csv(out / "included_enformer_tracks.csv", index=False)
    excluded[cols].to_csv(out / "excluded_enformer_tracks.csv", index=False)

    assay_counts = included["assay"].value_counts().to_dict()
    n_chip = (df["description"].str.split(":").str[0].str.upper() == "CHIP").sum()
    n_hist = df["description"].apply(lambda d: is_histone(d) is not None).sum()

    print("=" * 64)
    print("Enformer track filtering report")
    print("=" * 64)
    print(f"  human-head tracks         : {len(df)}")
    print(f"  assay == CHIP             : {n_chip}")
    print(f"  histone (H2A/H2B/H3/H4)   : {n_hist}")
    print(f"  eligible (ChIP, non-hist) : {len(included)}   {assay_counts}")
    print(f"  excluded                  : {len(excluded)}")
    print(f"  model geometry            : {ENF_SEQ_LEN} bp in -> {ENF_BINS} bins x 128 bp")
    print("-" * 64)
    print("Most frequent eligible factors:")
    print(included["factor"].value_counts().head(10).to_string())
    print("-" * 64)
    print(f"Wrote {out/'included_enformer_tracks.csv'} ({len(included)} rows)")
    print(f"Wrote {out/'excluded_enformer_tracks.csv'} ({len(excluded)} rows)")

    assert included["track_index"].max() < len(df)
    print("Validation: all eligible indices within Enformer human output dim — OK")


if __name__ == "__main__":
    main()
