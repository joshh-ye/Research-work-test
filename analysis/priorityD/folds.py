"""Chromosome splits. 'orig' = project split (val chr1+chr8, test chr9+chr22, others train; chrX/chrY tiles are train as in
borzoi_code/genome_tiler.py). Folds F2-F4 hold out other chromosomes (disjoint test sets); validation chromosomes per fold
are used only for early stopping / lambda / track selection."""
import numpy as np, pandas as pd
from common import CACHE, tile_table

FOLDS = {
    "orig": dict(val=["chr1", "chr8"], test=["chr9", "chr22"]),
    "F2": dict(val=["chr3"], test=["chr6", "chr21"]),
    "F3": dict(val=["chr4"], test=["chr11", "chr19", "chr20"]),
    "F4": dict(val=["chr5"], test=["chr7", "chr18"]),
}


def split_gids(name, tt=None):
    tt = tile_table() if tt is None else tt
    f = FOLDS[name]
    assert not set(f["val"]) & set(f["test"])
    te = tt[tt.chrom.isin(f["test"])].gid.values; va = tt[tt.chrom.isin(f["val"])].gid.values
    tr = tt[~tt.chrom.isin(f["val"] + f["test"])].gid.values
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    return tr, va, te
