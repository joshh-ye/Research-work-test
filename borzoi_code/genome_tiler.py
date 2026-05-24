from fasta_reader import FastaReader
from dataclasses import dataclass

@dataclass
class Interval:
    chrom: str
    start: int
    end: int
    name: str = "."

def tile_genome(fasta_path: str, seq_len: int = 524288,
                val_chroms={"chr1", "chr8"}, test_chroms={"chr9", "chr22"}):
    reader = FastaReader(fasta_path)
    chrom_sizes = reader.chrom_sizes()
    reader.close()

    splits = {"train": [],
              "val": [],
              "test": []}
    
    for chrom, length in chrom_sizes.items():
        if length < 1_000_000:
            continue
        if any(x in chrom for x in ("Un", "random", "alt", "fix", "M")):
            continue

    
        if chrom in val_chroms:                                                              
            split = "val"                                       
        elif chrom in test_chroms:
            split = "test"
        else:                                                                                
            split = "train"

        pos = 0
        while pos + seq_len <= length:
            splits[split].append(
                Interval(
                    chrom=chrom,
                    start=pos,
                    end=pos+seq_len,
                    name=f"{chrom}_{pos}_{pos+seq_len}"
              ))
            pos += seq_len   # non-overlapping windows

    for split, ivs in splits.items():
        print(f"{split}: {len(ivs)} intervals")

    return splits