import torch
from torch.utils.data import Dataset
from fasta_reader import FastaReader
from bigwig_loader import BigWigLoader
from sequence_utils import one_hot_encode, reverse_complement

class GenomicDataset(Dataset):
    def __init__(self, fasta_path, intervals, bigwig_loader=None,
                 seq_len=524288, center_bin_size=131072, bin_size=32):
        self.fasta = FastaReader(fasta_path)
        self.intervals = intervals
        self.bigwig_loader = bigwig_loader
        self.seq_len = seq_len
        self.center_bin_size = center_bin_size
        self.bin_size = bin_size

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        iv = self.intervals[idx]

        center = (iv.start + iv.end) // 2
        half   = self.seq_len // 2
        seq_str = self.fasta.fetch(iv.chrom, center - half, center + half)

        sequence = torch.from_numpy(one_hot_encode(seq_str))  # (seq_len, 4)

        targets = None
        if self.bigwig_loader is not None:
            half_c = self.center_bin_size // 2
            targets_np = self.bigwig_loader.load(
                iv.chrom, center - half_c, center + half_c
            )  # (n_center_bins, n_tracks)
            targets = torch.from_numpy(targets_np)

        return {"sequence": sequence, "targets": targets, "interval": str(iv)}

    def close(self):
        self.fasta.close()
        if self.bigwig_loader:
            self.bigwig_loader.close()
