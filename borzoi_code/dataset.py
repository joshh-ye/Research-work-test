import numpy as np
import torch
from torch.utils.data import Dataset
from fasta_reader import FastaReader
from bigwig_loader import BigWigLoader
from sequence_utils import one_hot_encode, reverse_complement

class GenomicDataset(Dataset):
    def __init__(self, fasta_path, intervals, bigwig_loader=None,
                 seq_len=524288, center_bin_size=131072, bin_size=32,
                 training=False, max_shift_bp=128, rc_prob=0.5):
        self.fasta = FastaReader(fasta_path)
        self.intervals = intervals
        self.bigwig_loader = bigwig_loader
        self.seq_len = seq_len
        self.center_bin_size = center_bin_size
        self.bin_size = bin_size
        # Augmentation (train split only). Borzoi-standard reverse-complement +
        # small random genomic shift; both keep sequence and targets aligned.
        self.training = training
        self.max_shift_bp = max_shift_bp
        self.rc_prob = rc_prob

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        iv = self.intervals[idx]

        center = (iv.start + iv.end) // 2

        # Random shift: move sequence and target windows together so they stay aligned.
        do_rc = False
        if self.training:
            if self.max_shift_bp > 0:
                # snap shift to whole bins so target bins stay aligned to the grid
                max_bins = self.max_shift_bp // self.bin_size
                if max_bins > 0:
                    center += int(np.random.randint(-max_bins, max_bins + 1)) * self.bin_size
            do_rc = np.random.rand() < self.rc_prob

        half   = self.seq_len // 2
        seq_str = self.fasta.fetch(iv.chrom, center - half, center + half)

        targets_np = None
        if self.bigwig_loader is not None:
            half_c = self.center_bin_size // 2
            targets_np = self.bigwig_loader.load(
                iv.chrom, center - half_c, center + half_c
            )  # (n_center_bins, n_tracks)

        if do_rc:
            # Reverse-complement the sequence and reverse the target bins to match.
            seq_str = reverse_complement(seq_str)
            if targets_np is not None:
                targets_np = targets_np[::-1].copy()

        sequence = torch.from_numpy(one_hot_encode(seq_str))  # (seq_len, 4)
        targets = torch.from_numpy(targets_np) if targets_np is not None else None

        return {"sequence": sequence, "targets": targets, "interval": str(iv)}

    def close(self):
        self.fasta.close()
        if self.bigwig_loader:
            self.bigwig_loader.close()
