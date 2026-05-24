import torch
from genome_tiler import tile_genome
from bigwig_loader import BigWigLoader
from dataset import GenomicDataset
from model import BorzoiTransferModel
from train import train

FASTA    = "path/to/hg38.fa"
BW_FILES = ["path/to/track1.bw", "path/to/track2.bw"]   # your BigWig files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

splits = tile_genome(FASTA)

bw_loader = BigWigLoader(BW_FILES, bin_size=32)
train_ds = GenomicDataset(FASTA, splits["train"], bigwig_loader=bw_loader)

model = BorzoiTransferModel(n_output_tracks=len(BW_FILES), device=device)

train(model, train_ds, val_dataset=None, n_epochs=3)

train_ds.close()
