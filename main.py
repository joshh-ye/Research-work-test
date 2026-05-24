"""
Borzoi Transfer Learning Pipeline

Headless script converted from the Colab notebook.
Runs: environment check -> BigWig signal check -> DNA encoding check ->
      genome tiling -> dataset inspect -> model load -> train -> eval plots.

Usage:
    python main.py [--data-root ./borzoi_data] [--ckpt-dir ./borzoi_ckpt]
                   [--output-dir ./results] [--epochs 4] [--batch-size 1]
                   [--lr 1e-4] [--n-bw N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Must be set before any HuggingFace imports so the library reads the right cache dir.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "hf_cache"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SLURM / headless runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pyBigWig
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from borzoi_code/ when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "borzoi_code"))

from sequence_utils import one_hot_encode
from fasta_reader import FastaReader
from bigwig_loader import BigWigLoader
from genome_tiler import tile_genome
from dataset import GenomicDataset
from model import BorzoiTransferModel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borzoi transfer learning pipeline")
    p.add_argument("--data-root",   default="./borzoi_data",
                   help="Root directory containing hg38/ and CRC_TFs_bw/")
    p.add_argument("--ckpt-dir",    default="./borzoi_ckpt",
                   help="Directory for model checkpoints")
    p.add_argument("--output-dir",  default="./results",
                   help="Directory for output figures and eval arrays")
    p.add_argument("--epochs",      type=int,   default=4)
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--n-bw",        type=int,   default=None,
                   help="Limit BigWig tracks to first N (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2 — environment check
# ---------------------------------------------------------------------------

def check_environment(device: torch.device) -> None:
    print(f"NumPy   {np.__version__}")
    print(f"PyTorch {torch.__version__}")
    print(f"Device  {device}")


# ---------------------------------------------------------------------------
# Step 5 — BigWig signal check
# ---------------------------------------------------------------------------

def check_bigwig_signal(
    bw_files: list[str], out_dir: Path
) -> tuple[str | None, int | None, int | None]:
    bw_demo = bw_files[0]
    bw = pyBigWig.open(bw_demo)
    window, bin_size = 50_000, 32
    n_bins = window // bin_size

    found_chrom, found_start, found_end, vals = None, None, None, None
    for chrom, length in bw.chroms().items():
        if length < window:
            continue
        for start in range(0, min(length - window, 20_000_000), window):
            end = start + window
            v = bw.stats(chrom, start, end, type="mean", nBins=n_bins)
            v = np.array([x if x is not None else 0.0 for x in v], dtype=np.float32)
            if v.max() > 0:
                found_chrom, found_start, found_end, vals = chrom, start, end, v
                break
        if found_chrom:
            break
    bw.close()

    if not found_chrom:
        print("Warning: no BigWig signal found in first 20 Mb of any chromosome")
        return None, None, None

    print(f"Signal found: {found_chrom}:{found_start:,}-{found_end:,}  "
          f"max={vals.max():.3f}  nonzero={np.count_nonzero(vals)}/{n_bins}")

    genomic_pos = np.linspace(found_start, found_end, n_bins)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(genomic_pos, vals, alpha=0.7)
    ax.set_xlabel(f"Position on {found_chrom} (bp)")
    ax.set_ylabel("Mean signal")
    ax.set_title(f"BigWig signal — {found_chrom}:{found_start:,}-{found_end:,} "
                 f"[{Path(bw_demo).name}]")
    plt.tight_layout()
    fig.savefig(out_dir / "bigwig_signal.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir}/bigwig_signal.png")

    return found_chrom, found_start, found_end


# ---------------------------------------------------------------------------
# Step 4 — DNA encoding check
# ---------------------------------------------------------------------------

def check_dna_encoding(
    fasta_path: str, chrom: str, start: int, end: int, out_dir: Path
) -> None:
    fasta = FastaReader(fasta_path)
    seq = fasta.fetch(chrom, start, end)
    encoded = one_hot_encode(seq)
    print(f"One-hot shape: {encoded.shape}  (positions x 4)")

    snippet = encoded[:80].T
    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(snippet, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("Base position")
    ax.set_title("One-hot DNA encoding (first 80 bp of signal window)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "dna_encoding.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir}/dna_encoding.png")


# ---------------------------------------------------------------------------
# BigWig integrity filter
# ---------------------------------------------------------------------------

def filter_bad_bigwigs(bw_files: list[str]) -> list[str]:
    bad = []
    for p in bw_files:
        try:
            bw = pyBigWig.open(p)
            bw.chroms()
            bw.close()
        except Exception:
            bad.append(p)
    if bad:
        print(f"Skipping {len(bad)} corrupted BigWig(s): "
              f"{[Path(p).name for p in bad]}")
    return [p for p in bw_files if p not in bad]


# ---------------------------------------------------------------------------
# Step 6 — genome tiling plot
# ---------------------------------------------------------------------------

def plot_genome_splits(splits: dict, out_dir: Path) -> None:
    names  = list(splits.keys())
    counts = [len(splits[s]) for s in names]
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, counts, color=colors[: len(names)], width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of 524 kb windows")
    ax.set_title("Genome tiling — intervals per split")
    plt.tight_layout()
    fig.savefig(out_dir / "genome_splits.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir}/genome_splits.png")


# ---------------------------------------------------------------------------
# Step 7 — dataset sample plots
# ---------------------------------------------------------------------------

def plot_dataset_sample(train_ds: GenomicDataset, out_dir: Path) -> None:
    sample = train_ds[0]

    snippet = sample["sequence"][11_000:11_080].numpy().T
    fig, ax = plt.subplots(figsize=(14, 2))
    ax.imshow(snippet, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("Base position")
    ax.set_title("Sequence snippet (80 bp at position 11000)")
    plt.tight_layout()
    fig.savefig(out_dir / "dataset_sequence.png", dpi=150)
    plt.close(fig)

    tgt0 = sample["targets"][:, 0].numpy()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(range(len(tgt0)), tgt0, alpha=0.7, color="darkorange")
    ax.set_xlabel("Bin (32 bp each)")
    ax.set_ylabel("Signal")
    ax.set_title("Target signal — track 0, center region")
    plt.tight_layout()
    fig.savefig(out_dir / "dataset_target.png", dpi=150)
    plt.close(fig)

    print(f"  -> dataset_sequence.png, dataset_target.png")
    print(f"     sequence shape: {sample['sequence'].shape}  "
          f"targets shape: {sample['targets'].shape}")


# ---------------------------------------------------------------------------
# Step 9 — training (resumable)
# ---------------------------------------------------------------------------

def train_resumable(
    model: BorzoiTransferModel,
    train_dataset: GenomicDataset,
    val_dataset: GenomicDataset,
    ckpt_dir: Path,
    out_dir: Path,
    n_epochs: int = 4,
    batch_size: int = 1,
    lr: float = 1e-4,
) -> dict:
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    history: dict = {"train_loss": [], "val_loss": []}
    start_epoch = 0

    head_pt = ckpt_dir / "model_head.pt"
    if head_pt.exists():
        model.head.load_state_dict(torch.load(head_pt, map_location=model.device))
        opt_pt = ckpt_dir / "optimizer.pt"
        if opt_pt.exists():
            optimizer.load_state_dict(torch.load(opt_pt, map_location=model.device))
        ep_txt = ckpt_dir / "epoch.txt"
        if ep_txt.exists():
            start_epoch = int(ep_txt.read_text())
        hist_json = ckpt_dir / "history.json"
        if hist_json.exists():
            history = json.loads(hist_json.read_text())
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found — starting from scratch")

    last_val_preds: list = []
    last_val_targets: list = []

    for epoch in range(start_epoch, n_epochs):
        # train
        model.head.train()
        epoch_losses: list[float] = []
        pbar = tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                    desc=f"Epoch {epoch+1}/{n_epochs} [train]")
        for batch in pbar:
            seq  = batch["sequence"].to(model.device)
            tgt  = batch["targets"].to(model.device)
            pred = model(seq)
            loss = F.poisson_nll_loss(pred, tgt, log_input=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        history["train_loss"].append(float(np.mean(epoch_losses)))

        # val
        model.head.eval()
        val_losses: list[float] = []
        ep_preds:   list = []
        ep_targets: list = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(val_dataset, batch_size=batch_size),
                              desc=f"Epoch {epoch+1}/{n_epochs} [val]"):
                seq  = batch["sequence"].to(model.device)
                tgt  = batch["targets"].to(model.device)
                pred = model(seq)
                val_losses.append(
                    F.poisson_nll_loss(pred, tgt, log_input=False).item()
                )
                ep_preds.append(pred.cpu().numpy())
                ep_targets.append(tgt.cpu().numpy())
        history["val_loss"].append(float(np.mean(val_losses)))
        last_val_preds, last_val_targets = ep_preds, ep_targets

        print(f"Epoch {epoch+1}/{n_epochs}  "
              f"train={history['train_loss'][-1]:.4f}  "
              f"val={history['val_loss'][-1]:.4f}")

        # checkpoint every epoch
        torch.save(model.head.state_dict(), head_pt)
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        (ckpt_dir / "epoch.txt").write_text(str(epoch + 1))
        (ckpt_dir / "history.json").write_text(json.dumps(history))
        print(f"  Checkpoint saved (epoch {epoch+1})")

    if last_val_preds:
        np.save(out_dir / "eval_preds.npy",
                np.concatenate(last_val_preds, axis=0))
        np.save(out_dir / "eval_targets.npy",
                np.concatenate(last_val_targets, axis=0))
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f)
        print(f"Saved eval arrays and history to {out_dir}")

    return history


# ---------------------------------------------------------------------------
# Step 10 — evaluation plots
# ---------------------------------------------------------------------------

def plot_evaluation(out_dir: Path) -> None:
    preds   = np.load(out_dir / "eval_preds.npy")
    targets = np.load(out_dir / "eval_targets.npy")
    n_intervals, center_bins, n_tracks = preds.shape
    print(f"Eval arrays: {preds.shape}  (intervals x bins x tracks)")

    with open(out_dir / "history.json") as f:
        hist = json.load(f)

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax.plot(ep, hist["train_loss"], marker="o", label="train")
    if hist["val_loss"]:
        ax.plot(ep, hist["val_loss"], marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Poisson NLL Loss")
    ax.set_title("Training curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # 2. Per-track Pearson R
    p_flat    = preds.reshape(-1, n_tracks)
    t_flat    = targets.reshape(-1, n_tracks)
    pearson_r = np.array(
        [pearsonr(p_flat[:, i], t_flat[:, i])[0] for i in range(n_tracks)]
    )
    order = np.argsort(pearson_r)[::-1]

    fig, ax = plt.subplots(figsize=(max(8, n_tracks * 0.2), 4))
    ax.bar(range(n_tracks), pearson_r[order], width=1.0, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Track (sorted by R)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"Per-track Pearson R  (mean={pearson_r.mean():.3f})")
    ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(out_dir / "pearson_r.png", dpi=150)
    plt.close(fig)

    # 3. Predicted vs actual — top-3 tracks, first interval
    top3 = order[:3]
    x    = np.arange(center_bins)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, ti in zip(axes, top3):
        r, _ = pearsonr(preds[0, :, ti], targets[0, :, ti])
        ax.plot(x, targets[0, :, ti], label="actual",    alpha=0.8, linewidth=0.8)
        ax.plot(x, preds[0,   :, ti], label="predicted", alpha=0.8, linewidth=0.8,
                linestyle="--")
        ax.set_ylabel("signal")
        ax.set_title(f"Track {ti}  (R={r:.3f})")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Genomic bin (32 bp)")
    fig.suptitle("Predicted vs actual — top-3 tracks, interval 0")
    plt.tight_layout()
    fig.savefig(out_dir / "pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # 4. Scatter — mean signal per interval x track
    p_mean   = preds.mean(axis=1).ravel()
    t_mean   = targets.mean(axis=1).ravel()
    r_all, _ = pearsonr(p_mean, t_mean)
    fig, ax  = plt.subplots(figsize=(5, 5))
    ax.scatter(t_mean, p_mean, s=4, alpha=0.3, rasterized=True)
    lim = max(t_mean.max(), p_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Actual mean signal")
    ax.set_ylabel("Predicted mean signal")
    ax.set_title(f"Predicted vs actual  (R={r_all:.3f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=150)
    plt.close(fig)

    print(f"Mean Pearson R across {n_tracks} tracks: {pearson_r.mean():.4f}")
    print(f"Figures saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    ckpt_dir  = Path(args.ckpt_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = str(data_root / "hg38" / "hg38.ml.fa")
    bw_files   = sorted(glob.glob(str(data_root / "CRC_TFs_bw" / "*.bw")))

    if not bw_files:
        sys.exit(f"No BigWig files found under {data_root}/CRC_TFs_bw/")

    # Step 2 — environment
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("=== Environment ===")
    check_environment(device)

    # Step 3 — paths
    print(f"\nFASTA : {fasta_path}")
    print(f"Tracks: {len(bw_files)} BigWig files")

    # Step 5 — BigWig signal
    print("\n=== BigWig signal check ===")
    chrom, start, end = check_bigwig_signal(bw_files, out_dir)

    # Step 4 — DNA encoding
    if chrom:
        print("\n=== DNA encoding check ===")
        check_dna_encoding(fasta_path, chrom, start, end, out_dir)

    # Filter corrupted BigWigs
    bw_files = filter_bad_bigwigs(bw_files)
    if args.n_bw:
        bw_files = bw_files[: args.n_bw]
        print(f"Using {len(bw_files)} BigWig tracks (--n-bw {args.n_bw})")

    # Step 6 — genome tiling
    print("\n=== Genome tiling ===")
    splits = tile_genome(fasta_path)
    for name, intervals in splits.items():
        print(f"  {name}: {len(intervals)} intervals")
    plot_genome_splits(splits, out_dir)

    # Step 7 — dataset sample
    print("\n=== Dataset sample ===")
    bw_loader = BigWigLoader(bw_files, bin_size=32)
    train_ds  = GenomicDataset(fasta_path, splits["train"], bigwig_loader=bw_loader)
    val_ds    = GenomicDataset(
        fasta_path, splits["val"],
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
    )
    print(f"Train: {len(train_ds)} intervals   Val: {len(val_ds)} intervals")
    plot_dataset_sample(train_ds, out_dir)

    # Step 8 — load model
    print("\n=== Loading model ===")
    n_tracks = len(bw_files)
    model = BorzoiTransferModel(n_output_tracks=n_tracks, device=device)
    frozen    = sum(p.numel() for b in model.backbones for p in b.parameters())
    trainable = sum(p.numel() for p in model.head.parameters())
    print(f"Frozen backbone params : {frozen:,}")
    print(f"Trainable head params  : {trainable:,}")

    # Step 9 — train
    print("\n=== Training ===")
    train_resumable(
        model, train_ds, val_ds, ckpt_dir, out_dir,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # Step 10 — evaluation plots
    print("\n=== Evaluation plots ===")
    plot_evaluation(out_dir)


if __name__ == "__main__":
    main()
