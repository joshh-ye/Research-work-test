"""
Joshua Ye
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

import csv
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
    p.add_argument("--num-workers", type=int,   default=8,
                   help="DataLoader worker processes for parallel BigWig I/O")
    # Head capacity / regularization
    p.add_argument("--head-hidden",  type=int,   default=1024,
                   help="Hidden width of the MLP head")
    p.add_argument("--head-dropout", type=float, default=0.2,
                   help="Dropout in the MLP head")
    # Training schedule
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Adam weight decay")
    p.add_argument("--patience",     type=int,   default=3,
                   help="Early-stopping patience on val loss (0 disables)")
    # Augmentation
    p.add_argument("--max-shift-bp", type=int,   default=128,
                   help="Max random genomic shift for train augmentation (0 disables)")
    p.add_argument("--no-rc",        action="store_true",
                   help="Disable reverse-complement augmentation")
    # Per-track loss balancing
    p.add_argument("--balance-tracks", action="store_true",
                   help="Weight the per-track loss by inverse mean signal so sparse "
                        "tracks are not drowned out by high-signal tracks")
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
    test_dataset: GenomicDataset,
    ckpt_dir: Path,
    out_dir: Path,
    n_epochs: int = 4,
    batch_size: int = 1,
    lr: float = 1e-4,
    num_workers: int = 16,
    weight_decay: float = 1e-4,
    patience: int = 3,
    track_weights: torch.Tensor | None = None,
) -> dict:
    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    if track_weights is not None:
        track_weights = track_weights.to(model.device)

    def weighted_poisson(pred, tgt):
        # Per-track-weighted Poisson NLL so sparse tracks contribute comparably.
        if track_weights is None:
            return F.poisson_nll_loss(pred, tgt, log_input=False)
        per = F.poisson_nll_loss(pred, tgt, log_input=False, reduction="none")
        return (per * track_weights).mean()

    history: dict = {"train_loss": [], "val_loss": []}
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    head_pt = ckpt_dir / "model_head.pt"
    if head_pt.exists():
        ckpt_sd = torch.load(head_pt, map_location=model.device)
        try:
            model.head.load_state_dict(ckpt_sd)
        except RuntimeError as e:
            print(f"  WARNING: checkpoint head is incompatible with current model "
                  f"architecture — starting head from scratch.\n  ({e})")
            # Rename rather than delete so the old checkpoint is recoverable.
            head_pt.rename(head_pt.with_suffix(".pt.bak"))
            (ckpt_dir / "model_head_best.pt").rename(
                (ckpt_dir / "model_head_best.pt.bak")
            ) if (ckpt_dir / "model_head_best.pt").exists() else None
        opt_pt = ckpt_dir / "optimizer.pt"
        if opt_pt.exists():
            optimizer.load_state_dict(torch.load(opt_pt, map_location=model.device))
        ep_txt = ckpt_dir / "epoch.txt"
        if ep_txt.exists():
            start_epoch = int(ep_txt.read_text())
        hist_json = ckpt_dir / "history.json"
        if hist_json.exists():
            history = json.loads(hist_json.read_text())
        best_json = ckpt_dir / "best_val.json"
        if best_json.exists():
            best_val_loss = json.loads(best_json.read_text())["val_loss"]
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found — starting from scratch")

    last_val_preds: list = []
    last_val_targets: list = []

    dl_context = "spawn" if num_workers > 0 else None

    # Data-loading workers never touch the GPU; hiding it stops the CUDA driver
    # from initialising a context in every spawn process, which wastes GPU memory
    # and can crash after a long training run when driver state is degraded.
    def _worker_init(worker_id: int) -> None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    worker_init_fn = _worker_init if num_workers > 0 else None

    for epoch in range(start_epoch, n_epochs):
        # train
        model.head.train()
        epoch_losses: list[float] = []
        pbar = tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, multiprocessing_context=dl_context,
                               worker_init_fn=worker_init_fn),
                    desc=f"Epoch {epoch+1}/{n_epochs} [train]")
        for batch in pbar:
            seq  = batch["sequence"].to(model.device)
            tgt  = batch["targets"].to(model.device)
            pred = model(seq)
            loss = weighted_poisson(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        history["train_loss"].append(float(np.mean(epoch_losses)))

        # Free the caching allocator's reserved-but-idle GPU blocks before
        # spawning validation workers.  After a full training epoch the
        # allocator can hold a large fragmented pool; releasing it gives the
        # CUDA driver a clean state and reduces the chance of OOM-killing the
        # new worker processes during their startup phase.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # val
        model.head.eval()
        val_losses: list[float] = []
        ep_preds:   list = []
        ep_targets: list = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(val_dataset, batch_size=batch_size,
                                         num_workers=num_workers, multiprocessing_context=dl_context,
                                         worker_init_fn=worker_init_fn),
                              desc=f"Epoch {epoch+1}/{n_epochs} [val]"):
                seq  = batch["sequence"].to(model.device)
                tgt  = batch["targets"].to(model.device)
                pred = model(seq)
                val_losses.append(weighted_poisson(pred, tgt).item())
                ep_preds.append(pred.cpu().numpy())
                ep_targets.append(tgt.cpu().numpy())
        history["val_loss"].append(float(np.mean(val_losses)))
        last_val_preds, last_val_targets = ep_preds, ep_targets

        print(f"Epoch {epoch+1}/{n_epochs}  "
              f"train={history['train_loss'][-1]:.4f}  "
              f"val={history['val_loss'][-1]:.4f}")

        # LR schedule on val loss
        current_val = history["val_loss"][-1]
        scheduler.step(current_val)

        # best checkpoint + early stopping
        if current_val < best_val_loss:
            best_val_loss = current_val
            epochs_no_improve = 0
            torch.save(model.head.state_dict(), ckpt_dir / "model_head_best.pt")
            (ckpt_dir / "best_val.json").write_text(
                json.dumps({"epoch": epoch + 1, "val_loss": best_val_loss})
            )
            print(f"  New best checkpoint (epoch {epoch+1}, val={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # save interval-0 predictions for epoch-progression plots
        np.save(out_dir / f"val_preds_ep{epoch+1}_int0.npy", ep_preds[0][0])

        # checkpoint every epoch
        torch.save(model.head.state_dict(), head_pt)
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        (ckpt_dir / "epoch.txt").write_text(str(epoch + 1))
        (ckpt_dir / "history.json").write_text(json.dumps(history))
        print(f"  Checkpoint saved (epoch {epoch+1})")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"  Early stopping: no val improvement for {patience} epochs")
            break

    # If the loop didn't run (already at n_epochs) but val arrays are missing,
    # run one validation pass to generate them.
    if not last_val_preds and not (out_dir / "val_preds.npy").exists():
        print("Training already complete — running validation pass for eval arrays...")
        model.head.eval()
        val_losses, last_val_preds, last_val_targets = [], [], []
        with torch.no_grad():
            for batch in tqdm(DataLoader(val_dataset, batch_size=batch_size,
                                         num_workers=num_workers, multiprocessing_context=dl_context,
                                         worker_init_fn=worker_init_fn),
                              desc="Final val eval"):
                seq  = batch["sequence"].to(model.device)
                tgt  = batch["targets"].to(model.device)
                pred = model(seq)
                loss = F.poisson_nll_loss(pred, tgt, log_input=False)
                val_losses.append(loss.item())
                last_val_preds.append(pred.cpu().numpy())
                last_val_targets.append(tgt.cpu().numpy())
        print(f"Final val loss: {float(np.mean(val_losses)):.4f}")

    if last_val_preds:
        np.save(out_dir / "val_preds.npy",
                np.concatenate(last_val_preds, axis=0))
        np.save(out_dir / "val_targets.npy",
                np.concatenate(last_val_targets, axis=0))
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f)
        print(f"Saved val arrays and history to {out_dir}")

    # Test set evaluation on best checkpoint (run once at the end)
    if not (out_dir / "test_preds.npy").exists():
        print("\nRunning test set evaluation...")
        best_ckpt = ckpt_dir / "model_head_best.pt"
        if best_ckpt.exists():
            model.head.load_state_dict(
                torch.load(best_ckpt, map_location=model.device)
            )
            best_info = json.loads((ckpt_dir / "best_val.json").read_text())
            print(f"  Loaded best checkpoint "
                  f"(epoch {best_info['epoch']}, val={best_info['val_loss']:.4f})")
        else:
            print("  No best checkpoint found — using current weights")
        model.head.eval()
        test_preds_list: list = []
        test_targets_list: list = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=dl_context),
                              desc="Test eval"):
                seq  = batch["sequence"].to(model.device)
                tgt  = batch["targets"].to(model.device)
                pred = model(seq)
                test_preds_list.append(pred.cpu().numpy())
                test_targets_list.append(tgt.cpu().numpy())
        np.save(out_dir / "test_preds.npy",
                np.concatenate(test_preds_list, axis=0))
        np.save(out_dir / "test_targets.npy",
                np.concatenate(test_targets_list, axis=0))
        print(f"Saved test eval arrays to {out_dir}")
    else:
        print("Test eval arrays already exist, skipping")

    return history


# ---------------------------------------------------------------------------
# Step 10 — evaluation plots
# ---------------------------------------------------------------------------

def plot_epoch_progression(
    out_dir: Path, top3: np.ndarray, targets_int0: np.ndarray
) -> None:
    """Overlay val predictions from each epoch for the top-3 tracks, interval 0."""

    ep_files = sorted(
        out_dir.glob("val_preds_ep*_int0.npy"),
        key=lambda p: int(p.stem.split("ep")[1].split("_")[0]),
    )
    if not ep_files:
        return

    n_epochs = len(ep_files)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(n_epochs - 1, 1)) for i in range(n_epochs)]
    x = np.arange(targets_int0.shape[0])

    fig, axes = plt.subplots(len(top3), 1, figsize=(13, 4 * len(top3)), sharex=True)
    if len(top3) == 1:
        axes = [axes]

    for ax, ti in zip(axes, top3):
        ax.plot(x, targets_int0[:, ti], color="black", linewidth=1.2,
                label="actual", zorder=10)
        for ep_idx, ep_file in enumerate(ep_files):
            ep_preds = np.load(ep_file)  # (center_bins, n_tracks)
            label = f"ep{ep_idx + 1}" if ep_idx in (0, n_epochs - 1) else None
            ax.plot(x, ep_preds[:, ti], color=colors[ep_idx], alpha=0.6,
                    linewidth=0.7, label=label)
        ax.set_ylabel("signal")
        ax.set_title(f"Track {ti}")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Genomic bin (32 bp)")
    fig.suptitle("Val predictions across epochs — top-3 tracks, interval 0")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_epochs))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Epoch", shrink=0.6)
    plt.tight_layout()
    fig.savefig(out_dir / "pred_vs_actual_epochs.png", dpi=150)
    plt.close(fig)
    print(f"  -> pred_vs_actual_epochs.png")


def plot_evaluation(
    out_dir: Path,
    bw_files: list[str] | None = None,
    split: str = "val",
) -> None:
    preds   = np.load(out_dir / f"{split}_preds.npy")
    targets = np.load(out_dir / f"{split}_targets.npy")
    n_intervals, center_bins, n_tracks = preds.shape
    print(f"{split} eval arrays: {preds.shape}  (intervals x bins x tracks)")

    # 1. Loss curve (val only)
    if split == "val":
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
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

    fig, ax = plt.subplots(figsize=(max(8, n_tracks * 0.05 + 2), 4))
    ax.bar(range(n_tracks), pearson_r[order], width=1.0, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Track (sorted by R)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"{split} per-track Pearson R  (mean={pearson_r.mean():.3f})")
    ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(out_dir / f"{split}_pearson_r.png", dpi=150)
    plt.close(fig)

    # 3. Per-track metrics CSV
    csv_path = out_dir / f"{split}_per_track_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_idx", "bw_filename", "pearson_r"])
        for i in range(n_tracks):
            fname = Path(bw_files[i]).name if bw_files else str(i)
            writer.writerow([i, fname, f"{pearson_r[i]:.6f}"])
    print(f"  -> {split}_per_track_metrics.csv  (mean R={pearson_r.mean():.4f})")

    # 4. Predicted vs actual — top-3 tracks, first interval
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
    fig.suptitle(f"{split} predicted vs actual — top-3 tracks, interval 0")
    plt.tight_layout()
    fig.savefig(out_dir / f"{split}_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # 5. Scatter — mean signal per interval x track
    p_mean   = preds.mean(axis=1).ravel()
    t_mean   = targets.mean(axis=1).ravel()
    r_all, _ = pearsonr(p_mean, t_mean)
    fig, ax  = plt.subplots(figsize=(5, 5))
    ax.scatter(t_mean, p_mean, s=4, alpha=0.3, rasterized=True)
    lim = max(t_mean.max(), p_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Actual mean signal")
    ax.set_ylabel("Predicted mean signal")
    ax.set_title(f"{split} predicted vs actual  (R={r_all:.3f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / f"{split}_scatter.png", dpi=150)
    plt.close(fig)

    print(f"Mean Pearson R across {n_tracks} tracks ({split}): {pearson_r.mean():.4f}")
    print(f"Figures saved to {out_dir}")

    # 6. Epoch progression (val only)
    if split == "val":
        plot_epoch_progression(out_dir, top3, targets[0])


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

    # Lock in the exact track list so resumes always use the same files.
    # On first run: save the list. On resume: load the saved list.
    bw_list_path = ckpt_dir / "bw_files.json"
    if bw_list_path.exists():
        bw_files = json.loads(bw_list_path.read_text())
        print(f"Loaded track list from checkpoint ({len(bw_files)} tracks)")
    else:
        bw_list_path.write_text(json.dumps(bw_files, indent=2))
        print(f"Saved track list to checkpoint ({len(bw_files)} tracks)")

    # Step 6 — genome tiling
    print("\n=== Genome tiling ===")
    splits = tile_genome(fasta_path)
    for name, intervals in splits.items():
        print(f"  {name}: {len(intervals)} intervals")
    plot_genome_splits(splits, out_dir)

    # Step 7 — dataset sample
    print("\n=== Dataset sample ===")
    sample_ds = GenomicDataset(
        fasta_path,
        splits["train"],
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
    )
    print(f"Train: {len(splits['train'])} intervals   Val: {len(splits['val'])} intervals   "
          f"Test: {len(splits['test'])} intervals")
    plot_dataset_sample(sample_ds, out_dir)
    del sample_ds

    # Create fresh datasets for DataLoader workers. Do not reuse the sample dataset above,
    # because plotting accesses sample_ds[0] and opens FASTA/BigWig handles, which are not
    # picklable under multiprocessing_context='spawn'.
    train_ds = GenomicDataset(
        fasta_path,
        splits["train"],
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
        training=True,
        max_shift_bp=args.max_shift_bp,
        rc_prob=0.0 if args.no_rc else 0.5,
    )
    val_ds = GenomicDataset(
        fasta_path,
        splits["val"],
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
    )
    test_ds = GenomicDataset(
        fasta_path,
        splits["test"],
        bigwig_loader=BigWigLoader(bw_files, bin_size=32),
    )

    # Step 8 — load model
    print("\n=== Loading model ===")
    n_tracks = len(bw_files)
    model = BorzoiTransferModel(
        n_output_tracks=n_tracks, device=device,
        hidden=args.head_hidden, dropout=args.head_dropout,
    )
    frozen    = sum(p.numel() for b in model.backbones for p in b.parameters())
    trainable = sum(p.numel() for p in model.head.parameters())
    print(f"Frozen backbone params : {frozen:,}")
    print(f"Trainable head params  : {trainable:,}")

    # Optional per-track loss weights (inverse mean signal), cached for resume.
    track_weights = None
    if args.balance_tracks:
        tw_path = ckpt_dir / "track_weights.npy"
        if tw_path.exists():
            weights_np = np.load(tw_path)
            print(f"Loaded cached track weights ({len(weights_np)} tracks)")
        else:
            print("Estimating per-track signal for loss balancing...")
            loader = BigWigLoader(bw_files, bin_size=32)
            half_c = train_ds.center_bin_size // 2
            sample_ivs = splits["train"][:: max(1, len(splits["train"]) // 50)]
            means = []
            for iv in tqdm(sample_ivs, desc="track-stats"):
                center = (iv.start + iv.end) // 2
                t = loader.load(iv.chrom, center - half_c, center + half_c)
                means.append(t.mean(axis=0))
            loader.close()
            track_mean = np.mean(means, axis=0) + 1e-6
            # inverse mean, normalized so the mean weight is 1 (keeps loss scale stable)
            weights_np = (1.0 / track_mean)
            weights_np = (weights_np / weights_np.mean()).astype(np.float32)
            np.save(tw_path, weights_np)
            print(f"Saved track weights to {tw_path}")
        track_weights = torch.from_numpy(np.asarray(weights_np, dtype=np.float32))

    # Step 9 — train
    print("\n=== Training ===")
    train_resumable(
        model, train_ds, val_ds, test_ds, ckpt_dir, out_dir,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay, patience=args.patience,
        track_weights=track_weights,
    )

    # Step 10 — evaluation plots
    print("\n=== Evaluation plots ===")
    plot_evaluation(out_dir, bw_files=bw_files, split="val")
    plot_evaluation(out_dir, bw_files=bw_files, split="test")


if __name__ == "__main__":
    main()
