# Borzoi Transfer Learning Pipeline

Transfer learning on frozen Borzoi backbones to predict ChIP-seq / ATAC-seq signal from DNA sequence.

---

## Project layout

```
.
├── main.py                        # Headless pipeline script (converted from notebook)
├── requirements-accre.txt         # pip deps for ACCRE GPU nodes
├── setup_accre_torch_gpu.slurm    # SLURM batch script for ACCRE A100
├── borzoi_code/                   # Core modules
│   ├── config.py                  # Path configuration (reads BORZOI_DATA_ROOT env var)
│   ├── sequence_utils.py          # One-hot DNA encoder + reverse complement
│   ├── fasta_reader.py            # FASTA genome reader (wraps pyfaidx)
│   ├── bigwig_loader.py           # BigWig signal loader (wraps pyBigWig)
│   ├── genome_tiler.py            # Slides 524 kb windows across genome, assigns splits
│   ├── dataset.py                 # PyTorch Dataset: sequence tensor + binned signal target
│   ├── model.py                   # Frozen Borzoi backbone ensemble + trainable linear head
│   ├── train.py                   # Training loop utilities
│   └── plot.py                    # Plotting helpers
├── borzoi_data/                   # (gitignored) hg38 FASTA + CRC_TFs_bw BigWigs
├── borzoi_ckpt/                   # (gitignored) model checkpoints
├── results/                       # (gitignored) output figures + eval arrays
└── Borzoi Pipeline.ipynb          # Original Colab notebook (reference)
```

### Data layout expected under `borzoi_data/`

```
borzoi_data/
├── hg38/
│   ├── hg38.ml.fa
│   └── hg38.ml.fa.fai
└── CRC_TFs_bw/
    ├── 1151.bw
    ├── 356.bw
    └── ...
```

---

## Running on ACCRE

```bash
sbatch setup_accre_torch_gpu.slurm
```

The SLURM script requests one A100 (40 GB) for 24 hours and runs `main.py` with default paths.

To customise paths or hyperparameters, edit the `python main.py` line in the `.slurm` file:

```bash
python main.py \
  --data-root  /path/to/borzoi_data \
  --ckpt-dir   /path/to/borzoi_ckpt \
  --output-dir ./results \
  --epochs 4 \
  --batch-size 1 \
  --n-bw 10          # use only first 10 BigWig tracks (fast test)
```

Training is **resumable**: if a checkpoint exists in `--ckpt-dir`, the script picks up from the last completed epoch.

---

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-accre.txt
python main.py --n-bw 5 --epochs 1   # quick smoke test
```

---

## Output figures

All figures are written to `--output-dir` (default `./results/`):

| File | Description |
|---|---|
| `bigwig_signal.png` | First non-zero 50 kb BigWig window (signal sanity check) |
| `dna_encoding.png` | One-hot heatmap of the first 80 bp in that window |
| `genome_splits.png` | Bar chart of train / val / test interval counts |
| `dataset_sequence.png` | Sequence snippet from the first training interval |
| `dataset_target.png` | Track-0 target signal for the first training interval |
| `loss_curve.png` | Train vs val Poisson NLL loss per epoch |
| `pearson_r.png` | Per-track Pearson R, sorted descending |
| `pred_vs_actual.png` | Predicted vs actual signal for top-3 tracks, interval 0 |
| `scatter.png` | Scatter of mean signal across all intervals × tracks |

Eval arrays `eval_preds.npy` and `eval_targets.npy` (shape: `intervals × center_bins × tracks`) are also saved for downstream analysis.

---

## Changes from the Colab notebook

| Area | Notebook | `main.py` |
|---|---|---|
| **Runtime** | Colab interactive cells | Headless Python script, `matplotlib.use("Agg")` |
| **Setup** | `!pip install`, Drive mount, file copy cells | Removed — handled by `requirements-accre.txt` + SLURM |
| **Paths** | Hardcoded `/content/data/...` | CLI `--data-root` arg; default `./borzoi_data` |
| **HF cache** | Symlinked to Drive | Standard `~/.cache/huggingface` (or set `HF_HOME`) |
| **Checkpoints** | Saved to Google Drive | Saved to `--ckpt-dir` (default `./borzoi_ckpt`) |
| **Figures** | `plt.show()` inline | `fig.savefig(...)` to `--output-dir`, figures closed after save |
| **tqdm** | `tqdm.notebook` | `tqdm` (plain terminal progress bars) |
| **BigWig filter** | Manual cell | Automatic — bad files skipped before training |
| **Track limit** | Hardcoded `BW_FILES[:10]` | `--n-bw N` flag (default: all) |
| **`config.py`** | `DATA_ROOT = "/content/data"` | Reads `BORZOI_DATA_ROOT` env var; falls back to `./borzoi_data` |
