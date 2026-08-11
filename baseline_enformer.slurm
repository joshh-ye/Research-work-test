#!/bin/bash
#SBATCH --job-name=enformer_base_test
#SBATCH --mail-user=joshua.z.ye@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=accre_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Original-Enformer best-match baseline, mirroring baseline_borzoi_filtered.slurm.
# No --from-matrix shortcut exists for Enformer, so both splits need this job.

echo "JOBID:      $SLURM_JOBID"
echo "HOST:       $(hostname)"
echo "START:      $(date)"
nvidia-smi

setup_accre_software_stack
module purge
module load python/3.12.4 cuda/12.6

unset PYTHONPATH || true
unset PYTHONHOME || true

source torch-venv/bin/activate

# one-time: pip install enformer-pytorch, pinning transformers so it doesn't
# clobber the version borzoi-pytorch needs (<4.51.0,>=4.34.1) in this same venv
python -c "import enformer_pytorch" || pip install enformer-pytorch "transformers<4.51.0,>=4.34.1"

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

# 0) track filtering report (cheap, also writes the included/excluded CSVs)
python -u filter_enformer_tracks.py \
    --data-root ./enformer_data \
    --out-dir   ./results_baseline_enformer

# 1) Enformer inference -> correlation matrix + best-match CSV (needs GPU)
#    NOTE: on ACCRE the transfer-model outputs live in ./results_full.
python -u baseline_enformer.py \
    --data-root     ./borzoi_data \
    --enformer-root ./enformer_data \
    --targets-dir   ./results_full \
    --results-dir   ./results_baseline_enformer \
    --split         test

# 2) Comparison + stats + figures (lightweight, no GPU)
python -u compare_transfer_enformer.py \
    --model-dir    ./results_full \
    --baseline-dir ./results_baseline_enformer \
    --out-dir      ./results_comparison_enformer \
    --split        test

echo "END: $(date)"
