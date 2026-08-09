#!/bin/bash
#SBATCH --job-name=borzoi_filt_test
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
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Full Borzoi inference for the filtered baseline. Needed for the TEST split
# (no saved correlation matrix exists for test). The VAL split can be produced
# on a laptop via --from-matrix and does not need this job.

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

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

# Test-split outputs are kept in their own directories so the val analysis in
# results_baseline_filtered/ + results_comparison/ is never overwritten.
# MODEL_DIR holds {test_preds,test_targets}.npy and test_per_track_metrics.csv —
# ./results_full on ACCRE, ./results_full_2 on the laptop.
MODEL_DIR=./results_full

mkdir -p results_baseline_filtered_test results_comparison_test

# 1) Borzoi inference -> test correlation matrix + best-match CSV (needs GPU)
python -u baseline_borzoi_filtered.py \
    --data-root   ./borzoi_data \
    --targets-dir "$MODEL_DIR" \
    --results-dir ./results_baseline_filtered_test \
    --n-folds     4 \
    --split       test

# 2) Comparison + stats + figures for test (lightweight, no GPU)
python -u compare_transfer_borzoi.py \
    --model-dir    "$MODEL_DIR" \
    --baseline-dir ./results_baseline_filtered_test \
    --out-dir      ./results_comparison_test \
    --split        test

echo "END: $(date)"
