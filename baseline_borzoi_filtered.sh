#!/bin/bash
# Full Borzoi inference for the filtered baseline — TEST split (the val split was
# produced on a laptop via --from-matrix and does not need this).
# Run on an ACCRE A100 interactive GPU node.
set -euo pipefail

echo "HOST:  $(hostname)"
echo "START: $(date)"
nvidia-smi

setup_accre_software_stack
module purge
module load python/3.12.4 cuda/12.6

unset PYTHONPATH || true
unset PYTHONHOME || true

source torch-venv/bin/activate

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

MODEL_DIR=./results_full
mkdir -p results_baseline_filtered_test results_comparison_test

# 1) Borzoi inference -> test correlation matrix + best-match CSV
python -u baseline_borzoi_filtered.py \
    --data-root   ./borzoi_data \
    --targets-dir "$MODEL_DIR" \
    --results-dir ./results_baseline_filtered_test \
    --n-folds     4 \
    --split       test

# 2) Comparison + stats + figures for test
python -u compare_transfer_borzoi.py \
    --model-dir    "$MODEL_DIR" \
    --baseline-dir ./results_baseline_filtered_test \
    --out-dir      ./results_comparison_test \
    --split        test

echo "END: $(date)"
