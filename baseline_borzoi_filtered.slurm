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

# NOTE: on ACCRE the transfer-model outputs live in ./results_full (adjust if it
# is results_full_2). Run test; optionally also run val for a from-scratch check.
python -u baseline_borzoi_filtered.py \
    --data-root   ./borzoi_data \
    --targets-dir ./results_full \
    --results-dir ./results_baseline_filtered \
    --n-folds     4 \
    --split       test

echo "END: $(date)"
