#!/bin/bash
#SBATCH --job-name=borzoi_baseline
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

echo "JOBID:      $SLURM_JOBID"
echo "HOST:       $(hostname)"
echo "START:      $(date)"
echo "PARTITION:  $SLURM_JOB_PARTITION"
echo "CPUS/TASK:  $SLURM_CPUS_PER_TASK"
echo "MEM (MB):   $SLURM_MEM_PER_NODE"
echo "CUDA:"
nvidia-smi

setup_accre_software_stack
module purge
module load python/3.12.4 cuda/12.6

unset PYTHONPATH || true
unset PYTHONHOME || true

source torch-venv/bin/activate

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

python -u baseline_borzoi.py \
    --data-root   ./borzoi_data \
    --targets-dir ./results_full \
    --results-dir ./results_baseline \
    --split val

echo "END: $(date)"
