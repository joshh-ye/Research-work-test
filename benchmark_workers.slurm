#!/bin/bash
#SBATCH --job-name=borzoi_benchmark
#SBATCH --mail-user=joshua.z.ye@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=accre_dgx_iacc
#SBATCH --partition=production
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out

echo "JOBID: $SLURM_JOBID"
echo "HOST:  $(hostname)"
echo "START: $(date)"
echo "CPUs:  $(nproc)"

setup_accre_software_stack
module purge
module load python/3.12.4 cuda/12.6

unset PYTHONPATH || true
unset PYTHONHOME || true

source torch-venv/bin/activate

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

python benchmark_workers.py --n-intervals 10 --max-workers 64

echo "END: $(date)"
