#!/bin/bash
#SBATCH --job-name=borzoi_eval
#SBATCH --mail-user=joshua.z.ye@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=accre_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out

echo "JOBID:      $SLURM_JOBID"
echo "HOST:       $(hostname)"
echo "START:      $(date)"
echo "PARTITION:  $SLURM_JOB_PARTITION"
echo "NODES:      $SLURM_JOB_NUM_NODES"
echo "NTASKS:     $SLURM_NTASKS"
echo "CPUS/TASK:  $SLURM_CPUS_PER_TASK"
echo "MEM (MB):   $SLURM_MEM_PER_NODE"
echo "TIMELIMIT:  $(squeue -j $SLURM_JOBID -h --format=%l 2>/dev/null || echo 'N/A')"
echo "WORKDIR:    $(pwd)"
echo "CUDA:"
nvidia-smi

setup_accre_software_stack
module purge
module load python/3.12.4 cuda/12.6

unset PYTHONPATH || true
unset PYTHONHOME || true

python -m venv torch-venv
source torch-venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-accre.txt
pip install borzoi-pytorch

export HF_HOME=/home/yejz1/Research-work-test/hf_cache

python -u main.py --n-bw 3 --epochs 1 --num-workers 4 --ckpt-dir ./borzoi_ckpt_smoke --output-dir ./results_smoke
