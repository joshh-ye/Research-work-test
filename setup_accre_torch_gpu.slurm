#!/bin/bash
#SBATCH --job-name=borzoi_eval
#SBATCH --mail-user=joshua.z.ye@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=accre_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

echo "JOBID: $SLURM_JOBID"
echo "HOST:  $(hostname)"
echo "START: $(date)"
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

python main.py
