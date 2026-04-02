#!/bin/bash
set -euo pipefail

# Usage:
#   bash setup_accre_tensorflow_gpu.sh
#
# Run this on an ACCRE GPU node after obtaining an allocation, for example:
#   salloc --account=YOUR_ACCOUNT --partition=batch_gpu --gres=gpu:1 --time=1:00:00

setup_accre_software_stack
module load python/3.12.4 scipy-stack/2025a cuda/12.6

python -m venv tf-venv
source tf-venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-accre.txt

python check_tensorflow_gpu.py
