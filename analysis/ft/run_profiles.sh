#!/usr/bin/env bash
# Profiling sweep (2 warm-up + 30 timed steps, batch 1 tile, 524,288 bp, bf16 autocast; 48 GB per-process cap; ckpt write + 20-tile val timing)
cd /home/yejz1/experimental_research
PY=./torch-venv/bin/python
run() { id=$1; shift
  echo "== $id $* | GPU used by others before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  $PY analysis/ft/train_ft.py --run-id $id --profile-steps 30 --val-limit 20 --mem-cap-gb 48 --num-workers 6 "$@" 2>&1 | grep -v -i warn | tail -4
  echo "   GPU used after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
run FT-P01 --mode frozen
run FT-P02 --mode partial --unfreeze blocks:1
run FT-P03 --mode partial --unfreeze blocks:2
run FT-P04 --mode partial --unfreeze blocks:4,up,final
run FT-P05 --mode partial --unfreeze blocks:8,up,final,horiz
run FT-P06 --mode lora --lora-targets attn --lora-r 8
run FT-P07 --mode lora --lora-targets attn,mlp,conv1x1 --lora-r 16
run FT-P08 --mode full --grad-ckpt transformer
run FT-P09 --mode full --grad-ckpt all
run FT-P10 --mode partial --unfreeze blocks:2 --grad-ckpt none
run FT-P11 --mode full --grad-ckpt none
echo ALL_DONE
