#!/bin/bash
# Waits for the two running variant-scoring jobs (never touches them), then builds the trunk cache (resumable; retried up to 5x).
cd /panfs/accrepfs.vampire/home/yejz1/experimental_research
LOG=results_phase2/trunk_cache/logs/queue_cache.log
echo "$(date -u) queue start; waiting for PIDs 3442888 3444921" >> $LOG
while kill -0 3444921 2>/dev/null; do sleep 120; done  # coordinator 2026-10-20: start after Enformer only, sharing the GPU with the Borzoi scoring job to save ~3 h
echo "$(date -u) GPU free of both scoring jobs; building cache" >> $LOG
for try in 1 2 3 4 5; do
  ./torch-venv/bin/python -u analysis/priorityD/01_build_cache.py --splits test val train --workers 8 >> results_phase2/trunk_cache/logs/build_cache.log 2>&1 && { echo "$(date -u) cache complete" >> $LOG; touch results_phase2/trunk_cache/CACHE_COMPLETE; exit 0; }
  echo "$(date -u) build attempt $try failed; retrying" >> $LOG; sleep 30
done
echo "$(date -u) cache build FAILED after 5 attempts" >> $LOG
