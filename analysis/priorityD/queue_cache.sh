#!/bin/bash
# 2026-10-20 slim recipe: no waiting (shares the GPU with Borzoi scoring; coordinator decision). Cache lives on /nobackup (symlinked
# at results_phase2/trunk_cache). Resumable; retried up to 5x; exit code 3 = quota guard stop (no retry).
cd /panfs/accrepfs.vampire/home/yejz1/experimental_research
C=results_phase2/trunk_cache; LOG=$C/logs/queue_cache.log
echo "$(date -u) slim build start" >> $LOG
for try in 1 2 3 4 5; do
  ./torch-venv/bin/python -u analysis/priorityD/01_build_cache.py --splits test val train --workers 8 >> $C/logs/build_cache.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "$(date -u) cache complete" >> $LOG; touch $C/CACHE_COMPLETE; exit 0; fi
  if [ $rc -eq 3 ]; then echo "$(date -u) QUOTA GUARD STOP; cache FAILED (incomplete)" >> $LOG; exit 3; fi
  echo "$(date -u) build attempt $try failed rc=$rc; retrying" >> $LOG; sleep 30
done
echo "$(date -u) cache build FAILED after 5 attempts" >> $LOG
