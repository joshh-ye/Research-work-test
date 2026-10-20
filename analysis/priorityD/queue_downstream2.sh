#!/bin/bash
# Waits for the trunk cache (queue_cache.sh -> CACHE_COMPLETE), then runs E4 > E5 seeds > E8 > E5 folds > extras. Every step is
# skipped if its output exists, so the script is safe to re-launch. Never touches other jobs.
cd /panfs/accrepfs.vampire/home/yejz1/experimental_research
PY=./torch-venv/bin/python; R=results_phase2; D=analysis/priorityD
mkdir -p $R/{e4_ridge,e5_seeds,e5_folds,e8_generalisation}/logs
LOG=$R/e4_ridge/logs/queue_downstream.log
say() { echo "$(date -u) $*" >> $LOG; }
left() { squeue -j 14717985 -h -o %L 2>/dev/null | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print 999999 }'; }
say "downstream queue start; waiting for CACHE_COMPLETE"
while [ ! -f $R/trunk_cache/CACHE_COMPLETE ]; do
  if grep -q "FAILED" $R/trunk_cache/logs/queue_cache.log 2>/dev/null; then say "cache build failed; abort"; exit 1; fi
  sleep 120
done
say "cache complete; time left $(left)s"
run() {  # run <done-file> <logfile> cmd...
  local done=$1 lg=$2; shift 2
  if [ -e "$done" ]; then say "skip (exists) $done"; return; fi
  if [ "$(left)" -lt 2400 ]; then say "skip (time) $*"; return; fi
  say "START $*"; "$@" >> $lg 2>&1 && say "OK $done" || say "FAIL $*"
}
# --- E4 + gate
run $R/e5_seeds/orig_ckpt_cache_eval/metrics.json $R/e5_seeds/logs/orig_ckpt_eval.log $PY -u $D/02_train_head.py --out $R/e5_seeds/orig_ckpt_cache_eval --seed 0 --eval-ckpt borzoi_ckpt_full/model_head_best.pt
run $R/e4_ridge/pretrained_valsel_orig/summary.json $R/e4_ridge/logs/valsel_orig.log $PY -u $D/04_pretrained_valsel.py --fold orig
run $R/e4_ridge/orig_pre_raw/summary.json $R/e4_ridge/logs/ridge_orig_pre_raw.log $PY -u $D/03_ridge.py --fold orig --feat pre_raw
run $R/e4_ridge/orig_probe/summary.json $R/e4_ridge/logs/ridge_orig_probe.log $PY -u $D/03_ridge.py --fold orig --feat probe
run $R/e4_ridge/orig_pre_log1p/summary.json $R/e4_ridge/logs/ridge_orig_pre_log1p.log $PY -u $D/03_ridge.py --fold orig --feat pre_log1p
$PY $D/06_summaries.py >> $R/e4_ridge/logs/summaries.log 2>&1
# --- E5 seeds 1-3
for s in 1 2 3; do run $R/e5_seeds/seed$s/metrics.json $R/e5_seeds/logs/seed$s.log $PY -u $D/02_train_head.py --out $R/e5_seeds/seed$s --seed $s; done
$PY $D/06_summaries.py >> $R/e4_ridge/logs/summaries.log 2>&1
# --- E8
EXC=$($PY -c "
import pandas as pd; d=pd.read_csv('exposure_audit/track_exposure.tsv',sep='\t'); print(','.join(map(str,d[d.cell_line.isin(['GP5d','LS174T','LS180','DLD-1'])].track_idx)))")
run $R/e8_generalisation/head_seed1/metrics.json $R/e8_generalisation/logs/head_seed1.log $PY -u $D/02_train_head.py --out $R/e8_generalisation/head_seed1 --seed 1 --exclude-tracks $EXC
run $R/e8_generalisation/SUMMARY.md $R/e8_generalisation/logs/e8_analysis.log $PY -u $D/05_e8.py
# --- E5 folds (seed 1)
for f in F2 F3 F4; do
  run $R/e4_ridge/pretrained_valsel_$f/summary.json $R/e5_folds/logs/valsel_$f.log $PY -u $D/04_pretrained_valsel.py --fold $f
  run $R/e4_ridge/${f}_pre_raw/summary.json $R/e5_folds/logs/ridge_$f.log $PY -u $D/03_ridge.py --fold $f --feat pre_raw --subsets union1888
  run $R/e5_folds/$f/seed1/metrics.json $R/e5_folds/logs/${f}_seed1.log $PY -u $D/02_train_head.py --out $R/e5_folds/$f/seed1 --seed 1 --split-name $f
done
$PY $D/06_summaries.py >> $R/e4_ridge/logs/summaries.log 2>&1
# --- extras: more seeds, fold seeds (no-RC ablation dropped: slim cache has no RC and heads are no-RC by construction)
for s in 4 5; do run $R/e5_seeds/seed$s/metrics.json $R/e5_seeds/logs/seed$s.log $PY -u $D/02_train_head.py --out $R/e5_seeds/seed$s --seed $s; done
for s in 2 3; do for f in F2 F3 F4; do run $R/e5_folds/$f/seed$s/metrics.json $R/e5_folds/logs/${f}_seed$s.log $PY -u $D/02_train_head.py --out $R/e5_folds/$f/seed$s --seed $s --split-name $f; done; done
$PY $D/06_summaries.py >> $R/e4_ridge/logs/summaries.log 2>&1
say "downstream queue finished"
