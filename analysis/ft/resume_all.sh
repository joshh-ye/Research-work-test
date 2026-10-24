#!/usr/bin/env bash
# Resume entry point for the fine-tuning program (new VM / new Slurm job / after a crash).  Idempotent.
#   bash analysis/ft/resume_all.sh
# 1. sanity (branch, GPU, quota, deadline);  2. mark every interrupted/stopped run that has checkpoints (RESUME_ME);
# 3. start the deadline supervisor (setsid nohup, pidfile) if it is not running.  The supervisor then resumes the marked runs
#    from find_latest_valid() one at a time (GPU free + deadline policy permitting) and continues QUEUE.json.
set -u
cd /home/yejz1/experimental_research || exit 1
PY=./torch-venv/bin/python
FT=research_management/finetuning
echo "branch: $(git branch --show-current)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || echo "WARNING: no GPU visible"
accre_storage 2>/dev/null | grep -E "^Home|Nobackup" || true
$PY - <<'PYEOF'
import sys; sys.path.insert(0, "analysis/ft")
import policy as P
d, ms = P.load_deadline()
print("effective stop:", d["effective_stop"], "| hours left to artifacts_saved (21:30):", round(P.hours_to(ms["artifacts_saved"]), 2),
      "| launch policy for a fresh 1 h run:", P.launch_allowed(1.0))
PYEOF
$PY analysis/ft/runqueue.py list
$PY analysis/ft/runqueue.py resume-all
if [ -f $FT/supervisor.pid ] && kill -0 "$(cat $FT/supervisor.pid)" 2>/dev/null && grep -q supervisor.py /proc/"$(cat $FT/supervisor.pid)"/cmdline 2>/dev/null; then
  echo "supervisor already running: pid $(cat $FT/supervisor.pid)"
else
  setsid nohup $PY analysis/ft/supervisor.py > $FT/supervisor.out 2>&1 < /dev/null &
  sleep 5; echo "supervisor started: pid $(cat $FT/supervisor.pid 2>/dev/null)"
fi
echo "status: $FT/supervisor_status.json   log: $FT/SUPERVISOR_LOG.md   registry: $FT/FT_REGISTRY.csv"
