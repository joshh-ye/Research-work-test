# Next VM / Slurm-session restart: fine-tuning program (created 2026-10-23)

Deadline: `research_management/finetuning/deadline.json` (effective stop Mon 2026-10-26 22:00 CDT; no training starts after 21:30).
Everything persistent is under home: code `analysis/ft/`, checkpoints `results_ft/runs/<run_id>/`, state `research_management/finetuning/`.
Node-local `/tmp/yejz1_ft/` (pooled cache copies etc.) is NOT persistent: regenerate (`analysis/ft/build_pooled.py`).

## One command
```
cd /home/yejz1/experimental_research && git switch research/crc-autonomous-study   # if not already
bash analysis/ft/resume_all.sh
```
It prints GPU / quota / deadline policy, lists the queue, marks every interrupted (`STATUS=running` with dead PID) or stopped run that
has checkpoints with `RESUME_ME`, and starts the supervisor (`setsid nohup`, pidfile `finetuning/supervisor.pid`; idempotent).
The supervisor resumes marked runs from `find_latest_valid()` (newest loadable checkpoint with matching config hash and tiles.csv hash),
one run at a time when the GPU is free (other jobs < 35 GB) and the deadline policy allows, then continues `finetuning/QUEUE.json`.

## What survives
- `results_ft/runs/<id>/ckpts/ckpt_step*.pt` (2 newest; trainable params + optimizer + scheduler + RNG + sampler position),
  `best_model.pt` (best full-validation mean r), `metrics.csv`, `STATUS.json`, `progress.json`, `run_cmd.json`, `val_epoch*_per_track_r.json`.
- `finetuning/FT_REGISTRY.csv` (append-only), `SUPERVISOR_LOG.md`, `supervisor_status.json`.
- Frozen trunk weights are NOT in checkpoints: they are re-read from `hf_cache/` (`johahi/borzoi-replicate-0`, revision recorded in each checkpoint).

## Manual operations
- Queue: `./torch-venv/bin/python analysis/ft/runqueue.py enqueue --est-hours H RUN_ID -- --mode partial --unfreeze blocks:2 --epochs 4 ...`
  or edit `finetuning/QUEUE.json` (list of {run_id, args, est_hours}); `runqueue.py list` shows states.
- Force a checkpoint: `touch results_ft/runs/<id>/REQUEST_CKPT` (or `kill -USR1 <pid>`); graceful stop: `kill -TERM <pid>` or `touch .../REQUEST_STOP`
  (checkpoint is written at the next optimizer-step boundary, exit code 7).  Remove REQUEST_STOP before resuming (resume_all does).
- Direct (no supervisor): `./torch-venv/bin/python analysis/ft/train_ft.py --run-id <id> <same args>` resumes automatically if `ckpts/` has a valid checkpoint
  (refuses with exit 8 if checkpoints exist but none is valid/compatible, instead of silently restarting).
- Exit codes: 0 done, 3 deadline policy refused, 4 storage guard (home > 175 GB or < 5 GB left), 5 OOM, 6 non-finite loss, 7 stopped by signal/request, 8 no valid checkpoint.
- Test evaluation is NOT part of train_ft.py (locked script only; see FT_PLAN.md).
