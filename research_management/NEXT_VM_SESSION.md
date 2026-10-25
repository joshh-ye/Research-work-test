# Next VM / Slurm-session restart: fine-tuning program (created 2026-10-23; updated 2026-10-25)

## Current state (2026-10-25): the FT program is COMPLETE apart from review of the final reports
- All planned training, the single locked test round and the final summaries are done; the queue is empty (all 21 `QUEUE.json` entries `finished`; `supervisor_status.json` lists no runs).
- Primary result (locked, commit `dc0bcd5`): 8-member fine-tuned ensemble with RC TTA, test Pearson r 0.7586 (183) / 0.7578 (152) vs E4 ridge 0.7426 / 0.7406; delta r +0.0173 on 152, TF-cluster CI [0.0123, 0.0252], two-way CI [0.0111, 0.0263] (both exclude 0).
  Secondaries are descriptive only (no ranking). Numbers: `results_ft/final/test_ci_table.csv`.
- Deliverables to review: `reports/FINETUNING_FINAL_REPORT.md`, `reports/FINETUNING_COMPARISON.md` (both git-ignored `reports/`), `research_management/FINETUNING_RESOURCE_REPORT.md` (section 8 actual usage), `results_ft/final/` (per-track test r table, figure, summaries),
  `analysis/ft/final_summary.py` (regenerates `results_ft/final/` from the existing `test_stats.npz` files in seconds, CPU only), `research_management/FINETUNING_PROGRESS.md` (chronology + locked table), `research_management/finetuning/{FT_PLAN,LOCK_PROPOSAL,TEST_LOCK.csv,FT_REGISTRY.csv}`.
  Not done: independent (scientific-reviewer) review of the final reports; no verdict is recorded in `REVIEW_FINDINGS.md` after the RV-FT-01 corrections.
- **No further test evaluation of locked ids is possible or allowed.** `predict_ft.py --split test` refuses an id whose `results_ft/ft_test_eval/<id>/test_stats.npz` exists (all six locked ids and the E4 gate have outputs) and any id not committed in `TEST_LOCK.csv`; `test_eval_cache.py` refuses ids not in `TEST_LOCK.csv` and re-evaluation (it does not check that the lock is committed). Do not edit `TEST_LOCK.csv`, `results_ft/ft_test_eval/` or `results_ft/cache_exps/test_eval/`.
  Any new configuration would need a new lock row committed before its first test run and must be labelled post-lock.
- Supervisor: at 2026-10-25 (check) PID 1932265 (`ps -p $(cat research_management/finetuning/supervisor.pid)`) is still alive, idle (no runs, GPU used by others about 6.5 GB, `allow_new_runs: true` in `LAUNCH_POLICY.json`). It would launch anything added to `QUEUE.json`; stop it with `kill -TERM $(cat research_management/finetuning/supervisor.pid)` when the coordinator decides (not done here).
  It dies with the Slurm job (14776988, ends 2026-10-26 22:53; effective stop 22:00 per `deadline.json`).

## What is persistent and what is not
- Persistent (home, backed up, git-ignored): `results_ft/` (about 24 GB on disk by `du -sh`, 19.9 GB apparent size, on 2026-10-25: runs 10.5 GB with `best_model.pt` of each run, preds 8.2 GB, test_runs 3.5 GB, cache_exps 0.7 GB incl. head checkpoints in `cache_exps/ckpt/`, selfcheck_val 0.6 GB, ft_test_eval 0.3 GB, final), code `analysis/ft/`, state `research_management/finetuning/`, `research_management/CHECKPOINT_INVENTORY.md` (sha256; regenerated 2026-10-25 00:03 after all runs finished).
- Resume checkpoints (`ckpts/ckpt_step*.pt`) of finished runs were deleted with user approval on 2026-10-24 (25.7 GB); `best_model.pt` kept for every run. Finished runs cannot be resumed (except FT-E4/FT-E5, which finished after the deletion and keep resume checkpoints), but validation/test predictions are regenerable from `best_model.pt` (about 2 min per run for validation).
- NOT persistent: node-local `/tmp/yejz1_ft/` (`pooled128` 43 GB and `ckpt/` copies of cache heads); regenerable with `./torch-venv/bin/python analysis/ft/build_pooled.py` (needs the /nobackup trunk cache, about 2.5 min). `/nobackup` trunk cache is read-only input (not backed up).
- Not in Git: everything under `results*/` and `reports/`; commits are the coordinator's.

## How to resume or re-run (only if a new, locked experiment is decided)
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
(Update 2026-10-25: the `ckpts/` resume files of all finished runs were deleted on 2026-10-24 with user approval; only `best_model.pt` and logs remain. A resumable checkpoint exists only for a run started after that date.)
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

Update 2026-10-25 00:07 CDT: the deadline supervisor (PID 1932265) was stopped (SIGTERM) after the queue drained and the final review passed (RV-FT-02 PASS). No training is running. Tracked copies of the final reports and test tables: research_management/finetuning/final/.
