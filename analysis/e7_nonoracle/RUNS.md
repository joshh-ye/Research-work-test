# E7 runs (2026-10-20, no git commits). GPU only for Enformer (shared A100, 19 GB used by other jobs at start; ~7 min val + ~3 min test).
1. `./torch-venv/bin/python -m pytest -q analysis/e7_nonoracle/tests` -> 7 passed
2. Enformer: `bash results_phase2/e7_nonoracle/logs/chain_enformer.sh` (val inference -> select_valsel.py enformer -> test inference)
3. AlphaGenome (hosted API, sequential, existing key file, 0 retries needed): `bash results_phase2/e7_nonoracle/logs/chain_alphagenome.sh` (val 750 + test 359 predict_interval calls, total wall time about 14 min including val and test)
4. `./torch-venv/bin/python analysis/e7_nonoracle/analyze.py enformer` and `... alphagenome` (10,000 bootstraps, seed 20261020, block 10; oracle reproduction gate passed for both)
Frozen selections: results_phase2/e7_nonoracle/<model>/selection_valsel.csv (sha256 in logs/*_select.log), written before test inference started.
Outputs per model: tile_stats_{val,test}.npz (per-tile cross sufficient statistics), per_tile_selected_pair_stats.npz (per-tile stats of TL/oracle/valsel/excl pairs), per_track_metrics.csv, model_summary.csv, delta_stats.csv, summary.json.
