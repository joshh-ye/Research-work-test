# Runs (CPU only; no GPU; no git commits)
2026-10-20, from /home/yejz1/experimental_research
1. `./torch-venv/bin/python -m pytest -q analysis/exposure_stratified/tests` -> 8 passed
2. `./torch-venv/bin/python analysis/exposure_stratified/run_exposure_stratified.py` (defaults: --n-boot 10000 --block 10 --seed 20261020; ~25 s). Reproduction gate passed (TL 0.6820, oracle 0.6640, val-selected 0.6609, delta +0.0180 [0.0092, 0.0260]).
Inputs: reports/qing_deliverables/evaluation_evidence/{tl_per_tile_sufficient_stats_shared128.npz, baseline_borzoi_rerun/{per_tile_stats_test_shared128.npz,track_mappings.csv,baseline_vs_tl_per_track.csv,tiles_test.csv}}, exposure_audit/track_exposure.tsv, results_benchmark/benchmark_comparison_test.csv.
Outputs: results_phase2/exposure_stratified/ (exposure_stratified_stats.csv, per_track_metrics_by_exposure.csv, unexposed_cellline_per_cluster.csv, signal_variance_sensitivity.csv, tl_vs_enformer_alphagenome_by_class.csv, per_tile_selected_pair_stats.npz, summary.json, SUMMARY.md).
