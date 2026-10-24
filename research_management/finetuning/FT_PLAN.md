# Borzoi performance-improvement program (FT): plan and pre-specified rules
Written 2026-10-23 22:15 CDT, before any FT experiment. Effective stop: 2026-10-26 22:00 CDT (`deadline.json`; Slurm 14776988 ends 22:53).

## Fixed evaluation contract (unchanged from E4; so numbers are comparable with r = 0.7426)
- hg38; tiles from `trunk_cache/tiles.csv`: train 4,659 / validation chr1+chr8 750 / test chr9+chr22 359. No split change.
- Targets: the 183 cached BigWig targets (`trunk_cache/tgt`, == `results_full/test_targets.npy`), evaluated on the shared 896 x 128 bp grid
  (crop 256 bins/side of the central 4,096 x 32 bp, mean-pool x4). Metric: mean over tracks of per-track Pearson r pooled over all bins of
  the split (same code path as E4, `analysis/priorityD/cmp_lib.py` / `estats.py`). r^2 and prediction R^2 reported separately, labelled.
- Primary track set for claims: 152 non-identical (SD-014); all 183 reported. Primary inference: TF-cluster bootstrap (SD-006), cell-line-cluster
  and tile-block alongside, 10,000 resamples, seed 20261020.
- **Model/hyperparameter/checkpoint selection uses validation chromosomes only.** Test predictions are produced only by `ft_test_eval` for configurations
  entered in `research_management/finetuning/TEST_LOCK.csv` (config id, selection rule, validation r, timestamp) BEFORE the test run. At most one
  test evaluation per locked configuration; no test-driven iteration.
- An "improvement over r = 0.743" is claimed only if, on the 152 tracks, Δr (new − ridge[pre_raw:union1888]) has a TF-cluster 95% CI excluding 0
  (and is reported with cell-line-cluster and tile-block CIs). Smaller or CI-crossing gains are reported as "not shown".
- MPRA data are not used for training, selection or tuning. Any MPRA re-scoring of a new model is a separate, labelled analysis; BL-007 (sign convention)
  stays open and P2 is not re-signed.

## Comparability notes (declared up front)
- Ridge 0.7426 uses 4-replicate pretrained outputs (1,888 channels); TL head / probe use the 4-replicate averaged trunk embedding. Fine-tuning experiments
  that update trunk weights use replicate 0 only (4 replicates x 186 M is ~4x the cost); each FT arm therefore has a matched frozen replicate-0 control
  (same head, same loss, same data) so the effect of unfreezing is attributable, and the final comparison with the 4-replicate ridge is reported as such.
- Input context (524,288 bp), target definition and grid are never changed to make training fit.
- Cache-based heads see shift-emulated, no-RC data (SD-018 cache recipe); FT runs that read sequence can use true shift + RC augmentation; the
  difference is labelled.

## Priority order (expected gain per GPU-hour; revised on validation evidence only)
1. Infra: checkpoint/resume library + test, deadline supervisor independent of the Claude session.
2. Cheap, cache-based (pooled 128 bp copy of the cache on node-local /tmp, regenerable): ridge optimisation (per-track lambda, feature transform,
   spatial context of neighbouring bins), combined output + trunk features, linear vs nonlinear heads on identical features and loss.
3. FT profiling on replicate 0 (memory, s/step, checkpoint size): partial (late blocks), LoRA/adapters on compatible modules, full.
4. Long FT run(s) of the best profiled arm with 30-min wall-clock checkpoints; validation each epoch; early stopping.
5. Locked test evaluation of selected configurations, bootstrap CIs, reports.

## Storage
- Persistent checkpoints: `/home/yejz1/experimental_research/results_ft/` (home, backed up; 68 GB free under the 186.3 GB soft quota at start).
  Guard: no checkpoint write if home usage > 175 GB; stop launching new runs and report.
- Regenerable scratch: `/tmp/yejz1_ft/` (node-local, NOT persistent across VMs). `/nobackup` (25 GB free) is not used for new data; the trunk cache is not modified or deleted.
