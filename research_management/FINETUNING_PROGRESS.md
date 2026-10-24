# Fine-tuning / performance-improvement progress

Effective stop: 2026-10-26 22:00 CDT (deadline.json). Plan and evaluation contract: `research_management/finetuning/FT_PLAN.md`.

## Status (updated 2026-10-23 22:20 CDT; ~71.7 h remaining)
- Current: Phase 1 audit done (deadline, hardware, storage). Two engineering tracks started:
  (A) cache-based ridge optimisation / combined features / identical-feature heads; (B) checkpoint library, deadline supervisor, FT profiling.
- Best validation r: none yet (reference E4 ridge[pre_raw:union1888]; test 0.7426 all183 / 0.7406 on 152).
- Best final test r: none (test evaluated only for locked configs; `finetuning/TEST_LOCK.csv`).
- Completed experiments: none. Active: FT-C* (cache), FT-P* (profiling).
- Blockers: none.
- Next: review profiling; choose the long FT arm; enqueue it under the supervisor.

## 2026-10-23 23:45 CDT (~70.2 h remaining)
- FT-S01 (frozen replicate-0 trunk, MLP head, Poisson, lr 1e-3 cosine, true shift+RC, 4 epochs, seed 1): full-val mean r 0.7280 (183) / 0.7282 (152), still rising (0.691, 0.706, 0.725, 0.728).
  Validation references (same grid, all 183): E4 ridge[pre_raw:union1888] 0.7337 (lambda 1e-4); 4-replicate trunk linear probe 0.7104.
  Interpretation (validation only): the original TL recipe (lr 1e-4, plateau schedule) appears under-trained; a 1-replicate frozen head nearly matches the 4-replicate ridge.
- Queue: FT-S05 full FT (running next) → FT-S03 partial 4 blocks+up+final → FT-S04 LoRA r16 → FT-S02 partial 2 blocks. Supervisor PID 1932265.

## 2026-10-24 01:20 CDT (~68.7 h remaining)
- **Cache experiments FT-C01..C85 done (validation only; `results_ft/cache_exps/`, `analysis/ft/README_cache_exps.md`).** E4 gate reproduced to 1e-11.
  Best ridge FT-C61 (sqrt features + context offsets ±1,2,4 + trunk centre bin, sqrt target back-transformed, shared lambda 3.2e-5): val r 0.7392 / 0.7370 (183/152), +0.0055 over E4 ridge.
  Best head FT-C84 (MLP, MSE, same 15,136 features): 0.7404 / 0.7393 (early stopping on validation, so slightly optimistic). FT-C83 trunk-only Poisson MLP at 128 bp: 0.7378 / 0.7376.
  Per-track lambda and SGD linear heads rejected (no cross-fit gain / worse than closed form). FT-C60 failed (OOM, 26.7k features).
  Head checkpoints copied from /tmp to `results_ft/cache_exps/ckpt/` (21 files, 521 MB); registry paths updated.
- **FT-S05 full fine-tuning (replicate 0, lr-trunk 3e-5, 4 epochs, seed 1): val r 0.7338 / 0.7331**, rising every epoch (0.696, 0.718, 0.730, 0.734).
  Matched frozen control FT-S01: 0.7280 / 0.7282, so unfreezing adds +0.006 (183) / +0.005 (152) on validation at 4 epochs. Epoch time 19 min (faster than profiled), peak 16.5 GB.
  One replicate-0 fine-tuned model already matches the 4-replicate E4 ridge on validation (0.7337 / 0.7317).
- Code: `train_ft.py --replicate {0..3}` added (for a fine-tuned replicate ensemble). `ckpt.config_hash` omits `replicate` at its default 0 so earlier runs stay resumable (verified: FT-S05 hash unchanged).
- Queue (FIFO): FT-S03 partial 4 blocks+up+final (running) → FT-S04 LoRA r16 → FT-S02 partial 2 blocks → FT-S09 full lr-trunk 1e-4 (4 ep, LR probe)
  → FT-S06 full lr-trunk 3e-5, 12 epochs, patience 3 → FT-S07 frozen 12 epochs (matched control for S06).
- Next decisions (validation only): choose lr from S05 vs S09; then fine-tune replicates 1-3 with the S06 recipe for an ensemble comparable to the 4-replicate ridge.

## 2026-10-24 01:41 CDT (~68.3 h remaining)
- New tools (`analysis/ft/README_predict.md`): `predict_ft.py` (val predictions; reproduces logged val r of FT-S01/S05 exactly; test mode refuses unless locked), `blend_val.py` (validation-only blends, weight chosen on val + 2-fold chr1/chr8 cross-fit), `compare_ft.py` (compare stage shared with `test_eval_cache.py`, outputs unchanged, verified diff 0). Tests: 10 new, 39 passed (GPU model tests not re-run).
- Validation blends (FT-B01..B03): FT-S05 + E4 ridge w=0.5 → 0.7436 / 0.7425 (val-selected w); FT-S05 + FT-C61 w=0.4 → 0.7465 / 0.7450.
  Honest cross-fit (chromosome-wise mean): +0.010 over the better component for +E4, +0.007 for +FT-C61. Same w chosen in both folds.
  Interpretation: fine-tuned sequence model and output-feature ridge make partly complementary errors. Candidate final configs: FT ensemble (+/-) ridge blend with w fixed on validation.
- Requested: locked test path for blends (`BLEND:<FT>|<ridge>|w=`) and an RC test-time-augmentation check on validation.

## 2026-10-24 01:43 CDT
- Locked blend test path added (`predict_ft.py --split test --config-id 'BLEND:<FT>|<ridge>|w=<w>'`; refuses unless that exact id is locked; 14 tests pass). No test evaluation run.
- RC test-time augmentation on validation, FT-S05: 0.7380 / 0.7370 vs 0.7338 / 0.7331 without (+0.004). Decision (validation evidence): FT components of final candidates use `--tta rc`; blend weights are re-selected on validation with TTA predictions.

## 2026-10-24 03:30 CDT (~66.5 h remaining)
- Arm comparison at 4 epochs, replicate 0, seed 1, Poisson MLP head (full val r 183 / 152; all still rising at epoch 4):
  | run | arm | trainable | val r |
  |---|---|---|---|
  | FT-S01 | frozen | 2.2 M | 0.7280 / 0.7282 |
  | FT-S05 | full, lr-trunk 3e-5 | 173.4 M | 0.7338 / 0.7331 |
  | FT-S03 | partial, last 4 blocks + up + final, lr-trunk 3e-5 | 77.6 M | 0.7373 / 0.7372 |
  | FT-S04 | LoRA r16 attn+mlp+conv1x1, lr 3e-4 | 4.9 M | 0.7364 / 0.7365 |
  Single seed; differences between S03/S04/S05 (0.003) are within plausible seed noise, to be checked with extra seeds of the selected arm.
- Queue revised on this validation evidence: FT-S02 (partial 2 blocks) → FT-S10 partial-4 lr-trunk 1e-4 (LR probe) → FT-S11 partial-4 12 epochs lr 3e-5 (edited to 1e-4 before start if S10 wins)
  → FT-S07 frozen 12 ep → FT-S09 full lr 1e-4 → FT-S06 full 12 ep. Previous queue saved in the session scratchpad.

## 2026-10-24 03:43 CDT (~66.3 h remaining)
- Round-2 validation blends (FT-B04..B12, `results_ft/blends/round2/`; all FT components with RC TTA; `blend_val.py` now accepts `RUN@rc`):
  FT-S03@rc 0.7410 / 0.7406; FT-S04@rc 0.7402 / 0.7399; FT-S05@rc 0.7380 / 0.7370; 3-run FT ensemble 0.7439 / 0.7434;
  3-run ensemble + FT-C61 ridge (w=0.6) 0.7489 / 0.7478 (w on val; cross-fit chromosome-wise 0.7359 vs 0.7305 ensemble-only and 0.7272 ridge-only).
- Implication: ensembling fine-tuned models and blending with the output-feature ridge both add on validation. Leading final-candidate family:
  ensemble of fine-tuned models (target: replicates 0-3 of the selected arm) with RC TTA, blended with FT-C61 at w fixed on validation.

## 2026-10-24 05:14 CDT
- FT-S02 partial 2 blocks (lr 3e-5, 4 ep): 0.7327 / 0.7331. FT-S10 partial-4 at lr-trunk 1e-4: 0.7285 / 0.7276 (vs 0.7373 at 3e-5) → keep lr-trunk 3e-5. FT-S11 (partial-4, 12 ep, 3e-5) running.
- Queue (validation-driven): FT-S11 → FT-R1/R2/R3 (replicates 1-3, FT-S11 recipe, for a 4-replicate fine-tuned ensemble comparable to the 4-replicate ridge) → FT-S07 frozen 12 ep control → FT-S12 (FT-S03 recipe, seed 2: seed-noise estimate) → FT-S06 full 12 ep.
  FT-S09 (full, lr 1e-4) dropped: lr 1e-4 was worse for partial and full is the weakest arm. Replicates 1-3 verified to load offline from hf_cache (171.27 M params each).
