# Final VM handoff — 2026-10-26 (Slurm 14776988, VM end 22:53 CDT; effective research deadline 22:00 CDT)
Branch `research/crc-autonomous-study`. No training was run in this sprint. GPU was used only for one inference pass of 8 members on the 359 test tiles (about 9 min).
Commits this sprint:
- `9aa33aa`: ablation lock, before any ablation test inference.
- The final sprint commit: see `git log`. Its message begins "Final VM sprint".

## 1. Completed experiments (this sprint)
| id | what | status | main output |
|---|---|---|---|
| EXP-022 | Validation audit of the locked FT result; weight-change check; checkpoint SHA verification | done | `reports/FINETUNING_VALIDATION_AUDIT.md` (tracked copy `research_management/finetuning/sprint_20261026/`) |
| EXP-020 | Ensemble ablation: 17 validation-selected configs, locked (9aa33aa), each evaluated once on test; FT-L2 diagnostic | done | `reports/ENSEMBLE_ABLATION.md`; `research_management/finetuning/ablation_summary.csv` |
| EXP-021 | Per-track biological analysis of the gain (post hoc) | done | `reports/FINETUNING_BIOLOGICAL_ANALYSIS.md`; `reports/ft_bio/` |
| — | Publication figures | done | `figures_ft_sprint/` (5 figures; PNG 300 dpi + PDF; values in `data/`) |
| — | Manuscript v2 update | done | `manuscript/manuscript_v2.md`: 2.8c terminology; 3.7.6; 3.7.7; Discussion 3a(v); Limitations 28-29; Figs 6-8 |
| — | Independent review | RV-FT-04 REVISE (1 major, 12 minor) → corrected → re-review PASS | `REVIEW_FINDINGS.md` |

`reports/` is gitignored, so the reports are also copied to `research_management/finetuning/sprint_20261026/`. `*.pdf` is gitignored, so the figure PDFs exist locally only and the PNGs are tracked.

## 2. Verified performance (test chr9+chr22, 359 tiles, hg38; mean per-track Pearson r)
| system | 152 non-identical tracks (claim set) | all 183 (secondary) |
|---|---|---|
| E4 ridge[pre_raw:union1888] (comparator) | 0.7406 | 0.7426 |
| **PRIMARY: 8-member FT ensemble, RC TTA (locked, dc0bcd5)** | **0.7578**: Δ +0.0173 (unrounded 0.01728); TF-cluster [0.0123, 0.0252]; two-way tile-block × TF [0.0111, 0.0263]; 145/152 improved | 0.7586: Δ +0.0160; 176/183 |
| validation-selected 2-member FT-R3+FT-S04 (post hoc) | 0.7586: Δ +0.0180 [TF 0.0133, 0.0259]; 149/152 | see `ablation_summary.csv` |
| validation-selected 4-member (post hoc) | 0.7607–0.7610: Δ +0.0201 to +0.0204 | |

- The primary was independently reproduced from stored per-member predictions: identical r, Δ and CIs.
- The claim rule (pre-specified) is met. Only the primary carries the claim.

**Best-performing model.** The best model *by the locked protocol* is the primary ensemble. Its checkpoints are:
- `results_ft/runs/{FT-S03,FT-R1,FT-R2,FT-R3}/best_model.pt` (partial-4; 310.4 MB each)
- `results_ft/runs/{FT-S04,FT-L1,FT-L2,FT-L3}/best_model.pt` (LoRA r16; 19.6 MB each)

The trainable state only is saved. Frozen weights reload from `hf_cache/hub/models--johahi--borzoi-replicate-{0..3}`.

Post hoc, the 2-member ensemble FT-R3 + FT-S04 matched the primary's gain at 1/4 of the inference cost. It did so only because the primary contains the failed member FT-L2, and it is about 0.002 below the FT-L2-free 4–6-member ensembles (0.7607–0.7610). None of these is a locked result; use them only after a new held-out evaluation.

## 3. Scientific conclusions
1. The fine-tuned system beats the pre-specified linear readout by a small margin: +0.017 r, about +0.024 in r² and prediction R². The gain is consistent across tracks and cell lines. No leakage was found.
2. The fine-tuning is partial (last 4 of 8 transformer blocks + U-Net decoder + `final_joined_convs`; 77.6 M trainable) or LoRA (frozen trunk, 53 rank-16 adapters; 4.9 M trainable). It is neither head-only nor full. Pretrained trunk weights changed in the partial arm (median relative change 0.02) and are unchanged by construction in the LoRA arm.
3. **Attribution is not established.** Frozen-feature MLP heads also beat the ridge on validation; about 0.005 of the test gain comes from RC test-time averaging; ensembling adds about 0.006–0.009 over single members.
4. **Ensembles.** Validation-selected ensembles of 3–6 members without FT-L2 score 0.7607–0.7610. The 2-member ensemble is about 0.002 lower (no paired CI). Member diversity comes mainly from the pretrained replicate (same-replicate pairs r 0.991 vs 0.97 across replicates).
5. **FT-L2 failure.** FT-L2 is normal on validation but fails on chr9:20–32 Mb of the test set: it over-predicts all tracks, median ×2.9. The equal-weight ensemble absorbs the failure, and the primary Δ is +0.0178 with that region excluded.
6. **Where the gains are.** Gains concentrate in tracks with weak ridge baselines and sparse signal (Spearman −0.34 / −0.36). These include tissue-lineage factors in less-represented lines (GATA6, HNF4A, FOXA2 in Caco-2/LS174T/LS180). POLR2A gains least. This is post hoc and confounded with baseline r.

## 4. Unresolved limitations
- **Test set.** Only two test chromosomes, already development-exposed. About 44% of test bp lie in Borzoi training folds.
- **Comparison and selection.** The comparison is system vs system. The ablation is post hoc, with no paired CIs between configurations. Each member is a single seed.
- **FT-L2.** Its root cause is unknown. Validation did not reveal it.
- **Variant scoring.** No variant-scoring or MPRA use of the fine-tuned models. BL-007 (MPRA sign convention) is still open.
- **Fold stratification.** No Borzoi-fold-stratified analysis of the FT gain.

## 5. Incomplete experiments / remaining computational work
None are running or half-finished. The ablation test evaluations are all complete: 17 of 17 configs, each evaluated once. Do not re-run test-eval; it refuses existing configs.

| item | compute | GPU estimate |
|---|---|---|
| Borzoi-fold-stratified Δ (existing per-tile stats + `exposure_audit/tile_fold_overlap.tsv`) | CPU, < 30 min | 0 |
| FT-L2 root cause: per-layer activation comparison FT-L2 vs FT-L1 on chr9:20–32 Mb; check adapter scale and which LoRA module diverges | GPU inference only | < 0.5 GPU-h |
| Second seed of FT-L2 (and of one other member) to test whether the failure is seed- or replicate-specific | training | about 1.1 GPU-h per LoRA member, about 1.0 per partial member (A100-80GB, 4 epochs, 18,636 steps, ≤ 4 GB) |
| New held-out evaluation of the post hoc 2-/4-member ensembles: retrain on a new chromosome split (e.g. fold F2 chr6+chr21) with fresh test lock | training 2–8 members | 2–9 GPU-h |
| Pre-registered variant scoring (MPRA P1) with the FT ensemble | inference: 5,729 variants × 2 alleles × 2 strands × members | about 1 GPU-h for 2 members, about 4 GPU-h for 8 |

## 6. Recommended next experiments (priority)
1. **Fold-stratified analysis** (free). It decides whether the gain survives on test sequence outside Borzoi training folds.
2. **FT-L2 diagnosis** (< 0.5 GPU-h), plus one extra seed (about 1 GPU-h). This is a robustness question for any single-model use.
3. **Independent held-out split** with a committed lock (2–9 GPU-h). It is the only way to confirm the smaller-ensemble results and to remove the chr9/chr22 development exposure.
4. **Factorial on the new split**: frozen head vs partial vs LoRA × RC TTA × 1/2/4 members. It separates trunk adaptation from ensembling and TTA.
5. **Variant scoring** with the FT ensemble under a new pre-registration (needs BL-007 resolved for P2).
Total suggested next-session GPU: about 6–12 A100-hours.

## 7. Storage, persistence and checkpoint verification
- **Persistent.** `results_ft/` resolves to `/panfs/accrepfs.vampire/home/yejz1/experimental_research/results_ft` (ACCRE home on panfs), about 25 GB, persistent across VM sessions. It is gitignored (`results*/`).
- **Checkpoint integrity.** `research_management/finetuning/checkpoint_verification.md` (2026-10-26): all 8 `best_model.pt` SHA-256[:16] match `CHECKPOINT_INVENTORY.md`.
- **Frozen weights.** HF weights are in `hf_cache/` in the repo, persistent; `HF_HUB_OFFLINE=1` is set in `model_ft.py`.
- **Home quota (2026-10-26 19:40).** 147.6 GB used of 186.3 GB soft.
- **NOT persistent.** `/tmp/yejz1_ft/pooled128` (43 GB) is node-local and needed by `ablation.py`, `bio_analysis.py` and `compare_ft.py`'s order check. Regenerate it with `./torch-venv/bin/python analysis/ft/build_pooled.py`, which reads the trunk cache `/nobackup/user/yejz1/crc_phase2/trunk_cache` (not backed up).
- **New artifacts.** Per-member test predictions are in `results_ft/ablation/test_members/<RUN>/test_pred_rc.npy` (fp16, 118 MB each) and ablation stats in `results_ft/ablation/test_eval/`.

## 8. Exact resume commands
```bash
cd /home/yejz1/experimental_research && git switch research/crc-autonomous-study
./torch-venv/bin/python analysis/ft/build_pooled.py                     # only if /tmp/yejz1_ft/pooled128 is missing (new VM)
./torch-venv/bin/python analysis/ft/verify_ckpts.py                     # checkpoint integrity
CUDA_VISIBLE_DEVICES= ./torch-venv/bin/python analysis/ft/ablation_report.py   # rebuild ablation tables/figures from saved evals
CUDA_VISIBLE_DEVICES= ./torch-venv/bin/python analysis/ft/bio_analysis.py      # rebuild biological analysis + figures
CUDA_VISIBLE_DEVICES= ./torch-venv/bin/python analysis/ft/ftl2_diag.py         # FT-L2 diagnostic
./torch-venv/bin/python analysis/ft/predict_ft.py --run FT-L2 --split val      # reload a member (refuses test unless locked)
# resume/extend training of a member (new run id!): copy argv from results_ft/runs/<RUN>/run_cmd.json, change --run-id/--seed, run analysis/ft/train_ft.py
```
Test evaluation of any **new** configuration requires a new row in a committed lock file (`TEST_LOCK.csv` / `ABLATION_LOCK.csv`). The scripts refuse otherwise.
