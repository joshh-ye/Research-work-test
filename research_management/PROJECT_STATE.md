# Project State

Last updated: 2026-10-19 (Phase 1 session)

## Current phase
**Phase 1 — inventory complete; strategist evaluation done; awaiting scientific-reviewer PASS before Phase 2.**
Full evidence: `PHASE1_INVENTORY.md`. Recovery report: `PHASE1_RECOVERY_REPORT.md`.

## Repository snapshot
- Branch `research/crc-autonomous-study` from `research/crc-manuscript-study` = `experimental_branch` = `ff88be2`.
  Other branches (`main`, `stable-version`, `experimentational_fix`) hold no unique results.
- Uncommitted pre-existing work left as is: `.gitignore` edit; `alphagenome_variant_effects/` (≈26 GB, untracked).
- Genome build: **hg38 throughout** (FASTA/BigWig chromosome lengths match for all 183 tracks; SNP REFs match hg38 1-based). No liftover.

## Completed work (do not repeat)
| Item | Output | Status |
|---|---|---|
| Borzoi-TL training (Slurm 12212537) | `borzoi_ckpt_full/model_head_best.pt` (epoch 5/8, val NLL 0.18523), `results_full/` | done; single seed, no seed set |
| Filtered pretrained-Borzoi baseline (13005163) | `results_baseline_filtered/` | done |
| Four-model benchmark (oracle mapping) | `results_benchmark/` | done; Enformer/AG producing runs unrecorded |
| Qing deliverables (2026-10-11/12) | `reports/qing_deliverables/` | done; two prior Opus reviews, no critical issues |
| AlphaGenome variant scoring (S6_CCV, hosted API) | `alphagenome_variant_effects/output/` | 10,148/10,148 SUCCESS, 0 REF mismatches; **unanalysed** |

## Verified headline numbers (test chr9+chr22, 359 tiles, 183 tracks)
- Borzoi-TL: r 0.6634 (native 32 bp); r 0.682, r² 0.4866, prediction R² 0.4809 (shared 128 bp).
- Pretrained Borzoi: oracle r 0.664; validation-selected r 0.661; validation-calibrated R² 0.444.
- Enformer 0.644, AlphaGenome 0.661 — August **oracle** r, not re-run.
- Δr TL − Borzoi (shared): +0.018 [0.009, 0.026]; +0.030 on 159 uncontaminated; −0.061 on 24 LoVo contaminated; TF-cluster CI [+0.004, +0.028].

## Do not quote (untraceable / conflicting)
- 0.6104 / 0.5022 "no augmentation" ablation and derived +28%/+9% (draft §5.1, manuscript.md, build_presentation.py).
- Validation filtered-Borzoi stats (r 0.625 etc.) — source dir `results_comparison/` missing.
- "42 same-experiment tracks" — use 24 (GSM-identical).
- Never mix native-grid 0.663 with shared-grid 0.682.

## Key gaps
No variant-effect analysis; no Borzoi/Enformer/Borzoi-TL variant scorer; no supervised baseline on pretrained
outputs; Enformer/AG not re-run with non-oracle mapping; no ablations; single split/seed; S6_CCV provenance unknown.
Draft inaccuracies: claims inverse-signal loss weighting (not used), "three cell lines" (10), stale Enformer text.
