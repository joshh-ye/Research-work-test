# Project State

Last updated: 2026-10-19 (Phase 1 session)

## Current phase
**Phase 1 — ACCEPTED 2026-10-19** (RV-002 REVISE → documentation corrections → RV-003 PASS). Phase 2 not started: it awaits the user's
decisions on SD-003..SD-010, framing, the E8 requirement and the CLAUDE.md contamination note, plus data blockers BL-001/002/004–006 and GPU authorization BL-003.
Full evidence: `PHASE1_INVENTORY.md`. Recovery report: `PHASE1_RECOVERY_REPORT.md`.

## Repository snapshot
- Branch `research/crc-autonomous-study` from `research/crc-manuscript-study` = `experimental_branch` = `ff88be2`.
  Other branches (`main`, `stable-version`, `experimentational_fix`) hold no unique results.
- Uncommitted pre-existing work left as is: `.gitignore` edit; `alphagenome_variant_effects/` (≈26 GB, untracked).
- Genome build: **hg38 throughout** (FASTA/BigWig chromosome lengths match for all 183 tracks; SNP REFs match hg38 1-based: 0 mismatches vs 7,757 if 0-based). No liftover performed here. GRCh37 origin of the SNP input is INFERRED (source table not opened).

## Completed work (do not repeat)
| Item | Output | Status |
|---|---|---|
| Borzoi-TL training (Slurm 12212537) | `borzoi_ckpt_full/model_head_best.pt` (epoch 5/8, val NLL 0.18523), `results_full/` | done; single seed, no seed set |
| Filtered pretrained-Borzoi baseline (13005163) | `results_baseline_filtered/` | done |
| Four-model benchmark (oracle mapping) | `results_benchmark/` | done; Enformer/AG producing runs unrecorded |
| Qing deliverables (2026-10-11/12) | `reports/qing_deliverables/` | done; two prior Opus reviews, no critical issues |
| AlphaGenome variant scoring (S6_CCV, hosted API) | `alphagenome_variant_effects/output/` | 10,148/10,148 SUCCESS, 0 REF mismatches; **unanalysed** |

## Verified headline numbers (test chr9+chr22, 359 tiles, 183 tracks)
- Borzoi-TL: r 0.6634 (native 32 bp); r 0.682, r² 0.4866, prediction R² 0.4810 (shared 128 bp).
- Pretrained Borzoi: oracle r 0.664; validation-selected r 0.661; validation-calibrated R² 0.444.
- Enformer 0.644, AlphaGenome 0.661 — August **oracle** r, not re-run.
- Δr TL − Borzoi (shared): +0.018 [0.009, 0.026]; TF-cluster CI [+0.004, +0.028]; **cell-line-cluster CI [−0.021, +0.038] (includes 0)**.
- Contamination: **≥28 tracks** are training experiments of pretrained Borzoi/Enformer = 24 LoVo GSE51142 (by GSM) + 4 ENCODE colon tracks 145–148 (ENCSR102CSD, ENCSR107EUS, ENCSR667UWT, ENCSR769WKR; by ENCSR; missed by Qing's GSM-only check). Lower bound (Qing A19 open).
  Δr 159/24 split (Qing): +0.030 / −0.061; RV-002 re-stratified 155/28: +0.030 / −0.051 (direction unchanged).

## Do not quote (untraceable / conflicting)
- 0.6104 / 0.5022 "no augmentation" ablation and derived +28%/+9% (draft §5.1, manuscript.md, build_presentation.py).
- Validation filtered-Borzoi stats (r 0.625 etc.) — source dir `results_comparison/` missing.
- "42 same-experiment tracks" (description-based) and "24 contaminated" — use "≥28 identical experiments (24 GSM + 4 ENCSR)".
- Never mix native-grid 0.663 with shared-grid 0.682.

## Key gaps
No variant-effect analysis; no Borzoi/Enformer/Borzoi-TL variant scorer; no supervised baseline on pretrained
outputs; Enformer/AG not re-run with non-oracle mapping; no ablations; single split/seed; S6_CCV provenance unknown.
Draft inaccuracies: claims inverse-signal loss weighting (not used), low-variance down-weighting (not used), "three cell lines" (10 named + 20 colon-tissue/unannotated tracks), stale Enformer text, untraceable validation/ablation claims (see SD-010).
SNP input: 4,409 surplus rows over 5,739 positions; GWAS-tested allele unidentifiable at 3,320 multi-ALT positions (7,729 rows) without the source table (ALTs alphabetical; pattern consistent with dbSNP multi-allelic expansion). 242 rows / 147 positions have chromosome-end-shifted windows (chr17/19/21 q-ter).
