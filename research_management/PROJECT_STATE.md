# Project State

Last updated: 2026-10-19 (setup session)

## Current phase
**Setup complete — Phase 1 not started.**

## Repository snapshot (at setup)
- Branch `research/crc-autonomous-study` created from `research/crc-manuscript-study` @ `ff88be2`.
- Uncommitted pre-existing work carried over, intentionally left uncommitted:
  - `.gitignore` modification (ignores `*.tar*`, `*.zip`, `results*/`, `reports/`).
  - `alphagenome_variant_effects/` (≈26 GB, untracked; contains venv, cache, inputs, outputs).
- No prior `.claude/` config, `CLAUDE.md`, or `research_management/` existed.

## Prior results (from previous sessions — re-verify in Phase 1, do not re-run)
- Borzoi-TL, test chr9+chr22, shared 896×128 bp grid: mean r 0.682, R² 0.481 (all 359 tiles).
- Pretrained Borzoi re-run: oracle r 0.664; validation-selected r 0.661; validation-calibrated R² 0.444.
- AlphaGenome 0.661 and Enformer 0.644: August oracle values, never re-run, no per-tile stats.
- 24 LoVo tracks (GSE51142) were training data of pretrained Borzoi and Enformer → contamination.
- August-slide ablation numbers (0.6104 / 0.5022) are untraceable; do not quote.
Source: `reports/qing_deliverables/` (Scientific_Review.md, Final_Acceptance_Review.md).

## Open questions
- Genome build of each dataset and of the SNP inputs (to confirm in Phase 1).
- Status and validity of the AlphaGenome variant-effects outputs.
