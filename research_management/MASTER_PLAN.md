# Master Plan — CRC Autonomous Study

Branch: `research/crc-autonomous-study` · Created 2026-10-19 · Coordinator: Opus main session

## Goal
Assess how well sequence-to-function models (pretrained Borzoi, fine-tuned Borzoi-TL, Enformer,
AlphaGenome) predict CRC regulatory activity, and use them to estimate SNP effects at CRC-relevant
variants, producing a reproducible, reviewed manuscript.

The precise scientific questions are to be refined in Phase 1 with research-strategist input and
recorded in `SCIENTIFIC_DECISIONS.md`.

## Phases
| # | Phase | Lead agents | Exit gate |
|---|---|---|---|
| 1 | Repository discovery and research inventory | research-investigator → strategist | reviewer PASS on inventory |
| 2 | SNP experiments and model analysis | computational-engineer | reviewer PASS per experiment |
| 3 | Statistical analysis and publication figures | computational-engineer | reviewer PASS on stats/figures |
| 4 | Manuscript preparation | coordinator + strategist | reviewer PASS on claims |
| 5 | Reproducibility and GitHub consolidation | computational-engineer | clean-room rerun + reviewer PASS |

Acceptance criteria: `PHASE_ACCEPTANCE.md`.

## Known prior work (do not repeat without justification)
- Qing deliverables (2026-10-11/12): Borzoi-TL vs pretrained Borzoi Pearson r and R², per-tile stats,
  contamination flags → `reports/qing_deliverables/`.
- Four-model benchmark (Aug 2026): `results_benchmark/benchmark_comparison_test.csv`
  (AlphaGenome and Enformer values are oracle-selected and not re-run).
- AlphaGenome variant-effects pipeline (S6_CCV batch): `alphagenome_variant_effects/` (untracked).
