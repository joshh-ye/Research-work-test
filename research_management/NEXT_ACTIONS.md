# Next Actions

Ordered; the coordinator picks from the top. Updated 2026-10-20 08:15 CDT (overnight session).

## Done this session (see PROJECT_STATE.md, EXPERIMENT_REGISTRY.csv EXP-005..016)
Exposure audit, E1 provenance, frozen scoring spec + E2 plan (A1–A3), E7 non-oracle, E2 interim (AG, Enformer), slim trunk cache (BL-008),
E4 ridge/probe + sensitivity, E5 seeds/folds, E8, RV-006 corrections, manuscript v2 track sections, Fig 6, provenance URLs.

## In progress
1. Borzoi/TL variant scoring (EXP-008, PID 3611891; resume `nohup ./torch-venv/bin/python variant_scoring/score_borzoi.py`).
2. Fold-seed extras (queue_downstream2.sh; rerun safe). After it ends: `./torch-venv/bin/python analysis/priorityD/06_summaries.py`
   then `./torch-venv/bin/python figures/scripts/fig6_e4_benchmark.py`.

## Next
3. When `results_phase2/variant_scores/borzoi/summary.tsv` exists: `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --out results_phase2/e2_validation/final`
   (Holm + §7 framing); log in EXECUTION_LOG; register EXP-008 complete + EXP-017 (E2 final); regenerate fig 5; add Borzoi to EXP-012
   (`analysis/exploratory/model_concordance.py --models ...`).
4. scientific-reviewer on E2 final (+ RV-006 corrections check) → apply §7 framing (SD).
5. Manuscript §3.5, abstract, title per framing; Phase 4 review; Phase 5 final review; morning report `reports/OVERNIGHT_PROGRESS_REPORT.md`.
6. Open (user): BL-007 MPRA sign convention (contact Law/Houlston or EGA); bigwig download source (PI/Qing); AlphaGenome training manifest.
