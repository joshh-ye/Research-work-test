# Next Actions

Ordered; the coordinator picks from the top. Updated 2026-10-20 (overnight session).

## Done this session
- User approved SD-003..SD-013; GPU authorized. E1 provenance (EXP-007), exposure audit (EXP-005), exposure-stratified
  stats (EXP-006), validation data availability, scoring spec frozen (SD-014, e041498), E2 plan frozen (SD-016, b59eea1).

## In progress
1. RV-004 scientific-reviewer design gate (E1, exposure audit, EXP-006, scoring spec, E2 plan, Priority D design).
2. GPU (after BL-008 home-quota incident): Borzoi/TL variant scoring resumed (EXP-008, PID 3611891; resumable — rerun
   `nohup ./torch-venv/bin/python variant_scoring/score_borzoi.py`); Enformer EXP-009 DONE. Slim trunk cache (SD-018) building via
   `analysis/priorityD/queue_cache.sh` → `queue_downstream2.sh` (E4 ridge union1888, probe, pretrained valsel, E5 seeds, E8, E5 folds).
   Cache + E7 outputs live in /nobackup/user/yejz1/crc_phase2/ (symlinked). If interrupted: rerun both queue scripts with nohup (skip-if-done).
   WATCH QUOTAS: `accre_storage` (home soft 186 GB / hard 232.8 GB).
3. E2 code (blinded) being implemented in `analysis/e2_validation/`.

## Next
4. E2 interim looks done for AlphaGenome and Enformer (EXP-011). When Borzoi summary.tsv exists: final 4-model run
   `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --out results_phase2/e2_validation/final` (Holm + framing).
5. Reviewer on E2 results + E4/E5/E8 results → framing decision (SD-011 rule in E2_ANALYSIS_PLAN §7).
6. Phase 3 figures/tables; Phase 4 manuscript revision (retire SD-010 claims); Phase 5 reproducibility docs.
7. Open: AlphaGenome non-oracle (needs validation-split API run, E7); AlphaGenome training manifest (identical-experiment check).
