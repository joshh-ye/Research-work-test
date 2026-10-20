# Next Actions

Ordered; the coordinator picks from the top. Updated 2026-10-20 (overnight session).

## Done this session
- User approved SD-003..SD-013; GPU authorized. E1 provenance (EXP-007), exposure audit (EXP-005), exposure-stratified
  stats (EXP-006), validation data availability, scoring spec frozen (SD-014, e041498), E2 plan frozen (SD-016, b59eea1).

## In progress
1. RV-004 scientific-reviewer design gate (E1, exposure audit, EXP-006, scoring spec, E2 plan, Priority D design).
2. GPU: Borzoi/TL variant scoring (EXP-008, PID 3442888, resumable: rerun the RUNS.md command); Enformer (EXP-009, PID 3444921).
   Queued: trunk cache (`analysis/priorityD/queue_cache.sh`, starts when Enformer exits) → `queue_downstream.sh`
   (E4 ridge, linear probe, pretrained non-oracle, E5 seeds, E8, E5 folds, extras). If the allocation died: rerun both queue scripts
   (wait loops reference old PIDs → they start immediately); every step is resumable/skip-if-done.
3. E2 code (blinded) being implemented in `analysis/e2_validation/`.

## Next
4. After RV-004 PASS (or corrections): unblind E2 for AlphaGenome (scores exist), then Borzoi/TL/Enformer when EXP-008/009 finish.
5. Reviewer on E2 results + E4/E5/E8 results → framing decision (SD-011 rule in E2_ANALYSIS_PLAN §7).
6. Phase 3 figures/tables; Phase 4 manuscript revision (retire SD-010 claims); Phase 5 reproducibility docs.
7. Open: AlphaGenome non-oracle (needs validation-split API run, E7); AlphaGenome training manifest (identical-experiment check).
