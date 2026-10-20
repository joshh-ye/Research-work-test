# Review Findings

One entry per scientific-reviewer / research-strategist invocation.

Template:
```
## RV-### · YYYY-MM-DD · Phase N · <checkpoint>
Reviewer: scientific-reviewer | research-strategist
Verdict: PASS | REVISE | BLOCKED
Scope: <files / commits / results>
Findings: [severity] issue — evidence — required fix — status (open|fixed in <commit>)
```

## Prior review (imported, not re-run)
- Qing deliverables were reviewed in a previous session: `reports/qing_deliverables/Scientific_Review.md`
  and `Final_Acceptance_Review.md`.


## RV-001 · 2026-10-19 · Phase 1 · strategy evaluation
Reviewer: research-strategist (opus; general-purpose fallback with the agent definition verbatim)
Verdict: n/a (strategy) — see PHASE1_RECOVERY_REPORT.md §Strategy
Key findings:
- [critical] S6_CCV input is a multi-allelic expansion (5,739 positions → 10,148 rows); GWAS-tested allele unknown — coordinator-verified — fix: E1 — status open.
- [major] No supervised baseline; TL gain (+0.018 r) may be readout recalibration — fix: E4 — open.
- [major] Single seed/split, chr9/22 development-exposed; 0.610 ablation untraceable — fix: E5 — open.
- [major] Enformer/AlphaGenome comparisons oracle-only — fix: E7 — open.
- [major] LD confounding and multiple testing in SNP arm — fix: SD-004/SD-005 — open.
- [info] Method novelty limited (Li et al. 2026 PLOS Genet, doi 10.1371/journal.pgen.1012145, fine-tuned Enformer on breast/prostate TF ChIP).

## RV-002 · Phase 1 · scientific-reviewer gate — NOT YET RUN
