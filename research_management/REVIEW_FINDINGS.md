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

## RV-002 · 2026-10-19 · Phase 1 · inventory gate
Reviewer: scientific-reviewer (opus)
Verdict: **REVISE** (documentation corrections only; no new experiments required)
Scope: PHASE1_INVENTORY.md, PHASE1_RECOVERY_REPORT.md, SCIENTIFIC_DECISIONS.md, PROJECT_STATE.md, PHASE_ACCEPTANCE.md;
re-derivation from `results_full/test_{preds,targets}.npy`, `results_benchmark/*.csv`, `summary_numbers*.json`,
`alphagenome_variant_effects/` input/output, `hg38.ml.fa`.
Re-derived and matching: SNP rows/positions/ALT distribution (10,148 / 5,739 / 2,419-2,231-1,089); REF 0 / 7,757 mismatches
(1- / 0-based); 10,148 SUCCESS; TL r 0.6634 native, 0.6820 / r² 0.4866 / prediction R² 0.4810 shared; Borzoi oracle r 0.6640;
Δr +0.0180 [0.0092, 0.0260]; 159/24 strata +0.0299/−0.0609; TF-cluster CI [0.0044, 0.0281]; validation-selected r 0.6609 /
validation-calibrated R² 0.4442; 51 zero-signal tiles; 24 GSM-contaminated tracks. Metric labelling correct in all state docs.
Findings:
- F1 [major] CONFIRMED (reviewer + coordinator grep): contamination ≥28, not 24. Tracks 145–148 are ENCSR-identical in both vocabularies. Re-stratified 155/28: +0.0304 / −0.0509 — fixed (docs) — status fixed in the Phase 1 review commit (2026-10-19); the A19 lookup remains open (BL-004).
- F2 [minor] CONFIRMED: 10 named cell lines + 20 unannotated tracks; cell-line bootstrap handling SUSPECTED — docs fixed; check open.
- F3 [minor] CONFIRMED: prediction R² 0.4810, not 0.4809 — fixed.
- F4 [minor] CONFIRMED: "~4,400 probably wrong" overstated → 4,409 surplus rows; tested allele unidentifiable at 3,320 positions / 7,729 rows — fixed.
- F5 [minor] CONFIRMED: GRCh37 origin stated as fact → INFERRED; 5,741 vs 5,739 noted — fixed.
- F6 [minor] CONFIRMED: shifted windows = 242 rows / 147 positions — fixed.
- F7 [minor] CONFIRMED: additional unsupported draft claims (draft:95, manuscript:32/55/107, build_presentation.py:517, README "deflate") — added to SD-010 / Recovery §9.
- F8 [minor] CONFIRMED: cell-line-cluster CI [−0.021, +0.038] includes 0, was omitted — added.
- F9 [info] CONFIRMED: chr9/22 backbone exposure unresolvable in repo — BL-005.
- F10 [info] CONFIRMED: zero-signal tiles and pooled Pearson correctly characterised.
SD-003..SD-010 recommendations: see SCIENTIFIC_DECISIONS.md (RV-002 table).
Not applied by coordinator: the CLAUDE.md "24 LoVo" note — awaiting user approval (NEXT_ACTIONS 0c).

## RV-003 · 2026-10-19 · Phase 1 · re-review of RV-002 corrections
Reviewer: scientific-reviewer (opus; same reviewer instance, scope limited to the research_management/ diff)
Verdict: **PASS**
Findings: all 10 RV-002 corrections are applied, accurate and consistent across files; no new errors. Two cleanups were applied
before the commit: (1) Recovery Report "Open items" updated (gate status; BL-004/005/006); (2) RV-002 F1 status wording.
Optional EXP-004 registry note was also updated. CLAUDE.md note and SD-003..SD-010 remain pending user decision.
→ Phase 1 accepted.
