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

## RV-004 · 2026-10-20 · Phase 2 · design gate before unblinding (E1, exposure audit, EXP-006, scoring spec, E2 plan, Priority D design)
Reviewer: scientific-reviewer (opus, native subagent)
Verdict: **REVISE** (documentation/pre-registration only; no GPU or new data)
Verified: E1 counts 5,729/4,396/23/12, 5,741−2=5,739; liftover 5,737/5,739 (2 unmapped), Ensembl 5,739/5,739; 0 REF mismatches;
14 rows checked vs raw GWAS Catalog; SD-015 acceptable. Exposure 31 = 24+3+4 (3 ENCSR in both vocabularies; 18 other HCT-116
ENCODE absent); fold fractions reproduced (Basenji2 first-party bed 0.726 vs HF 0.730). EXP-006 0.682/0.664/0.661, Δr +0.0305/−0.0432,
val-calibrated R² 0.448/0.444 recomputed. Scoring centering, bins (symmetric → exact RC flip), pseudocount on sums, padding counts,
TL mask = 31 identical tracks verified. pytest 23 passed.
Findings:
- [major] E2 power statement not computed — 4,323 tested / 134 pos / 4,098 neg / 91 flag-1; 58 informative regions — fixed in plan Amendment A1 (MDA pending from labels code).
- [major] Track-set confound (pretrained 132 tracks, 118 LoVo; TL 152 HCT-116-dominated) — track-matched comparator + per-cell-line breakdown pre-registered (A1.3).
- [minor] 2 allele swaps at chr17 q-ter (hm_code 11; rs34400960 palindromic) — excluded from P2 (A1.5); logFC-vs-REF source quoted (ST4 legend) (A1.2).
- [minor] AUROC ties (AG quantile compressed) — midranks; raw = tie-free sensitivity (A1.4).
- [minor] EXP-006 causal wording; LoVo confound — rephrased, within-LoVo contrast added (identical −0.061 vs non-identical +0.009) in SUMMARY.md.
- [minor] sign-flip vs CI different estimands — labelled in SUMMARY.md.
- [minor] held-out-chromosome sensitivity — S6 descriptive (A1.7).
- [minor] flag-definition contradiction / "8 tests" wording — fixed (A1.2, A1.6).
- [minor] Borzoi scores in scaled/soft-clipped space — limitation declared (A1.8).
- [minor] palindrome caveat affects orientation only — noted.
- [info] Priority D: label E8 "validation-selected same-TF transfer"; stratify E4 by exposure class; test RC shift-offset sign — forwarded.

## RV-005 · 2026-10-20 · Phase 2 · re-review of RV-004 corrections + blinded E2 implementation
Reviewer: scientific-reviewer (opus; same instance as RV-004)
Verdict: **REVISE** (one code change)
Verified: POWER.md counts independently (134/4,098/91; 58 regions, 130/2,518; held-out 713/19); all 10 A2 decisions match code; paired deltas
share resamples; framing rule correct; 73 channels match targets_human.csv; AG cache 33 tracks × 10,148 rows, 242 shifted flagged; SUMMARY fixes correct. pytest 25 passed.
Findings:
- [major] `--allow-partial` AlphaGenome-first run would silently unblind Borzoi/TL/Enformer on partial chunks — fix: per-model `--models`, finalised runs only, Holm/framing deferred to final run (plan A3) — code change in progress.
- [minor] "track-matched" comparator is mostly non-same-TF, non-CRC (38/152 same-TF) — relabelled; same-TF sensitivity pre-registered (A3).
- [info] AG quantile primary has 254 distinct values; midranks verified.
Unblinding permitted for AlphaGenome after the A3 code change; other models when finalised.

## RV-006 · 2026-10-20 · Priority D milestone (E4 EXP-013, E4 sensitivity EXP-014, E5 EXP-015, E8 EXP-016, SD-018..023) — scientific-reviewer (Opus)
Verdict: **REVISE** (documentation/traceability only; numbers recompute; SD-019 outcome holds).
Findings:
1. [major] SD-019 cites TL−ridge CIs for seeds 1–3 / F2 that no saved file contains (reviewer recompute agrees: seeds −0.059..−0.060, F2 −0.066, F3 −0.083, F4 −0.068, all CIs exclude 0) → add paired TL−ridge and TL−probe per seed/fold to E5 outputs.
2. [major] SD-020 Framing-A title overgeneralises ("head-only transfer learning") — evidence covers one recipe (Poisson MLP, batch 1, lr 1e-4, 8 epochs); probe on same embedding also beats TL, so loss/optimisation is an untested explanation → scope to "the original head-only fine-tuning recipe".
3. [minor] E4 SUMMARY header/limitations say 7,611 features; primary is union1888 (06_summaries.py:71,76).
4. [minor] E4 sensitivity parts C/D were specified after the primary E4 result (SD-023 07:13 vs results 06:12) → "post-primary-result sensitivity analyses".
5. [minor] Quote SD-006 primary inference (TF-cluster, not_identical152: −0.0555 [−0.0756, −0.0428]) plus cell-line-cluster CI.
6. [minor] EXP-015 "TL > pretrained valsel in every fold" rests on tile-block CIs; cell-line-cluster CI includes 0 on orig.
7. [minor] E5 SUMMARY stale (seeds 4–5 done: 0.6819, 0.6822) → regenerate after the queue.
8. [minor] R² labels: E5 folds pretrained valsel and E8 per-track R² columns need "validation-calibrated" vs raw labels.
9. [minor] E8: LEAKY reference should also be reported for the cache-recipe seed1 head (0.5494); selection freedom differs (1,882 any-TF vs few same-TF); LS174T derives from LS180 (≈3 donor clusters; general knowledge); not zero-shot (validation labels of held-out tracks used) → wording "no transfer advantage".
Verified: train-only standardisation, val-only λ, no refit, disjoint folds, no target-derived features, parity of tiles/targets/metric, cluster-CI construction, SD-008 rule and variant set committed before results (a896b9e), worst variant reported. Probe should be foregrounded as the exposure-symmetric comparison.
Actions: SD text (2, 5, 6) by coordinator; code/summary fixes (1, 3, 4, 7, 8, 9) dispatched to computational-engineer.
RV-006 corrections applied 2026-10-20 ~08:00 (engineer; coordinator-verified): #1 e5_seed_vs_ridge.csv / e5_fold_vs_ridge.csv (TL−ridge seeds 1–5 −0.059..−0.061, folds −0.065..−0.083; tile-block, TF- and cell-line-cluster CIs all exclude 0; match reviewer recompute); #3 E4 SUMMARY features text fixed; #4 C/D relabelled post-primary; #7 seeds 4–5 included, run-status line (fold-seed extras pending, queue regenerates); #8 R² labels; #9 E8 second LEAKY reference (cache seed1 0.5494), LS180 CEBPB/CDX2 = 94% of gap, interpretation limits. Re-review deferred to the E2-final review (doc-only fixes).

## RV-007 · 2026-10-20 · E2 final 4-model MPRA validation + framing (SD-024) + RV-006 corrections — scientific-reviewer (Opus)
Verdict: **PASS** (Framing B correctly adopted under the pre-registered rule; numbers recompute; A3 unblinding protocol followed; RV-006 corrections adequate).
Minor findings: (1) rename "held-out-chromosome" subset → "Borzoi-TL val/test chromosomes"; underpowered, not leakage evidence; (2) add qualifiers:
annotation-count baseline alone 0.618 [0.556, 0.684]; HCEC-1CT (normal) ≈ tumour lines → no CRC-specific signal; (3) clogit LR p anticonservative
(LD) → exploratory; region-bootstrap CI before citing; (4) "no model differs" = 6 headline pairs; TL − valsel-mapping +0.032 [0.007, 0.052] unadjusted
(BH q 0.39); annotation-count deltas mislabelled family; (5) P1 weight: top 3 regions 45% → report + equal-weight sensitivity (exploratory);
(6) P2 anti-concordance must not be used to infer REF/ALT; (7) AG RNA vs GTEx eQTL direction 0.5008 — sanity-check before reporting;
(8) stale docs: analysis/e2_validation/RUNS.md, results_phase2/exploratory/README.md, EXP-012 row, manuscript_v2.md:67.
Verified: Borzoi/TL scoring per frozen spec (recompute ≤1.2e-7), resume after BL-008 clean (20 chunks, unique row ids), timings, Holm, permutation, §7.
Actions: (1)(2)(4)(6) SD-024 text updated by coordinator; (3)(5)(7)(8) dispatched to computational-engineer as exploratory/doc fixes.

## RV-008 · 2026-10-20 · Phase 4 manuscript + Phase 5 reproducibility — scientific-reviewer (Opus)
Verdict: **Phase 4 PASS** (≈60 numbers spot-checked, all match; minor 4.1–4.7); **Phase 5 REVISE** (2 major).
Phase 4 minor: 4.1 power figure 0.5847 is the final-run recompute, pre-unblinding = 0.5834/SE 0.0335 (labels/power_statement.tsv); 4.2 abstract
wording implies equivalence; 4.3 data availability says checkpoint not committed (it is); 4.4 "reviewed in lee2023" → "described in";
tuupanen2009/wright2010 uncited; 4.5 "independent models", "clearly better than distance"; 4.6 FIGURES.md Fig 6 legend stale; 4.7 source for
GTEx 0.57 p 0.17; shared-grid crop size; fold-stratification window.
Phase 5: 5.1 [major] borzoi_ckpt_full/*.pt (8.6+8.6+17.2 MB) and 43 results_comparison_*/ files are tracked since ff88be2 (pre-project, already
on origin) — docs say not in git → correct docs; untracking (git rm --cached, no history rewrite) is a user decision (BL-009); 5.2 [major]
README.md not updated; 5.3 .gitignore edit uncommitted (user's own pending edit — left for user, BL-009); 5.4 GWAS Catalog summary-stat columns
in tracked variant_provenance TSVs — licence/sharing is a user decision (BL-009); 5.5 reproduce_cpu.sh rm -rf guard; state same-host clean room.
Verified: clean room 92 files + 24 numbers bit-identical; no restricted data (external_data/EGA/MPRA/GTEx/AG) tracked; no secret patterns.
Also: Phase 4 acceptance requires a strategist PASS (pending).
