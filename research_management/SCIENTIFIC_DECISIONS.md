# Scientific Decisions

Record each decision as: ID · date · decision · rationale · evidence · alternatives considered · decided by.

## SD-001 · 2026-10-19 · Metric definitions
- Decision: report Pearson r, squared Pearson r (r²), and prediction R² (1 − SS_res/SS_tot) as distinct,
  explicitly labelled quantities. R² for pretrained baselines is labelled "validation-calibrated R²".
- Rationale: raw-scale R² of an uncalibrated pretrained track is meaningless (previously observed −193).
- Evidence: `reports/qing_deliverables/Assumptions.md`, `Scientific_Review.md`.
- Decided by: user requirement (setup).

## SD-002 · 2026-10-19 · Reuse Qing's completed analysis
- Decision: treat `reports/qing_deliverables/` as completed; re-verify, do not re-run, unless a new
  question requires it (record the justification here).
- Decided by: user requirement (setup).

---
## Proposed by research-strategist (2026-10-19) — status: PROPOSED, pending user approval / reviewer gate

- **SD-003 Variant set.** Provisionally Chen et al., Nat Commun 15:3557 (2024), doi 10.1038/s41467-024-47399-x,
  Supplementary Data 6 (5,741 CRC CCVs; source build INFERRED, not yet opened) as present in hg38. Coordinator-verified: input has 5,739 unique
  positions expanded to 10,148 rows (2,419 / 2,231 / 1,089 positions with 1 / 2 / 3 ALTs) → most extra rows are
  not the GWAS-tested allele. Unit of analysis = CCV position with the GWAS-reported allele; extra ALTs exploratory.
  Identity UNVERIFIED until the supplementary file is matched (E1).
- **SD-004 Primary SNP validation** = Law et al., Nat Genet 56:2104 (2024), doi 10.1038/s41588-024-01900-w, MPRA allelic
  hits, tested within locus with locus-clustered inference. Secondary = GTEx v8 colon eQTL direction.
- **SD-005 Pre-specified primary variant score** per model (proximal |log2 ALT/REF| over 501 bp centre, max over
  CRC-cell-line DNase + TF ChIP; AlphaGenome quantile version), frozen before validation labels are joined.
- **SD-006 TL inference unit** = TF-clustered bootstrap; uncontaminated tracks primary (159 as proposed; RV-002: 155 after excluding ≥28), 183 secondary;
  track-level tests descriptive only.
- **SD-007** Validation-selected (non-oracle) track matching is primary for all pretrained baselines; oracle = upper bound.
- **SD-008** No "CRC-specific adaptation" claim without a supervised baseline (ridge/linear probe on pretrained
  outputs) on identical splits.
- **SD-009** chr9/chr22 are development-exposed; robustness claims require ≥3 seeds and chromosome-fold CV.
- **SD-010** Retire untraceable/contradicted draft claims: 0.610/0.502 ablation and +28%/+9%; inverse-signal
  weighting; "three cell lines"; "generous baseline ⇒ conservative gain"; track-level p≈1e-10 as primary inference;
  149-target subset means; "beats AlphaGenome/Enformer" (until E7); any causal/clinical SNP language.
  RV-002 additions: low-variance "automatic down-weighting"; "outperforms on held-out validation"; "converging
  evidence" from the ablation; performance-motivated trunk tap; val t 0.00017 / Wilcoxon 1.1e-9; README "deflate";
  "24 contaminated" → "≥28".

---
## RV-002 scientific-reviewer recommendations on SD-003..SD-010 (2026-10-19) — for user decision
| SD | Reviewer recommendation | Modification proposed |
|---|---|---|
| SD-003 | approve with modification | Unit = position (5,739). Positions whose GWAS allele cannot be matched to Supp Data 6 are **excluded, not guessed**. Reconcile 5,741 vs 5,739. Verify hg38 positions against rsIDs (dbSNP/Ensembl). Drop "GRCh37" until the file is opened. |
| SD-004 | approve with modification | Conditional on BL-002. Pre-specify a fallback if only a hit list is available (within-locus hit vs non-hit). Report CCV∩MPRA overlap and power before unblinding. eQTL concordance is only partly independent (AlphaGenome/Borzoi trained on GTEx RNA-seq). Harmonise eQTL alleles to the tested allele. |
| SD-005 | approve with modification | Add a **signed** companion score (needed for the direction sign test). Freeze the track-list file, pseudocount, 501 bp aggregation (sum vs mean), RC averaging, and the AlphaGenome quantile definition. Add a mean-over-tracks sensitivity (max favours models with more tracks). |
| SD-006 | approve with modification | Primary set = 155 (excluding ≥28), pending A19. Report the cell-line-cluster CI (or two-way bootstrap) alongside the TF-cluster CI. State that region uncertainty spans only 2 chromosomes. |
| SD-007 | approve with modification | Selection on chr1/chr8 (validation) only. Add a sensitivity analysis excluding identical-experiment candidates from the vocabularies. AlphaGenome needs a validation-split API run. |
| SD-008 | approve | Ridge on pretrained outputs at the matched grid, same training chromosomes, λ chosen on validation. |
| SD-009 | approve with clarification | Chromosome-fold CV tests head robustness only. It does not remove backbone sequence exposure. |
| SD-010 | approve with modification | Additions above. |
Status of all SD-003..SD-010: still PROPOSED until the user decides.

---
## USER DECISIONS · 2026-10-20 (overnight autonomous session instructions)
SD-003..SD-010 are **APPROVED with the RV-002 scientific-reviewer modifications** (table above), with the following
user-specified details. Decided by: user (2026-10-20), recorded by coordinator.

- **SD-003 (approved, modified).** Two datasets:
  (a) **Exploratory set** = the complete 10,148-row S6_CCV input (all ALTs) is preserved for exploratory
  mutational-sensitivity analysis; no valid prediction is discarded.
  (b) **Allele-matched set** = a separate dataset of positions whose reported allele(s) are confirmed against the original
  source (Chen 2024 Supp Data 6 or equivalent); this is the only set used for claims about the reported CRC-associated variants.
  Missing alleles are never guessed; unmatched positions are excluded from (b) and listed. 5,741 vs 5,739 must be reconciled.
- **SD-004 (approved, modified)** as RV-002; conditional on BL-002.
- **SD-005 (approved, modified)** as RV-002: signed + absolute scores, frozen track-list file, pseudocount, aggregation, RC
  averaging, AlphaGenome quantile definition, mean-over-tracks sensitivity — frozen in a committed file **before** any validation label is joined.
- **SD-006, SD-007, SD-008, SD-009, SD-010 (approved)** with RV-002 modifications.
- **SD-011 Manuscript framing.** Provisionally adopt **Framing B** (computational annotation of CRC-associated variants with
  multiple genomic foundation models). If independent MPRA validation cannot be obtained (BL-002), fall back to **Framing A**
  (leakage-aware benchmark of transfer learning for CRC TF binding).
- **SD-012 Generalisation (E8 approved).** Investigate the ~10 cell lines / tracks possibly absent from pretrained training
  data. Actual training exposure (by GSM/ENCSR/experiment identity in the Borzoi/Enformer target vocabularies) must be verified
  before the analysis is interpreted as generalisation.
- **SD-013 CLAUDE.md contamination note** updated to "≥28 (24 LoVo GSE51142 by GSM + 4 ENCODE colon ENCSR, tracks 145–148);
  lower bound pending BL-004" — approved by user 2026-10-20.
- **GPU authorization (BL-003 resolved).** User authorized use of the allocated A100 for justified Phase 2 experiments (2026-10-20).
- **SD-014 · 2026-10-20 · Frozen variant-scoring specification.** `variant_scoring/SCORING_SPEC.md` + `variant_scoring/track_lists/`
  frozen and committed before any validation label (Law 2024 MPRA, GTEx/CEDAR eQTL) is joined. Borzoi-TL primary = 152 tracks
  (31 identical-experiment excluded); pretrained Borzoi / Enformer = 132 keyword-selected CRC tracks (126 TF ChIP + 6 DNase; 118 LoVo —
  noted imbalance; per-cell-line sensitivity to be reported); AlphaGenome headline = quantile version, raw log2 = sensitivity.
  512 bp centre (bin-aligned; deviation from the 501 bp in SD-005), sum over bins, pseudocount 1, RC-averaged. Decided by coordinator under approved SD-005.
- **SD-015 · 2026-10-20 · Allele source for the allele-matched set (deviation from SD-003 wording).** Chen 2024 Supp Data 6
  (5,741 CCVs, hg19, unique rsIDs) contains no alleles. Alleles are taken from the GWAS Catalog study cited in Chen 2024's data
  availability (GCST90129505; harmonised GRCh38 file, GRCh37 original for 4 rows). "Matched" = REF/ALT equals the GWAS-tested pair;
  it does not identify the causal/risk allele (risk-by-beta-sign column provided). 5,729 matched; 12 S6 CCVs unmatched (10 no allele
  info, 2 rsID drift: rs149685934, rs1312157506); 5,741 vs 5,739 reconciled. 544 palindromic rows rely on harmonisation strand.
  Accepted by coordinator (scientifically equivalent primary source; no guessed alleles). Evidence: `variant_provenance/PROVENANCE.md`.
- **SD-016 · 2026-10-20 · E2/E6 analysis plan frozen** (`variant_scoring/E2_ANALYSIS_PLAN.md`) before unblinding: Law MPRA ST4
  positives vs flag-0 negatives, within Chen region; stratified AUROC + region bootstrap + permutation; direction sign test; Holm
  over 4 models; distance-to-TSS and annotation-count baselines; framing rule.
- **SD-017 · 2026-10-20 · MPRA direction (P2) is uninterpretable until the logFC sign convention is resolved (post-unblinding).**
  After the interim AlphaGenome look showed P2 concordance 0.311 [0.232, 0.395], a model-independent check
  (`external_data/validation/MPRA_SIGN_CONVENTION.md`) found the ST4 sign UNRESOLVED: legend says "with respect to the REF allele";
  MPRAnalyze logFC = log(non-baseline/baseline) with alphabetical default baseline (would give REF/ALT if coded alt/ref; Law code not public);
  rs6983267 literature (G allele stronger TCF4 binding: Tuupanen 2009 doi 10.1038/ng.406; Wright 2010 doi 10.1128/mcb.01384-09) with ST4
  logFC +0.16 (ALT T) suggests REF/ALT; GTEx v10 eQTL concordance under ALT/REF 0.57 (p 0.17) weakly suggests ALT/REF. Decision: P1
  (sign-free within-locus discrimination) remains the primary validation; P2 is reported exactly as pre-specified with an explicit
  "sign convention unresolved" caveat and is excluded from the framing rule; no post-hoc re-signing. The model-agreement pattern across
  models (sign-agnostic |concordance − 0.5|) may be reported as post-hoc exploratory only. BL-007 opened. Decided by coordinator.
- **SD-018 · 2026-10-20 · Slim trunk cache (storage-forced deviation, BL-008).** Cache keeps forward trunk embeddings, targets and
  pretrained Borzoi outputs for 1,888 channels (1,882 filtered ChIP-TF + 6 CRC DNase); RC embeddings dropped (regenerable intermediates;
  905 already-written RC files removed by the engineer). Retrained heads (E5 seeds/folds, E8) = "cache recipe: shift-emulated, no-RC";
  the original checkpoint remains the headline TL model. E4 ridge primary features = 1,888 channels (not all 7,611); sub-variants 1,882 and
  132 CRC tracks. Quota guard stops the build above 172 GB. Decided by coordinator.
- **SD-019 · 2026-10-20 · SD-008 rule triggered (E4, EXP-013).** TL − best pre-specified ridge is negative with CI excluding 0 (all ridge
  variants, the trunk linear probe, seeds 1–3 and fold F2). Claims of Borzoi-TL superiority/CRC-specific adaptation are withdrawn; TL results
  are reported descriptively. Borzoi-TL remains one of the four pre-registered E2 models; it is not the methodological contribution.
  Strategist memo 2026-10-20; decided by coordinator (pre-registered rule, no post-hoc change).
  Primary inference (SD-006, TF-cluster, not_identical152): TL − ridge[pre_raw:union1888] −0.0555 [−0.0756, −0.0428]; cell-line-cluster
  [−0.0828, −0.0476]; TL − trunk probe −0.0378 TF-cluster [−0.0525, −0.0289] (exposure-symmetric comparison). Per-seed/per-fold paired CIs: see
  E5 outputs (RV-006 #1). TL > pretrained valsel is not robust to cell-line clustering (orig all183 [−0.018, 0.041]).
- **SD-020 · 2026-10-20 · Framing still decided by E2_ANALYSIS_PLAN §7 at the final 4-model run.** Titles fixed before E2 final —
  B: "Leakage-aware evaluation of genomic foundation models for colorectal cancer risk-variant annotation and transcription-factor binding";
  A: "A leakage-aware benchmark shows the original head-only Borzoi fine-tuning recipe does not exceed a linear readout for colorectal-cancer TF binding" (RV-006: scoped to the one recipe tested — Poisson MLP head, batch 1, lr 1e-4, 8 epochs; loss/optimisation as the explanation is untested).
  The track-prediction benchmark is reported as a negative result under either framing.
- **SD-021 · 2026-10-20 · No new head architectures (e.g. MLP) are evaluated on chr9/chr22 in this phase** (forking paths on a
  development-exposed test set; the linear probe already settles the comparison).
- **SD-022 · 2026-10-20 · Ridge-head variant scoring NOT run this session.** It would require new GPU REF/ALT raw scoring after two interim
  E2 looks and a new amendment; value judged lower than the risk of forking paths. May be proposed later as exploratory (outside Holm/§7).
- **SD-023 · 2026-10-20 · E4 reporting.** TL − ridge reported with tile-block, TF-cluster and cell-line-cluster CIs (SD-006). Exposure-controlled
  ridge (feature channels matching the target's GSM, or same cell line + TF, removed) and Borzoi-training-fold stratification of test tiles are
  labelled sensitivity analyses; they cannot overturn SD-019.
- **SD-024 · 2026-10-20 · Framing B adopted per E2_ANALYSIS_PLAN §7 (pre-registered rule; E2 final, EXP-017).** All four headline models meet
  the three criteria (Holm p < 0.05; CI excludes 0.5; Δ vs distance baseline CI excludes 0). Title = SD-020 B. Required qualifiers: no model is
  shown to exceed the Chen annotation-count baseline (paired Δ CIs include 0); no model differs from another (paired Δ CIs include 0); signal is
  weak on the held-out-chromosome subset (descriptive, 10 regions); P2 direction uninterpretable (SD-017); MPRA is episomal; Borzoi/AlphaGenome
  training includes GTEx and ENCODE data overlapping CRC-relevant assays. Pending scientific-reviewer confirmation (RV-007).
