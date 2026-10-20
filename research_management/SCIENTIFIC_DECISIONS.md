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
