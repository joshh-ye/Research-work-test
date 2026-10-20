# E2/E6 Pre-specified Validation Analysis Plan (frozen before unblinding)

Status: FROZEN 2026-10-20 by coordinator, before any model score was joined to any validation label.
Complements `SCORING_SPEC.md` (SD-005/SD-014). Decisions: SD-003, SD-004, SD-015, SD-016.
Label-only counts below were computed without looking at model scores.

## 1. Variant sets
- **Allele-matched set (primary)** = `variant_provenance/allele_matched_variants.tsv` (5,729 rows / positions;
  REF/ALT = allele pair tested in GWAS Catalog GCST90129505, Chen 2024 S6 CCVs, hg38 1-based). Joined to model scores by
  `S6_CCV_input_key` (`chr:pos:ref:alt`).
- **Exploratory set** = all 10,148 rows (`variant_provenance/exploratory_all_rows.tsv`); used only for mutational-sensitivity
  description (score distributions, model–model concordance), never for validation claims.
- Locus unit = Chen 2024 `chen_region` (142 regions). Signal (`chen_lead_snp`) is a secondary cluster unit.

## 2. Validation labels
### 2a. Law et al. 2024 MPRA (primary; SD-004)
- Tested universe = matched CCVs whose rsID (or hg38 position) is in Law ST2 (8,880 tested variants).
- **Primary positive** = in ST4 (allelic FDR < 1e-3 in ≥1 of HT29, SW403, HCEC-1CT); equals ST2 flag = 2 for 265/275 rsIDs.
- **Negative** = tested, ST2 flag = 0 in all three lines.
- **Excluded from primary** = ST2 flag = 1 in any line without ST4 membership (168 variants genome-wide; meaning of flag 1 is not
  documented in the table — treated as ambiguous).
- Sensitivity S1: positives = any flag ≥ 1.
- Sensitivity S2: per-cell-line labels (HT29, SW403 tumour; HCEC-1CT normal).
- Allele orientation: ST4 REF/ALT compared with the matched REF/ALT; swapped rows have logFC sign flipped; any non-matching pair
  (not a swap) → excluded and listed.

### 2b. eQTL (secondary; partly NOT independent — AlphaGenome/Borzoi trained on GTEx RNA-seq)
- CEDAR transverse colon (eQTL Catalogue credible sets; microarray, more independent) and GTEx v10 colon transverse / sigmoid
  significant pairs (`chr_pos_ref_alt_b38`, slope = ALT effect). Allele-harmonised to the matched REF/ALT; swaps flipped.
- Label = CCV is a significant eQTL (any gene) vs not, within locus. Direction analysis only for AlphaGenome RNA-seq gene-level
  scorers (colon biosamples) vs eQTL slope for the same gene; chromatin/TF scores have no gene-level direction and are not used for direction.

## 3. Scores (from SCORING_SPEC.md, unchanged)
Per model: primary = max-over-tracks |log2FC| (512 bp, RC-averaged); signed = signed log2FC at arg-max track;
sensitivities = mean |log2FC| and mean signed over tracks. Models: AlphaGenome (quantile headline; raw sensitivity),
pretrained Borzoi (132 tracks), Borzoi-TL (152 primary; 183 sensitivity), Enformer (132 tracks).

## 4. Primary statistics
- **P1 Within-locus discrimination.** For each region with ≥1 positive and ≥1 negative: AUROC of the primary score
  (positives vs negatives within region). Summary = mean of per-region AUROC weighted by n_pos × n_neg (= stratified
  Mann–Whitney). Null = 0.5. Inference: region-clustered bootstrap (10,000 resamples, seed 20261020, percentile 95% CI) and
  within-region label permutation p (10,000).
- **P2 Direction.** Among MPRA positives with |signed score| > 0: fraction with sign(model signed score) = sign(MPRA logFC
  ALT vs REF; cell line with smallest FDR). Null 0.5; region-clustered bootstrap CI; exact binomial p as descriptive.
- Multiple testing: Holm across the 4 models' headline scores for P1, and separately for P2 (8 primary tests total).
  All sensitivities, eQTL analyses and pairwise model differences are secondary (BH-FDR within family, labelled exploratory).
- **Model comparison:** paired Δ(stratified AUROC) between models with the same region bootstrap (secondary).

## 5. Pre-specified controls / sensitivities
- Distance to nearest GENCODE v46 protein-coding TSS (from `alphagenome_variant_effects/input/annotations`): within-locus AUROC of
  −|distance| as a trivial baseline; conditional logistic regression (strata = region) of label on score + log10 distance.
- Chen 2024 epigenomic annotation count (S6 0/1 flags summed) as a second trivial baseline.
- Exclude palindromic variants (S3); exclude chromosome-end padded windows (S4); exclude the 242 AlphaGenome shifted windows (AG only).
- Signal-level (lead SNP) clustering instead of region (S5).
- Leakage note: MPRA cell lines (HT29, SW403, HCEC-1CT) are not among the Borzoi-TL target cell lines; HT-29 has 1 TL track.

## 6. Power statement (before unblinding)
Report n regions informative for P1, n positives, n negatives, n flag-1 excluded, and the minimum detectable stratified AUROC
(approx., region bootstrap under null) before computing any score–label statistic.

## 7. Interpretation rules
- Framing B (SD-011) is supported only if P1 for ≥1 model has Holm-adjusted p < 0.05 and its CI excludes 0.5, and it exceeds the
  distance baseline (CI of Δ vs distance excludes 0). Otherwise Framing A.
- Larger predicted effects do not imply greater biological accuracy; no causal or clinical claims.
