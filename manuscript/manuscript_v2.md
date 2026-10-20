# Manuscript v2 (skeleton, Phase 4 preparation)

Status: SKELETON, written 2026-10-20 from frozen specs and completed results only. Provisional framing B (SD-011), with fallback to framing A
if the pre-registered framing rule (`variant_scoring/E2_ANALYSIS_PLAN.md` §7) is not met in the final all-model E2 run.
Convention: `[PENDING: EXP-xxx ...]` = result not yet computed or not yet final; no number is given for it.
Every number carries an HTML comment with its source file. Citations are keys in `manuscript/references.tsv` (all DOI-verified).
Supersedes neither existing draft in place (`research_paper_draft.md`, `research_paper_prep/manuscript.md` are untouched); see `manuscript/CLAIMS_AUDIT.md`.

---

## Title (provisional)

Multi-model computational annotation of colorectal cancer risk variants with genomic foundation models, and a leakage-aware benchmark of transfer learning for colorectal transcription-factor binding

(Fallback title under framing A: "A leakage-aware benchmark of transfer learning from Borzoi to colorectal-cancer transcription-factor binding".)

## Abstract (structure only)

**Background.** [Colorectal cancer GWAS and fine-mapping yield thousands of credible causal variants (CCVs), mostly noncoding; sequence-to-function models can score them; no benchmark exists that is both leakage-aware and validated against independent functional data.]
**Methods.** [One sentence each: (i) 5,741 CCVs from Chen et al. with alleles recovered from the GWAS Catalog (5,729 allele-matched); (ii) four models scored with one frozen protocol (AlphaGenome, pretrained Borzoi, Enformer, Borzoi-TL, a frozen-trunk MLP head on 183 CRC TF ChIP-seq tracks); (iii) validation against the Law et al. MPRA, within-locus, region-clustered inference; (iv) training-exposure audit of test tracks and tiles.]
**Results.**
- Model evaluation: [PENDING: final Borzoi-TL vs ridge baseline, EXP-E4; seeds, EXP-E5] plus verified exposure-stratified delta r (see Results 2).
- Variant annotation: [PENDING: EXP-E2 final four-model P1 stratified AUROC with Holm-adjusted p; region-clustered CI]. Interim AlphaGenome-only look exists (interim, Holm deferred).
- Direction concordance: [PENDING: P2, interpretable only after BL-007 / SD-017].
- Distance-to-TSS and annotation-count baselines: [PENDING: final run].
**Conclusions.** [PENDING: depends on framing rule outcome. No causal or clinical claim.]

## 1. Introduction

**1.1 CRC genetics.** [Colorectal cancer is heritable; GWAS have identified many risk loci (huyghe2019). Trans-ancestry fine-mapping of over 254,000 individuals defined 238 independent signals in 142 regions and 5,741 credible causal variants (chen2024); the CCV definition is variants whose conditional P is within two orders of magnitude of the signal's most significant variant (no posterior inclusion probability exists; `variant_provenance/PROVENANCE.md`). Most CCVs are noncoding. High-throughput MPRA has tested 8,880 candidate variants in colon cell lines (law2024).]
**1.2 Sequence-to-function models.** [Basenji (kelley2018), Enformer (avsec2021), Borzoi (linder2025) and AlphaGenome (avsec2026) predict chromatin, TF-binding and expression tracks from sequence and are used to score variants. Their training data (ENCODE, GEO, GTEx) overlap many public CRC assays; this makes test-set exposure a first-order concern for any downstream benchmark. Cross-individual variant-effect prediction is known to be limited (huang2023).]
**1.3 Transfer learning.** [Adapting a pretrained model to a disease-specific track panel by retraining a head has been shown for Enformer on breast and prostate cancer TF ChIP-seq (li2026) and parameter-efficient fine-tuning for Borzoi-family models has been described (yuan2026). Genomic augmentation is reviewed in lee2023. Novelty of the method itself is limited; the contribution here is the leakage-aware evaluation and the variant-annotation validation.]
**1.4 Aims.** [(i) Benchmark a frozen-trunk Borzoi head (Borzoi-TL) against pretrained Borzoi, Enformer and AlphaGenome on 183 CRC TF tracks with exposure-stratified inference. (ii) Score CRC CCVs with the same models under one frozen protocol and test, within locus, whether scores discriminate MPRA-active from inactive variants beyond trivial baselines.]
Note for authors: the Introduction sentences above are outlines. Only the citations in `references.tsv` may be used; no other reference has been verified.

## 2. Methods

### 2.1 Genome build and coordinates
hg38 throughout (FASTA `borzoi_data/hg38/hg38.ml.fa`; chromosome lengths match all 183 BigWigs) <!-- research_management/PROJECT_STATE.md; PHASE1_RECOVERY_REPORT.md §5 -->. SNP coordinates are 1-based; REF alleles were asserted against the FASTA at index POS-1 (0 mismatches of 10,148 rows; a 0-based read gives 7,757 mismatches) <!-- variant_provenance/PROVENANCE.md; PROJECT_STATE.md -->. Source CCV positions are hg19 (Chen supplementary tables are labelled "Chr (hg19)"); hg38 positions of the input were verified by UCSC liftover (5,737/5,739; two unmapped) and Ensembl REST release 116 (5,739/5,739) <!-- variant_provenance/PROVENANCE.md -->. The upstream hg19-to-hg38 conversion that created the S6_CCV scoring input is undocumented.

### 2.2 Transcription-factor ChIP-seq targets
183 BigWig tracks (list: `borzoi_ckpt_full/bw_files.json`), 179 with a GEO GSM and 4 with an ENCODE ENCSR accession. Composition: 10 cell lines (HCT-116 105, LoVo 41, Caco-2 7, GP5d 6, LS174T 5, LS180 5, SW480 4, RKO 3, DLD-1 1, HT-29 1) plus 4 transverse-colon tissue tracks and 1 primary colon tumour sample (12 labels); 77 TFs <!-- exposure_audit/track_exposure.tsv cell_line column; EXPOSURE_AUDIT.md §1; exposure_stratified SUMMARY (77 TF clusters) -->. Provenance: the original CSV Cell_line field named 10 cell lines and was empty for 20 tracks; these were resolved from GEO/ENCODE metadata (source column: csv 163, geo_text 16, encode 4). [PENDING: processing of BigWigs to coverage targets (normalisation/transform) as implemented in `borzoi_code/bigwig_loader.py`; to be described from code in the supplement. `GEO series list`: PENDING table of GSM/GSE per track, derivable from `exposure_audit/track_exposure.tsv`.]

### 2.3 Tiles and splits
The genome is cut into non-overlapping 524,288 bp windows from position 0 (`borzoi_code/genome_tiler.py`). Training: all other assembled chromosomes (4,659 tiles); validation chr1 + chr8 (750 tiles); test chr9 + chr22 (359 tiles) <!-- exposure_audit/EXPOSURE_AUDIT.md §4 (tile counts reconstructed from hg38.ml.fa.fai; 359 matches the known test count) -->. Targets are the 32 bp-binned signal over the central 131,072 bp (4,096 bins); inputs are one-hot 524,288 bp <!-- borzoi_code/dataset.py:10 -->. Evaluation on a shared 128 bp grid (4-bin means) is used for cross-model comparison; native-grid (32 bp) numbers are reported separately and never mixed with shared-grid numbers.
**These chromosomes are not held out from the pretrained backbones.** Test chr9/chr22 tiles: 43.6% of bp lie in Borzoi training folds (30.4% in the Borzoi test fold) and 73.0% in Enformer training folds; validation tiles chr1/chr8: 65.7% (Borzoi) and 89.7% (Enformer) in training folds <!-- exposure_audit/EXPOSURE_AUDIT.md §4 table; Enformer fold file is the third-party Genentech reproduction, Basenji2 first-party bed 0.726 per RV-004 -->. The chr9/chr22 test set is also development-exposed (results were viewed during development; SD-009).

### 2.4 Borzoi-TL architecture and training (as actually run)
Four pretrained Borzoi replicates (`johahi/borzoi-replicate-0..3`, HF port via `borzoi-pytorch`) are frozen; the 1,920-channel trunk embedding that feeds `human_head` is captured with a forward pre-hook, averaged across the four replicates, centre-cropped to 4,096 bins, and passed to the only trainable module: `Linear(1920,1024) -> GELU -> Dropout(0.2) -> Linear(1024,183) -> Softplus` (2,154,679 parameters) <!-- borzoi_code/model.py; PHASE1_RECOVERY_REPORT.md §4 -->.
Training (Slurm 12212537, `main.py --epochs 8 --ckpt-dir ./borzoi_ckpt_full --output-dir ./results_full`, `training_run.slurm`): Adam on head parameters, lr 1e-4, weight decay 1e-4, batch size 1, Poisson negative log-likelihood (`log_input=False`), `ReduceLROnPlateau` (factor 0.5, patience 1), early-stopping patience 3, augmentation on training tiles only: random shift up to +/-128 bp snapped to whole 32 bp bins (+/-4 bins) and reverse-complement with p = 0.5 <!-- main.py:62-85, 265-282; borzoi_code/dataset.py:27-53 -->. **Not used:** inverse-signal per-track loss weighting (`--balance-tracks` was not passed), and no low-variance down-weighting (`low_variance_flag` is False for all 183 tracks) <!-- research_management/EXPERIMENT_REGISTRY.csv EXP-001; PHASE1_RECOVERY_REPORT.md §9 -->. The checkpoint used is epoch 5 of 8 (validation NLL 0.18523; epochs 6-8 did not improve) <!-- borzoi_ckpt_full/ (epoch 5); results_full/history.json -->. **Single seed; no seed was set; single train/val/test split.** The choice of trunk embedding as the feature tap was not ablated.

### 2.5 Baselines and track mapping
- **Pretrained Borzoi** (4 replicates, 7,611 human tracks). The comparator for each of the 183 targets is one pretrained output track. *Oracle mapping* = the track with the highest test Pearson r (selection on test; upper bound). *Validation-selected mapping* (primary, SD-007) = the track chosen on the validation chromosomes (chr1, chr8) only; sensitivity analysis excluding identical-experiment candidates from the vocabulary <!-- research_management/SCIENTIFIC_DECISIONS.md SD-007; analysis/e7_nonoracle/SELECTION_RULE.md (pre-specified before any E7 statistic) -->.
- **Enformer** (`EleutherAI/enformer-official-rough`, 5,313 human tracks; snapshot revision used for the August oracle run is unknown) and **AlphaGenome** (hosted API, ChIP-TF head, 1,617 tracks) were mapped with the same two rules (E7). The August oracle runs were reproduced exactly (Enformer 0.64432 vs 0.64432; AlphaGenome 0.66118 vs 0.66117); the AlphaGenome hosted API differs slightly between calls (max per-target |delta r| 2.1e-4) <!-- results_phase2/e7_nonoracle/SUMMARY.md -->.
- [PENDING: EXP-E4 supervised ridge/linear probe on pretrained-model outputs at the matched grid, same training chromosomes, lambda on validation (SD-008). No "adaptation helps" claim is made without it.]
- Ridge/probe and multi-seed designs: [PENDING: EXP-E4, EXP-E5 (>=3 seeds, chromosome-fold CV, no-RC ablation)].

### 2.6 Training-exposure audit
Track identity was resolved from GEO text (179 GSM, 0 errors) and ENCODE REST (4 ENCSR; ENCODE `dbxrefs` GEO:GSM to ENCSR for HCT116/Caco-2). A TL target is **identical-experiment** if its GSM/ENCSR is present in the pretrained model's target vocabulary file paths. Result: 31 tracks = 24 LoVo GSE51142 (GSM) + 4 colon ENCSR (head indices 145-148) + 3 HCT-116 ENCSR (ENCSR000DTO CTCF, ENCSR000BML POLR2A, ENCSR000BSA JUND); same set for Borzoi and Enformer; a **lower bound** (only HCT116/Caco-2 ENCODE cross-references searched; reprocessed duplicates not detected; AlphaGenome manifest not obtained, so AlphaGenome exposure is undetermined) <!-- exposure_audit/EXPOSURE_AUDIT.md §1 -->. Exposure class hierarchy: identical > same cell line + same target > same cell line other target > same target other cell line > unexposed (Borzoi counts 31/22/112/13/5) <!-- exposure_audit/EXPOSURE_AUDIT.md §2 --> (the unexposed classes use cell-line-level regex matching and are descriptive). Primary TL analysis set = the 152 non-identical tracks (SD-014), all 183 secondary.

### 2.7 Metrics
Three quantities are distinct and always labelled:
- **Pearson r** (primary): per-track correlation over all test bins (pooled over tiles), averaged over tracks.
- **Squared Pearson r (r^2)**: the square of that correlation (per track, then averaged).
- **Prediction R^2 = 1 - SS_res/SS_tot** on the raw scale. For pretrained models this is **validation-calibrated R^2** (a linear map fitted on validation chromosomes, then applied to test); uncalibrated raw-scale R^2 of pretrained tracks is meaningless (previously observed -193) and is not reported. TL R^2 is uncalibrated on test, so TL-minus-baseline R^2 differences compare unlike objects and are secondary.
MPRA validation uses a different quantity, the stratified within-region AUROC, which is not r, r^2 or R^2. 51 of 359 test tiles have zero signal and inflate pooled correlations; a sensitivity analysis excludes them.

### 2.8 Bootstrap designs for model comparison
Delta r = TL minus baseline (tracks matched), 10,000 resamples, seed 20261020, percentile 95% CI <!-- results_phase2/exposure_stratified/SUMMARY.md -->: (a) track resampling (descriptive only; tracks are not independent); (b) **TF-cluster** (77 clusters); (c) **cell-line-cluster** (12 labels incl. two tissue labels); (d) **tile-block** (37 blocks of 10 contiguous tiles, within chromosome; only two chromosomes, so chromosome-level resampling is impossible); (e) **two-way** tile-block x cluster, with both models recomputed per resample. Cluster-level sign-flip tests on cluster-mean deltas (exact for <=16 clusters). Subsets with <6 clusters are descriptive only. The pretrained R^2 calibration is fixed and not resampled. The sign-flip test (unweighted mean of cluster means) and the CIs (track-weighted mean) estimate different quantities.

### 2.9 CCV set, provenance and allele matching
Source: Chen et al. Supplementary Data 6 (5,741 CCVs, hg19, unique rsIDs, 238 signals, 142 regions; no alleles, no PIP) <!-- variant_provenance/PROVENANCE.md --> (chen2024). Scoring input: `alphagenome_variant_effects/input/S6_CCV_ChrPosRefAlt_rsid.tsv`, 10,148 rows over 5,739 positions (2,419/2,231/1,089 positions with 1/2/3 ALTs); the extra ALTs are a multi-allelic expansion and not the GWAS-tested allele.
- **Allele-matched set (primary for claims)**: alleles from GWAS Catalog GCST90129505 (harmonised GRCh38 file; original GRCh37 for 4 rows), the deposited summary statistics of the same trans-ancestry GWAS (100,204 cases, 154,587 controls). A row is matched iff {REF, ALT} equals {effect_allele, other_allele}. Result: 5,729 matched positions (822 palindromic; no strand flips needed), 4,396 non-reported ALT rows, 23 rows (10 rsIDs) without allele data, 12 S6 CCVs without a matched row (10 no allele info; rs149685934 and rs1312157506 lost to rsID drift); 5,741 - 2 = 5,739 reconciled. Deviation from SD-003 wording (S6 contains no alleles) recorded as SD-015.
- **Exploratory set**: all 10,148 rows; used only for score distributions and model concordance, never for validation claims.
"Matched" certifies the tested allele pair, not the causal or risk allele; a risk-allele-by-beta-sign column is provided. 544 palindromic matched variants outside Chen S15 rely on the GWAS Catalog harmonisation strand.

### 2.10 Variant scoring (frozen before any validation label was joined; SD-005, SD-014)
`variant_scoring/SCORING_SPEC.md`, track lists in `variant_scoring/track_lists/`. Windows: Borzoi 524,288 bp, Enformer 196,608 bp, AlphaGenome 1,048,576 bp, variant placed at window index L/2; windows running off a chromosome are N-padded for Borzoi/Enformer (149 / 30 rows) and shifted for AlphaGenome (242 rows / 147 positions, flagged). Aggregation: sum over the 512 bp centre (Borzoi 16 bins, Enformer 4 bins; AlphaGenome 501 bp mask, `DIFF_LOG2_SUM`), forward + reverse-complement averaged (Borzoi, Enformer), pseudocount 1 on aggregated sums. Per-track signed effect lfc = log2((ALT+1)/(REF+1)). **Primary score** = max over frozen tracks of |lfc|; signed companion = lfc at the arg-max track; sensitivity = mean |lfc| over tracks. Track sets: Borzoi-TL 152 (31 identical excluded; all 183 secondary); pretrained Borzoi and Enformer 132 keyword-selected CRC tracks (126 TF ChIP + 6 DNase; 118 LoVo, so max is LoVo-weighted; histone ChIP, CAGE, RNA excluded); AlphaGenome 33 tracks (DNase 6, ATAC 1, ChIP-TF 26; HCT116 27). AlphaGenome headline = provider quantile score (SD-014), raw log2 = like-for-like sensitivity. Borzoi scores are in the scaled/soft-clipped training output scale, not Borzoi's own variant-scoring convention. Cross-model comparison uses ranks, not raw magnitudes.

### 2.11 Validation design (SD-004, SD-016; `variant_scoring/E2_ANALYSIS_PLAN.md`, frozen before unblinding)
- **MPRA (primary)**: Law et al. ST2 (8,880 tested variants) and ST4 (275 hits, FDR < 1e-3 in HT29, SW403 or HCEC-1CT; REF/ALT and logFC) (law2024). Tested universe = matched CCVs in ST2 (4,323). Positive = ST4 (134), negative = ST2 flag 0 in all three lines (4,098), flag-1 "hit" excluded from primary (91; positive in sensitivity S1). Allele orientation checked against matched REF/ALT (132 kept, 2 swapped and excluded from P2, 0 dropped). The MPRA is episomal and per-variant effect sizes of non-hits are not public (EGA controlled), so a hit-versus-non-hit design is used.
- **P1 (primary)**: within-region stratified AUROC (weights n_pos x n_neg; midranks), 58 informative regions (130 positives, 2,518 negatives); region-clustered bootstrap (10,000, seed 20261020) and within-region permutation p (one-sided). Minimum detectable AUROC about 0.583-0.585 (null bootstrap SE 0.033-0.034) <!-- results_phase2/e2_validation/labels/POWER.md (A2: 0.583); interim_alphagenome/SUMMARY.md (0.5847) -->.
- **P2 (direction)**: fraction of MPRA positives with sign(model signed score) = sign(MPRA logFC). **Caveat (SD-017, BL-007): the ST4 logFC sign convention (ALT/REF vs REF/ALT) is unresolved**; P2 is reported exactly as pre-specified, is excluded from the framing rule, and is not re-signed post hoc <!-- external_data/validation/MPRA_SIGN_CONVENTION.md -->.
- **Controls**: distance to nearest GENCODE v46 protein-coding TSS, Chen annotation count (sum of 7 S6 flags), conditional logistic regression (strata = region), exclusions (palindromic, padded windows, 242 AlphaGenome shifted windows), signal-level clustering, held-out-chromosome subset chr1/8/9/22 (descriptive).
- **eQTL (secondary; partly not independent, GTEx RNA-seq is in model training data)**: CEDAR transverse colon (eQTL Catalogue credible sets) and GTEx v10 colon transverse/sigmoid significant pairs, harmonised to matched REF/ALT (gtex2020 for v8).

### 2.12 Multiple testing
P1: Holm across the four headline models (one-sided permutation p); P2: separate Holm family of four (bootstrap two-sided p). Everything else (sensitivities, eQTL, pairwise model differences, clogit) is secondary/exploratory with BH-FDR within family. Holm, paired model deltas and the framing rule are computed only in the final run with all four models complete (Amendment A3); every look at labels is logged in `research_management/EXECUTION_LOG.md`. Framing rule: B is supported only if P1 for >=1 model has Holm p < 0.05, CI excluding 0.5, and exceeds the distance baseline with CI of the difference excluding 0.

## 3. Results

### 3.1 Training-exposure audit (EXP-005)
- 31 of 183 test tracks are identical experiments in the Borzoi and Enformer target vocabularies (24 LoVo + 4 colon ENCSR + 3 HCT-116 ENCSR); lower bound. <!-- exposure_audit/track_exposure.tsv; EXPOSURE_AUDIT.md §1 -->
- Exposure classes (Borzoi): identical 31, same cell line same target 22, same cell line other target 112, same target other cell line 13, unexposed 5. <!-- EXPOSURE_AUDIT.md §2 -->
- Test tiles (chr9 + chr22): 43.6% of bp in Borzoi training folds and 73.0% in Enformer training folds. <!-- EXPOSURE_AUDIT.md §4; fold file for Enformer is third-party (HF Genentech) -->
- 17 cell-line tracks (GP5d 6, LS174T 5, LS180 5, DLD-1 1) have no cell-line exposure in either vocabulary; they test pretrained-model exposure, not TL generalisation, since TL was trained on them. <!-- EXPOSURE_AUDIT.md §3 -->

### 3.2 Borzoi-TL versus pretrained Borzoi on 359 test tiles (EXP-006; shared 128 bp grid, Pearson r unless stated)
Verified headline values: Borzoi-TL r 0.682 (r^2 0.487; prediction R^2 0.481); pretrained Borzoi oracle r 0.664, validation-selected r 0.661, validation-calibrated R^2 0.444 (validation-selected) / 0.448 (oracle). <!-- results_phase2/exposure_stratified/SUMMARY.md §0, §1-3; research_management/PROJECT_STATE.md -->
Native 32 bp grid (separate, not comparable): Borzoi-TL r 0.6634. <!-- PHASE1_RECOVERY_REPORT.md §6 -->

**Table 1 (delta r = TL - pretrained Borzoi, oracle mapping; CIs from EXP-006).**

| Subset | n tracks (TF / cell clusters) | TL r | Borzoi oracle r | delta r | TF-cluster 95% CI | cell-line-cluster 95% CI |
|---|---|---|---|---|---|---|
| all 183 | 183 (77/12) | 0.682 | 0.664 | +0.018 | [0.005, 0.028] | [-0.020, 0.037] (includes 0) |
| primary 152 | 152 (62/11) | 0.685 | 0.655 | +0.030 | [0.021, 0.041] | [0.009, 0.038] |
| identical-experiment 31 | 31 (25/3) | 0.667 | 0.710 | -0.043 | [-0.077, -0.020] | [-0.061, 0.029] (<6 clusters; descriptive) |
<!-- results_phase2/exposure_stratified/SUMMARY.md §1-2; exposure_stratified_stats.csv. Validation-selected mapping: all 183 delta +0.021 [0.007, 0.031] TF-cluster / [-0.018, 0.041] cell-line; primary 152 +0.034; identical 31 -0.043 -->

Interpretation constraints (stated in text): the TL advantage is concentrated in non-identical tracks and TL is worse than pretrained Borzoi on identical-experiment tracks, consistent with (not proof of) a training-exposure advantage for the pretrained model; the -0.043 is confounded with LoVo (within LoVo: identical 24 tracks -0.061 vs non-identical 17 tracks +0.009; HCT-116: identical 3 tracks +0.029 vs non-identical 102 +0.036). <!-- SUMMARY.md §1-2 (coordinator, RV-004) -->
Other metrics (TF-cluster CIs, oracle mapping): all 183 r^2 delta +0.027 [0.009, 0.040]; prediction R^2 TL 0.481 vs validation-calibrated 0.448 (delta +0.033 [0.014, 0.047]); primary 152 r^2 delta +0.043 [0.029, 0.057]. <!-- SUMMARY.md §3 -->
Sensitivity: excluding the 51 zero-signal tiles (308 tiles) delta r +0.016 [0.002, 0.027] (TF cluster), conclusions unchanged. Low-variance tracks are harder for both models; delta by variance tertile (oracle) +0.009/+0.024/+0.020 (not monotone). <!-- SUMMARY.md §4 -->
Unexposed cell lines (17 tracks, 4 clusters; DESCRIPTIVE): pooled TL 0.545, oracle 0.521, valsel 0.514; all four cell lines have positive delta; no CI claim. <!-- SUMMARY.md §5 -->
[PENDING: EXP-E4 supervised ridge baseline (is the TL gain readout recalibration?), EXP-E5 seeds/chromosome folds, ablations; none of the TL recipe components has been ablated.]

### 3.3 Enformer and AlphaGenome, non-oracle baselines (EXP-010)

| model | oracle r | validation-selected r | validation-selected excl. identical r |
|---|---|---|---|
| Enformer | 0.6443 | 0.6406 | 0.6247 |
| AlphaGenome | 0.6612 | 0.6569 | 0.6570 (proxy exclusion; not exposure-corrected) |
| Borzoi-TL (reference) | 0.6820 | n/a | n/a |
<!-- results_phase2/e7_nonoracle/SUMMARY.md table; oracle values reproduce the August benchmark -->

Delta r TL - model, validation-selected: Enformer all 183 +0.0413 (TF-cluster [0.0256, 0.0558]; cell-line-cluster [-0.0060, 0.0753], includes 0); AlphaGenome all 183 +0.0251 (TF [0.0121, 0.0428]; cell-line [0.0161, 0.0419]). <!-- e7_nonoracle/SUMMARY.md delta table -->
Qualifiers: Enformer validation selection picked the identical-experiment candidates for 80/183 targets; "TL beats Enformer" holds only after excluding them or on the primary-152 subset (primary 152 delta +0.0587 [0.0458, 0.0767]). AlphaGenome exposure is unverified, so its identical-31 numbers are not exposure-corrected. Validation-calibrated R^2 deltas (valsel): Enformer +0.0614, AlphaGenome +0.0408 (TF-cluster CIs [0.0401, 0.0797] and [0.0232, 0.0630]). Region uncertainty spans two chromosomes only. <!-- e7_nonoracle/SUMMARY.md Notes -->

### 3.4 Variant provenance and allele matching (EXP-007)
5,741 S6 CCVs; 10,148 scoring rows over 5,739 positions; 0 REF mismatches; 5,729 allele-matched positions (822 palindromic), 4,396 non-reported ALT rows, 23 rows without allele information, 12 S6 CCVs unmatched; liftover agrees at 5,737/5,739 (2 unmapped), Ensembl at 5,739/5,739. <!-- variant_provenance/PROVENANCE.md; variant_provenance/build_provenance.log -->
Label-only validation counts (no scores used): 4,323 matched CCVs tested in Law ST2; 134 ST4 positives; 4,098 negatives; 91 flag-1 excluded; 58 informative regions; 132 positives orientation-concordant, 2 swapped. <!-- variant_scoring/E2_ANALYSIS_PLAN.md Amendments A1-A2; results_phase2/e2_validation/labels/POWER.md -->

### 3.5 Variant-effect annotation and MPRA validation (EXP-E2)
- **AlphaGenome (interim, Holm deferred, single-model, unadjusted p).** Stratified P1 AUROC [PENDING: final; interim value reported only in `results_phase2/e2_validation/interim_alphagenome/SUMMARY.md`, not to be quoted as a result until the four-model Holm run]. <!-- interim only; research_management/EXECUTION_LOG.md look #1 -->
- Pretrained Borzoi, Borzoi-TL, Enformer: [PENDING: EXP-008/009 finalised scores; EXP-E2 unblinding per model].
- P1 across four models with Holm: [PENDING: EXP-E2 final].
- Comparison with distance-to-TSS and annotation-count baselines (A1): [PENDING: final].
- Direction concordance P2: [PENDING: final]. **Any P2 value must be reported with this caveat: the Law ST4 logFC sign convention is unresolved (SD-017, BL-007); P2 is excluded from the framing rule.** An interim AlphaGenome concordance below 0.5 was observed before the sign check; it is uninterpretable until BL-007 is resolved and is not re-signed post hoc. A sign-agnostic model-agreement pattern may be reported only as post-hoc exploratory.
- eQTL secondary analyses (CEDAR, GTEx v10): [PENDING: EXP-E6].
- Framing decision (B vs A): [PENDING: final all-model run and framing rule, `E2_ANALYSIS_PLAN.md` §7].

## 4. Discussion (skeleton)
1. Principal findings: [PENDING: depends on final E2/E4/E5]. Current verified statements only: the TL-minus-pretrained-Borzoi difference is small (+0.018 all-track) and is positive on non-identical tracks and negative on identical-experiment tracks; the cell-line-cluster CI for all 183 includes 0.
2. Training exposure as a design issue for CRC benchmarks: public CRC ChIP-seq (especially LoVo GSE51142 and ENCODE colon/HCT-116) is in the pretraining vocabularies; fold-level overlap of any "held-out" chromosome with backbone training must be reported.
3. Relation to prior work: closest is li2026 (Enformer retrained on breast/prostate TF ChIP-seq, with variant and TWAS analyses); method novelty here is limited; the added value is exposure-stratified inference, four-model non-oracle comparison and MPRA-based validation.
4. Interpretation of variant scores: associations with MPRA activity at fine-mapped loci do not establish causal variants or mechanisms; larger predicted effects do not imply greater biological accuracy.
5. Why the MPRA/eQTL readouts are only partly independent of training (GTEx RNA-seq in model training; Law ST2 Tier annotations used model/ATAC/ChIP; use MPRA columns only).
6. [PENDING: what the sign-convention resolution (BL-007) implies for P2.]
7. Future work: backbone fine-tuning (yuan2026-style), additional MPRA/STARR datasets (Chen 2026 Sci Adv, Fabo 2025 bioRxiv, Kellman 2025: listed in `VALIDATION_AVAILABILITY.md`, not yet retrieved/verified here and therefore not cited).

## 5. Limitations (complete list from state files)
Test/data
1. Single split, single seed (no seed set); no ablation of any recipe component (no-RC, augmentation, feature tap); no supervised baseline yet [PENDING E4/E5].
2. chr9/chr22 are development-exposed (SD-009) and ~44% (Borzoi) / ~73% (Enformer) of test bp lie in pretrained training folds; chr1/chr8 validation also overlaps (66% / 90%). Chromosome-fold CV would test head robustness only, not backbone exposure. Enformer fold file is a third-party reproduction; Borzoi HF replicate weights assumed to share the README folds (unverified).
3. Region-level uncertainty spans two test chromosomes; chromosome-level resampling is impossible.
4. The count of 31 identical-experiment tracks is a lower bound (true count is >=31) (GSM/ENCSR identity only; ENCODE xrefs searched for HCT116/Caco-2 only; reprocessed duplicates missed). AlphaGenome training exposure is undetermined (manifest not obtained, ChIP-TF metadata lacks accessions); the AlphaGenome exclusion is a proxy.
5. Oracle mapping selects on test (upper bound); validation-selected mapping is deployable. The earlier 1,882-track "filtered Borzoi" comparison used test-selected tracks.
6. Non-independence: 183 tracks, 77 TFs, 12 cell-line/tissue labels (identical-experiment, same-cell-line-same-target and unexposed subsets have <6 clusters). The sign-flip test and the CIs estimate different quantities. 17 "unexposed" cell-line tracks test pretrained exposure, not TL generalisation.
7. 51 zero-signal tiles inflate pooled Pearson r; pooled-over-tiles correlation; r, r^2 and R^2 are not interchangeable; TL R^2 (uncalibrated) vs baseline validation-calibrated R^2 compare unlike objects.
8. Cell-type, resolution and context mismatch between models (524 kb/32 bp, 196 kb/128 bp, 1 Mb); the 118/132 LoVo imbalance of the pretrained track set and HCT-116 dominance of TL and AlphaGenome sets; max-over-tracks favours models with more tracks (mean-|lfc| sensitivity pre-specified).
9. Enformer snapshot revision used in the August oracle run unknown; AlphaGenome hosted API output varies slightly between calls (max 2.1e-4 in per-target r).
Variants
10. S6 contains no alleles; alleles come from marginal trans-ancestry GWAS summary statistics (GCST90129505, harmonised with Ensembl 95/dbSNP 151), not Chen's conditional analyses. "Matched" certifies the tested allele pair only; the risk allele is by beta sign; 544 palindromic variants outside S15 rely on the harmonisation strand; 10 CCVs (23 rows) have no allele data and are excluded, not guessed; two rsID-drift CCVs are lost.
11. The upstream hg19-to-hg38 conversion that built the scoring input is undocumented (verified post hoc by liftover and Ensembl). 4,396 additional ALTs at matched positions are exploratory.
12. Chromosome-end windows: 149 (Borzoi) / 30 (Enformer) rows N-padded, 242 AlphaGenome rows shifted; sensitivity exclusions pre-specified.
13. Borzoi scores are summed in the scaled/soft-clipped output scale; pc = 1 is a convention; 512 vs 501 bp and bin granularity (Enformer 128 bp bins blur the variant position); AlphaGenome quantile background is provider-defined and tie-heavy (midranks; raw log2 is the tie-free sensitivity); AlphaGenome RC averaging is server-side.
14. LD within loci and about 143 independent regions (58 informative for P1); MPRA is episomal; non-hits may reflect low activity or low power; per-variant effect sizes of non-hits are not public; minimum detectable AUROC about 0.583.
15. P2 direction is uninterpretable until the Law ST4 logFC sign convention is resolved (SD-017, BL-007); post-hoc analyses are labelled as such.
16. eQTL concordance is only partly independent (GTEx RNA-seq in training); GTEx v10 sign concordance with ALT/REF was 0.57 (p 0.17) and gives no sign information.
17. No causal, clinical or mechanistic claims.
Process
18. Interim E2 look at AlphaGenome was made before the other models; Holm and framing rule are deferred to the final run (A3). Several state-file results await reviewer sign-off (EXP-005/006/007/010 listed as "reviewer pending" in the registry).

## 6. Figure legends (placeholders)
- **Fig. 1. Study design.** [PENDING] Data sources (Chen CCVs, GWAS Catalog alleles, Law MPRA, GTEx/CEDAR), models, track sets, and the pre-registered analysis flow.
- **Fig. 2. Training-exposure audit.** [PENDING: from `exposure_audit/track_exposure.tsv`, `tile_fold_overlap.tsv`] Exposure class per track; fold overlap of test/validation tiles with Borzoi and Enformer training folds.
- **Fig. 3. Model comparison.** [PENDING: from EXP-006/010 per-track metrics] Per-track Pearson r (shared 128 bp grid), TL versus pretrained Borzoi (validation-selected mapping), Enformer and AlphaGenome, coloured by exposure class; delta r with TF-, cell-line- and tile-block bootstrap CIs. State metric (Pearson r) and mapping rule in the legend.
- **Fig. 4. Supervised baseline and robustness.** [PENDING: EXP-E4, EXP-E5]
- **Fig. 5. Variant-score validation against MPRA.** [PENDING: EXP-E2 final] Stratified within-region AUROC with region-clustered CIs per model, with distance-to-TSS and annotation-count baselines; metric is AUROC, not r/r^2/R^2.
- **Fig. 6. Direction concordance (P2).** [PENDING; include the SD-017 sign-convention caveat].
- **Fig. S-series.** Palindromic/padded/shifted-window sensitivities, per-cell-line breakdowns, zero-signal-tile sensitivity, mean-versus-max score sensitivity [PENDING].

## 7. Supplementary methods (outline)
S1 BigWig processing and target construction (`bigwig_loader.py`) [PENDING write-up]. S2 Full track table with GSM/ENCSR, cell line, exposure class (Borzoi, Enformer, AlphaGenome). S3 Fold files and md5 (Borzoi sequences_human.bed.gz, Basenji2, Genentech Enformer intervals). S4 Validation-selection rules (`analysis/e7_nonoracle/SELECTION_RULE.md`) and track-matched / same-TF comparators. S5 Variant provenance: sources and md5 table, reconciliation 5,741 vs 5,739, allele-matching algorithm. S6 Scoring specification, frozen track lists, window conventions, padding counts. S7 E2 plan with Amendments A1-A3; power statement; every unblinding look. S8 Ridge/probe and multi-seed designs [PENDING]. S9 MPRA sign-convention investigation (`external_data/validation/MPRA_SIGN_CONVENTION.md`). S10 Software/versions and environments.

## 8. Data and code availability
Branch `research/crc-autonomous-study` of this repository (code: `borzoi_code/`, `main.py`, `variant_scoring/`, `variant_provenance/`, `exposure_audit/`, `analysis/`). Checkpoint: `borzoi_ckpt_full/model_head_best.pt` (not committed; release location [PENDING]). Public inputs: Chen 2024 Supplementary Data (Nature static content; md5 in `variant_provenance/PROVENANCE.md`); GWAS Catalog GCST90129505 (academic, non-commercial licence); Law 2024 ST2/ST4 (Springer Nature supplement); GTEx v8/v10 colon eQTL (open); CEDAR/eQTL Catalogue credible sets. Not public: Law MPRA per-variant data for non-hits (EGA EGAD50000000596, controlled access). AlphaGenome scores were produced with the hosted API (key not shared); pretrained weights via HuggingFace (`johahi/borzoi-replicate-0..3`, `EleutherAI/enformer-official-rough`). Large artefacts (`alphagenome_variant_effects/` about 26 GB, `results*/`) are git-ignored; deposit plan [PENDING]. CRC cell-line ChIP-seq sources are GEO/ENCODE (accessions in `exposure_audit/track_exposure.tsv`).

## References
See `manuscript/references.tsv` (verified keys only): huyghe2019, chen2024, law2024, kelley2018, avsec2021, linder2025, avsec2026, li2026, yuan2026, huang2023, lee2023, gtex2020, tuupanen2009, wright2010. [Reference list to be formatted from the TSV at submission.]
