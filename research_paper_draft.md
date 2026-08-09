# Data Augmentation and Feature-Tap Transfer Learning Improve Borzoi for Colorectal-Cancer Transcription-Factor Binding Prediction

**Joshua Ye**
2026

## Abstract

Borzoi is a large pre-trained sequence-to-function model that predicts thousands of genomic tracks (RNA-seq, ATAC-seq, ChIP-seq) directly from DNA sequence. Its native human output head targets ~7,611 generic tracks and cannot be queried for arbitrary new assays without retraining. We adapt frozen Borzoi to a panel of 183 colorectal-cancer (CRC) transcription-factor (TF) ChIP-seq tracks across three CRC cell lines (Caco-2, HCT-116, LoVo) by (1) tapping Borzoi's 1920-dimensional trunk embedding instead of its lossy human-head output, (2) training a small MLP head on top of that embedding while keeping the entire Borzoi backbone frozen, and (3) applying two DNA-valid data augmentations during training — random genomic shift (±128 bp, bin-quantized) and reverse-complementing (p=0.5) — together with weight decay, dropout, inverse-signal per-track loss weighting, and early stopping. Against a same-track-count internal ablation, augmentation and the training recipe raise mean Pearson r from 0.502 to 0.643 on validation (+28% relative) and from 0.610 to 0.663 on held-out test (+9% relative). Against a matched external baseline — the best correlating track among 1,882 eligible (ChIP, non-histone) tracks from the frozen, unmodified Borzoi human head, chosen per CRC target — our transfer model outperforms original Borzoi on 122/183 test tracks (66.7%, 34 losses, 27 ties; paired t p = 7.7×10⁻⁵, Wilcoxon p = 1.9×10⁻¹⁰, Cohen's d = 0.30), raising mean test Pearson r from 0.646 to 0.663. Restricting to the 149 targets where the transfer model does not lose to Borzoi raises the achievable mean r to 0.685 (median 0.725), quantifying the residual gap concentrated in a small, identifiable subset of targets (REST, RFX2, ZBTB7B, NFYA, ZNF143, CTCF, and related factors). We report this residual as an open limitation rather than as evidence for wholesale track removal, and discuss why.

## 1. Introduction

Deep sequence-to-function models such as Enformer and Borzoi predict genome-wide regulatory tracks from raw DNA sequence and have become a standard substrate for downstream regulatory-genomics work. In practice, a lab studying a specific biological system — here, transcription-factor binding in colorectal cancer — usually wants predictions for a bespoke, non-generic panel of tracks rather than Borzoi's pre-trained output vocabulary. Two paths exist: (a) fully fine-tune or retrain a model on the new track panel, which is data-hungry and risks destroying the pre-trained representation, or (b) freeze the pre-trained backbone and train a small, task-specific head on top of it. We take path (b), which is cheap, less prone to overfitting on a modest CRC ChIP-seq panel, and lets us isolate exactly what the added training recipe contributes.

The central methodological question we address is whether a standard, low-cost machine-learning technique — data augmentation — transfers cleanly into this genomics setting, using transformations that are biologically valid for DNA rather than the pixel-space transformations (crop, flip, rotate) that motivate it in computer vision. We combine this with two further changes: reading a richer intermediate representation out of the frozen backbone (its 1920-dim trunk embedding, rather than its 7,611-track human-head output), and a conventional but non-trivial training recipe (regularization, per-track loss balancing, scheduling, early stopping). We evaluate the result against both an internal ablation (recipe on vs. off, same architecture) and an external baseline (the original, unmodified Borzoi, best-matched per target), on held-out validation and test genomic intervals.

## 2. Related Work

Data augmentation for genomic deep neural networks is established in the literature. EvoAug (Lee, Tang, Toneyan & Koo, *Genome Biology* 2023;24:105) introduces evolution-inspired augmentations (mutation, insertion, deletion, translocation, in addition to shift and reverse-complement) and shows they improve generalization and interpretability of genomic DNNs trained with limited data, followed by a fine-tuning phase on the original, unaugmented data. Borzoi and Enformer themselves use stochastic sequence shifting and reverse-complementing as standard training-time augmentations at pre-training scale. Our work applies the same class of augmentation — but at head-only fine-tuning scale, on a frozen large pre-trained backbone, and quantifies the effect via a controlled ablation rather than assuming it from the literature.

## 3. Data and Model

**Targets.** 183 ChIP-seq BigWig tracks over three CRC cell lines (Caco-2, HCT-116, LoVo; colon tissue), covering transcription factors including POLR2A, CTCF, RAD21, SMC1A/SMC3, MYC, JUND, REST, and others (`borzoi_data/CRC_TFs_bw/`).

**Genome splits.** Chromosome-held-out splits from `hg38`: validation = {chr1, chr8}; test = {chr9, chr22}; all other autosomes/assembled chromosomes = train (`borzoi_code/genome_tiler.py`). Non-train chromosomes are touched only for evaluation, never for head training or model selection beyond early stopping on validation.

**Sequence and target geometry.** 524,288 bp input windows; predictions and targets are read at 32 bp resolution over the center 131,072 bp (4,096 bins per window), matching Borzoi's native bin size.

**Backbone.** Four Borzoi replicate folds (`johahi/borzoi-replicate-{0..3}` via `borzoi-pytorch`/HuggingFace) are loaded, fully frozen, and their per-fold 1920-channel trunk embeddings (captured via a forward pre-hook on `human_head`, so no dependence on internal Borzoi method names) are averaged across folds before being passed to the trainable head. This taps Borzoi's internal representation rather than its 7,611-track human output, which is optimized for Borzoi's own pre-training tracks and would otherwise force our new tracks through an information bottleneck.

**Head.** A two-layer MLP is the only trainable component: `Linear(1920→1024) → GELU → Dropout(0.2) → Linear(1024→183) → Softplus` (Softplus keeps predicted coverage non-negative while remaining smooth, unlike ReLU). (`borzoi_code/model.py`)

**Training recipe** (`main.py`):
- Optimizer: Adam on head parameters only, weight decay 1×10⁻⁴.
- Loss: Poisson negative log-likelihood, with per-track inverse-mean-signal weighting so sparse tracks are not drowned out by high-signal ones.
- Schedule: `ReduceLROnPlateau` (factor 0.5, patience 1) plus early stopping on validation loss (patience 3).
- Augmentation (train split only, `borzoi_code/dataset.py`): random genomic shift up to ±128 bp, snapped to 32 bp bins so target bins stay grid-aligned (true Borzoi/Enformer shift is single-bp resolution; ours is coarser to avoid re-binning targets); reverse-complement with probability 0.5, reversing target bins to match. Both keep sequence and targets aligned and are disabled on validation/test.

## 4. Experimental Design

Two comparisons are reported, evaluated on the same held-out genomic intervals for both models:

**(A) Internal ablation.** The transfer model trained with vs. without the augmentation + training recipe described above, same architecture and track set, mean Pearson r over all 183 tracks.

**(B) External baseline — filtered original Borzoi.** Original Borzoi's human head outputs ~7,611 generic tracks that do not correspond 1:1 to our 183 CRC targets, so a track-matching procedure is required for a fair comparison (`filter_borzoi_tracks.py`, `borzoi_baseline_utils.py`):
1. Restrict to Borzoi output channels 2186–6069 (`targets_human.csv`), the block of ChIP-relevant tracks → 3,884 candidates.
2. Keep descriptions containing `"chip"`; drop histone-modification marks (H3K27ac, H3K4me1/2/3, H3K9ac/me2/me3, H3K27me3, H3K36me3, H3K79me2, H3K4ac, H3K56ac, H2A/H2B/H4 marks, H2A.Z) → 1,882 eligible TF/DNA-binding-protein tracks.
3. For each of our 183 CRC targets, compute Pearson r against all 1,882 eligible tracks, streamed over held-out bins/intervals (memory-mapped, chunked — no full matrix materialized in RAM).
4. Take the arg-max track per CRC target as that target's Borzoi baseline prediction, logging the matched track's description, factor, and cell/tissue for a biological-plausibility check (same TF / related TF-family / different TF).

This baseline is deliberately generous to Borzoi: it is allowed to pick its single best-correlating track per target from 1,882 candidates, with no penalty for using a track from an unrelated cell line or a related-but-not-identical TF.

**Significance testing.** Paired t-test and Wilcoxon signed-rank test across the 183 (transfer r, Borzoi r) pairs, plus paired Cohen's d. We flag this as a lower-confidence read below (Section 6).

## 5. Results

### 5.1 Internal ablation: augmentation + training recipe

| Split | No augmentation | + Augmentation & recipe | Δ | Relative |
|---|---|---|---|---|
| Validation | 0.502 | **0.643** | +0.141 | +28% |
| Test | 0.610 | **0.663** | +0.053 | +9% |

The validation/test relationship also becomes more internally consistent: without the recipe, validation (0.502) sits well below test (0.610), an unusual pattern; with it, validation (0.643) sits just under test (0.663), the expected ordering when validation has any indirect influence via early stopping. This is corroborating, not conclusive, evidence that the gain reflects a real improvement in generalization rather than noise.

### 5.2 External baseline: transfer model vs. filtered original Borzoi

**Validation (n=183 targets):**

| Metric | Filtered Borzoi | Transfer model |
|---|---|---|
| Mean Pearson r | 0.625 | **0.643** (+0.018) |
| Median Pearson r | 0.649 | **0.679** |
| High-confidence tracks (r ≥ 0.7) | 34.4% | **45.9%** |
| Tracks won | 33 (18.0%) | **119 (65.0%)**, 31 ties |

**Test (n=183 targets):**

| Metric | Filtered Borzoi | Transfer model |
|---|---|---|
| Mean Pearson r | 0.646 | **0.663** (+0.018) |
| Median Pearson r | 0.666 | **0.690** |
| High-confidence tracks (r ≥ 0.7) | 38.3% | **47.0%** |
| Tracks won | 34 (18.6%) | **122 (66.7%)**, 27 ties |

Paired statistics on test: mean paired difference = +0.0178 (95% CI [0.0088, 0.0260]); paired t p = 7.7×10⁻⁵; Wilcoxon p = 1.9×10⁻¹⁰; Cohen's d = 0.30 (small-to-moderate). Results are directionally identical to validation, which is the intended confirmatory role of the held-out test split here.

### 5.3 Where the residual gap concentrates

The 34 test-split targets where the transfer model loses to filtered Borzoi (`results_comparison_test/transfer_vs_filtered_borzoi_test.csv`, `winner == "borzoi"`) are disproportionately a small number of TFs repeated across LoVo/HCT-116/Caco-2, dominated by REST (Δr ≈ −0.31, −0.27), RFX2 (Δr ≈ −0.26), ZBTB7B (Δr ≈ −0.16), NFYA, ZNF143, CTCF, JUND, and SMC1A/SMC3/NIPBL/E2F8 (cohesin-pathway factors). Restricting the comparison to the 149 targets *not* in this set — i.e., where transfer at minimum ties Borzoi — raises the transfer model's mean test Pearson r to 0.685 (median 0.725) and its win rate to 81.9% against the same 149-track Borzoi baseline (`results_targets_filtered_test/summary_before_after_test.csv`). We report this as a characterization of where the model currently falls short, not as a track-selection procedure for a final model: dropping a target from evaluation does not improve the model's actual predictions for that target, and several of the losing factors (REST, CTCF, cohesin components) are biologically important, so this residual is flagged for future modeling work (Section 7) rather than exclusion.

## 6. Discussion: Statistical Caveats

The reported p-values (7.7×10⁻⁵, 1.9×10⁻¹⁰) are correct arithmetic but likely overstate confidence: the 183 tracks are not independent samples — many are the same assay (e.g., CTCF ChIP-seq) repeated across the three cell lines, so per-track errors are correlated. Treating them as 183 independent observations is pseudoreplication and inflates apparent significance. The more trustworthy signal is the consistency and direction of the effect: a positive mean gain that replicates on an independent, previously-untouched test split, a majority win rate (65–67%) that holds across both splits, and a shift toward more high-confidence tracks (r ≥ 0.7) in both. A track-cluster-aware or assay-grouped significance test (e.g., clustering by TF or by cell line and testing at the cluster level) is listed as future work rather than attempted here.

A second caveat: at least one track per run has near-zero target/prediction variance (flagged via a `low_variance_flag` in the per-target table), which produces an undefined or unstable Pearson r; such tracks are automatically down-weighted in the correlation computation but were not manually re-verified for identical handling across the baseline and transfer numbers.

## 7. Limitations and Future Work

- **Shift resolution.** Augmentation shift is quantized to 32 bp bins rather than Borzoi's native single-bp resolution, to avoid re-binning targets on the fly; single-bp shift with on-the-fly rebinning is a natural next step.
- **Augmentation scope.** Only two augmentations were tested (shift, reverse-complement). EvoAug-style mutation/insertion/deletion augmentations, applied during head-only fine-tuning, are untested here.
- **Single dataset.** All results are on one CRC TF ChIP-seq panel across three cell lines; generalization to other tissue types or assay classes (RNA-seq, ATAC-seq) is unverified.
- **Baseline generosity vs. specificity trade-off.** The filtered-Borzoi baseline is intentionally generous (best of 1,882 candidate tracks per target, any cell line), which likely *overstates* Borzoi's baseline performance relative to a fixed, single matched track — making our reported gain, if anything, a conservative estimate.
- **Residual underperformance.** REST, RFX2, ZBTB7B, NFYA, ZNF143, CTCF, and cohesin-pathway factors (SMC1A/SMC3/NIPBL) remain below the matched Borzoi baseline; whether this reflects head undercapacity, insufficient training signal for these specific factors, or a genuine limit of the frozen-trunk-embedding approach for these binding patterns is unresolved and the top candidate for follow-up work (e.g., factor-specific loss reweighting, or selectively unfreezing late backbone layers for these tracks only).
- **Statistical independence.** As discussed in Section 6, a cluster-aware significance test is needed for a fully rigorous claim of significance.
- **Foundation-model benchmarking.** No comparison yet against Enformer or other genomic foundation models on the same CRC TF panel (harness for Enformer exists in this repo — `baseline_enformer.py`, `filter_enformer_tracks.py`, `compare_transfer_enformer.py` — but end-to-end results are not yet finalized here).

## 8. Conclusion

Standard data augmentation — random genomic shift and reverse-complementing, transformations that are genuine symmetries of double-stranded DNA rather than borrowed pixel-space heuristics — combines with a richer feature tap (trunk embedding vs. lossy output head) and a conventional regularized training recipe to produce a consistent, modest improvement in transfer learning from a frozen Borzoi backbone onto a bespoke 183-track CRC transcription-factor panel. Internally, the recipe raises mean Pearson r by 28% (validation) and 9% (test) over an ablated baseline. Externally, against the original, unmodified Borzoi (given every benefit of the doubt via best-of-1,882-tracks matching), the resulting model wins on two-thirds of targets and raises mean Pearson r by 0.018 on both validation and held-out test, with the gain concentrated in high-confidence tracks. The residual gap is not uniform: it concentrates in a small, identifiable set of factors that motivate targeted follow-up rather than track removal. Backbone weights are never updated — every reported gain comes from a lightweight, cheap-to-train head and its training recipe, not from additional pre-training data or a larger model.

## Data and Code Availability

All code, configuration, and result tables referenced above are in this repository. Key entry points: `main.py` (training), `borzoi_code/model.py` (architecture), `borzoi_code/dataset.py` (augmentation), `filter_borzoi_tracks.py` / `borzoi_baseline_utils.py` (baseline construction), `compare_transfer_borzoi.py` (paired comparison, figures, statistics), `identify_underperforming_targets.py` (residual-gap characterization). Result tables: `results_comparison/` (validation), `results_comparison_test/` (test), `results_targets_filtered_test/` (residual-gap subset).

## References

1. Linder, J. et al. Borzoi: predicting RNA-seq and other functional genomic tracks from DNA sequence. (Borzoi model this work builds on.)
2. Avsec, Ž. et al. Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods* (2021). (Enformer.)
3. Lee, N.K., Tang, Z., Toneyan, S., Koo, P.K. EvoAug: improving generalization and interpretability of genomic deep neural networks with evolution-inspired data augmentations. *Genome Biology* 24, 105 (2023). doi:10.1186/s13059-023-02941-w, PMCID: PMC10161416.
