# PI update, 21 October 2026: speaker script

Generated from `reports/pi_update_2026-10-21/deck_content.py` (same text as the PowerPoint speaker notes).
11 slides; target about 10 minutes.

## Introduction

### 0:00 — Slide 1: Evaluating genomic sequence models for colorectal cancer TF binding and risk variants

Today I'll give an update on the colorectal cancer project, where we evaluate genomic sequence models for two tasks. The first is predicting CRC transcription-factor binding, and the second is prioritising CRC risk variants. The main results are now final, all analysis phases have passed our internal, agent-based review checkpoints, and a full manuscript draft exists. I'll go through the design and the two main results, then the limitations, and finish with a few questions where I'd value your input.

### 0:30 — Slide 2: Research overview and questions

The motivation is that CRC GWAS fine-mapping gives thousands of credible causal variants, and most of them are noncoding, so their function is hard to interpret. Sequence models like Borzoi, Enformer and AlphaGenome predict regulatory signals directly from DNA sequence, so in principle they can score each variant. We asked two questions. First, does fine-tuning Borzoi on CRC ChIP-seq data actually improve CRC TF-binding prediction over what the pretrained model already knows? Second, do these models rank variants that are functional in an MPRA above variants that are not, within the same risk locus? The MPRA analysis plan and the ridge decision rule were frozen before results were seen, although the chr9 and chr22 test set had been used in earlier development, and we audited how much of our test data the pretrained models had already seen during training.

## Design

### 1:30 — Slide 3: Models, baselines and how each is used

Here are the models and baselines and where each one is used. Borzoi-TL is our fine-tuned model: a new prediction head trained on CRC ChIP-seq on top of the frozen pretrained Borzoi trunk. Ridge regression is not another foundation model; it is a supervised baseline that learns a regularised linear combination of pretrained Borzoi outputs, with the penalty chosen on validation chromosomes only. The linear probe is the same ridge, but on the trunk embedding that the Borzoi-TL head sees, so it is the fairest comparison in terms of training-data exposure. For variant scoring, each model takes the maximum effect over its own set of CRC tracks, and those sets differ in size, which matters when we compare models later.

### 2:15 — Slide 4: Borzoi fine-tuning setup

This is the fine-tuning setup as it was actually run. The input is a 524 kilobase sequence, and the four pretrained Borzoi replicates are frozen; we average their 1,920-channel trunk embedding. Only a small head is trained: linear to 1,024 units, GELU, dropout, then linear to 183 outputs with a Softplus, about 2.15 million parameters. The targets are 183 CRC TF ChIP-seq tracks covering 77 TFs, mostly HCT-116 and LoVo, trained with a Poisson loss, and the checkpoint is epoch 5 chosen on validation. One important caveat is that the test chromosomes, chr9 and chr22, are not new to the pretrained backbone: 44% of their bases lie in Borzoi training folds and 73% in Enformer training folds.

## Question 1: TF-binding prediction

### 3:00 — Slide 5: Linear readouts outperform head-only Borzoi fine-tuning

This is the first main result. All numbers are mean per-track Pearson r on the same 359 test tiles and the same 128 bp grid, on the 152 tracks that are not identical to pretraining experiments, with confidence intervals from a bootstrap over TF clusters. Both linear readouts beat Borzoi-TL: the ridge by about 0.056 in r and the probe by about 0.038, and those intervals are well away from zero. This replicates across five head seeds and across three other chromosome splits, with the ridge refit per fold, although those heads used a slightly simpler cached training recipe without reverse-complement augmentation, and the probe was not repeated on the other folds. Borzoi-TL is a little better than a single mapped pretrained track; on all 183 tracks that gain is not robust to resampling cell lines (+0.021, CI −0.018 to 0.041), and both supervised baselines exceed it by more. So the conclusion is narrow: this head-only fine-tuning recipe did not add value beyond a linear readout of pretrained Borzoi; it is not a general statement that ridge beats neural fine-tuning.

## Question 2: variant prioritisation

### 4:15 — Slide 6: CRC variant-effect scoring

For the second question we scored the CRC credible variants with all four models. The existing scoring input had 10,148 rows, because some positions were scored with two or three alternative alleles; that is valid as a measure of how sensitive a position is to mutation, but it does not tell us about the actual CRC allele. So we matched each position to the allele pair tested in the GWAS summary statistics, which gave 5,729 positions, and only this set is used for validation claims. The variant score is the largest predicted change between REF and ALT across each model's CRC tracks; for AlphaGenome, from the existing hosted-API run, it is the provider's quantile-normalised score, and window sizes and reverse-complement handling also differ, so we compare models by ranks, not raw magnitudes. The scoring spec was frozen before any validation label was joined. These are predicted regulatory effects, so on their own they are hypotheses, which is why the MPRA comparison comes next.

### 5:20 — Slide 7: MPRA validation design

An MPRA, or massively parallel reporter assay, puts each allele into a reporter construct and measures whether the two alleles drive different expression, here in two CRC lines and one normal colon line. We used the published strong hits as positives and tested variants with no effect as negatives; the intermediate hit class is excluded because its definition is unclear. The key design choice is that we compare variants only within the same risk region and then pool, so a model cannot score well just by distinguishing active loci from inactive ones. Uncertainty comes from resampling whole regions, and significance from permuting labels within regions, with Holm correction across the four models. We added two simple baselines: distance to the nearest gene start, and the count of Chen's own epigenomic annotations, which is the more demanding one.

### 6:15 — Slide 8: Four-model MPRA results

Here are the final pre-registered results; AUROC here is a ranking measure where 0.5 is chance, and it is not a correlation. All four models rank MPRA hits above non-hits within loci, with AUROCs from 0.64 to 0.67, and all beat distance to TSS, which is at chance. The Holm p of 0.0004 is the smallest attainable Holm-adjusted value with 10,000 permutations, so it should not be read as an effect size. The important caveat is the last column: the Chen annotation count alone reaches 0.618, and no model is shown to beat it, and no model is shown to differ from any other. The pre-registered paired difference does not show added value over annotations, which is absence of evidence, not equivalence; a post hoc conditional logit suggests the score stays associated after adjusting for annotation count, with region-bootstrap intervals excluding zero, but that is exploratory and is why we want an independent nested test.

## Interpretation

### 7:15 — Slide 9: Interpretation and limitations

I'll separate limitations we have confirmed from questions that are still open. Confirmed: at least 31 of our 183 tracks are the exact experiments Borzoi and Enformer were trained on, that count is a lower bound, and the test chromosomes overlap their training folds, so the benchmark is exposure-aware but not leakage-free; the pretrained models have also seen the reference sequence of every locus and ENCODE and GTEx data, some of the same data types that underlie the epigenomic annotations. Model comparisons are not perfectly fair, because each model's score is a maximum over a different number and mix of tracks, and our pretrained track set is dominated by LoVo while the fine-tuned set is dominated by HCT-116. In exploratory per-line analyses we see no tumour-specific signal, the MPRA itself is episomal, in three cell lines, with the weight concentrated in about 12 effective regions, and the small chr1, 8, 9 and 22 subset gives only weak descriptive AUROCs of 0.45 to 0.56. The main open issue is the MPRA sign convention: we do not know whether the published fold changes are ALT over REF or the reverse, so the direction test, which came out at 0.27 to 0.36 and is significantly below 0.5 for all four models, cannot be interpreted yet, and we deliberately do not use it to infer the convention.

## Conclusions

### 8:45 — Slide 10: Conclusions and manuscript direction

To summarise: all four models capture within-locus signal that matches MPRA activity, but the pre-registered test does not show more than existing annotations, and the post hoc adjustment result is only suggestive. The fine-tuning result is negative for this specific recipe, and the linear baselines make that clear. A broader lesson is that public CRC ChIP-seq data is already inside the pretraining sets, so any benchmark like this has to report exposure. The manuscript draft is written around this leakage-aware evaluation, with the framing chosen by the rule we fixed before unblinding rather than by preference, and it runs from the variant analysis to the exposure audit to the track benchmark as a negative result.

### 9:30 — Slide 11: Next steps and questions for discussion

The cheapest and most valuable next step is to confirm the MPRA sign convention with the authors, because it decides whether the direction analysis means anything. The second is an independent validation set, ideally another MPRA or a colon caQTL dataset, with a pre-specified test of whether model scores add to annotations. The remaining manuscript work is the design figure, the supplementary methods for track processing, and the introduction. For the title, the current manuscript title is: Leakage-aware evaluation of genomic foundation models for colorectal cancer risk-variant annotation and transcription-factor binding, and the alternative term is sequence-to-function models. I'd like your view on three things: whether this framing is right for publication, which validation dataset is worth pursuing and whether to try fine-tuning the backbone, and the title, venue and data-sharing questions.
