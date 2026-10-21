"""Slide text and speaker notes for the 2026-10-21 PI update. Every number is traced in PI_UPDATE_SOURCE_AUDIT.md."""

N = "−"  # minus sign

SLIDES = [
    # 1 ------------------------------------------------------------------------------------------------------
    dict(
        kind="title", section="Introduction", start="0:00",
        title="Evaluating genomic sequence models for colorectal cancer TF binding and risk variants",
        subtitle=["Research progress update", "PI meeting, 21 October 2026"],
        notes=[
            "Today I'll give an update on the colorectal cancer project, where we evaluate genomic sequence models for two tasks.",
            "The first is predicting CRC transcription-factor binding, and the second is prioritising CRC risk variants.",
            "The main results are now final, all analysis phases have passed our internal, agent-based review checkpoints, and a full manuscript draft exists.",
            "I'll go through the design and the two main results, then the limitations, and finish with a few questions where I'd value your input.",
        ],
    ),
    # 2 ------------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Introduction", start="0:30",
        title="Research overview and questions",
        bullets=[
            "CRC fine-mapping defines 5,741 credible causal variants (Chen 2024), mostly noncoding",
            "Sequence-to-function models (Borzoi, Enformer, AlphaGenome) can score such variants",
            "Q1: Does supervised adaptation of Borzoi improve CRC TF-binding prediction?",
            ("183 tracks (152 non-identical); chromosomes held out from fine-tuning only", 1),
            "Q2: Can sequence models prioritise experimentally functional CRC variants?",
            ("Validation: Law et al. 2024 MPRA, compared within risk loci", 1),
        ],
        notes=[
            "The motivation is that CRC GWAS fine-mapping gives thousands of credible causal variants, and most of them are noncoding, so their function is hard to interpret.",
            "Sequence models like Borzoi, Enformer and AlphaGenome predict regulatory signals directly from DNA sequence, so in principle they can score each variant.",
            "We asked two questions. First, does fine-tuning Borzoi on CRC ChIP-seq data actually improve CRC TF-binding prediction over what the pretrained model already knows?",
            "Second, do these models rank variants that are functional in an MPRA above variants that are not, within the same risk locus?",
            "The MPRA analysis plan and the ridge decision rule were frozen before results were seen, although the chr9 and chr22 test set had been used in earlier development, and we audited how much of our test data the pretrained models had already seen during training.",
        ],
    ),
    # 3 ------------------------------------------------------------------------------------------------------
    dict(
        kind="table", section="Design", start="1:30",
        title="Models, baselines and how each is used",
        table=dict(
            name="Models table",
            header=["Model", "What is fitted to CRC data", "TF-binding benchmark", "Variant scoring"],
            col_widths=[3.0, 4.4, 2.4, 2.333],
            row_h=0.55,
            rows=[
                ["Borzoi-TL (fine-tuned)", "New prediction head on frozen Borzoi trunk", "Main model", "152 tracks"],
                ["Pretrained Borzoi", "Nothing; one output track mapped per target", "Comparator", "132 CRC tracks"],
                ["Ridge on pretrained Borzoi", "Linear readout of 1,888 pretrained outputs", "Supervised baseline", "Not used"],
                ["Linear probe", "Ridge on the same trunk embedding as Borzoi-TL", "Supervised baseline", "Not used"],
                ["Enformer", "Nothing; mapped track", "Supplementary", "132 CRC tracks"],
                ["AlphaGenome", "Nothing; hosted API", "Supplementary", "33 CRC tracks"],
            ],
        ),
        notes=[
            "Here are the models and baselines and where each one is used.",
            "Borzoi-TL is our fine-tuned model: a new prediction head trained on CRC ChIP-seq on top of the frozen pretrained Borzoi trunk.",
            "Ridge regression is not another foundation model; it is a supervised baseline that learns a regularised linear combination of pretrained Borzoi outputs, with the penalty chosen on validation chromosomes only.",
            "The linear probe is the same ridge, but on the trunk embedding that the Borzoi-TL head sees, so it is the fairest comparison in terms of training-data exposure.",
            "For variant scoring, each model takes the maximum effect over its own set of CRC tracks, and those sets differ in size, which matters when we compare models later.",
        ],
    ),
    # 4 ------------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Design", start="2:15",
        title="Borzoi fine-tuning setup",
        bullets=[
            dict(text="Input: 524,288 bp one-hot DNA sequence (hg38)", numbered=True),
            dict(text="Frozen trunk: 4 pretrained Borzoi replicates, 1,920-channel embedding averaged", numbered=True),
            dict(text="Trainable head only: two linear layers with GELU and Softplus (2.15 M parameters)", numbered=True),
            dict(text="Targets: 183 CRC TF ChIP-seq tracks, 77 TFs, 10 cell lines, colon tissue and one tumour", numbered=True),
            dict(text="Training: Poisson loss, Adam, learning rate 1e-4; best epoch 5 of 8 on validation", numbered=True),
            dict(text="Splits: train other chromosomes; validation chr1, chr8; test chr9, chr22 (359 tiles)", numbered=True),
        ],
        notes=[
            "This is the fine-tuning setup as it was actually run.",
            "The input is a 524 kilobase sequence, and the four pretrained Borzoi replicates are frozen; we average their 1,920-channel trunk embedding.",
            "Only a small head is trained: linear to 1,024 units, GELU, dropout, then linear to 183 outputs with a Softplus, about 2.15 million parameters.",
            "The targets are 183 CRC TF ChIP-seq tracks covering 77 TFs, mostly HCT-116 and LoVo, trained with a Poisson loss, and the checkpoint is epoch 5 chosen on validation.",
            "One important caveat is that the test chromosomes, chr9 and chr22, are not new to the pretrained backbone: 44% of their bases lie in Borzoi training folds and 73% in Enformer training folds.",
        ],
    ),
    # 5 ------------------------------------------------------------------------------------------------------
    dict(
        kind="table", section="Question 1: TF-binding prediction", start="3:00",
        title="Linear readouts outperform head-only Borzoi fine-tuning",
        table=dict(
            name="Track benchmark table",
            header=["Comparison (152 non-identical tracks)", "Comparator r", "Borzoi-TL r",
                    "Δr, TL minus comparator [95% TF-cluster CI]"],
            col_widths=[4.6, 1.9, 1.9, 3.733],
            row_h=0.6,
            rows=[
                ["Ridge on pretrained Borzoi outputs", "0.741", "0.685", f"{N}0.0555 [{N}0.0756, {N}0.0428]"],
                ["Linear probe on frozen trunk", "0.723", "0.685", f"{N}0.0378 [{N}0.0525, {N}0.0289]"],
                ["Pretrained Borzoi, validation-selected track", "0.651", "0.685", "+0.0341 [0.0238, 0.0453]"],
                ["Ridge; 5 seeds (cache recipe)", "", "", f"{N}0.0537 to {N}0.0554; tile, TF, cell-line CIs exclude 0"],
                ["Ridge; 3 other folds × 3 seeds (cache recipe)", "", "", f"{N}0.0615 to {N}0.0775; tile, TF, cell-line CIs exclude 0"],
            ],
        ),
        notes=[
            "This is the first main result. All numbers are mean per-track Pearson r on the same 359 test tiles and the same 128 bp grid, on the 152 tracks that are not identical to pretraining experiments, with confidence intervals from a bootstrap over TF clusters.",
            "Both linear readouts beat Borzoi-TL: the ridge by about 0.056 in r and the probe by about 0.038, and those intervals are well away from zero.",
            "This replicates across five head seeds and across three other chromosome splits, with the ridge refit per fold, although those heads used a slightly simpler cached training recipe without reverse-complement augmentation, and the probe was not repeated on the other folds.",
            "Borzoi-TL is a little better than a single mapped pretrained track; on all 183 tracks that gain is not robust to resampling cell lines (+0.021, CI −0.018 to 0.041), and both supervised baselines exceed it by more.",
            "So the conclusion is narrow: this head-only fine-tuning recipe did not add value beyond a linear readout of pretrained Borzoi; it is not a general statement that ridge beats neural fine-tuning.",
        ],
    ),
    # 6 ------------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Question 2: variant prioritisation", start="4:15",
        title="CRC variant-effect scoring",
        bullets=[
            "5,739 of 5,741 CCVs scored as 10,148 REF/ALT rows; REF checked against hg38 (0 mismatches)",
            "Several ALTs per position measure positional mutational sensitivity, not disease alleles",
            "Allele-matched set: 5,729 positions with the GWAS-tested allele pair, used for all claims",
            ("Matched means the tested pair, not a confirmed causal or risk allele", 1),
            "Score: max over CRC tracks of |log2 FC| (ALT vs REF); AlphaGenome: max |quantile score|",
            "All four scored under a spec frozen before labels; windows and normalisation differ by model",
        ],
        notes=[
            "For the second question we scored the CRC credible variants with all four models.",
            "The existing scoring input had 10,148 rows, because some positions were scored with two or three alternative alleles; that is valid as a measure of how sensitive a position is to mutation, but it does not tell us about the actual CRC allele.",
            "So we matched each position to the allele pair tested in the GWAS summary statistics, which gave 5,729 positions, and only this set is used for validation claims.",
            "The variant score is the largest predicted change between REF and ALT across each model's CRC tracks; for AlphaGenome, from the existing hosted-API run, it is the provider's quantile-normalised score, and window sizes and reverse-complement handling also differ, so we compare models by ranks, not raw magnitudes. The scoring spec was frozen before any validation label was joined.",
            "These are predicted regulatory effects, so on their own they are hypotheses, which is why the MPRA comparison comes next.",
        ],
    ),
    # 7 ------------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Question 2: variant prioritisation", start="5:20",
        title="MPRA validation design",
        bullets=[
            dict(text="Labels: Law et al. 2024 MPRA of allelic activity in HT29, SW403 and HCEC-1CT", numbered=True),
            dict(text="Positives: strong allelic hits; negatives: tested non-hits; weaker hits excluded", numbered=True),
            dict(text="Within each risk region: AUROC of model score, positives vs negatives", numbered=True),
            dict(text="Pool 58 informative regions (130 positives, 2,518 negatives); region bootstrap CI", numbered=True),
            dict(text="Baselines: distance to nearest TSS; Chen annotation count (7 epigenomic flags)", numbered=True),
            dict(text="Pre-registered rule: Holm p < 0.05 (4 models), CI > 0.5, ΔAUROC vs distance CI > 0", numbered=True),
        ],
        notes=[
            "An MPRA, or massively parallel reporter assay, puts each allele into a reporter construct and measures whether the two alleles drive different expression, here in two CRC lines and one normal colon line.",
            "We used the published strong hits as positives and tested variants with no effect as negatives; the intermediate hit class is excluded because its definition is unclear.",
            "The key design choice is that we compare variants only within the same risk region and then pool, so a model cannot score well just by distinguishing active loci from inactive ones.",
            "Uncertainty comes from resampling whole regions, and significance from permuting labels within regions, with Holm correction across the four models.",
            "We added two simple baselines: distance to the nearest gene start, and the count of Chen's own epigenomic annotations, which is the more demanding one.",
        ],
    ),
    # 8 ------------------------------------------------------------------------------------------------------
    dict(
        kind="table", section="Question 2: variant prioritisation", start="6:15",
        title="Four-model MPRA results",
        table=dict(
            name="MPRA results table",
            header=["Score", "Within-locus AUROC [95% CI]", "Holm p",
                    "ΔAUROC vs distance [95% CI]", "ΔAUROC vs annotation [95% CI]"],
            col_widths=[2.533, 2.9, 1.2, 2.75, 2.75],
            row_h=0.55,
            rows=[
                ["Enformer", "0.674 [0.607, 0.756]", "0.0004", "+0.159 [0.060, 0.280]", f"+0.056 [{N}0.006, 0.124]"],
                ["AlphaGenome (quantile)", "0.666 [0.590, 0.759]", "0.0004", "+0.151 [0.049, 0.272]", f"+0.048 [{N}0.035, 0.138]"],
                ["Borzoi-TL", "0.661 [0.584, 0.768]", "0.0004", "+0.145 [0.036, 0.293]", f"+0.042 [{N}0.014, 0.126]"],
                ["Pretrained Borzoi", "0.640 [0.554, 0.744]", "0.0004", "+0.125 [0.004, 0.263]", f"+0.022 [{N}0.063, 0.116]"],
                ["Distance to TSS (baseline)", "0.515 [0.445, 0.567]", "—", "—", "—"],
                ["Annotation count (baseline)", "0.618 [0.556, 0.684]", "—", "—", "—"],
            ],
        ),
        notes=[
            "Here are the final pre-registered results; AUROC here is a ranking measure where 0.5 is chance, and it is not a correlation.",
            "All four models rank MPRA hits above non-hits within loci, with AUROCs from 0.64 to 0.67, and all beat distance to TSS, which is at chance.",
            "The Holm p of 0.0004 is the smallest attainable Holm-adjusted value with 10,000 permutations, so it should not be read as an effect size.",
            "The important caveat is the last column: the Chen annotation count alone reaches 0.618, and no model is shown to beat it, and no model is shown to differ from any other.",
            "The pre-registered paired difference does not show added value over annotations, which is absence of evidence, not equivalence; a post hoc conditional logit suggests the score stays associated after adjusting for annotation count, with region-bootstrap intervals excluding zero, but that is exploratory and is why we want an independent nested test.",
        ],
    ),
    # 9 ------------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Interpretation", start="7:15",
        title="Interpretation and limitations",
        bullets=[
            "Confirmed: ≥31 of 183 tracks are Borzoi/Enformer training data; 44% / 73% of test bp in their training folds",
            "Confirmed: unequal comparisons; scores are maxima over 152, 132, 132 or 33 tracks with different windows",
            "Confirmed: track selection; pretrained set LoVo-heavy, Borzoi-TL and AlphaGenome sets HCT-116-heavy",
            "Observed (exploratory): no tumour-specific signal; normal-colon labels give similar AUROC",
            "Unresolved: MPRA fold-change sign convention, so the direction test (0.27–0.36) is uninterpretable",
            "Unresolved: AlphaGenome training exposure; no second independent validation dataset yet",
        ],
        notes=[
            "I'll separate limitations we have confirmed from questions that are still open.",
            "Confirmed: at least 31 of our 183 tracks are the exact experiments Borzoi and Enformer were trained on, that count is a lower bound, and the test chromosomes overlap their training folds, so the benchmark is exposure-aware but not leakage-free; the pretrained models have also seen the reference sequence of every locus and ENCODE and GTEx data, some of the same data types that underlie the epigenomic annotations.",
            "Model comparisons are not perfectly fair, because each model's score is a maximum over a different number and mix of tracks, and our pretrained track set is dominated by LoVo while the fine-tuned set is dominated by HCT-116.",
            "In exploratory per-line analyses we see no tumour-specific signal, the MPRA itself is episomal, in three cell lines, with the weight concentrated in about 12 effective regions, and the small chr1, 8, 9 and 22 subset gives only weak descriptive AUROCs of 0.45 to 0.56.",
            "The main open issue is the MPRA sign convention: we do not know whether the published fold changes are ALT over REF or the reverse, so the direction test, which came out at 0.27 to 0.36 and is significantly below 0.5 for all four models, cannot be interpreted yet, and we deliberately do not use it to infer the convention.",
        ],
    ),
    # 10 -----------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Conclusions", start="8:45",
        title="Conclusions and manuscript direction",
        bullets=[
            "All four models prioritise MPRA-active CRC variants within loci above chance and distance",
            "No model is shown to beat the annotation-count baseline or another model",
            "Head-only Borzoi fine-tuning did not beat linear readouts of the pretrained model",
            "CRC benchmarks need explicit reporting of pretraining-data exposure",
            "Manuscript v2: leakage-aware evaluation, variant-annotation framing chosen by pre-registered rule",
            ("Post hoc: score still associated after annotation adjustment (suggestive)", 1),
        ],
        notes=[
            "To summarise: all four models capture within-locus signal that matches MPRA activity, but the pre-registered test does not show more than existing annotations, and the post hoc adjustment result is only suggestive.",
            "The fine-tuning result is negative for this specific recipe, and the linear baselines make that clear.",
            "A broader lesson is that public CRC ChIP-seq data is already inside the pretraining sets, so any benchmark like this has to report exposure.",
            "The manuscript draft is written around this leakage-aware evaluation, with the framing chosen by the rule we fixed before unblinding rather than by preference, and it runs from the variant analysis to the exposure audit to the track benchmark as a negative result.",
        ],
    ),
    # 11 -----------------------------------------------------------------------------------------------------
    dict(
        kind="bullets", section="Conclusions", start="9:30",
        title="Next steps and questions for discussion",
        bullets=[
            "Next: confirm the MPRA sign convention with the Law/Houlston group",
            "Next: independent MPRA or colon caQTL test of model plus annotation vs annotation alone",
            "Next: Fig. 1 schematic, supplementary methods, introduction, final review",
            "Question: is a leakage-aware benchmark with a negative fine-tuning result the right framing?",
            "Question: which validation dataset to obtain, and is backbone fine-tuning worth testing?",
            "Question: title wording, venue, and what data can be shared in a public repository",
        ],
        notes=[
            "The cheapest and most valuable next step is to confirm the MPRA sign convention with the authors, because it decides whether the direction analysis means anything.",
            "The second is an independent validation set, ideally another MPRA or a colon caQTL dataset, with a pre-specified test of whether model scores add to annotations.",
            "The remaining manuscript work is the design figure, the supplementary methods for track processing, and the introduction.",
            "For the title, the current manuscript title is: Leakage-aware evaluation of genomic foundation models for colorectal cancer risk-variant annotation and transcription-factor binding, and the alternative term is sequence-to-function models.", "I'd like your view on three things: whether this framing is right for publication, which validation dataset is worth pursuing and whether to try fine-tuning the backbone, and the title, venue and data-sharing questions.",
        ],
    ),
]
