# SD-005 variant-scoring specification (FROZEN 2026-10-20, before any validation label is joined)

Status: frozen by the computational engineer for coordinator review. No MPRA / eQTL / GWAS-allele label has been
opened or joined in building this spec, the track lists or the scores. Any later change = new dated version + SD entry.

Genome build: **hg38** everywhere (variants, `borzoi_data/hg38/hg38.ml.fa`, model tracks). Variant input:
`alphagenome_variant_effects/input/S6_CCV_ChrPosRefAlt_rsid.tsv` (10,148 rows = exploratory set; 5,739 unique positions;
`chr:pos:ref:alt`, POS **1-based**). The allele-matched subset (SD-003b) is a later JOIN on `row_id` / `input_variant`.
REF is asserted against the FASTA for every row (index `POS-1`, 0-based); any mismatch aborts the run (`ref_check.tsv`
is written). Alleles are reference-strand and never complemented. Code: `variant_scoring/{common,runner,score_borzoi,score_enformer}.py`.

## 1. Per-model windows, bins, RC (all scripts)

| | Borzoi-TL / pretrained Borzoi | Enformer | AlphaGenome (existing run) |
|---|---|---|---|
| input length | 524,288 bp | 196,608 bp | 1,048,576 bp (`SEQUENCE_LENGTH_1MB`) |
| variant placement | window `[p0-L/2, p0+L/2)`, `p0=POS-1`; variant base at window index **L/2** (first base of the right half; identical to `borzoi_code/dataset.py` centre convention) | same | `Variant.reference_interval.resize(L)` -> offset L/2-1 (1 bp difference, irrelevant because scoring uses a variant-centred mask) |
| output | 6144 bins x 32 bp (central 196,608 bp) | 896 bins x 128 bp (central 114,688 bp) | scorer-internal (hosted API), `CenterMaskScorer` |
| aggregation window | **512 bp = `[p0-256, p0+256)`** = output bins **3064..3079** (16 bins; bin 3072 starts at the variant base) | 512 bp = bins **446..449** (4 bins; bin 448 starts at the variant base) | 501 bp centred on variant (`width=501`) |
| aggregation | **sum** over bins of the strand-averaged prediction (sum vs mean is irrelevant to the ratio except via the pseudocount; sum is fixed) | same | `DIFF_LOG2_SUM` (sum then log2; server-side epsilon not exposed) |
| RC averaging | yes: forward and reverse-complement sequence; RC output flipped along bin axis and averaged (the bin slice is symmetric about the output centre, so bins coincide; tested) | yes | not controlled by us (server-side; no explicit RC step) |
| replicates | Borzoi: 4 replicates (`johahi/borzoi-replicate-0..3`), outputs averaged (pretrained) / trunk embeddings averaged (TL, as `borzoi_code/model.py`) | single model `EleutherAI/enformer-official-rough` (enformer_pytorch, torch-venv; project tf-venv has no loader) | hosted API, "all folds" server-side |
| one-hot | ACGT channels, N = zeros; RC = reverse positions + flip channels | same | n/a |

The 512 bp window is the closest whole-bin window to 501 bp for both 32 bp and 128 bp
bins; it is symmetric about the variant bin boundary, not about the variant base (variant is base 256 of 512).

**Chromosome-end rule (Borzoi, Enformer):** windows that run off a chromosome are **N-padded** (zero one-hot), the variant
stays at window index L/2; no shifting, no row dropping. `variants.tsv` records `n_pad_left`, `n_pad_right`,
`window_padded`, `n_N_in_agg_window`. Counts in the 10,148 rows: Borzoi 149 rows / 91 positions padded; Enformer 30 rows /
17 positions (computed from the .fai; tested). Sensitivity analyses should exclude `window_padded` rows and rows with
`n_N_in_agg_window>0`.
**AlphaGenome** (already run, unchanged): the 1 Mb window is **shifted** to fit (variant off-centre; `interval_shifted=True`
for 242 rows / 147 positions in `output/variant_run_status.tsv`); these rows are flagged, not re-run; scores use the
variant-centred 501 bp mask so remain defined. Sensitivity analyses exclude those 242 rows.

## 2. Score definitions (identical for all models)

For track *t*: `REF_t`, `ALT_t` = aggregated (sum over the aggregation bins, RC-averaged) model output, **raw model output units**
(softplus outputs; no un-squashing/unscaling), REF scored once per position, ALT per row.
* signed per-track effect: `lfc_t = log2((ALT_t + pc) / (REF_t + pc))`, **pseudocount pc = 1.0** added to the aggregated sums
  (the Borzoi SAD convention `log2(sum+1)`). The raw `REF_t`/`ALT_t` sums are stored (`ref_sum_*.npy`, `alt_sum_*.npy`), so pc
  sensitivity (e.g. 0.1, 10) can be recomputed without GPU. Note raw outputs of Borzoi/Enformer are in squashed model scale, so pc=1 is
  not equivalent across models; cross-model comparison uses ranks / within-model standardisation, not raw magnitudes.
* **PRIMARY score** = `max_t |lfc_t|` over the frozen track list (column `primary_max_abs_log2fc`).
* **Signed companion** (for the direction sign test) = `lfc` at the arg-max-|lfc| track (`signed_at_argmax`; arg-max track index in
  `argmax_track_pos`), plus `mean_signed_log2fc` (mean of signed `lfc_t` over tracks).
* **Mean-over-tracks sensitivity** = `mean_t |lfc_t|` (`mean_abs_log2fc`) (max favours models with more tracks, so this is reported alongside).
* Ties/NaN: none expected; NaN would be reported, not imputed.

## 3. Frozen track lists (`variant_scoring/track_lists/`, built by `make_track_lists.py`, metadata keywords only, no performance info)

* `borzoi_tl_tracks.tsv`: Borzoi-TL heads 0..182 (order = `borzoi_ckpt_full/bw_files.json` = `results_full/test_per_track_metrics.csv`,
  verified identical; `track_idx` is the 0-based head index, checked against `bw_files.json` DCid order). **All 183 scored (secondary set `all183`); primary = 152 (`primary152`)**:
  exclude the 31 `identical_experiment` tracks of `exposure_audit/track_exposure.tsv` (`Borzoi_class`): head indices 13..36 (24 LoVo GSE51142), 78, 81, 93 (HCT-116 ENCODE) and
  145..148 (ENCODE colon ENCSR). Per-track `borzoi/enformer/alphagenome_exposure_class` are recorded in the list file. (Supersedes the earlier 155/28 figure of SD-013/RV-002.)
  Rationale: those assays were training targets of the pretrained trunk; MPRA/eQTL labels are different data, so all183 stays a legitimate secondary analysis.
* `borzoi_pretrained_crc_tracks.tsv` (132 tracks of `borzoi_data/targets_human.csv`, 0-based `track_index` = output channel,
  index==row asserted) and `enformer_pretrained_crc_tracks.tsv` (132 tracks of `enformer_data/targets_human.txt`, 5,313 channels):
  assay prefix in {CHIP, DNASE, ATAC} AND description matches a CRC cell-line regex
  `HCT116|HT-29|LoVo|Caco-2|SW480|SW620|RKO|COLO205|DLD-1|T84|LS174T|LS180|GP5d`; **histone-mark ChIP excluded** (same regex semantics as
  `borzoi_baseline_utils.is_histone`, checked identical on the matched set); CAGE/RNA excluded; non-malignant colon (HCEC, colonic mucosa) excluded.
  Result: 126 TF ChIP + 6 DNase (HCT116, HT-29, LoVo, Caco-2, SW480, RKO); no CRC ATAC tracks exist. **118 of 132 are LoVo ChIP tracks**
  (GEO LoVo TF batch) - the max statistic is therefore LoVo-weighted; report a per-cell-line breakdown as sensitivity.
  Both lists contain the same 132 assays/cell lines (Enformer indices differ). 24 LoVo GSE51142 tracks are training targets of both
  models (A18); that is irrelevant for variant-effect labels but should be stated.
* Whole-vocabulary arrays are stored too (`log2fc_pretrained.npy`: all 7,611 / 5,313 tracks), so any other track subset can be
  derived post hoc but would be exploratory.
* `alphagenome_crc_tracks.tsv`: from `alphagenome_variant_effects/output/variant_all_scores_per_track/*.parquet`, scorer_kind `diff`,
  modality in {DNASE, ATAC, CHIP_TF}, same CRC cell-line regex on `biosample_name`: 33 tracks (6 DNASE, 1 ATAC, 26 CHIP_TF; HCT116 27, Caco-2 2,
  HT-29, RKO, SW480, LoVo 1 each). Key = (modality, track_name, track_strand). HCEC 1CT (non-malignant) excluded for consistency.
  CHIP_HISTONE, CAGE, PROCAP, RNA-seq, splicing, contact maps are not part of the primary AlphaGenome score.

## 4. AlphaGenome score (no re-run; computed from the cache)

Scorers (alphagenome 0.8.0 `get_recommended_scorers`): `CenterMaskScorer(requested_output in {DNASE, ATAC, CHIP_TF}, width=501,
aggregation_type=DIFF_LOG2_SUM)`; `raw_score` = `log2(sum ALT) - log2(sum REF)` over the 501 bp centre mask (same quantity as `lfc`, but
epsilon/pseudocount fixed server-side and no RC averaging by us). The column `quantile_score` is the provider's per-track
quantile of the raw score against its own background of variants (signed, range about [-1, 1]); the background distribution is NOT exposed by the
client and was not recomputed - treated as a provider-defined number.
* **AlphaGenome raw-version primary** (cross-model comparable) = `max_t |raw_score_t|` over the 33 tracks; signed companion = `raw_score` at the arg-max track;
  mean-sensitivity = `mean_t |raw_score_t|`.
* **AlphaGenome quantile version** (SD-005 "quantile version"; per-track-calibrated, recommended by the provider for comparing tracks) =
  `max_t |quantile_score_t|`; signed = `quantile_score` at the arg-max track.
Both are frozen and reported; which is the headline AlphaGenome number is **left for the coordinator** (this spec does not choose the winner post hoc:
the raw version is the like-for-like model comparator, the quantile version is the AlphaGenome-native SD-005 statistic).
`output/variant_run_status.tsv` (status SUCCESS for all 10,148 rows, 0 REF mismatches) is the AlphaGenome row-level audit.

## 5. Outputs (git-ignored) `results_phase2/variant_scores/<model>/`

`variants.tsv` (rows, pad/N flags, `pos_id`), `positions.tsv`, `ref_check.tsv`, `ref_sum_<head>.npy` (positions x T), `alt_sum_<head>.npy` (rows x T),
`log2fc_<head>.npy` (rows x T, signed, pc=1), `summary.tsv` (per row, per head and track set: primary, signed_at_argmax, argmax_track_pos,
mean_signed, mean_abs), `chunks/` (checkpoints every 300 positions ~ 530 rows; reruns resume). heads: Borzoi `pretrained` (7611), `tl` (183); Enformer `pretrained` (5313).
Track column index = model output channel index (Borzoi/Enformer: targets file `track_index`; TL: head index).

## 6. Known limitations (to be carried into any write-up)
* Pretrained Borzoi/Enformer/AlphaGenome were trained on genome-wide sequence incl. chr9/chr22 and on GTEx/ENCODE data; eQTL concordance is only partly independent (RV-002).
* Raw output scale differs across models; pc=1 is a convention, not an optimum.
* 512 vs 501 bp and 16/4-bin granularity; Enformer 128 bp bins blur the variant position more than Borzoi 32 bp bins.
* Track lists are LoVo-dominated for Borzoi/Enformer, HCT116-dominated for AlphaGenome.
* 149 (Borzoi) / 30 (Enformer) / 242 (AlphaGenome) rows have chromosome-end windows handled differently (pad vs shift).

## Coordinator decision (2026-10-20, before any validation label was joined)
AlphaGenome headline score = **quantile version** (`max_t |quantile_score_t|`, signed companion at arg-max), as specified in
approved SD-005. The raw-log2 version is a pre-specified sensitivity / like-for-like comparator. Recorded as SD-014.
