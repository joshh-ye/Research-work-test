# Phase 1 Inventory — repository discovery and research inventory

Written 2026-10-19 by research-investigator (Sonnet). Branch `research/crc-autonomous-study` @ `34209c0` (not switched).
Labels: **VERIFIED** = seen directly in a file/log/command this session; **INFERRED** = reasoning stated; **UNKNOWN** = not recoverable.
Metric types are named per row: **r** = Pearson r; **r²** = squared Pearson r; **R²** = prediction R² (1 − SSE/SST); **R²-valcal** = validation-calibrated R² (2-parameter linear fit on chr1+chr8, applied unchanged to test).
No code or results were modified; nothing was run on GPU. CPU reads only (CSV/JSON/headers, a pyfaidx REF check, pyBigWig header check).
All paths are relative to `/home/yejz1/experimental_research` unless stated. `/home/yejz1/experimental_research_clean` (a separate working copy, commits `1cad259`, `cfaa74a`) holds `DEVELOPMENT_MILESTONES.md`, `FINALIZED_PIPELINE.md`, `archive/` — used as corroborating evidence only.

---

## 1. Original research objectives

| Claim | Evidence | Label |
|---|---|---|
| Project began as transfer learning on frozen Borzoi backbones to predict ChIP-seq/ATAC-seq signal from DNA sequence; repo = Colab notebook → headless `main.py` | `README.md:1-4`; "Changes from the Colab notebook" table in `README.md`; commits `e6a63c4`, `8553c05` (2026-05-23) | VERIFIED |
| First commits (2026-04-02: `eb19a34`, `1801e6c`, `f04d14d`, `8fbc662`) were only ACCRE GPU/TensorFlow bring-up, no genomics | `git log --all`; clean-copy `DEVELOPMENT_MILESTONES.md` Stage 0 | VERIFIED |
| Objective as stated in the manuscript draft: adapt frozen Borzoi to 183 CRC TF ChIP-seq tracks and test whether (A) augmentation + training recipe improves it (internal ablation) and (B) it beats original Borzoi best-matched per target | `research_paper_draft.md` Abstract, §4 | VERIFIED (as stated); the ablation arm (A) has no traceable data, see §6 |
| Objective as stated for Qing (2026-10-11): predict CRC TF ChIP coverage from sequence and quantify whether the adapted model explains the tracks better than off-the-shelf Borzoi, Enformer, AlphaGenome | `reports/qing_deliverables/Qing_Research_Summary.md` §1 | VERIFIED |
| Objective in the new autonomous study: assess model prediction of CRC regulatory activity and estimate SNP effects at CRC-relevant variants | `research_management/MASTER_PLAN.md` Goal | VERIFIED |
| The manuscript draft explicitly lists variant-effect scoring as not done: "we do not yet extend these per-track accuracy gains to variant-effect scoring…" | `research_paper_prep/manuscript.md` Limitations ("Downstream applications") | VERIFIED |
| The structural template for the manuscript is a PLOS Genetics paper (Li et al., "Tissue-specific transfer learning improves functional variant and therapeutic target discoveries in breast and prostate cancer", 22(5):e1012145, 2026) | `research_paper_prep/journal.pgen.1012145.xml` `<article-title>`, `<volume>22`, `<issue>5`, `<elocation-id>e1012145` (opened) | VERIFIED (that the file is that article; I did not independently verify the article's content or existence beyond the file) |
| Original motive for the AlphaGenome variant project (S6_CCV) and its link to CRC | no document records it; input file has no header/provenance; the README only says "source data (read-only)" | UNKNOWN |

Presentations: `build_presentation.py` (585 lines) and `pi_update_presentation.py` (387 lines) were committed together in `28f1cfb` (2026-08-09). They are python-pptx builders; they restate the draft's narrative (augmentation ablation; filtered-Borzoi baseline; val then test results; caveat about pseudoreplication). Not re-read in full; numbers extracted in §6.

---

## 2. Completed experiments — result directory → script → commit

Branch topology (VERIFIED, `git merge-base`, `git ls-tree`):
- `stable-version` (`ec22f4f`, 2026-05-30) diverges from `main` at `fe8febb` (2026-05-28); contains only the early 10-track-era pipeline; no results, no baselines.
- `main` local `b9a3639` (2026-05-29) is an ancestor of `origin/main` `0897410` (2026-06-22). Neither has `training_run.slurm`, baselines or results code (only `benchmark_workers.py/.slurm` in local main).
- `experimentational_fix` (`e65657f`, 2026-06-27) branches from `a230dcb` (2026-06-01); adds worker-crash probes, `baseline_borzoi.py`, `training_run.slurm`. Strict ancestor of `experimental_branch`.
- `experimental_branch` = `research/crc-manuscript-study` = `ff88be2` (2026-08-21): superset — adds all baselines (Borzoi filtered, Enformer, AlphaGenome), comparison/plot scripts, `research_paper_*`, presentations, committed `borzoi_ckpt_full/*.pt` (42 MB) and `results_comparison_*`/`results_benchmark` files. `research/crc-autonomous-study` = `ff88be2` + `34209c0` (setup only; adds CLAUDE.md, `.claude/`, `research_management/`).
- Net: all science is on `experimental_branch`/`ff88be2`; other branches are behind with no unique results (also stated in `reports/qing_deliverables/Reproducibility.md`; confirmed by the ancestry checks above).

Key commits for the science (VERIFIED from `git log -S`, `git log -- <path>`):
- `8553c05` 2026-05-23: `main.py` + `borzoi_code/*`; fixed 524,288 bp tiling, chr1/chr8 val, chr9/chr22 test, Poisson NLL (`borzoi_code/genome_tiler.py`).
- `e57d537` 2026-06-12 "high performance volatile commit": trunk-embedding MLP head, RC + ±128 bp shift augmentation, weight decay, ReduceLROnPlateau, early stopping, optional `--balance-tracks` (all introduced in one commit via `git log -S max_shift_bp / weighted_poisson / ReduceLROnPlateau`).
- `3d84156` 2026-08-04 first original-Borzoi test benchmark; `28f1cfb` 2026-08-09 track filtering + Enformer comparison + presentations + draft; `9411535` 2026-08-21 Enformer fixes; `ff88be2` 2026-08-21 AlphaGenome benchmark + checkpoint + result CSVs.

| Result dir / file | Producing script | Slurm log / date | Commit | Content / status |
|---|---|---|---|---|
| `results/` (5 PNGs, 2026-05-30) | early `main.py` | `borzoi_eval-11436632.out` (2026-05-29; 10 tracks, linear head on 7,611-d Borzoi output) | `cc7f4f4`/`fe8febb` era | Sanity figures only. Log: "Mean Pearson R across 10 tracks: 0.4078" (r, val). Architecture later abandoned. VERIFIED |
| `results_full/` (2026-06-28→07-01) | `main.py --epochs 8 --ckpt-dir ./borzoi_ckpt_full --output-dir ./results_full` (`training_run.slurm`) | `borzoi_eval-12212537.out` (start Sun Jun 28 22:33 CDT) | code at `e57d537` (model/dataset) + `e65657f` (main.py) per `Reproducibility.md`; `training_run.slurm` first in `experimentational_fix` | Definitive 183-track run: val r 0.6434, test r 0.6634 (native 32 bp grid). Test/val preds+targets `.npy` (1.08 GB / 2.25 GB), per-track CSVs, history. VERIFIED (log lines: "mean R=0.6434", "mean R=0.6634") |
| `borzoi_ckpt_full/` | same run | same | checkpoint committed in `ff88be2` | `model_head_best.pt` (epoch 5, val NLL 0.18523, `best_val.json`), `model_head.pt` (epoch 8, `epoch.txt`=8), `optimizer.pt`, `history.json`, `bw_files.json` (183 files). VERIFIED |
| `borzoi_ckpt/` | early runs | May 2026 | — | only `bw_files.json` (10 files); no weights. VERIFIED |
| `results_baseline_filtered/borzoi_filtered_*_test.{npy,csv}` (2026-08-05) | `baseline_borzoi_filtered.py` (+`filter_borzoi_tracks.py`, `borzoi_baseline_utils.py`) | `borzoi_filt_test-13005163.out` (Aug 5 01:50) | `3d84156`/`28f1cfb` | pretrained Borzoi best-match vs 183 targets, native 32 bp grid: mean r 0.6456, median 0.6658 (log). VERIFIED |
| `results_baseline_filtered/*_enformergrid.*` (2026-08-21) | `baseline_borzoi_filtered.py --grid enformer` | no .out on disk; `results_benchmark/runtime_test.csv` says 9m39s on A100-80GB | `9411535`/`ff88be2` | shared 896×128 bp grid. INFERRED run was interactive |
| `results_baseline_enformer/` (2026-08-21) | `baseline_enformer.py`, `filter_enformer_tracks.py`, `enformer_baseline_utils.py` | `enformer_base_test-13349427.out` FAILED (ModuleNotFoundError `enformer_pytorch`, then FileNotFoundError `enformer_data/targets_human.txt`); `-13349446.out` (Aug 10) and `-13349441.out` (Aug 13) are 24-line headers with no result | `28f1cfb`, fix `9411535` | 1,990 eligible tracks + best-match CSV + Pearson matrix. Run that produced files: UNKNOWN (no log); `runtime_test.csv` says 1m08s |
| `results_baseline_alphagenome/` (2026-08-21) | `baseline_alphagenome.py` (hosted API) | none on disk; `runtime_test.csv` 2m48s, 359 gRPC calls | `ff88be2` | 1,617 TF-ChIP tracks, best-match + Pearson matrix. VERIFIED files; run log UNKNOWN |
| `results_benchmark/` (2026-08-21) | `benchmark_compare_all.py`, `plot_benchmark_comparison.py` | config says git `b5fcf93` (HEAD at run time, before `ff88be2`) | `ff88be2` | Four-model comparison on shared grid; headline table §6. VERIFIED |
| `results_comparison_borzoi/`, `results_comparison_enformer/` (2026-08-21) | `compare_transfer_borzoi.py`, `compare_transfer_enformer.py` | — | `28f1cfb`, outputs in `ff88be2` | test-split paired stats vs filtered baselines (native grid). Script defaults still say `results_full_2/` and `results_comparison/`; on-disk names differ. INFERRED renamed by hand |
| `results_comparison_R/` | `plot_benchmark_comparison.R` | — | `ff88be2` | 4-model figures + `summary_by_baseline_test.csv` |
| `reports/qing_deliverables/` (2026-10-11/12) | `reports/qing_deliverables/scripts/*.py` | ad hoc session (not Slurm) | untracked/ignored (`reports/` in `.gitignore`) | §3 |
| Pretrained-Borzoi rerun with per-tile stats | `scripts/rerun_borzoi_baseline.py` | 2026-10-12 | untracked | 11 GB `evaluation_evidence/baseline_borzoi_rerun/`; reproduces August matrix to 3e-8, argmax 183/183. VERIFIED (`summary.json` `reproduction`) |

Referenced but ABSENT on this working copy (also absent from every git branch): `results_full_2/`, `results_comparison/` (validation comparison), `results_comparison_test/`, `results_targets_filtered_test/` (`git log --all -- <dir>` returns nothing), `results_baseline_filtered_test/`, `results_smoke`. Consequence: the validation-split baseline numbers and the 149-target table in the draft have no data file here (see §6). The 149-subset test numbers were recomputed from `results_comparison_borzoi/transfer_vs_filtered_borzoi_test.csv` this session and match (§6).

Failed / aborted Slurm jobs (VERIFIED from `*.out`): `11270065` state_dict size mismatch (10→183 tracks, checkpoint from earlier run); `11288969` FileNotFoundError `results/eval_preds.npy`; `11459308`, `11460150` (Slurm "CANCELLED … SIGNAL Terminated", 3-track and 183-track attempts), `11464728` (8-epoch run cancelled mid-epoch 2 on 2026-06-03 after "Epoch 1/8 train=0.1996 val=0.2007"), `11556196` (TIME LIMIT in epoch 1); `11988360` (Jun 18: epoch 1 completed, then crash "Parent directory borzoi_ckpt_full does not exist"); `12087841` (Jun 22 DataLoader worker exit); `12115416` (pickle error); `12115849`/`12115998` (num_workers benchmarks, best =8 workers); `12115321` (3-track smoke run: val r 0.7557, test r 0.7770, 3 tracks only — not comparable); `12013167` (baseline failed: missing `borzoi_data/targets_human.csv`).

---

## 3. Qing's completed Borzoi evaluation (summary only; not redone)

Files: `reports/qing_deliverables/{Qing_Research_Summary,Scientific_Review,Final_Acceptance_Review,Assumptions,Reproducibility,Progress_Log,Qing_Ready_To_Send}.md`, `summary_numbers.json`, `summary_numbers_followup.json`, `Pearson_Comparison.csv`, `R2_Comparison.csv`, `evaluation_evidence/`, `scripts/`.

What was done (VERIFIED by reading `Qing_Research_Summary.md` §2–10, `Final_Acceptance_Review.md`, JSONs):
1. Recomputed from saved arrays Borzoi-TL per-track Pearson r, r², and prediction R² (test chr9+chr22, 359 tiles) on native 32 bp and shared 896×128 bp grids; recomputation matches historical CSVs (max |Δ| 2.3e-5 native; 2e-14 shared; `summary_numbers.json` `historical_cross_checks`).
2. Re-ran pretrained Borzoi inference (2026-10-12) saving per-tile sufficient statistics; reproduces the August Pearson matrix (max |Δr| 3e-8; same argmax 183/183). Added validation-selected (non-oracle) and same-TF track mappings and val-calibrated R².
3. Identified 24 contaminated LoVo tracks (GSE51142, GSM1208593–GSM1208794) present by GSM in `borzoi_data/targets_human.csv` and `enformer_data/targets_human.txt`; oracle picked the identical experiment for 22 (Borzoi) / 23 (Enformer).
4. Identified 51 zero-signal test tiles (28 chr9 + 23 chr22: 48 all-N assembly gaps, 3 sequence-present with no signal); showed they raise both models' scores by ≈0.02 and move paired differences ≤0.006.
5. Statistics: track-bootstrap CIs, TF-cluster bootstrap (77 TFs), tile-level paired bootstrap, Wilcoxon; stratified 183/159/24.
6. Two independent Claude Opus reviews (rounds 2026-10-11 and 10-12): arithmetic sound; interpretation of the first round corrected for leakage; no CRITICAL findings in round 2; 3 MAJOR wording items addressed (`Final_Acceptance_Review.md`).

Conclusions recorded: frozen-backbone + MLP head explains ≈46–48 % of per-track variance (R² 0.459 masked / 0.481 all tiles); Borzoi-TL beats the re-run pretrained Borzoi by Δr +0.018 to +0.021 (all 183) and +0.029 to +0.033 (159 uncontaminated); loses −0.06 on the 24 identical-experiment tracks; advantage is largest where the TF is absent from the baseline vocabulary and ≈ parity where an equivalent experiment exists; TF-clustered CIs lower bound is small (e.g. [+0.004, +0.028] vs pretrained Borzoi); cell-line-clustered CIs (10 clusters) include 0 for pretrained Borzoi and Enformer.
Recorded outstanding items (`Final_Acceptance_Review.md`, `Qing_Research_Summary.md` §9–10): AlphaGenome and Enformer not re-run (oracle, all-tile, August only); no supervised baseline on pretrained outputs (ridge over 1882 tracks / linear probe on trunk); no backbone unfreezing; no random seed; Enformer HF revision unrecorded; HCT-116 ENCODE matches not checkable offline (A19).

Note: Qing's own interpretation that the baseline "oracle" is optimistic and the 0.6104→0.6634 slide numbers are untraceable is corroborated here (§6, §7).

---

## 4. Existing model checkpoints and sources

| Artifact | Detail | Evidence | Label |
|---|---|---|---|
| Borzoi-TL head | `borzoi_ckpt_full/model_head_best.pt` (epoch 5 of 8, val weighted-Poisson NLL 0.18523); SHA-256 `16ba1d73…` (from Reproducibility.md, not recomputed here) | `borzoi_ckpt_full/best_val.json` `{"epoch": 5, "val_loss": 0.18523}`; `epoch.txt`=8; `history.json` (8 epochs, val min at index 4); log "Early stopping: no val improvement for 3 epochs … Loaded best checkpoint (epoch 5, val=0.1852)" | VERIFIED |
| Architecture | 4 frozen Borzoi replicates (`johahi/borzoi-replicate-0..3`, 743,570,796 params), forward-pre-hook on `human_head` taps 1920-ch trunk embedding, averaged across folds, center 4096 of ~6144 bins; head `Linear(1920,1024)→GELU→Dropout(0.2)→Linear(1024,183)→Softplus`, 2,154,679 trainable params (arithmetic checked: 1920·1024+1024+1024·183+183) | `borzoi_code/model.py:22-62`; log 12212537 "Trainable head params: 2,154,679" | VERIFIED |
| Training data | 183 CRC TF ChIP BigWigs (hg38), tiles chr≠{1,8,9,22}: 4,659 train / 750 val / 359 test; target = mean coverage over center 131,072 bp at 32 bp (4096 bins); batch 1; lr 1e-4; Adam wd 1e-4; ReduceLROnPlateau(0.5, patience 1); early stop patience 3; RC p=0.5; shift ±128 bp bin-snapped; Poisson NLL | `borzoi_code/genome_tiler.py`, `dataset.py`, `main.py` (args lines 62–90, train loop 261–412), `training_run.slurm` (command `python -u main.py --epochs 8 …`), log tile counts | VERIFIED |
| Loss weighting | Per-track inverse-mean weighting exists (`--balance-tracks`) but the final run did NOT use it: `training_run.slurm` passes no such flag; log 12212537 has 0 matches for "track-stats/Estimating per-track/cached track weights"; no `track_weights.npy` in `borzoi_ckpt_full/`; clean-copy milestone doc says "never used" (line ~540) | as stated | VERIFIED |
| Training time/seed | ~54 h per Qing summary; no seed set in `main.py` (grep for "seed" returns nothing) | `main.py` | VERIFIED (no seed) |
| Splits | val chr1, chr8; test chr9, chr22; non-overlapping 524,288 bp windows; scaffold/`alt`/`Un`/`random` and chromosomes <1 Mb skipped; chrX trained (chrY/chrM absent from FASTA) | `genome_tiler.py` | VERIFIED |
| Pretrained Borzoi weights | `hf_cache/hub/models--johahi--borzoi-replicate-{0..3}`, snapshots `521db103…`, `ad05f7a1…`, `871a31b5…`, `da3b9002…` (each `config.json` + `model.safetensors`); output vocabulary 7,611 human tracks (`borzoi_data/targets_human.csv` 7,613 lines incl. header) | directory listing; wc | VERIFIED. That these are the revisions used in June: INFERRED (single snapshot per repo) |
| Enformer | `hf_cache/hub/models--EleutherAI--enformer-official-rough`, two snapshots `8df78695…`, `affe5713…`; revision used unknown; `enformer-pytorch 0.8.11` used in Aug (`benchmark_config_test.json`; `.out` shows 0.8.12 was pip-installed on Aug 10); vocabulary `enformer_data/targets_human.txt` (5,313 tracks + header); 196,608 bp input, 128 bp bins | files named | VERIFIED; revision UNKNOWN |
| AlphaGenome | hosted API (`gdmscience.googleapis.com`), no version exposed; also `alphagenome_variant_effects/external/alphagenome_research` + `ag-local-venv` (local weights path, needs HF token); no weights downloaded here (no `cache/weights`, no `cache/hf_home`) | listings | VERIFIED |
| Training data/splits of the pretrained models | fold BEDs and test regions are not in the repo; Qing A05 states "in all likelihood genome-wide with region folds" | `Assumptions.md` A05 | UNKNOWN from repo (do not quote literature conventions without opening a source) |

---

## 5. Existing SNP analyses

**5.1 AlphaGenome S6_CCV pipeline (`alphagenome_variant_effects/`, untracked, ~26 GB)** — VERIFIED unless noted.
- Source of variants: `input/S6_CCV_ChrPosRefAlt_rsid.tsv`, header `chr:pos:ref:alt<TAB>rs`, 10,148 rows, all distinct SNVs (ref/alt length 1/1), 22 autosomes, 5,739 distinct rsIDs (3,320 rsIDs recur across rows; max 3 rows/rsID — i.e. rsID is not a unique key). Origin/biological meaning of "S6_CCV" (supplementary table? credible causal variants?) is **UNKNOWN**; no provenance file.
- Genome build: hg38 declared (`README.md`, `config.py`) and checked: `borzoi_data/hg38/hg38.ml.fa` chr1=248,956,422 bp (hg38; hg19 chr1 = 249,250,621 — INFERRED from general knowledge, not from a file); the AlphaGenome reference `input/reference/GRCh38.p13.genome.fa` also present.
- REF verification: pipeline validates REF vs hg38 before scoring (`scripts/hg38_reference.py`, README "REF / ALT handling"); `output/run_summary.json`: `ref_mismatches: 0`. **Independently re-checked this session**: all 10,148 REF alleles equal the base at 1-based `POS` in `hg38.ml.fa` (0 mismatches); reading as 0-based would give 7,757 mismatches, so 1-based convention is correct. ALT is not verifiable against the FASTA (by design). Strand: REF/ALT on reference strand, never reverse-complemented (README).
- Coordinates: README documents 1-based `Variant.position`, 0-based half-open `Interval`, 1 Mb window centred (`SEQUENCE_LENGTH_1MB`); 242 variants had windows shifted at chromosome ends (`interval_shifted` True in `variant_run_status.tsv`) — variant not centred for those.
- Scorers: `variant_scorers.get_recommended_scorers(HOMO_SAPIENS)` — 19 scorers (RNA-seq gene-mask LFC/active, CenterMask 501 bp for ATAC/DNase/ChIP-TF/CAGE/PRO-cap, 2001 bp for ChIP-histone, contact maps, splicing, polyadenylation); 371 merged RNA-seq tracks; ~42 k per-track rows/variant (`status` n_score_rows 12,006–122,460). GENCODE v46 gene annotation.
- Completion: `output/run_summary.json`: 10,148/10,148 SUCCESS, 0 failed, 0 not processed; cache has 10,148 `.json` + 10,148 `.parquet`; log: batch done in 365.3 min, 27.8 variants/min, last pass `{'SUCCESS': 10141, 'API_ERROR': 1}` then rerun/summarise gives 10,148 SUCCESS. Expression summary: 309,269 variant×gene rows, 5,316 genes; direction counts at |log2FC|≥0.1375: 117 increased, 130 decreased, 309,022 minimal.
- Outputs (not read beyond headers): `output/variant_expression_summary.tsv` (133 MB), `variant_all_scores.tsv` (205 MB), `variant_run_status.tsv`, `variant_all_scores_per_track/chr*.parquet`, `output/raw_predictions/chr1_109817260_A_G/` (one smoke variant only), `figures/` (3 PNGs for that variant). Transfer tarballs built 2026-09-16 (`*.tar*` in repo root).
- NOT done: the local-weights backend (`run_alphagenome_local_a100.slurm`, `scripts/compare_backends.py`) — no `output_local/` or `cache/scores_local/` exist and no `.hf_token`; no analysis of the scores (no CRC/colon-specific track selection, no ranking, no statistical analysis, no comparison with Borzoi-TL); effect size is mostly tiny (only 247 of 309,269 gene-level rows exceed 10 % predicted change).
- Quantile-score and "minimal change" threshold are reporting conveniences (README), not significance claims.

**5.2 Borzoi/Enformer variant scoring: does not exist.** `grep -w vcf|snp|snv|variant|variants|rsid` over repo-root `*.py/*.slurm/*.sh` and `borzoi_code/*.py` returns no matches; the only variant code is under `alphagenome_variant_effects/`. Borzoi-TL checkpoint has no variant-scoring wrapper. VERIFIED.

---

## 6. Verified numerical results

Test split = chr9+chr22, 359 tiles, 183 tracks, hg38. "shared128" = central 114,688 bp at 128 bp (896 bins); "native32" = center 131,072 bp at 32 bp (4096 bins).

| # | Quantity | Value | Metric type | Source (file) | Label |
|---|---|---|---|---|---|
| 1 | Borzoi-TL val mean over 183 tracks (native32) | 0.6434 | r | `borzoi_eval-12212537.out` "mean R=0.6434" | VERIFIED |
| 2 | Borzoi-TL test mean (native32) | 0.6634 (median 0.6904) | r | same log; `results_comparison_borzoi/summary_stats_test.csv`; `summary_numbers.json` `pearson_r_native32` | VERIFIED |
| 3 | Borzoi-TL test mean (shared128), all 359 tiles | 0.6820 (median 0.7083) | r | `results_benchmark/aggregate_metrics_test.csv`; `summary_numbers.json` | VERIFIED |
| 4 | Borzoi-TL test mean (shared128) | 0.4810 (0.480966; median 0.4948) | prediction R² | `summary_numbers.json` `prediction_R2_shared128` | VERIFIED |
| 5 | Borzoi-TL test mean (shared128) | 0.4866 | r² | same, `pearson_r_squared_shared128` | VERIFIED |
| 6 | Borzoi-TL native32 prediction R² | 0.4578 | R² | same | VERIFIED |
| 7 | Borzoi-TL, 51 zero-signal tiles masked (308 tiles) | r 0.6629, R² 0.4595 | r, R² | `summary_numbers_followup.json` | VERIFIED |
| 8 | Pretrained Borzoi oracle best-match (shared128), re-run | r 0.6640; r² 0.4597; masked r 0.647 | r (oracle, test-selected) | `results_benchmark/aggregate_metrics_test.csv`; Qing §7.4 | VERIFIED |
| 9 | Pretrained Borzoi validation-selected (non-oracle) | r 0.661 (masked 0.644); R²-valcal 0.444 (masked 0.428) | r; R²-valcal | `Qing_Research_Summary.md` §7.4 table, `evaluation_evidence/baseline_borzoi_rerun/summary.json` | VERIFIED (from reports; not recomputed) |
| 10 | Pretrained Borzoi raw-scale R² | mean −193 (172/183 negative) | R² (raw, inapplicable) | Qing §7.2 | VERIFIED (reported) |
| 11 | Enformer best-match (shared128), Aug oracle | r 0.6443; r² 0.4352 | r, r² | `aggregate_metrics_test.csv`; `summary_numbers.json` | VERIFIED; never re-run; oracle |
| 12 | AlphaGenome best-match (shared128), Aug oracle | r 0.6612; r² 0.4590 | r, r² | same | VERIFIED; never re-run; oracle |
| 13 | Filtered Borzoi best-match (native32) | mean r 0.6456, median 0.6658 | r (oracle) | `borzoi_filt_test-13005163.out`; `summary_stats_test.csv` | VERIFIED |
| 14 | TL − pretrained Borzoi (shared128, all 183) | Δr +0.0180 [0.0092, 0.0260]; Wilcoxon p 1.66e-10; d 0.309; W/L/T 121/33/29 | r | `results_benchmark/paired_stats_test.csv` | VERIFIED |
| 15 | TL − pretrained, native32 | Δr +0.01776 [0.00884, 0.02598]; paired t p 7.74e-5; Wilcoxon p 1.89e-10; d 0.299; 122/34/27 | r | `results_comparison_borzoi/paired_stats_test.csv` | VERIFIED |
| 16 | TL − Enformer / − AlphaGenome (shared128) | +0.0377 [0.0262,0.0482] / +0.0208 [0.0111,0.0295]; p 4.7e-15 / 1.9e-10 | r | `paired_stats_test.csv` | VERIFIED |
| 17 | Same, 159 uncontaminated tracks | Δr +0.0299 (vs Borzoi, oracle), +0.0199 (AlphaGenome), +0.0531 (Enformer); 24 contaminated: −0.0609 / +0.0269 / −0.0644 | r | Qing §7.1 / `evaluation_evidence/stratified_by_contamination.csv` | VERIFIED (reported) |
| 18 | TF-cluster bootstrap CI of Δr (77 TFs), all 183 | vs Borzoi [+0.004,+0.028]; vs AlphaGenome [+0.008,+0.038]; vs Enformer [+0.022,+0.051] | r | `evaluation_evidence/cluster_robust_stats.csv`; Qing §7.1 | VERIFIED (reported) |
| 19 | 149 non-losing targets (test, native32): TL mean r 0.6853, median 0.7250, win rate 81.9 % | recomputed by me from `results_comparison_borzoi/transfer_vs_filtered_borzoi_test.csv` (winner != "borzoi"): n=149, mean 0.68528, median 0.72500, 0.8188 | r | that CSV | VERIFIED (recomputed; selection conditioned on outcome) |
| 20 | Draft test table: 66.7 % wins (122/183), 34 losses, 27 ties; high-conf (r≥0.7) 47.0 % vs 38.3 % | match `summary_stats_test.csv`, `paired_stats_test.csv` | r | those CSVs | VERIFIED |
| 21 | Loss list in draft: REST Δ −0.31/−0.27, RFX2 −0.26, ZBTB7B −0.16 | CSV gives REST(LoVo) −0.3006, REST(HCT-116) −0.2659, RFX2(LoVo) −0.2569, ZBTB7B −0.1649 | r | same CSV | VERIFIED (rounding: −0.30 not −0.31) |
| 22 | Biological match of best Borzoi track | same TF 66, related 27, different 90 | — | same CSV `biological_match` | VERIFIED |

Untraceable / conflicting numbers are listed in the dedicated section below.

---

## 7. Unfinished experiments

| Item | Evidence | Label |
|---|---|---|
| No ablation of the recipe (augmentation / trunk tap / head / wd / LR schedule / early stopping) — all bundled in `e57d537` | clean-copy `DEVELOPMENT_MILESTONES.md` lines ~218, 544; `Assumptions.md` A11; no log has a no-augmentation 183-track run (`--no-rc`, `--max-shift-bp 0` never passed) | VERIFIED |
| `--balance-tracks` implemented, never run | §4 | VERIFIED |
| Backbone unfreezing/partial fine-tuning never attempted | `Qing_Research_Summary.md` §9; milestones doc | VERIFIED |
| Supervised baseline on pretrained outputs (ridge on 1,882 tracks / linear probe on frozen trunk) never run | Qing §9–10 | VERIFIED |
| Enformer and AlphaGenome baselines never re-run with per-tile stats/non-oracle mapping; Enformer needs ~1 min GPU | Qing §9; `Final_Acceptance_Review.md` | VERIFIED |
| Validation-split baseline for pretrained Borzoi existed only in the draft (0.625); per-tile val statistics now exist from the 2026-10-12 rerun (`per_tile_stats_val_*.npz`) but no val comparison table was produced against the draft's val numbers | `evaluation_evidence/baseline_borzoi_rerun/` listing | VERIFIED |
| AlphaGenome local-weights backend never run (no `output_local`, no HF token) | §5.1 | VERIFIED |
| No scoring of variants by Borzoi-TL / Enformer; no analysis of AlphaGenome S6_CCV outputs | §5 | VERIFIED |
| Cluster-aware significance test flagged as "future work" in the draft (done later by Qing) | `research_paper_draft.md` §6–7; `cluster_robust_stats.csv` | VERIFIED |
| `compare_transfer_*.py` defaults point to `results_full_2/`, `results_comparison/` (do not match on-disk dirs); `pi_update_presentation.py` expects `results_comparison_test/`, `results_baseline_filtered_test/`, `run_test_analysis.sh` (not in repo) — scripts are not re-runnable as committed | `compare_transfer_borzoi.py:45-47`; `pi_update_presentation.py:106-110,198-200,378` | VERIFIED |
| Draft says the Enformer end-to-end comparison "is not yet finalized"; it now exists in `results_comparison_enformer/` and `results_benchmark/` | `research_paper_draft.md` §7 vs files | VERIFIED (draft stale) |
| TODO/TBD markers | none found in `manuscript.md` (grep for TODO/TBD returned nothing) | VERIFIED |
| `borzoi_ckpt_full/optimizer.pt` + `model_head.pt` (epoch 8) not used; resume state only | Reproducibility.md | VERIFIED |

---

## 8. Scientific limitations (evidence-based)

1. **Training-set leakage for baselines:** 24 of 183 targets (all LoVo, GSE51142) are training experiments of pretrained Borzoi and Enformer (and therefore of the frozen backbone used by Borzoi-TL). `Assumptions.md` A18; `contamination_flags.csv`. An earlier, description-based stratification in `results_benchmark/stratified_by_assay_overlap_test.csv` flags 42 "same_experiment_in_borzoi" tracks (TL 0.7001 vs 0.7241, Δ −0.024), not 24 — a different definition (assay description match, not GSM); the two numbers must not be conflated.
2. **Sequence overlap:** the frozen backbone was pretrained on hg38 genome-wide; chr9/chr22 sequence is likely not novel to it (only the head is strictly held out). Fold definitions are not in the repo (UNKNOWN; do not assert Borzoi/Enformer/AlphaGenome held-out chromosomes without citing a source actually opened). AlphaGenome training overlap not checkable (A22).
3. **Oracle selection:** baselines' best-match track chosen with test targets; upper bound on off-the-shelf performance. For pretrained Borzoi, validation-selected mapping shows small optimism (0.664 → 0.661). Not done for Enformer/AlphaGenome.
4. **Calibration:** baseline raw R² is meaningless (−193); only R²-valcal (asymmetric — favors baseline slightly) or r². Borzoi-TL R² ≈ r² − 0.0056 (raw output on the observed scale).
5. **Single split / single seed / single run:** one fixed chromosome split (val chr1,8; test chr9,22), no seed set, no replicate runs; no CV across chromosomes. Checkpoint chosen on val NLL only (val curve is very flat: 0.1852–0.1875 over 8 epochs).
6. **Development-time test exposure:** August slides compared configurations on the test split (0.6104 → 0.6634), so test results were visible during design (`Assumptions.md` A20); source of those numbers untraceable.
7. **Non-independence:** 183 tracks = 77 TFs × 10 cell lines (HCT-116 90, LoVo 41, Caco-2 7, GP5d 6, LS180 5, LS174T 5, SW480 4, RKO 3, DLD-1 1, HT29 1); track-level p ≤1e-10 is anti-conservative; TF-cluster CIs are far wider; cell-line clustering inconclusive.
8. **Cell-type mismatch:** pretrained best-match tracks often come from other cell types/TFs (only 66/183 same TF; 90 different TF; Borzoi vocabulary has target TF in target cell line for only 48/183 — Qing §6).
9. **Zero-signal tiles:** 51/359 test tiles (assembly gaps) inflate r/R² by ≈0.02 for TL and pretrained Borzoi; cannot be removed for Enformer/AlphaGenome (only whole-split matrices saved). `results_benchmark/README.md` says these "slightly deflate" scores — contradicted by Qing's analysis (they inflate).
10. **Pooled Pearson:** r computed over flattened tile×bin pairs pools between-tile mean differences; no per-tile / peak-level / Spearman metrics for baselines.
11. **Resolution/context mismatch:** Enformer sees 196,608 bp (vs 524,288), 128 bp native bins; AlphaGenome hosted, unversioned; shared grid is 128 bp for all.
12. **Vocabulary size asymmetry:** 1,882 / 1,990 / 1,617 candidate tracks give oracle selection different multiple-comparison advantage.
13. **Shift augmentation quantised to 32 bp** (not single-bp). Draft's own caveat.
14. **Genome build:** hg38 for FASTA, BigWigs (header chr lengths match `hg38.ml.fa.fai` for chr1 and chr22 in all 183 files (chr9 checked on one file) — checked this session), Borzoi/Enformer targets and AlphaGenome inputs. No liftover anywhere (none found). VERIFIED; the build of the *original* BigWig processing (Cistrome/ChIP-Atlas pipeline) is not documented — INFERRED hg38 only from chromosome lengths.
15. **Single dataset/assay:** TF ChIP only; no RNA/ATAC/DNase evaluation; no evidence the TL tracks transfer to variant scoring.
16. **Draft/manuscript flaws:** see §9 (claims about the recipe components not matching the run).
17. **Selection bias in "149 kept targets" analysis:** subset defined by the model not losing; reporting its higher mean is outcome-conditioned.

---

## 9. Manuscript progress

`research_paper_draft.md` (119 lines, commit `28f1cfb`/`36838d0`): Title, Abstract, §1 Intro, §2 Related work, §3 Data and Model, §4 Experimental design, §5 Results (5.1 ablation, 5.2 external baseline, 5.3 residual gap), §6 Statistical caveats, §7 Limitations, §8 Conclusion, Data/Code availability, 3 references.
`research_paper_prep/manuscript.md` (160 lines; PLOS-Genetics-structured: Abstract, Author summary, Intro, Results ×4, Discussion, Limitations, Materials and methods ×6, Data/code availability, References ×4), `manuscript.tex` (same content), 11 PNG figures, template XML. Last modified 2026-08-10; predates AlphaGenome/Enformer benchmark, the leakage finding, Qing's R² work.

| Claim in draft | Traceable? | Evidence / issue |
|---|---|---|
| Test mean r 0.663 vs filtered Borzoi 0.646, Δ +0.018, 122/34/27, paired t 7.7e-5, Wilcoxon 1.9e-10, d 0.30, CI [0.0088,0.0260] | YES | `results_comparison_borzoi/*_test.csv` (row 15, 20) |
| Median 0.690 vs 0.666; ≥0.7 share 47.0 % vs 38.3 % | YES | `summary_stats_test.csv` |
| 149-target subset: mean 0.685, median 0.725, 81.9 % | YES (recomputed) | row 19; source dir `results_targets_filtered_test/` named in draft does not exist |
| Residual-gap factors (REST, RFX2, ZBTB7B, NFYA, ZNF143…) | YES (largest losers) | transfer_vs_filtered CSV |
| Validation: Borzoi 0.625, TL 0.643, median 0.649/0.679, 33 vs 119 wins, 34.4 % vs 45.9 % ≥0.7 | TL val 0.643 YES (log); **Borzoi val numbers NO** | `results_comparison/` (val) and `results_baseline_filtered/*_val*` absent; only PNGs `research_paper_prep/figures/*_val_*.png` remain |
| Internal ablation: no-augmentation 0.502 val / 0.610 test vs 0.643 / 0.663; "+28 % / +9 %" | **NO** | no log/CSV contains 0.5022/0.6104 (grep over `*.out`, CSVs; Qing A11); no matching run exists; can't attribute gain to augmentation, trunk tap or recipe (milestones doc). The 0.643/0.663 half is traceable |
| "inverse-signal per-track loss weighting" part of the recipe (draft §3, Abstract; manuscript Abstract & methods) | **NO — contradicted** | final run did not use `--balance-tracks` (§4) |
| "183 tracks across three CRC cell lines (Caco-2, HCT-116, LoVo)" (draft §3; manuscript methods) | **NO — contradicted** | `TF_184tracks.csv`: 10 cell lines among the 183 BigWigs |
| "Backbone: four folds… frozen", "MLP head", "Softplus", augmentation specs | YES | `model.py`, `dataset.py` |
| "Splits: validation chr1,chr8; test chr9,chr22" | YES | `genome_tiler.py` |
| "Baseline is deliberately generous… so reported gain is a conservative estimate" | PARTLY | true that oracle is optimistic (validation-selected 0.661 vs 0.664), but the 24 LoVo tracks are baseline training experiments (bias in baseline's favor on those) and the baseline is not tested for baseline-trained-on-same-supervision; "conservative" needs the supervised-baseline control (§7) |
| "Enformer comparison not finalized" | STALE | results exist (§2) |
| Contamination of 24 tracks, R² metrics, zero-signal tile effect, TF-cluster inference | **ABSENT from draft** | exist only in `reports/qing_deliverables/` |
| References: Linder (Borzoi) given without journal/year/DOI in draft; EvoAug (Genome Biol 24:105, doi 10.1186/s13059-023-02941-w); Enformer (Nat Methods 2021;18(10):1196–1203, doi 10.1038/s41592-021-01252-x) | UNVERIFIED — I did not open any external source; the PLOS Genetics file in the repo was verified only as a local file | do not rely on them until checked |
| Draft: Cohen's d 0.30; build_presentation.py slide: d 0.28, Wilcoxon 1.1e-9, paired t 0.00017 (val) | val-split stats untraceable (no val comparison dir) |  |

Presentation numbers (`build_presentation.py:420-421, 495-496, 517`): val ablation 0.5022→0.6434 (+0.1412), test 0.6104→0.6634 (+0.0530) — baseline halves untraceable; val baseline R 0.625, wins 119/183 (65 %), 46 % vs 34 % ≥0.7 — untraceable data file; Wilcoxon 1.1e-9/d 0.28 (val) — no source.

Overall: the draft is a complete skeleton with traceable *test-split* TL-vs-filtered-Borzoi results, but its headline "augmentation ablation" and "inverse-signal weighting" claims are unsupported/contradicted, it omits leakage, and it predates the four-model benchmark; variant-effect work is not started.

---

## 10. Next research priorities (what the evidence suggests remains)

1. **Fix the claim base before any new writing:** remove/relabel the unsupported ablation (0.502/0.610) and the inverse-weighting statement; correct "three cell lines" to ten; add contamination stratification (159/24) and metric definitions. (Evidence: §9.)
2. **Supervised baseline on pretrained outputs** (ridge over 1,882 tracks and/or linear probe on frozen trunk, same train chromosomes) — the only way to attribute the +0.02–0.03 Δr to CRC-specific adaptation (`Qing_Research_Summary.md` §9). Optionally a genuine single-factor ablation (augmentation off) now that the pipeline is known.
3. **Re-run Enformer (≈1 min GPU) and AlphaGenome (API) with per-tile stats, validation-selected mapping, masked tiles** so the same non-oracle/Analysis-B comparisons exist for all models (outstanding in `Final_Acceptance_Review.md`).
4. **Variant-effect analysis for CRC:** AlphaGenome S6_CCV scores exist for 10,148 SNVs (0 REF mismatches, all SUCCESS) but have no downstream analysis and unknown provenance of the variant set; Borzoi-TL / Enformer variant scoring does not exist. Required for the stated goal in `MASTER_PLAN.md` and flagged by the draft as future work. Needs: confirm variant-set origin and CRC relevance, select CRC-relevant tracks (colon/CRC cell lines, TF/histone), handle recurring rsIDs, report the 242 non-centred windows, add a Borzoi-TL scorer with REF/ALT inserted into the FASTA window.
5. **Uncertainty/robustness of Borzoi-TL itself:** multiple seeds/chromosome folds (single split, no seed); TF- and tile-bootstrap already exist for the comparison.
6. **Optional:** partial backbone unfreezing evaluated on the 159 uncontaminated tracks; per-tile/peak-level and rank metrics; resolve HCT-116 ENCODE overlap (A19) via accession lookup.

---

## Provenance (datasets, tracks, builds, splits)

- **Targets:** 183 BigWigs `borzoi_data/CRC_TFs_bw/<DCid>.bw` (order in `borzoi_ckpt_full/bw_files.json`); metadata `borzoi_data/TF_184tracks.csv` (184 rows; header `DCid, Species, GSMID, Factor, Cell_line, Cell_type, Tissue_type, FastQC, UniquelyMappedRatio, PBC, PeaksFoldChangeAbove10, FRiP, PeaksUnionDHSRatio`). Row DCid 42890 (GSM1239630, ZNF83, LoVo) has no BigWig (VERIFIED; no BigWig is absent from the CSV). Species Homo sapiens only. 77 distinct factors; cell lines HCT-116 90, LoVo 41, Caco-2 7, GP5d 6, LS180 5, LS174T 5, SW480 4, RKO 3, DLD-1 1, HT29 1. Source database/pipeline (the DCid + QC-metric columns look like Cistrome DB; **INFERRED**, not stated in repo). Accessions are GEO GSM ids (e.g. GSM575225 CDX2 Caco-2); LoVo GSE51142 series per Qing.
- **Per-file BigWig header:** version 4, chromosomes match hg38 (24 contigs listed); no per-file normalisation recorded (targets used raw: "no normalisation, no log transform").
- **Reference genome:** `borzoi_data/hg38/hg38.ml.fa` (3.09 GB, `.fai`; chr1 248,956,422; chr9 138,394,717; chr22 50,818,468 = hg38). hg38 end-to-end for training/eval, baselines, AlphaGenome variant pipeline (README/config + REF check). No hg19 and no liftover found.
- **Pretrained model track vocabularies:** Borzoi `borzoi_data/targets_human.csv` (7,611 tracks; eligible ChIP non-histone idx 2186–6069 → 1,882, `filter_borzoi_tracks.py`); Enformer `enformer_data/targets_human.txt` (5,313 tracks; 1,990 eligible); AlphaGenome 1,617 TF-ChIP tracks (`results_baseline_alphagenome/alphagenome_tracks_test.csv`).
- **Splits:** train = all chromosomes ≥1 Mb except chr1, chr8, chr9, chr22 → 4,659 tiles; val chr1+chr8 → 750; test chr9+chr22 → 359; non-overlapping 524,288 bp; whole-chromosome holdout (`genome_tiler.py`; log 12212537).
- **Software:** borzoi-pytorch 0.4.4, enformer-pytorch 0.8.11, torch 2.11.0+computecanada, python 3.12.4 (`benchmark_config_test.json`); AlphaGenome client 0.8.0 (README).

## Untraceable / conflicting numbers

1. 0.6104 (test) and 0.5022 (val) "no augmentation" ablation: appear in `research_paper_draft.md` §5.1, `research_paper_prep/manuscript.md:52-53`, `build_presentation.py:420-421`; no log/CSV/script produces them (Qing A11; my grep of all `*.out` agrees). Do not quote. The derived "+28 % / +9 %" and the "val/test consistency" argument are likewise unsupported.
2. Validation-split filtered-Borzoi baseline (mean r 0.625, median 0.649, 33/119/31 wins, 34.4 % ≥0.7), val paired stats (p 0.00017, Wilcoxon 1.1e-9, d 0.28): data dir `results_comparison/` absent; baseline val matrix never saved in `results_baseline_filtered/`. Only PNGs remain. The draft itself gives no val Wilcoxon p (slide has 1.1e-9).
3. Draft statement that the transfer loss uses inverse-signal weighting: contradicted by run config (flag never passed).
4. "Three CRC cell lines": contradicted (10 cell lines).
5. Same-experiment overlap count: 42 (`results_benchmark/stratified_by_assay_overlap_test.csv`, description-based) vs 24 (Qing, GSM-identical). Use 24 for "training experiments"; 42 is a looser assay match.
6. `results_benchmark/README.md` (limitation 5) says zero-signal tiles "slightly deflate" scores and are "the all-N chr22 p-arm"; Qing shows they inflate and include 28 chr9 + 23 chr22 tiles (48 all-N + 3 no-signal).
7. Two cleaned-up result directory names used in the scripts (`results_full_2`, `results_comparison`, `results_comparison_test`, `results_targets_filtered_test`) vs on-disk (`results_full`, `results_comparison_borzoi`): mapping INFERRED, not recorded.
8. Draft REST Δr "≈ −0.31": file gives −0.3006 (LoVo) and −0.2659 (HCT-116); "−0.27" matches HCT-116 only when rounded; "−0.31" is not in the CSV (minor).
9. Native-grid vs shared-grid means (0.6634 vs 0.6820) are different quantities; "0.663" appears in the draft (native), "0.682" in Qing/benchmark (shared). Do not mix.
10. Draft p-value 7.7e-5 is a paired t on native grid; benchmark's shared-grid paired t is 4.6e-5.

## Open questions

1. What is the S6_CCV variant set (publication/table, disease context, why 3,320 rsIDs recur across variants) and is it CRC-relevant?
2. What were the exact held-out chromosomes/regions of pretrained Borzoi, Enformer, AlphaGenome, and does the test set (chr9, chr22) overlap their training intervals? (fold files not in repo; literature must be opened and cited.)
3. Which Enformer HF snapshot (`8df78695…` vs `affe5713…`) produced `results_baseline_enformer/`?
4. Which Slurm/interactive runs produced the Aug 21 Enformer-grid Borzoi, Enformer and AlphaGenome outputs (no logs on disk)?
5. Is the 10-HCT-116 ENCODE CTCF/ZFX match an identical experiment to the CRC track (A19)?
6. Where did the 0.5022/0.6104 values come from (a run on another machine/branch, e.g. `~/stable_research` or `~/Research-work-test` referenced in logs, now missing)? Logs reference `/home/yejz1/Research-work-test` and `/home/yejz1/stable_research`; neither directory exists now.
7. Original source pipeline/genome build of the 183 BigWigs (Cistrome DB/ChIP-Atlas?) — only chromosome lengths confirm hg38.
8. Is the validation-split comparison (draft 0.625) reproducible from `per_tile_stats_val_*.npz` and what are the validation-selected values? (Not computed.)
9. Whether the colon/CRC-specific AlphaGenome tracks (RNA-seq 371 merged tracks, ontology CURIEs) cover CRC cell lines adequately for the intended SNP analysis (not checked).
10. Why 242 variants have shifted windows and whether any lie within CRC-relevant loci.

---
## Errata from RV-002 scientific-reviewer gate (2026-10-19)
These corrections supersede the inventory text above where they conflict.
- **§3 / §8.1 contamination:** ≥28 tracks, not 24. Tracks 145–148 (DCid 62707/62737/64170/64457 = ENCSR102CSD/ENCSR107EUS/
  ENCSR667UWT/ENCSR769WKR; CTCF/POLR2A, colon) each appear in `borzoi_data/targets_human.csv` and
  `enformer_data/targets_human.txt` (coordinator grep-confirmed). `contamination_flags.csv` marks them False (GSM-only check).
  The oracle picked the identical experiment for targets 145 and 147. The count is a lower bound (A19 open).
- **§8 item 7 cell lines:** the listed counts sum to 163. 20 tracks have `Cell_line=None` (colon tissue/unannotated).
  Whether Qing's 10-cluster cell-line bootstrap dropped or merged them is unchecked.
- **Row 4:** prediction R² = 0.4810 (0.480966).
- **SNP input:** 4,409 surplus rows; the tested allele is unidentifiable at 3,320 positions / 7,729 rows. ALTs are alphabetical.
  The pattern suggests a dbSNP multi-allelic expansion. The GRCh37 origin is INFERRED.
- **Shifted windows:** 242 rows = 147 positions (chr19 133, chr21 93, chr17 16 rows).
- **chr9/chr22 backbone exposure:** UNVERIFIED is correct. No fold BEDs exist in the repo.
