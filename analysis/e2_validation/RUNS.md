# analysis/e2_validation RUNS (2026-10-20; cwd /home/yejz1/experimental_research). Rows 1-7 are label-only / score-only / synthetic. Score-label joins (unblinded looks) are rows 8-10 and post-hoc row 11.

| # | command | output | notes |
|---|---|---|---|
| 1 | `./ag-venv/bin/python analysis/e2_validation/labels.py convert-gtex` | `results_phase2/e2_validation/labels/cache/gtex_v10_*.ccv.tsv.gz` | parquet->TSV, matched positions only |
| 2 | `./torch-venv/bin/python analysis/e2_validation/labels.py` | `results_phase2/e2_validation/labels/{variant_labels.tsv,label_counts.tsv,eqtl_pairs_aligned.tsv.gz,st4_resolution.tsv,power_statement.tsv,POWER.md}` | label-only; counts printed |
| 3 | `./ag-venv/bin/python analysis/e2_validation/scores.py build-ag-cache` | `results_phase2/e2_validation/scores/alphagenome_{scores.tsv,rna_colon.tsv.gz}` | scores only; asserts 33 tracks/variant; 242 shifted windows |
| 4 | `./ag-venv/bin/python analysis/e2_validation/stats.py build-tss` | `results_phase2/e2_validation/tss_protein_coding_gencode_v46_hg38.tsv` | 78,776 TSS; GAPDH 12:6534517 checked |
| 5 | `./torch-venv/bin/python -m pytest analysis/e2_validation/tests -q` | stdout | synthetic data only |
| 6 | `./torch-venv/bin/python analysis/e2_validation/run_e2.py` | refuses (no --unblind) | verified; real run NOT executed |
| 7 | `./torch-venv/bin/python -m pytest analysis/e2_validation/tests -q` (after Amendment A3) | stdout | 29 passed; `--models` option, finalised-only loading, interim labelling added; real run NOT executed |
| 8 | `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --models alphagenome_quantile --out results_phase2/e2_validation/interim_alphagenome` | `results_phase2/e2_validation/interim_alphagenome/` | UNBLIND look #1, 2026-10-20 01:04 (EXECUTION_LOG; RV-005 permitted). Interim, single-model, unadjusted p (Holm deferred): P1 AUROC 0.666 [0.590, 0.759]; P2 0.311 (n=132) |
| 9 | `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --models enformer --out results_phase2/e2_validation/interim_enformer` | `results_phase2/e2_validation/interim_enformer/` | UNBLIND look #2, 2026-10-20 02:36 (EXECUTION_LOG; frozen Enformer run finalised). Interim, unadjusted: P1 0.674 [0.607, 0.756]; P2 0.364 (n=132) |
| 10 | `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --out results_phase2/e2_validation/final` | `results_phase2/e2_validation/final/` (log `final_run.log`; EXP-017) | FINAL unblind, all 4 headline models (Borzoi pretrained and Borzoi-TL unblinded for the first time), 2026-10-20 10:55 CDT (outputs written 10:56). Holm, paired deltas and framing rule computed only here -> Framing B (SD-024). Pre-registered record; NOT to be modified |
| 11 | `./ag-venv/bin/python analysis/e2_validation/post_final_exploratory.py prep-ag` then `./torch-venv/bin/python analysis/e2_validation/post_final_exploratory.py run` | `results_phase2/e2_validation/post_final_exploratory/` | POST-HOC EXPLORATORY (RV-007 minor findings 3, 5, 7): clogit region-bootstrap CI, region-weight concentration / equal-weight / leave-region-out AUROC, eQTL direction sanity. Re-reads the same scores and labels; not pre-registered; does not modify `final/` |

Total unblinded pre-registered looks: 3 (rows 8-10). Row 11 is post hoc and labelled as such.
