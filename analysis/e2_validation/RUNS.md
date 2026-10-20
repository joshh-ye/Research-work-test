# analysis/e2_validation RUNS (2026-10-20; cwd /home/yejz1/experimental_research). No score-label join has been executed.

| # | command | output | notes |
|---|---|---|---|
| 1 | `./ag-venv/bin/python analysis/e2_validation/labels.py convert-gtex` | `results_phase2/e2_validation/labels/cache/gtex_v10_*.ccv.tsv.gz` | parquet->TSV, matched positions only |
| 2 | `./torch-venv/bin/python analysis/e2_validation/labels.py` | `results_phase2/e2_validation/labels/{variant_labels.tsv,label_counts.tsv,eqtl_pairs_aligned.tsv.gz,st4_resolution.tsv,power_statement.tsv,POWER.md}` | label-only; counts printed |
| 3 | `./ag-venv/bin/python analysis/e2_validation/scores.py build-ag-cache` | `results_phase2/e2_validation/scores/alphagenome_{scores.tsv,rna_colon.tsv.gz}` | scores only; asserts 33 tracks/variant; 242 shifted windows |
| 4 | `./ag-venv/bin/python analysis/e2_validation/stats.py build-tss` | `results_phase2/e2_validation/tss_protein_coding_gencode_v46_hg38.tsv` | 78,776 TSS; GAPDH 12:6534517 checked |
| 5 | `./torch-venv/bin/python -m pytest analysis/e2_validation/tests -q` | stdout | synthetic data only |
| 6 | `./torch-venv/bin/python analysis/e2_validation/run_e2.py` | refuses (no --unblind) | verified; real run NOT executed |
| 7 | `./torch-venv/bin/python -m pytest analysis/e2_validation/tests -q` (after Amendment A3) | stdout | 29 passed; `--models` option, finalised-only loading, interim labelling added; real run NOT executed |
