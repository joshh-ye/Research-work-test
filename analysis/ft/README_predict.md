# predict_ft.py / blend_val.py / compare_ft.py (hg38; val = chr1+chr8, test = chr9+chr22)

All commands from `/home/yejz1/experimental_research`, python = `./torch-venv/bin/python`. GPU memory of predict_ft is capped at 20 GB (`--max-gb`).
Val prediction of one run: ~140 s (750 tiles, bf16, shared A100). Ridge val-prediction regeneration (once per cache model): E4 12 s, FT-C61 ~260 s (peak ~10 GB).

## 1. Validation predictions (per run; re-runnable, skips existing files unless `--force`)
    ./torch-venv/bin/python analysis/ft/predict_ft.py --runs FT-S02,FT-S03,FT-S04 --split val
    ./torch-venv/bin/python analysis/ft/predict_ft.py --run FT-S05 --split val --tta rc      # RC-averaged TTA, written to *_rc files (select TTA on val only)
Outputs in `results_ft/preds/<RUN>/`: `val_pred[_rc].npy` (fp16, 750 x 896 x 183), `val_gid.npy`, `val_stats[_rc].npz` (per-tile n,sx,sy,sxy,sx2,sy2,sse; gid),
`val_repro[_rc].json`. The run reconstructs the model from `run_cmd.json` (incl. `--replicate`), checks the best_model.pt trainable keys key-for-key, and
asserts the mean val r reproduces the logged best full-val r (`meta.val_mean_pearson_r_full`) within 1e-3 (exit code 3 otherwise; not applied to --tta) and that
loader targets equal the pooled-128 targets (max abs diff < 1e-4; observed 1.4e-6). Transform = train_ft.evaluate (forward strand, no shift/RC, bf16, crop 256 bins/side, pool x4).

## 2. Blends / ensembles (validation only)
    ./torch-venv/bin/python analysis/ft/blend_val.py --cands "FT-S05;FT-S05+E4;FT-S05+FT-C61;ENS:FT-S02,FT-S05+FT-C61"
`A+B`: w*A + (1-w)*B, w = weight on the FT/ensemble prediction (w=1 -> FT alone, w=0 -> ridge alone), w in {0,0.1,..,1}. `ENS:a,b` = mean of val predictions.
E4 = FT-C02 config (ridge[pre_raw:union1888]); ridge val predictions are refit on train tiles from `/tmp/yejz1_ft/pooled128` (node-local; rebuild with analysis/ft/build_pooled.py
if missing) and cached in `results_ft/preds/{E4_ridge,FT-C61}/`; they must reproduce the logged val r (0.7337 / 0.7392, assertion, 1e-3). Outputs:
`results_ft/blends/blend_val_results.csv` (one row per candidate, upserted; `w_ft_selected_on_val` is VALIDATION-selected = optimistic;
`crossfit_r_*` = honest 2-fold estimate: choose w on one chromosome, score on the other, mean of the two chromosome scores; compare it with
`chromwise_mean_r_all183_*` of the components, NOT with the pooled r) and `blend_val_grid.csv` (all w). Registry ids FT-B01.. are appended by hand (atomic flock append, columns as FT_REGISTRY header).

## 3. Locked test evaluation (coordinator only; never run during development)
    # 1) add the config id to research_management/finetuning/TEST_LOCK.csv (selection rule + val numbers)
    ./torch-venv/bin/python analysis/ft/predict_ft.py --run FT-S05 --split test                                    # config id defaults to the run id
    ./torch-venv/bin/python analysis/ft/predict_ft.py --runs FT-S05,FT-S06 --split test --config-id ENS:FT-S05+FT-S06   # ensemble = mean of run predictions
Refuses (before loading anything) unless the config id is a TEST_LOCK.csv row and `results_ft/ft_test_eval/<CONFIG>/test_stats.npz` does not exist
(':' is replaced by '_' in the directory name). Blends with ridge are NOT supported by predict_ft test (the test path is pure FT). Writes `test_stats.npz`
(same keys/order as test_eval_cache; test tiles in pooled order), `run_info.json`, then runs `compare_ft.py` (subprocess, CPU): Pearson r / r^2 / prediction R^2,
paired deltas vs E4 ridge[pre_raw:union1888] and vs Borzoi-TL, tile-block / TF-cluster / cell-line-cluster / two-way CIs, 10,000 resamples, seed 20261020,
all183 and not_identical152 -> `test_comparison.csv`. `--tta rc` is only legitimate if the TTA choice was fixed on validation first.
`compare_ft.py` was moved verbatim out of test_eval_cache.compare(); rerun on E4-ridge-repro stats gives a bit-identical test_comparison.csv.

## Tests
    ./torch-venv/bin/python -m pytest analysis/ft/tests/test_predict_ft.py -q

## 4. Locked fixed-w blend on test (coordinator only)
    # lock row id (w is part of the id and never chosen on test):  BLEND:<FTpart>|<RIDGEcfg>|w=0.4     FTpart = run id or ENS:A+B; RIDGEcfg = E4 | FT-C61 | any configs/<id>.json ridge
    ./torch-venv/bin/python analysis/ft/predict_ft.py --split test --config-id 'BLEND:FT-S05|FT-C61|w=0.4'
Refuses unless that exact id is in TEST_LOCK.csv and `results_ft/ft_test_eval/<id>/test_stats.npz` is absent. Blend = w*FT + (1-w)*ridge (w = FT weight). FT test predictions (predict_ft path) and ridge test
predictions (refit on train tiles, as test_eval_cache) are cached under `results_ft/ft_test_eval/_components/` only as a by-product of a locked evaluation and reused by other locked blends.
