# Borzoi fine-tuning (FT) program: comparison tables

Genome build hg38. Validation = chr1 + chr8 (750 tiles); test = chr9 + chr22 (359 tiles); train 4,659 tiles (`results_phase2/trunk_cache/tiles.csv`). Metric = mean over tracks of per-track Pearson r pooled over all bins of the
split, on the shared 896 x 128 bp grid; "183" = all tracks, "152" = non-identical tracks (31 Borzoi/Enformer-identical experiments excluded; `exposure_audit/track_exposure.tsv`).
Pearson r, squared Pearson r (r^2) and prediction R^2 (= 1 - SS_res/SS_tot) are separate quantities and are labelled in every column. Narrative and caveats: `reports/FINETUNING_FINAL_REPORT.md`.
All tables are generated from files listed under each table; the test tables come from `results_ft/final/` (regenerate with `./torch-venv/bin/python analysis/ft/final_summary.py`).

## (a) Single arms on VALIDATION (replicate-specific trunks; one seed per run)
Source: `results_ft/final/final_run_table.csv` (built from `results_ft/runs/<run>/{stdout.log,metrics.csv,run_cmd.json}` and `research_management/finetuning/FT_REGISTRY.csv`: peak GPU GB = registry `gpu_mem_gb`, runtime = registry `runtime_s`);
RC-TTA column from `results_ft/blends/round{2,3,4}/blend_val_results.csv`. val r = best full-validation mean per-track Pearson r (early stopping on validation; the best epoch was the last one in every run listed).
s/step = median of `metrics.csv` `sec_per_step` (batch of 1 tile = 524,288 bp, bf16 autocast; GPU shared with other jobs, so +-10-20% timing noise). min/epoch = mean interval between consecutive `EPOCH` log lines (training epoch + ~1-2 min validation; for the 4-epoch runs it is the mean over epochs 2-4). All runs completed in a single attempt (`attempts.json` n = 1), so runtime = wall-clock.
All arms use the Poisson loss unless stated, MLP head (1920-1024-183, 2.2 M params; 2.15 M counted as trainable in frozen mode), cosine schedule, lr-head 1e-3, true shift + RC augmentation, seed 1 unless stated.

| run | arm / change | rep | ep | trainable params | peak GPU GB (registry) | median s/step | min/epoch (incl. val) | runtime h | val r183 | val r152 | val r183 / r152 with RC TTA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FT-S01 | frozen trunk + MLP head, 4 ep (matched control) | 0 | 4 | 2,154,679 | 2.9 | 0.137 | 12.7 | 0.87 | 0.7280 | 0.7282 | - |
| FT-S07 | frozen, 12 ep (control for longer training) | 0 | 12 | 2,154,679 | 2.9 | 0.081 | 8.0 | 1.71 | 0.7288 | 0.7291 | 0.7329 / 0.7329 |
| FT-S02 | partial: last 2 blocks, lr-trunk 3e-5 | 0 | 4 | 33,671,351 | 3.2 | 0.118 | 10.2 | 0.81 | 0.7327 | 0.7331 | - |
| FT-S03 | partial: last 4 blocks + up + final, lr-trunk 3e-5 (selected recipe) | 0 | 4 | 77,600,823 | 3.7 | 0.155 | 14.4 | 1.02 | 0.7373 | 0.7372 | 0.7410 / 0.7406 |
| FT-S12 | FT-S03 recipe, seed 2 (seed-noise estimate) | 0 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7370 | 0.7371 | 0.7409 / 0.7407 |
| FT-S10 | partial-4, lr-trunk 1e-4 (LR probe) | 0 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7285 | 0.7276 | - |
| FT-S11 | partial-4, 12 ep cosine | 0 | 12 | 77,600,823 | 3.7 | 0.154 | 13.1 | 2.62 | 0.7366 | 0.7368 | - |
| FT-S05 | full fine-tuning, lr-trunk 3e-5 | 0 | 4 | 173,426,647 | 16.5 | 0.229 | 19.0 | 1.42 | 0.7338 | 0.7331 | 0.7380 / 0.7370 |
| FT-S04 | LoRA r16 (attn+mlp+conv1x1), lr-lora 3e-4 | 0 | 4 | 4,892,855 | 3.2 | 0.204 | 17.0 | 1.13 | 0.7364 | 0.7365 | 0.7402 / 0.7399 |
| FT-E1 | FT-S03 + mse128 loss | 0 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.87 | 0.7321 | 0.7313 | - |
| FT-E2 | FT-S03 + 8 blocks + up + final | 0 | 4 | 140,634,167 | 4.4 | 0.221 | 18.4 | 1.23 | 0.7361 | 0.7357 | - |
| FT-E3 | FT-S03 + head hidden 2048 | 0 | 4 | 79,755,319 | 3.7 | 0.157 | 13.4 | 0.89 | 0.7386 | 0.7385 | - |
| FT-E4 | FT-S03 + lr-head 3e-4 | 0 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7373 | 0.7374 | - |
| FT-E5 | FT-S03 + trunk dropout | 0 | 4 | 77,600,823 | 4.1 | 0.169 | 14.3 | 0.95 | 0.7361 | 0.7361 | - |
| FT-R1 | FT-S03 recipe, replicate 1 | 1 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7360 | 0.7354 | 0.7402 / 0.7395 |
| FT-R2 | FT-S03 recipe, replicate 2 | 2 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7303 | 0.7296 | 0.7362 / 0.7354 |
| FT-R3 | FT-S03 recipe, replicate 3 | 3 | 4 | 77,600,823 | 3.7 | 0.154 | 13.1 | 0.88 | 0.7376 | 0.7373 | 0.7421 / 0.7414 |
| FT-L1 | LoRA (FT-S04 recipe), replicate 1 | 1 | 4 | 4,892,855 | 3.2 | 0.204 | 17.0 | 1.14 | 0.7357 | 0.7355 | 0.7400 / 0.7398 |
| FT-L2 | LoRA, replicate 2 | 2 | 4 | 4,892,855 | 3.2 | 0.204 | 17.0 | 1.14 | 0.7307 | 0.7301 | 0.7358 / 0.7351 |
| FT-L3 | LoRA, replicate 3 | 3 | 4 | 4,892,855 | 3.2 | 0.204 | 17.0 | 1.13 | 0.7362 | 0.7361 | 0.7411 / 0.7406 |

Not run: FT-S06 (full, 12 epochs) and FT-S09 (full, lr 1e-4) were dropped before launch on validation evidence (`FINETUNING_PROGRESS.md`, 2026-10-24 05:14 and 07:51 entries).
Profiling-only arms (FT-P01..P11, 30 timed steps, not trained) are in `research_management/FINETUNING_RESOURCE_REPORT.md` section 5.

### (a2) Cache-based experiments on VALIDATION (FT-C01..C85; no Borzoi forward pass; `results_ft/cache_exps/{ridge,heads}_val_results.csv`, `research_management/finetuning/FT_REGISTRY.csv`)
| id | model | features / transform | val r183 | val r152 | note |
|---|---|---|---|---|---|
| FT-C02 | ridge (E4 reproduction) | pretrained outputs, raw, 1888 ch, shared lambda 1e-4 | 0.7337 | 0.7317 | gate; val and test reproduce the published E4 values to < 5e-11 (`results_ft/cache_exps/gate_E4_repro.json`) |
| FT-C08 | ridge | outputs raw, target sqrt back-transformed | 0.7361 | 0.7344 | |
| FT-C28 | ridge | outputs sqrt, target sqrt back-transformed | 0.7362 | 0.7343 | |
| FT-C40 | ridge | outputs + trunk embedding (3808 ch), sqrt | 0.7376 | 0.7357 | |
| FT-C48 | ridge | outputs, context offsets +-1,2,4 (13,216 features) | 0.7389 | 0.7367 | |
| FT-C63 | ridge | outputs, offsets +-1,2,4,8 (16,992) | 0.7389 | 0.7367 | no gain over +-4 |
| FT-C61 | ridge | outputs + trunk centre bin + offsets +-1,2,4 (15,136), sqrt target back-transformed, shared lambda 3.2e-5 | 0.7392 | 0.7370 | best ridge; locked as secondary FT-T05 |
| FT-C62 | ridge | same as C61, per-track lambda | 0.7397 | 0.7375 | rejected: no cross-fit gain (cross-fit r183 0.7270 per-track vs 0.7271 shared lambda) |
| FT-C71 | MLP, MSE | trunk embedding 1920 | 0.7322 | 0.7317 | |
| FT-C83 | MLP, Poisson | trunk embedding 1920 | 0.7378 | 0.7376 | values from `FINETUNING_PROGRESS.md` / `CHECKPOINT_INVENTORY.md` (rows wider than CSV header) |
| FT-C84 | MLP, MSE | same 15,136 features as C61 | 0.7404 | 0.7393 | early stopping on validation (optimistic) |
| FT-C60 | ridge | 26.7 k features | failed (OOM) | | not rerun |
Linear SGD heads (FT-C65..C70) 0.6687-0.7073 (183): worse than closed form. 62 ridge rows and 21 head rows in the two CSVs; see them for all variants.

## (b) Ensembles, blends and cross-fit on VALIDATION
Source: `results_ft/blends/blend_val_results.csv` (round 1; non-TTA predictions) and `results_ft/blends/round{2,3,4}/blend_val_results.csv` (members use RC TTA). `w on val` = weight on the FT/ensemble prediction chosen on validation (grid 0..1 step 0.1) = optimistic.
Cross-fit = choose w on one validation chromosome, score on the other, mean of the two chromosome scores; compare it with the chromosome-wise component means in the two adjacent columns, NOT with the pooled r. ENS8-members = FT-S03, R1, R2, R3, S04, L1, L2, L3.
In the round-3 six-member ensemble the members are FT-S03, R1, R2, R3, S04 and FT-S05 (full, replicate 0), a validation-only candidate that was not locked.

| round | candidate (round-1 rows without @rc; all others RC TTA; ENS: = mean of member predictions; A+B = w*A+(1-w)*B) | w on val | val r183 | val r152 | val r^2 (sq. Pearson) 183 | val pred. R^2 183 | chromosome-wise mean r183, FT/ensemble part | same, ridge part | cross-fit r183 / r152 (chromosome-wise) |
|---|---|---|---|---|---|---|---|---|---|
| round1 | FT-S05+E4 | 0.5 | 0.7436 | 0.7425 | 0.5701 | 0.5634 | 0.7218 | 0.7212 | 0.7313 / 0.7333 |
| round1 | FT-S05+FT-C61 | 0.4 | 0.7465 | 0.7450 | 0.5744 | 0.5580 | 0.7218 | 0.7272 | 0.7343 / 0.7361 |
| round2 | ENS:FT-S03,FT-S04,FT-S05 | 1.0 | 0.7439 | 0.7434 | 0.5702 | 0.5636 | 0.7305 | - | - |
| round2 | FT-S03+FT-C61 | 0.5 | 0.7479 | 0.7467 | 0.5768 | 0.5638 | 0.7272 | 0.7272 | 0.7346 / 0.7364 |
| round2 | FT-S04+FT-C61 | 0.5 | 0.7477 | 0.7465 | 0.5764 | 0.5636 | 0.7265 | 0.7272 | 0.7344 / 0.7363 |
| round2 | FT-S05+FT-C61 | 0.5 | 0.7475 | 0.7461 | 0.5757 | 0.5617 | 0.7265 | 0.7272 | 0.7354 / 0.7371 |
| round2 | ENS:FT-S03,FT-S04,FT-S05+FT-C61 | 0.6 | 0.7489 | 0.7478 | 0.5779 | 0.5666 | 0.7305 | 0.7272 | 0.7359 / 0.7378 |
| round2 | ENS:FT-S03,FT-S04,FT-S05+E4 | 0.7 | 0.7468 | 0.7461 | 0.5747 | 0.5681 | 0.7305 | 0.7212 | 0.7336 / 0.7358 |
| round3 | ENS:FT-S03,FT-R1,FT-R2,FT-R3 | 1.0 | 0.7475 | 0.7468 | 0.5758 | 0.5688 | 0.7349 | - | - |
| round3 | ENS:FT-S03,FT-R1,FT-R2,FT-R3+FT-C61 | 0.7 | 0.7498 | 0.7487 | 0.5793 | 0.5696 | 0.7349 | 0.7272 | 0.7372 / 0.7392 |
| round3 | ENS:FT-S03,FT-R1,FT-R2,FT-R3,FT-S04,FT-S05 | 1.0 | 0.7480 | 0.7474 | 0.5764 | 0.5693 | 0.7353 | - | - |
| round3 | ENS:FT-S03,FT-R1,FT-R2,FT-R3,FT-S04,FT-S05+FT-C61 | 0.7 | 0.7502 | 0.7492 | 0.5798 | 0.5700 | 0.7353 | 0.7272 | 0.7376 / 0.7397 |
| round3 | ENS:FT-S03,FT-R1,FT-R2,FT-R3+E4 | 0.8 | 0.7486 | 0.7478 | 0.5775 | 0.5705 | 0.7349 | 0.7212 | 0.7359 / 0.7382 |
| round4 | ENS:FT-S04,FT-L1,FT-L2,FT-L3 | 1.0 | 0.7477 | 0.7472 | 0.5760 | 0.5691 | 0.7353 | - | - |
| round4 | ENS:FT-S04,FT-L1,FT-L2,FT-L3+FT-C61 | 0.7 | 0.7499 | 0.7489 | 0.5795 | 0.5698 | 0.7353 | 0.7272 | 0.7374 / 0.7396 |
| round4 | ENS:ENS8-members | 1.0 | 0.7482 | 0.7476 | 0.5768 | 0.5698 | 0.7357 | - | - |
| round4 | ENS:ENS8-members+FT-C61 | 0.7 | 0.7501 | 0.7491 | 0.5799 | 0.5701 | 0.7357 | 0.7272 | 0.7376 / 0.7398 |
| round4 | ENS:ENS8-members+E4 | 0.8 | 0.7489 | 0.7482 | 0.5780 | 0.5710 | 0.7357 | 0.7212 | 0.7364 / 0.7388 |
| round4 | ENS:FT-S03,FT-R1,FT-R2,FT-R3,FT-S12 | 1.0 | 0.7476 | 0.7470 | 0.5759 | 0.5689 | 0.7347 | - | - |
| round4 | ENS:FT-S03,FT-R1,FT-R2,FT-R3,FT-S12+FT-C61 | 0.7 | 0.7499 | 0.7489 | 0.5795 | 0.5698 | 0.7347 | 0.7272 | 0.7371 / 0.7391 |

Rows that are the locked configurations: `ENS:ENS8-members` (primary, 0.7482 / 0.7476), `ENS:FT-S03,FT-R1,FT-R2,FT-R3` (0.7475 / 0.7468), `ENS:ENS8-members+FT-C61` w 0.7 (0.7501 / 0.7491, cross-fit 0.7376 / 0.7398). The no-TTA variant of the primary (0.7474 / 0.7470) is from the validation self-check (`results_ft/selfcheck_val/`, `TEST_LOCK.csv`).
Ensemble gains saturate at about four replicate members (4 members 0.7475 / 0.7468; 8 members 0.7482 / 0.7476; differences at noise level, see seed note in the final report).

## (c) LOCKED TEST results (single round; lock committed in `dc0bcd5` before any FT test evaluation)
Comparator: E4 ridge[pre_raw:union1888], test r 0.7426 (183) / 0.7406 (152). Delta r = config - E4 ridge, paired. CIs: 10,000 resamples, seed 20261020; TF-cluster resamples tracks by TF (tiles fixed), tile-block resamples blocks of 10 contiguous tiles
(tracks fixed), two-way = tile blocks x TF clusters (or x cell-line clusters), cell-line-cluster resamples tracks by cell line. 152 tracks: 62 TF clusters, 11 cell-line clusters; 183 tracks: 77 and 12.
Source: `results_ft/final/test_ci_table.csv` (collected from `results_ft/ft_test_eval/<id>/test_comparison.csv` and `results_ft/cache_exps/test_eval/FT-C61/test_comparison.csv`, produced by `analysis/ft/compare_ft.py`).
Only FT-T01 (PRIMARY) carries a claim; FT-T02..T06 are secondary and descriptive (they are NOT ranked; FT-T03 scoring above FT-T01 on test is not used for selection).

### 152 non-identical tracks (claim set)
| config | mean Pearson r | r tile-block 95% CI | mean r^2 (sq. Pearson) | mean pred. R^2 | delta r vs E4 | TF-cluster CI | two-way tile x TF CI | tile-block CI | cell-line-cluster CI | two-way tile x cell-line CI |
|---|---|---|---|---|---|---|---|---|---|---|
| FT-T01 PRIMARY ENS8 RC | 0.7578 | [0.7372, 0.7754] | 0.5898 | 0.5837 | +0.0173 | [0.0123, 0.0252] | [0.0111, 0.0263] | [0.0142, 0.0211] | [0.0129, 0.0414] | [0.0114, 0.0424] |
| FT-T02 ENS8 no TTA | 0.7527 | [0.7270, 0.7737] | 0.5826 | 0.5757 | +0.0121 | [0.0072, 0.0196] | [0.0007, 0.0238] | [0.0022, 0.0195] | [0.0076, 0.0371] | [0.0011, 0.0389] |
| FT-T03 ENS4 partial RC | 0.7613 | [0.7426, 0.7773] | 0.5948 | 0.5893 | +0.0208 | [0.0160, 0.0287] | [0.0136, 0.0309] | [0.0163, 0.0274] | [0.0164, 0.0432] | [0.0136, 0.0441] |
| FT-T04 BLEND ENS8+C61 w0.7 RC | 0.7603 | [0.7404, 0.7773] | 0.5935 | 0.5887 | +0.0197 | [0.0151, 0.0272] | [0.0141, 0.0282] | [0.0170, 0.0234] | [0.0158, 0.0412] | [0.0144, 0.0422] |
| FT-T05 FT-C61 ridge | 0.7466 | [0.7260, 0.7643] | 0.5751 | 0.5548 | +0.0061 | [0.0037, 0.0093] | [0.0027, 0.0105] | [0.0040, 0.0086] | [0.0032, 0.0100] | [0.0026, 0.0111] |
| FT-T06 FT-S03 RC | 0.7522 | [0.7328, 0.7687] | 0.5810 | 0.5748 | +0.0117 | [0.0063, 0.0199] | [0.0036, 0.0223] | [0.0064, 0.0189] | [0.0077, 0.0332] | [0.0041, 0.0342] |
| E4 ridge repro (gate) | 0.7406 | [0.7193, 0.7585] | 0.5663 | 0.5596 | -0.0000 | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] |

### All 183 tracks
| config | mean Pearson r | r tile-block 95% CI | mean r^2 (sq. Pearson) | mean pred. R^2 | delta r vs E4 | TF-cluster CI | two-way tile x TF CI | tile-block CI | cell-line-cluster CI | two-way tile x cell-line CI |
|---|---|---|---|---|---|---|---|---|---|---|
| FT-T01 PRIMARY ENS8 RC | 0.7586 | [0.7367, 0.7768] | 0.5899 | 0.5842 | +0.0160 | [0.0118, 0.0225] | [0.0106, 0.0236] | [0.0129, 0.0198] | [0.0120, 0.0387] | [0.0106, 0.0391] |
| FT-T02 ENS8 no TTA | 0.7537 | [0.7277, 0.7750] | 0.5829 | 0.5765 | +0.0110 | [0.0069, 0.0172] | [0.0003, 0.0214] | [0.0015, 0.0182] | [0.0073, 0.0344] | [0.0006, 0.0354] |
| FT-T03 ENS4 partial RC | 0.7618 | [0.7415, 0.7785] | 0.5945 | 0.5894 | +0.0191 | [0.0152, 0.0258] | [0.0129, 0.0282] | [0.0150, 0.0254] | [0.0144, 0.0403] | [0.0126, 0.0407] |
| FT-T04 BLEND ENS8+C61 w0.7 RC | 0.7613 | [0.7401, 0.7790] | 0.5939 | 0.5891 | +0.0186 | [0.0148, 0.0249] | [0.0138, 0.0259] | [0.0160, 0.0222] | [0.0152, 0.0387] | [0.0139, 0.0390] |
| FT-T05 FT-C61 ridge | 0.7485 | [0.7265, 0.7670] | 0.5766 | 0.5560 | +0.0059 | [0.0040, 0.0085] | [0.0029, 0.0099] | [0.0039, 0.0084] | [0.0040, 0.0098] | [0.0030, 0.0107] |
| FT-T06 FT-S03 RC | 0.7526 | [0.7317, 0.7700] | 0.5805 | 0.5749 | +0.0100 | [0.0056, 0.0167] | [0.0029, 0.0195] | [0.0051, 0.0166] | [0.0058, 0.0303] | [0.0032, 0.0308] |
| E4 ridge repro (gate) | 0.7426 | [0.7201, 0.7612] | 0.5679 | 0.5619 | -0.0000 | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] | [-0.0000, 0.0000] |

The last row is the E4 reproduction through the same pooled-128 path (gate): delta 0 to numerical precision.

### Per-track distribution of test Pearson r (computed from the sufficient statistics in `test_stats.npz`; `results_ft/final/test_summary.csv`, per-track values in `results_ft/final/per_track_test_r.csv`)
152 non-identical tracks
| config | mean r | median r | Q25 - Q75 | IQR | tracks improved vs E4 | mean delta r | median delta r |
|---|---|---|---|---|---|---|---|
| E4 ridge (comparator) | 0.7406 | 0.7862 | 0.6604 - 0.8443 | 0.1839 | - | - | - |
| FT-T01 PRIMARY ENS8 RC | 0.7578 | 0.7971 | 0.6812 - 0.8518 | 0.1707 | 145/152 (95.4%) | +0.0173 | +0.0108 |
| FT-T02 ENS8 no TTA | 0.7527 | 0.7930 | 0.6758 - 0.8461 | 0.1702 | 126/152 (82.9%) | +0.0121 | +0.0071 |
| FT-T03 ENS4 partial RC | 0.7613 | 0.8019 | 0.6873 - 0.8565 | 0.1692 | 150/152 (98.7%) | +0.0208 | +0.0140 |
| FT-T04 BLEND ENS8+C61 w0.7 RC | 0.7603 | 0.8011 | 0.6840 - 0.8553 | 0.1712 | 151/152 (99.3%) | +0.0197 | +0.0130 |
| FT-T05 FT-C61 ridge | 0.7466 | 0.7897 | 0.6706 - 0.8466 | 0.1760 | 112/152 (73.7%) | +0.0061 | +0.0025 |
| FT-T06 FT-S03 RC | 0.7522 | 0.7903 | 0.6797 - 0.8450 | 0.1653 | 118/152 (77.6%) | +0.0117 | +0.0060 |

All 183 tracks
| config | mean r | median r | Q25 - Q75 | IQR | tracks improved vs E4 | mean delta r | median delta r |
|---|---|---|---|---|---|---|---|
| E4 ridge (comparator) | 0.7426 | 0.7753 | 0.6626 - 0.8445 | 0.1819 | - | - | - |
| FT-T01 PRIMARY ENS8 RC | 0.7586 | 0.7927 | 0.6808 - 0.8508 | 0.1700 | 176/183 (96.2%) | +0.0160 | +0.0103 |
| FT-T02 ENS8 no TTA | 0.7537 | 0.7860 | 0.6751 - 0.8482 | 0.1731 | 154/183 (84.2%) | +0.0110 | +0.0066 |
| FT-T03 ENS4 partial RC | 0.7618 | 0.7964 | 0.6865 - 0.8543 | 0.1678 | 181/183 (98.9%) | +0.0191 | +0.0136 |
| FT-T04 BLEND ENS8+C61 w0.7 RC | 0.7613 | 0.7965 | 0.6839 - 0.8561 | 0.1722 | 182/183 (99.5%) | +0.0186 | +0.0129 |
| FT-T05 FT-C61 ridge | 0.7485 | 0.7836 | 0.6705 - 0.8478 | 0.1773 | 138/183 (75.4%) | +0.0059 | +0.0029 |
| FT-T06 FT-S03 RC | 0.7526 | 0.7874 | 0.6791 - 0.8448 | 0.1657 | 135/183 (73.8%) | +0.0100 | +0.0047 |

"Tracks improved" = per-track r strictly above E4 ridge's per-track r. Per-cell-line summary of the primary (152 tracks; `results_ft/final/per_cell_line_test_r.csv`; a per-TF table, 62 TFs in the 152 set, is in `per_TF_test_r.csv`; no TF-family mapping exists in the repository, so TF-family summaries were not made):

| cell line (152 non-identical tracks) | n tracks | mean r primary | mean r E4 ridge | mean delta r | tracks improved |
|---|---|---|---|---|---|
| HCT-116 | 102 | 0.7928 | 0.7801 | +0.0127 | 96/102 |
| LoVo | 17 | 0.7281 | 0.7176 | +0.0104 | 16/17 |
| Caco-2 | 7 | 0.6089 | 0.5330 | +0.0760 | 7/7 |
| GP5d | 6 | 0.7793 | 0.7582 | +0.0212 | 6/6 |
| LS180 | 5 | 0.6199 | 0.5891 | +0.0308 | 5/5 |
| LS174T | 5 | 0.5236 | 0.4838 | +0.0398 | 5/5 |
| SW480 | 4 | 0.7056 | 0.6882 | +0.0174 | 4/4 |
| RKO | 3 | 0.8313 | 0.8203 | +0.0110 | 3/3 |
| DLD-1 | 1 | 0.6123 | 0.5842 | +0.0281 | 1/1 |
| HT-29 | 1 | 0.6509 | 0.6453 | +0.0056 | 1/1 |
| primary_colon_tumor | 1 | 0.7132 | 0.7031 | +0.0101 | 1/1 |

## (d) Historical references (test unless stated; all Pearson r, shared 128 bp grid)
| reference | value | source |
|---|---|---|
| Original Borzoi-TL (4-replicate averaged trunk embedding, MLP head), test, 183 tracks | 0.6820 (0.682; r^2 0.4866, prediction R^2 0.4810) | `research_management/PROJECT_STATE.md` line 15; `research_management/PHASE1_RECOVERY_REPORT.md` (`results_benchmark/aggregate_metrics_test.csv`). On the 152 set the same TL = 0.6851 (`results_ft/ft_test_eval/*/test_comparison.csv`, `mean_a` of the "TL - new" rows). Native-grid 0.663 is a different quantity and is not mixed here. |
| Pretrained Borzoi, validation-selected track mapping, test | 0.6609 (0.661) | `research_management/EXECUTION_LOG.md` (EXP-006); `PHASE1_RECOVERY_REPORT.md` |
| 4-replicate trunk linear probe, test | 0.7223 | `research_management/PROJECT_STATE.md` line 17; `EXECUTION_LOG.md` (E4, EXP-013) |
| E4 ridge[pre_raw:union1888], test (183 / 152) | 0.7426 / 0.7406 | same sources; `results_phase2/e4_ridge/orig_pre_raw/union1888_test_stats.npz`; recomputed here from stats |
| E4 ridge, validation (183 / 152) | 0.7337 / 0.7317 | `FT_REGISTRY.csv` FT-C02; `FINETUNING_PROGRESS.md` |
| 4-replicate trunk linear probe, validation (183) | 0.7104 | `FINETUNING_PROGRESS.md` 2026-10-23 23:45 entry |
Absolute r is not comparable across splits (E4 ridge: 0.7337 validation vs 0.7426 test); only paired test deltas are unbiased comparisons.
