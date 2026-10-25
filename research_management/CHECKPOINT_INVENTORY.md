# Checkpoint inventory (generated 2026-10-25 00:03 by analysis/ft/make_ckpt_inventory.py; do not hand-edit)

Genome build hg38; all runs fine-tune/probe Borzoi (frozen_source column = pretrained trunk re-loaded from HF cache). Checkpoints hold trainable params only (+ optimizer/RNG for ckpts/ files). `results_ft/` is gitignored (`results*/`); no checkpoint is in Git.

Reload: `analysis/ft/predict_ft.py --run <ID> --split val` rebuilds the model from `run_cmd.json` argv via `train_ft.parse()/build()` and loads `best_model.pt` key-for-key; resume training from newest `ckpts/ckpt_step*.pt` via the run's original `train_ft.py` argv.

## Training runs (results_ft/runs)

| run | mode / arm | rep | loss | ep | seed | state | best ep | best full-val r183 | r152 | best_model.pt (size, sha256[:16]) | frozen trunk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FT-E1 | partial unfreeze=blocks:4,up,final | 0 | mse128 | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7321 | 0.7313 | 310.4 MB, `524122f682b97d5c` | johahi/borzoi-replicate-0@521db10337 |
| FT-E2 | partial unfreeze=blocks:8,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7361 | 0.7357 | 562.6 MB, `f4983c4caee6df58` | johahi/borzoi-replicate-0@521db10337 |
| FT-E3 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7386 | 0.7385 | 319.1 MB, `855b0b98faac933c` | johahi/borzoi-replicate-0@521db10337 |
| FT-E4 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7373 | 0.7374 | 310.4 MB, `c4d8d4f33d08e121` | johahi/borzoi-replicate-0@521db10337 |
| FT-E5 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7361 | 0.7361 | 310.4 MB, `1af3b56d9a08af7e` | johahi/borzoi-replicate-0@521db10337 |
| FT-L1 | lora lora-targets=attn,mlp,conv1x1 lora-r=16 | 1 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7357 | 0.7355 | 19.6 MB, `2ff253b3665a2704` | johahi/borzoi-replicate-1@ad05f7a1c7 |
| FT-L2 | lora lora-targets=attn,mlp,conv1x1 lora-r=16 | 2 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7307 | 0.7301 | 19.6 MB, `9874ecbb0746f7f1` | johahi/borzoi-replicate-2@871a31b59e |
| FT-L3 | lora lora-targets=attn,mlp,conv1x1 lora-r=16 | 3 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7362 | 0.7361 | 19.6 MB, `b65010d218c01bfb` | johahi/borzoi-replicate-3@da3b900237 |
| FT-P12 | partial unfreeze=blocks:2 | 0 | ? | 1 | ? | finished (ep 1, step 600) | 1 | 0.6821 | 0.6840 | 134.7 MB, `71b46dbed46499a8` | johahi/borzoi-replicate-0@521db10337 |
| FT-R1 | partial unfreeze=blocks:4,up,final | 1 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7360 | 0.7354 | 310.4 MB, `cd3a1a5758f07a2e` | johahi/borzoi-replicate-1@ad05f7a1c7 |
| FT-R2 | partial unfreeze=blocks:4,up,final | 2 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7303 | 0.7296 | 310.4 MB, `066c36512e82eff9` | johahi/borzoi-replicate-2@871a31b59e |
| FT-R3 | partial unfreeze=blocks:4,up,final | 3 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7376 | 0.7373 | 310.4 MB, `cf592f92b09ff4df` | johahi/borzoi-replicate-3@da3b900237 |
| FT-S01 | frozen | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7280 | 0.7282 | 8.6 MB, `a195f021d3149aae` | johahi/borzoi-replicate-0@521db10337 |
| FT-S02 | partial unfreeze=blocks:2 | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7327 | 0.7331 | 134.7 MB, `cab6a91878437e69` | johahi/borzoi-replicate-0@521db10337 |
| FT-S03 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7373 | 0.7372 | 310.4 MB, `bbb1ffc79a3359c2` | johahi/borzoi-replicate-0@521db10337 |
| FT-S04 | lora lora-targets=attn,mlp,conv1x1 lora-r=16 | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7364 | 0.7365 | 19.6 MB, `9ec8320c541eefa8` | johahi/borzoi-replicate-0@521db10337 |
| FT-S05 | full | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7338 | 0.7331 | 693.8 MB, `7d052b42550cf22b` | johahi/borzoi-replicate-0@521db10337 |
| FT-S07 | frozen | 0 | poisson | 12 | 1 | finished (ep 12, step 55908) | 12 | 0.7288 | 0.7291 | 8.6 MB, `962582cd8f610a50` | johahi/borzoi-replicate-0@521db10337 |
| FT-S10 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 1 | finished (ep 4, step 18636) | 4 | 0.7285 | 0.7276 | 310.4 MB, `1f51b75a0afb86b7` | johahi/borzoi-replicate-0@521db10337 |
| FT-S11 | partial unfreeze=blocks:4,up,final | 0 | poisson | 12 | 1 | finished (ep 12, step 55908) | 12 | 0.7366 | 0.7368 | 310.4 MB, `f2c79d62af1e795a` | johahi/borzoi-replicate-0@521db10337 |
| FT-S12 | partial unfreeze=blocks:4,up,final | 0 | poisson | 4 | 2 | finished (ep 4, step 18636) | 4 | 0.7370 | 0.7371 | 310.4 MB, `a551ff499cd73e31` | johahi/borzoi-replicate-0@521db10337 |

best ep / r183 / r152 are the max over `EPOCH n full val` lines in each run's stdout.log (full validation set, 750 tiles; r = mean per-track Pearson r over 183 / 152 tracks). Paths: `results_ft/runs/<run>/best_model.pt`.

### Recent resume checkpoints (results_ft/runs/<run>/ckpts/)

| run | files (size) |
|---|---|
| FT-E1 | none |
| FT-E2 | none |
| FT-E3 | none |
| FT-E4 | ckpt_step00013977.pt (931.3 MB); ckpt_step00018636.pt (931.3 MB) |
| FT-E5 | ckpt_step00013977.pt (931.3 MB); ckpt_step00018636.pt (931.3 MB) |
| FT-L1 | none |
| FT-L2 | none |
| FT-L3 | none |
| FT-P12 | none |
| FT-R1 | none |
| FT-R2 | none |
| FT-R3 | none |
| FT-S01 | none |
| FT-S02 | none |
| FT-S03 | none |
| FT-S04 | none |
| FT-S05 | none |
| FT-S07 | none |
| FT-S10 | none |
| FT-S11 | none |
| FT-S12 | none |

### Profile-only runs (no checkpoints; STATUS.json state)

| run | state | argv (mode / key args) |
|---|---|---|
| FT-P01 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=frozen |
| FT-P02 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=partial unfreeze=blocks:1 |
| FT-P03 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=partial unfreeze=blocks:2 |
| FT-P04 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=partial unfreeze=blocks:4,up,final |
| FT-P05 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=partial unfreeze=blocks:8,up,final,horiz |
| FT-P06 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=lora lora-targets=attn lora-r=8 |
| FT-P07 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=lora lora-targets=attn,mlp,conv1x1 lora-r=16 |
| FT-P08 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=full grad-ckpt=transformer |
| FT-P09 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=full grad-ckpt=all |
| FT-P10 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=partial unfreeze=blocks:2 grad-ckpt=none |
| FT-P11 | profile_done | profile-steps=30 val-limit=20 mem-cap-gb=48 num-workers=6 mode=full grad-ckpt=none |

## Cache-based heads (results_ft/cache_exps)

Heads trained on the frozen trunk cache (`/nobackup/.../trunk_cache`); config per id in `results_ft/cache_exps/configs/<id>.json`; val metrics from `results_ft/cache_exps/heads_val_results.csv` (column names as in that file: val_r_all183 = mean Pearson r over 183, val_r_152). Rows of the CSV wider than its header (mlp heads) are parsed from the row tail (fields -4/-3 = r183/r152; verified against FT-C83..85 rows). Config `checkpoint` field points to `/tmp/yejz1_ft/ckpt/` (node-local); the persistent copy is `results_ft/cache_exps/ckpt/`.

| id | family | best_ep | val r183 | val r152 | ckpt (size) | sha256[:16] |
|---|---|---|---|---|---|---|
| FT-C65 | linear_mse | 14 | 0.7073 | 0.7081 | 1.4 MB | `e2aa95de8ad5c1d7` |
| FT-C66 | linear_mse | 14 | 0.7015 | 0.7006 | 11.1 MB | `3876becafa3c5b92` |
| FT-C67 | linear_mse | 14 | 0.7007 | 0.7016 | 1.4 MB | `493b0ba971e16f86` |
| FT-C68 | linear_mse | 6 | 0.6781 | 0.6754 | 11.1 MB | `b836d71cac0f7e6c` |
| FT-C69 | linear_mse | 10 | 0.6870 | 0.6875 | 1.4 MB | `8cc22f0267773f4c` |
| FT-C70 | linear_mse | 10 | 0.6687 | 0.6694 | 11.1 MB | `1c21436a0caa2da0` |
| FT-C71 | mlp_mse | 8 | 0.7322 | 0.7317 | 8.6 MB | `a95ff6d3791a3555` |
| FT-C72 | mlp_mse | 8 | 0.7206 | 0.7199 | 8.6 MB | `58d49fcfc81a2786` |
| FT-C73 | mlp_mse | 14 | 0.7360 | 0.7345 | 62.8 MB | `f694e4f347cb9259` |
| FT-C74 | mlp_mse | 15 | 0.7328 | 0.7321 | 8.6 MB | `3f369360997fab10` |
| FT-C75 | mlp_poisson | 15 | 0.7350 | 0.7345 | 8.6 MB | `d28dfd749966333f` |
| FT-C76 | mlp_mse | 8 | 0.7223 | 0.7220 | 62.8 MB | `1246ba1019a68f02` |
| FT-C77 | mlp_poisson | 8 | 0.7265 | 0.7267 | 8.6 MB | `7ac141dedd1baf05` |
| FT-C78 | mlp_poisson | 10 | 0.7146 | 0.7155 | 8.6 MB | `9a4f392c8e4445f3` |
| FT-C79 | mlp_mse | 13 | 0.7332 | 0.7322 | 62.8 MB | `ab88eef39bb00307` |
| FT-C80 | mlp_poisson | 8 | 0.7283 | 0.7275 | 62.8 MB | `799bff0764423373` |
| FT-C81 | mlp_poisson | 8 | 0.7170 | 0.7163 | 62.8 MB | `9966744e0ec3dbf9` |
| FT-C82 | mlp_poisson | 14 | 0.7203 | 0.7197 | 62.8 MB | `f4617ebabafb700d` |
| FT-C83 | mlp_poisson | 24 | 0.7378 | 0.7376 | 8.6 MB | `974a5b3089f39f6b` |
| FT-C84 | mlp_mse | 21 | 0.7404 | 0.7393 | 62.8 MB | `bb867c2ce6bbed34` |
| FT-C85 | mlp_mse | 18 | 0.7337 | 0.7332 | 8.6 MB | `15e34c78a17ed3d0` |

## Validation predictions (results_ft/preds)

| dir | total size | files |
|---|---|---|
| E4_ridge | 246.0 MB | info.json (0.0 MB), val_gid.npy (0.0 MB), val_pred.npy (246.0 MB) |
| FT-C61 | 246.0 MB | info.json (0.0 MB), val_gid.npy (0.0 MB), val_pred.npy (246.0 MB) |
| FT-L1 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-L2 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-L3 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-R1 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-R2 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-R3 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S01 | 253.7 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_repro.json (0.0 MB), val_stats.npz (7.7 MB) |
| FT-S03 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S04 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S05 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S07 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S11 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |
| FT-S12 | 507.3 MB | val_gid.npy (0.0 MB), val_pred.npy (246.0 MB), val_pred_rc.npy (246.0 MB), val_repro.json (0.0 MB), val_repro_rc.json (0.0 MB), val_stats.npz (7.7 MB), val_stats_rc.npz (7.7 MB) |

## Blends (results_ft/blends)

| path | size |
|---|---|
| blends/blend_val_grid.csv | 0.0 MB |
| blends/blend_val_results.csv | 0.0 MB |
| blends/round2/blend_val_grid.csv | 0.0 MB |
| blends/round2/blend_val_results.csv | 0.0 MB |
| blends/round3/blend_val_grid.csv | 0.0 MB |
| blends/round3/blend_val_results.csv | 0.0 MB |
| blends/round4/blend_val_grid.csv | 0.0 MB |
| blends/round4/blend_val_results.csv | 0.0 MB |

## Storage and persistence

- `results_ft/` total: 19.91 GB (du -sb). Subdirs: blends 0.0 MB, cache_exps 576.6 MB, final 0.2 MB, ft_test_eval 257.6 MB, preds 6.83 GB, runs 8.80 GB, selfcheck_val 538.1 MB, test_runs 2.90 GB.
- `results_ft/` lives on home: persistent and backed up (FT_PLAN.md). Gitignored (`results*/`); checkpoints are NOT in Git.
- Home / nobackup usage (`accre_storage`, columns usage/quota/limit then files):
```
Home (user):		141.6G	186.3G	232.8G		227158	3000000	3500000	
Nobackup (user):	160.6G	186.3G	232.8G		18811	3000000	3500000
```
- `/tmp/yejz1_ft` is node-local, NOT persistent, not backed up: ckpt 545.7 MB, pooled128 43.14 GB. `pooled128` is regenerable via `analysis/ft/build_pooled.py` (needs trunk cache); `ckpt/` copies of cache heads are duplicated in `results_ft/cache_exps/ckpt/` (verify sha before relying on either).
- Trunk cache `/nobackup/user/yejz1/crc_phase2/trunk_cache`: present, 127.81 GB; read-only input, on /nobackup (not backed up).
