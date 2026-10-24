# analysis/ft (cache experiments, files listed below; other files in this dir belong to the fine-tuning profiling work): cache-based performance program (FT-C experiments)
Cheap experiments on the existing Phase-2 trunk cache (never modified; no Borzoi forward passes). hg38, shared 896 x 128 bp grid,
183 targets, train 4659 / val chr1+chr8 750 / test chr9+chr22 359 tiles. Selection on validation only; test only via `test_eval_cache.py`
for ids locked in `research_management/finetuning/TEST_LOCK.csv`. Registry: `research_management/finetuning/FT_REGISTRY.csv`;
outputs: `results_ft/cache_exps/` (configs/FT-Cxx.json, per_track_val_r/, ridge_val_results.csv, heads_val_results.csv, logs/).

| file | purpose |
|---|---|
| poolcache.py, build_pooled.py | build `/tmp/yejz1_ft/pooled128` (node-local, regenerable; ~43 GB, ~2.5 min): emb pooled to 128 bp (central [4:4100] bins, crop 256/side, mean-pool x4, fp16), pre128 as is, tgt pooled via the E4 path |
| ftlib.py | feature builder (transforms, neighbouring-bin context, zero-padded in standardised space), streamed fp64 Gram + eigh ridge, validation streaming stats, lambda choice (shared / per-track / chromosome cross-fit), registry helper |
| ftheads.py, 12_heads.py | linear / MLP heads (MSE on standardised targets; Poisson NLL + Softplus) on identical features, early stopping on val mean r |
| 10_gate.py | gate: E4 ridge[pre_raw:union1888] reproduced from the pooled copy |
| 11_ridge_search.py | stages s1 (feature x target transform), s2 (feature sets A=outputs, T=trunk, C=combined), s3 (context offsets) |
| test_eval_cache.py | locked test evaluation + paired bootstrap vs E4 ridge and TL |
| tests/test_ft.py | unit tests (pooling, shift/context, ridge closed form, streaming metrics, lock refusal) |

Pearson r = mean over tracks of per-track r pooled over all bins; r^2 = squared Pearson; R^2 = 1-SSE/SST (not meaningful for predictions
that are not back-transformed to the raw target scale).
