# TEST_LOCK round (single round; written before any FT test evaluation)
Revision 2 after scientific review RV-FT-01 (REVISE). All selection used validation chromosomes chr1+chr8 (750 tiles). Test = chr9+chr22 (359 tiles), one evaluation per
config id with `predict_ft.py --split test` (FT, ensembles, blends) or `test_eval_cache.py` (FT-C61). TEST_LOCK.csv is committed to Git before the first test run;
each `run_info.json` records the commit hash.

## Locked configurations
Exact config ids (literal `|`; FT ids carry the TTA choice; `--runs` passed in id order). Validation r = mean per-track Pearson r on the 896 x 128 bp grid (183 / 152 tracks).

| # | config id | role | selection rule | val r 183 / 152 |
|---|---|---|---|---|
| 1 | `ENS:FT-S03+FT-R1+FT-R2+FT-R3+FT-S04+FT-L1+FT-L2+FT-L3#tta=rc` | PRIMARY | highest-validation pure fine-tuned configuration (no fitted weight): 8 members = partial-4 and LoRA-r16 on Borzoi replicates 0-3. Near-tie disclosed: 4-replicate partial ensemble 0.7475 / 0.7468 (the earlier declared target) and LoRA-only 4-member 0.7477 / 0.7472; differences are at noise level. | 0.7482 / 0.7476 |
| 2 | `ENS:FT-S03+FT-R1+FT-R2+FT-R3+FT-S04+FT-L1+FT-L2+FT-L3#tta=none` | secondary | primary without RC test-time augmentation (separates the TTA contribution) | from self-check |
| 3 | `ENS:FT-S03+FT-R1+FT-R2+FT-R3#tta=rc` | secondary | earlier declared target: 4 replicates of the best single-model arm (FT-S03 recipe) | 0.7475 / 0.7468 |
| 4 | `BLEND:ENS:FT-S03+FT-R1+FT-R2+FT-R3+FT-S04+FT-L1+FT-L2+FT-L3\|FT-C61\|w=0.7#tta=rc` (literal pipes in the CSV) | secondary | primary + best cache ridge; w on validation grid 0..1 step 0.1; both chromosome folds chose 0.7 | 0.7501 / 0.7491 (w on val); cross-fit 0.7376 / 0.7398 (chromosome-wise mean) |
| 5 | `FT-C61` | secondary | best cache ridge (feature transform, context offsets, lambda on validation); no TTA (n/a) | 0.7392 / 0.7370 |
| 6 | `FT-S03#tta=rc` | secondary | best single fine-tuned model (replicate 0) | 0.7410 / 0.7406 |

## Claim rule (tightened before any test run)
- Comparator: E4 ridge[pre_raw:union1888] (test r 0.7426 / 0.7406, already public).
- An improvement is claimed only for the PRIMARY and only if, on the 152 non-identical tracks, the 95% CIs of delta r (primary - ridge) from both the TF-cluster
  bootstrap and the two-way (tile-block x TF) bootstrap exclude 0. The tile-block and cell-line-cluster CIs are reported alongside. The TF-cluster CI is conditional on the
  test regions (it resamples tracks, tiles fixed).
- If the primary fails the rule, no improvement is claimed from any secondary. Secondaries are reported descriptively with the same CIs and are never ranked to pick a
  "best test" model. The "TL - new" rows of compare_ft are descriptive only.

## Labelling and caveats
- The comparison is system vs system: "fine-tuned ensemble (replicate-specific trunks, RC TTA) vs 4-replicate pretrained-output ridge (shift-emulated features, no RC)".
  Row 2 isolates the RC TTA contribution; on validation TTA added about +0.004 for single models.
- Validation numbers are optimistic: early stopping per member on validation, TTA chosen on validation, primary picked among near-tied ensembles (<= ~0.001), about 120
  validation-evaluated configurations overall. Absolute r is not comparable across splits (E4 ridge: 0.7337 val vs 0.7426 test); only the test delta r is unbiased.
- Exposure: about 44% of test bp lie in Borzoi training folds. The ridge reads the pretrained output heads, which include the >= 31 experiments identical to project tracks;
  the fine-tuned models replace the head and partly retrain the trunk. The direction of the resulting bias is unknown and plausibly favours the ridge on the 183-track set;
  the 152 non-identical tracks are therefore the claim set.
- Two test chromosomes (359 tiles): region-level uncertainty is substantial.
- Pearson r is the primary metric; squared Pearson r and prediction R^2 are reported with explicit labels.
- Test predictions of components cached under results_ft/ft_test_eval/_components/ may not be reused for new (post-hoc) configurations; any such use is labelled post-hoc.
