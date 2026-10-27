# Ensemble ablation of the locked 8-member fine-tuned ensemble (EXP-020; post hoc, descriptive)
Run 2026-10-26 19:10–19:30 CDT. Code: `analysis/ft/ablation.py` (select / test-preds / test-eval) and `analysis/ft/ablation_report.py` (summary + figures).
Tables: `research_management/finetuning/{ablation_summary.csv, ablation_per_track_test_r.csv, ablation_val_selection.json, ABLATION_LOCK.csv, ftl2_region_sensitivity.json}`.
Figures: `figures_ft_sprint/fig_individual_vs_ensemble.{png,pdf}`, `figures_ft_sprint/fig_ensemble_size.{png,pdf}`.

## Status and test discipline
- **The primary was not test-selected.** The locked primary (`ENS:FT-S03+FT-R1+FT-R2+FT-R3+FT-S04+FT-L1+FT-L2+FT-L3#tta=rc`) was chosen on validation and committed in `TEST_LOCK.csv` (dc0bcd5) before any fine-tuned test evaluation (`FINETUNING_VALIDATION_AUDIT.md` §1 item 7). There is no test-set ensemble-selection limitation to disclose for it.
- **Ablation selection used validation only.** Every ablation configuration was selected on validation (chr1+chr8, 750 tiles):
  - individual members;
  - best equal-weight 2-subset (exhaustive, 28 candidates) and 4-subset (70 candidates), selected by validation mean r over all 183 tracks;
  - a greedy forward curve over sizes 1–8;
  - a weighted 8-member ensemble (simplex weights by SLSQP).
  `ablation.py select` wrote 16 rows; the greedy size-4 row (which line 149 of the script skipped at 9aa33aa; fixed since) was added by hand from `val_selection.json` (validation values 0.7482 / 0.7477) before the commit (RV-FT-04 #4). These 17 configurations were committed (**9aa33aa, 19:16 CDT**) **before** any per-member test prediction existed.
- **Per-member test predictions.** These were generated afterwards: one RC-TTA pass per member, `predict_ft.run_prediction`, gated on the committed lock.
- **Each config evaluated once.** Every configuration was evaluated once with the same `compare_ft.run_compare` CIs as the primary. One evaluation was interrupted before its statistics were written (FT-R1 single) and was rerun.
- **All ablation test numbers are post hoc relative to the primary lock.** They are descriptive; they do not replace the primary or form new claims. Differences between configurations have no paired CI (only each configuration minus the E4 ridge does).
- **Reproduction check.** `ABL:greedy8`, which has the same 8 members as the primary, recomputed from stored per-member fp16 predictions reproduces the locked result to < 1e-5: r 0.7578, Δ +0.0173, same CIs at the printed precision, max per-track difference 7.8e-6 (fp16 storage of member predictions). On validation the recomputed 8-member ensemble gives 0.74819 / 0.74761, against the locked 0.7482 / 0.7476.

## Results (test chr9+chr22, 359 tiles; mean per-track Pearson r on the 152 non-identical tracks unless stated)
Δ = configuration − E4 ridge (0.7406). CIs are 95% percentile, 10,000 resamples: TF-cluster (62 clusters) and two-way tile-block × TF.

| configuration (selected on validation) | members | val r152 | test r152 | Δ152 | TF-cluster CI | two-way CI | tracks improved /152 | share of 8-member gain | GPU s (test, 359 tiles) |
|---|---|---|---|---|---|---|---|---|---|
| FT-S03 (partial-4, rep 0) | 1 | 0.7406 | 0.7522 | +0.0117 | [0.0063, 0.0199] | [0.0036, 0.0223] | 118 | 0.68 | 61 |
| FT-R1 (partial-4, rep 1) | 1 | 0.7395 | 0.7541 | +0.0136 | [0.0088, 0.0215] | [0.0066, 0.0235] | 138 | 0.79 | 58 |
| FT-R2 (partial-4, rep 2) | 1 | 0.7354 | 0.7528 | +0.0122 | [0.0076, 0.0194] | [0.0051, 0.0219] | 130 | 0.71 | 58 |
| FT-R3 (partial-4, rep 3) | 1 | 0.7414 | 0.7549 | +0.0144 | [0.0101, 0.0215] | [0.0072, 0.0240] | 147 | 0.83 | 58 |
| FT-S04 (LoRA, rep 0) | 1 | 0.7399 | 0.7509 | +0.0103 | [0.0049, 0.0187] | [0.0030, 0.0206] | 111 | 0.60 | 62 |
| FT-L1 (LoRA, rep 1) | 1 | 0.7398 | 0.7517 | +0.0112 | [0.0058, 0.0196] | [0.0026, 0.0222] | 117 | 0.65 | 61 |
| **FT-L2 (LoRA, rep 2)** | 1 | 0.7351 | **0.6297** | **−0.1108** | [−0.1324, −0.0912] | [−0.1986, 0.0114] | 8 | −6.41 | 61 |
| FT-L3 (LoRA, rep 3) | 1 | 0.7406 | 0.7545 | +0.0139 | [0.0093, 0.0213] | [0.0062, 0.0239] | 144 | 0.80 | 61 |
| best 2 = greedy 2: FT-R3+FT-S04 | 2 | 0.7463 | 0.7586 | +0.0180 | [0.0133, 0.0259] | [0.0110, 0.0280] | 149 | 1.04 | 120 |
| greedy 3: +FT-R1 | 3 | 0.7476 | 0.7608 | +0.0203 | [0.0155, 0.0282] | [0.0132, 0.0301] | 150 | 1.17 | 178 |
| best 4: FT-S03+FT-R3+FT-L1+FT-L3 | 4 | 0.7478 | 0.7607 | +0.0201 | [0.0153, 0.0283] | [0.0127, 0.0304] | 150 | 1.17 | 241 |
| greedy 4: FT-R3+FT-S04+FT-R1+FT-L3 | 4 | 0.7477 | 0.7610 | +0.0204 | [0.0157, 0.0283] | [0.0133, 0.0303] | 150 | 1.18 | 239 |
| greedy 5: +FT-S03 | 5 | 0.7479 | 0.7609 | +0.0203 | [0.0155, 0.0284] | [0.0131, 0.0304] | 150 | 1.18 | 300 |
| greedy 6: +FT-L1 | 6 | 0.7481 | 0.7610 | +0.0205 | [0.0156, 0.0287] | [0.0132, 0.0306] | 150 | 1.18 | 361 |
| greedy 7: +FT-L2 | 7 | 0.7480 | 0.7567 | +0.0162 | [0.0111, 0.0241] | [0.0095, 0.0254] | 139 | 0.94 | 422 |
| **all 8 (= locked PRIMARY)** | 8 | 0.7476 | **0.7578** | **+0.0173** | [0.0123, 0.0252] | [0.0111, 0.0263] | 145 | 1.00 | 480 |
| weighted 8 (simplex weights on val) | 8 | 0.7482 | 0.7605 | +0.0199 | [0.0150, 0.0280] | [0.0131, 0.0297] | 150 | 1.15 | 480 |

All-183 values, tile-block CIs, cell-line CIs, r² and prediction R² for every configuration are in `ablation_summary.csv`. Inference cost is the measured A100 wall time for 359 test tiles: about 60 s per member, two forward passes (forward + RC) per tile per member, so cost is linear in members. Model loading (about 1 min per member) is excluded.

## Findings
1. **Smaller ensembles match the primary, partly because the primary contains a failed member.**
   - The validation-selected 2-member ensemble (FT-R3 + FT-S04, one partial and one LoRA model) reaches Δ +0.0180 (104% of the 8-member gain) at 1/4 of the inference cost. It matches the primary only because FT-L2 penalises the primary (finding 3).
   - Against the validation-selected ensembles that exclude FT-L2 it is lower on test: 0.7586 vs 0.7607–0.7610 (−0.002, no paired CI). Validation also ranked it below the 8-member ensemble (0.7463 vs 0.7476, 152 tracks).
   - The validation-selected 4-member ensembles reach Δ +0.0201 / +0.0204 at 1/2 the cost.
   - These are point estimates; differences between configurations (0.001–0.003) are of the size of seed noise and have no paired CI.
   - Validation predicted saturation beyond about 3–4 members (0.7476 → 0.7482). Test agrees for sizes 3–6.
2. **Every single member except one beats the ridge on test.**
   - Seven of the eight members alone exceed the E4 ridge: Δ +0.010 to +0.014, every TF-cluster lower bound > 0.
   - Ensembling adds about +0.006 to +0.009 over the members' typical +0.012, consistent with the validation estimate of about +0.007.
3. **FT-L2 fails on a test region; validation did not detect it.**
   - FT-L2 (LoRA r16, Borzoi replicate 2) is normal on validation (0.7351) but reaches only 0.6297 on test.
   - Diagnosis (`analysis/ft/ftl2_diag.py` → `ftl2_diag.json`; `ftl2_region_sensitivity.json`): FT-L2 over-predicts every track across **chr9:20–32 Mb (hg38; 23 of 359 test tiles)**. Its predictions there are a median ×2.9 (range ×1.4–×6.9 across tracks) those of FT-L1. In the same tiles the targets and the other seven members agree with FT-L1 (per-track median ratios in `ftl2_diag.json`: targets/FT-L1 about 1.04, other members/FT-L1 about 1.00–1.04).
   - Excluding those 23 tiles (post hoc, descriptive), FT-L2 scores 0.7536, in line with the other members.
   - On validation, no tile showed a comparable FT-L2/FT-L1 level ratio (99th percentile 1.30, max 2.16 against up to 11 on test). The 51 test tiles with all-zero targets (assembly gaps) enter the pooled r. There the FT-L2/FT-L1 ratio is mostly a constant 0.391 (three tiles 1.6–2.6), which carries no diagnostic information about the chr9 region.
   - The cause is not established. It is a member-specific, region-specific failure (LoRA adapters on one replicate), not a pipeline error, because the same files reproduce the locked primary to < 1e-5.
4. **Effect of FT-L2 on the primary.**
   - Including FT-L2 lowers the ensemble by about 0.003–0.004 (greedy 6 0.7610 → greedy 7 0.7567; the primary 0.7578).
   - The equal-weight ensemble absorbs most of the failure: the primary still improves 145/152 tracks.
   - Excluding chr9:20–32 Mb, primary − ridge is +0.0178 (149/152 improved), against +0.0173 on all tiles. The region was chosen after the fact as the place where one member pulls the ensemble down, so this check can only show that the region **depresses rather than inflates** the gain; it is not an independent robustness test.
   - The locked primary stays the reported result. The ablation shows the primary is, if anything, slightly penalised by one faulty member, not inflated.
5. **Validation-fitted weights did not help out of sample.**
   - The weighted 8-member ensemble gave almost all weight to the seven well-behaved members. FT-R2 received 0.002, FT-L2 0.059 and the others 0.11–0.22.
   - Its test r (0.7605) is close to the equal-weight subsets that exclude FT-L2.
   - In the chromosome cross-fit on validation (fit chr1, evaluate chr8 and vice versa), the fitted weights were slightly *worse* than equal weights in both folds (0.6941 vs 0.6946; 0.7761 vs 0.7768). The weighting therefore has no validation support beyond what it could not see.
6. **Diversity.**
   - Member predictions are highly correlated: mean per-track Pearson r between members on validation is 0.973.
   - The two arms on the *same* replicate correlate more (0.991) than any pair on different replicates (0.970 within an arm, 0.969 across arms).
   - The pretrained replicate is therefore the main source of diversity; partial vs LoRA adds little. This matches the validation-selected 2-member ensemble pairing two different replicates (rep 3 partial + rep 0 LoRA).

## Conclusions for the manuscript
- The locked result (+0.0173) stands, with no test-set selection.
- Post hoc and descriptive: a validation-selected 2-member ensemble (1/4 of the cost) matched the primary's gain, but only because the primary contains the failed FT-L2. It is about 0.002 below the FT-L2-free 3–6-member ensembles. 4-member ensembles at half the cost retained the gain of the FT-L2-free ensembles. A 2-fold cost reduction is supported on these data; a 4-fold reduction costs about 0.002 r (no paired CI).
- One of eight members fails on a contiguous 12 Mb test region that validation did not flag. Ensembles are more robust than single models here, and **per-member test-region QC is a limitation of the locked design**. It cannot be fixed post hoc without new held-out data.

## Limitations
- The ablation evaluations are post hoc (17 extra test evaluations) and use the same development-exposed chr9/chr22.
- There are no paired CIs between ablation configurations.
- Each member is a single seed.
- The FT-L2 region analysis is a diagnostic. Excluding regions is not a valid way to re-estimate performance and is not used for any claim.
- Inference timings are from one shared A100 and include data loading.
