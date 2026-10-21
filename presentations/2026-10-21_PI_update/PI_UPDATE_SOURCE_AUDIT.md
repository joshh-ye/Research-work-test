# PI update 2026-10-21: source verification audit

Deck: `reports/PI_UPDATE_2026-10-21.pptx` (11 slides), built by `reports/pi_update_2026-10-21/build_deck.py` from
`deck_content.py`. Every numerical claim below was checked by the coordinator against the raw result file (source priority 1)
on 2026-10-20 and cross-checked against `manuscript/manuscript_v2.md` (priority 4). Paths are relative to the repo root.
Rounding: r to 3 decimals, Δr to 4 decimals (as in the manuscript), AUROC and ΔAUROC to 3 decimals.
Independent review: see the "Scientific review" section at the end.

## Slide 2: Research overview and questions
| Claim | Value | Source |
|---|---|---|
| CRC credible causal variants | 5,741 (Chen 2024 Supp. Data 6) | `variant_provenance/PROVENANCE.md`; manuscript §2.9 |
| CRC TF ChIP-seq tracks | 183 | `borzoi_ckpt_full/bw_files.json`; `exposure_audit/track_exposure.tsv` |
| MPRA source | Law et al. 2024 (ST2, ST4) | `variant_scoring/E2_ANALYSIS_PLAN.md` §2a; `manuscript/references.tsv` (law2024) |

## Slide 3: Models, baselines and how each is used
| Claim | Value | Source |
|---|---|---|
| Borzoi-TL = head on frozen trunk | n/a | `borzoi_code/model.py`; manuscript §2.4 |
| Ridge features | 1,888 pretrained output channels (union1888 = 1,882 ChIP-TF + 6 CRC DNase) | `results_phase2/e4_ridge/SUMMARY.md`; `SCIENTIFIC_DECISIONS.md` SD-018; manuscript §2.5a |
| Linear probe on 1,920-d trunk embedding | n/a | `results_phase2/e4_ridge/SUMMARY.md`; manuscript §2.5a |
| λ selected on validation only | grid 1e-6 to 100 | `analysis/priorityD/ridgelib.py`; `results_phase2/e4_sensitivity/SUMMARY.md` part B |
| Variant-scoring track sets | TL 152; pretrained Borzoi 132; Enformer 132; AlphaGenome 33 | `variant_scoring/SCORING_SPEC.md`; `variant_scoring/track_lists/`; manuscript §2.10 |
| Enformer/AlphaGenome track benchmark is supplementary | SR2 | manuscript §3.7.4 and §7 (SD-025) |

## Slide 4: Borzoi fine-tuning setup
| Claim | Value | Source |
|---|---|---|
| Input length | 524,288 bp, hg38 | `borzoi_code/dataset.py:10`; manuscript §2.3 |
| Frozen trunk | 4 replicates (`johahi/borzoi-replicate-0..3`), 1,920-channel embedding averaged | `borzoi_code/model.py`; `PHASE1_RECOVERY_REPORT.md` §4 |
| Head | Linear(1920,1024)–GELU–Dropout(0.2)–Linear(1024,183)–Softplus; 2,154,679 parameters | `borzoi_code/model.py`; manuscript §2.4 |
| Targets | 183 tracks, 77 TFs, 10 cell lines + 4 colon tissue + 1 tumour | `exposure_audit/track_exposure.tsv`; `exposure_audit/EXPOSURE_AUDIT.md` §1 |
| Training | Poisson NLL, Adam lr 1e-4, batch 1, epoch 5 of 8 (val NLL 0.18523) | `main.py:62-85`; `results_full/history.json`; manuscript §2.4 |
| Splits | test chr9+chr22 = 359 tiles; val chr1+chr8 = 750; train 4,659 | `exposure_audit/EXPOSURE_AUDIT.md` §4 |
| Test chromosomes in pretraining folds (slides 4 notes, 9) | 43.6% bp Borzoi (30.4% in Borzoi test fold), 73.0% Enformer | `exposure_audit/EXPOSURE_AUDIT.md` §4 |

## Slide 5: Linear readouts outperform Borzoi fine-tuning (Pearson r, 359 test tiles, shared 128 bp grid)
| Claim | Value | Source |
|---|---|---|
| Ridge r, 152 tracks | 0.7406 | `results_phase2/e4_ridge/e4_comparison.csv` row `ridge[pre_raw:union1888],not_identical152` |
| Probe r, 152 tracks | 0.7229 | `results_phase2/e4_sensitivity/A_cluster_cis.csv` (mean_b, TL vs probe_trunk1920) |
| Pretrained Borzoi val-selected r, 152 | 0.6510 | `e4_comparison.csv` row `pretrained_Borzoi_valsel_nonoracle,not_identical152` |
| Borzoi-TL r, 152 | 0.6851 | `e4_comparison.csv` row `Borzoi-TL original checkpoint,not_identical152` |
| TL − ridge, TF-cluster CI | −0.0555 [−0.0756, −0.0428] | `A_cluster_cis.csv` row `TL,ridge_union1888_raw,not_identical152` |
| TL − probe | −0.0378 [−0.0525, −0.0289] | `A_cluster_cis.csv` row `TL,probe_trunk1920,not_identical152` |
| TL − pretrained | +0.0341 [0.0238, 0.0453] | `A_cluster_cis.csv` row `TL,pretrained_valsel,not_identical152` |
| Not robust to cell-line clustering (notes) | all 183: +0.0211, cell-line CI [−0.0183, 0.0414] | manuscript §3.7.4 from `A_cluster_cis.csv` |
| Seeds 1–5, TL − ridge, 152 | −0.0537 to −0.0554; tile-block, TF-cluster, cell-line CIs exclude 0 | `results_phase2/e5_seeds/e5_seed_vs_ridge.csv` |
| Folds F2/F3/F4 × seeds 1–3, TL − ridge, 152 | −0.0615 to −0.0775; all CIs exclude 0 | `results_phase2/e5_folds/e5_fold_vs_ridge.csv` |
| All-183 r (notes) | TL 0.682, ridge 0.743, probe 0.722 | `e4_comparison.csv`; manuscript Table 5 |
| Seed/fold heads use the cache recipe (no RC, shift-emulated, fp16) | n/a | `SCIENTIFIC_DECISIONS.md` SD-018; `results_phase2/e5_seeds/SUMMARY.md` |

## Slide 6: CRC variant-effect scoring
| Claim | Value | Source |
|---|---|---|
| CCVs present in scoring input | 5,739 of 5,741 | `variant_provenance/PROVENANCE.md` |
| Scoring rows / positions | 10,148 rows at 5,739 positions (2,419/2,231/1,089 with 1/2/3 ALTs) | `variant_provenance/PROVENANCE.md`; manuscript §2.9 |
| REF check | 0 mismatches of 10,148 against `hg38.ml.fa` | `variant_provenance/PROVENANCE.md`; `EXPERIMENT_REGISTRY.csv` EXP-008 |
| Allele-matched positions | 5,729 (GCST90129505 allele pair) | `variant_provenance/allele_matched_variants.tsv`; `PROVENANCE.md`; SD-015 |
| Score definition | Borzoi/TL/Enformer: max over tracks of \|log2((ALT+1)/(REF+1))\|, 512 bp centre, RC-averaged; AlphaGenome: max \|quantile score\| (501 bp, server-side RC/epsilon) | `variant_scoring/SCORING_SPEC.md` §1–2; `E2_ANALYSIS_PLAN.md` §3 |
| Frozen before labels joined | n/a | `variant_scoring/E2_ANALYSIS_PLAN.md` header; `EXECUTION_LOG.md` |

## Slide 7: MPRA validation design
| Claim | Value | Source |
|---|---|---|
| Cell lines | HT29, SW403, HCEC-1CT | `E2_ANALYSIS_PLAN.md` §2a |
| Positives / negatives / excluded (all) | 134 / 4,098 / 91 | `results_phase2/e2_validation/final/tables/power_statement.tsv`; Amendment A1.1 |
| Informative regions and counts used | 58 regions, 130 positives, 2,518 negatives | `results_phase2/e2_validation/final/tables/baselines.tsv` (n_regions_informative, n_pos, n_neg) |
| Inference | region bootstrap 10,000; within-region permutation 10,000; Holm over 4 | `E2_ANALYSIS_PLAN.md` §4, A2.1 |
| Annotation count = sum of 7 Chen S6 flags | n/a | `E2_ANALYSIS_PLAN.md` A2.5 |
| Framing rule | `E2_ANALYSIS_PLAN.md` §7 | |

## Slide 8: Four-model MPRA results (stratified within-region AUROC; not r, r² or R²)
| Claim | Value | Source |
|---|---|---|
| Enformer | 0.6743 [0.6067, 0.7564] | `results_phase2/e2_validation/final/tables/p1.tsv` |
| AlphaGenome (quantile) | 0.6662 [0.5896, 0.7593] | `p1.tsv` |
| Borzoi-TL (152 tracks) | 0.6606 [0.5844, 0.7677] | `p1.tsv` |
| Pretrained Borzoi (132 tracks) | 0.6398 [0.5541, 0.7442] | `p1.tsv` |
| Holm p 0.0004 (floor, 4 × 1/10,001) | n/a | `p1.tsv` perm_p_greater; manuscript Table 1 note |
| Distance baseline | 0.5153 [0.4446, 0.5665] | `baselines.tsv` |
| Annotation count baseline | 0.6182 [0.5560, 0.6845] (upper bound 0.68449 → 0.684) | `baselines.tsv` |
| Δ vs distance | Enf +0.159 [0.060, 0.280]; AG +0.151 [0.049, 0.272]; TL +0.145 [0.036, 0.293]; Borzoi +0.125 [0.004, 0.263] | `paired.tsv` family `paired_delta_vs_distance` |
| Δ vs annotation count | Enf +0.056 [−0.006, 0.124]; AG +0.048 [−0.035, 0.138]; TL +0.042 [−0.014, 0.126]; Borzoi +0.022 [−0.063, 0.116] | `paired.tsv` rows with model_b `baseline_chen_annot_count` |
| No model differs (notes) | all 6 pairwise CIs include 0 | `paired.tsv` family `paired_delta_models` |
| Framing criteria met by all 4 | True | `final/tables/framing.tsv`; SD-024; RV-007 |

## Slide 9: Interpretation and limitations
| Claim | Value | Source |
|---|---|---|
| ≥31 identical-experiment tracks (lower bound) | 24 LoVo GSM + 4 colon ENCSR + 3 HCT-116 ENCSR | `exposure_audit/EXPOSURE_AUDIT.md` §1 |
| Track-set sizes | 152/132/132/33 | `variant_scoring/SCORING_SPEC.md` |
| Pretrained set LoVo-heavy | 118 of 132 LoVo | manuscript §2.10; Limitation 9 |
| TL set HCT-116-heavy; AlphaGenome 27 of 33 HCT116 | HCT-116 105 of 183 | `exposure_audit/track_exposure.tsv`; manuscript §2.10 |
| No tumour specificity | HCEC-1CT AUROC 0.66–0.71 vs HT29 0.69–0.72, SW403 0.58–0.64 | `final/tables/secondary_P1.tsv` S2 rows |
| P2 direction | 0.273–0.364 (n = 132) | `final/tables/p2.tsv`; SD-017; BL-007 |
| Post hoc clogit, score coef with distance + annotation count (slides 8 notes, 10) | AG 0.772 [0.533, 1.085]; Borzoi 0.651 [0.408, 0.945]; TL 0.764 [0.486, 1.155]; Enformer 0.655 [0.417, 0.917] (region bootstrap, 2,000; suggestive only, SD-026) | `results_phase2/e2_validation/post_final_exploratory/SUMMARY.md`; `final/tables/clogit.tsv` |
| chr1/8/9/22 subset (notes) | 10 regions, 19 positives, AUROC 0.45–0.56, descriptive | `final/tables/heldout_chr_descriptive.tsv` |
| P2 below 0.5 (notes) | Holm p ≤ 0.004 for all four | `final/tables/p2.tsv` |
| ~12 effective regions (notes) | 11.9; top 3 regions 45.0% of weight (post hoc) | `results_phase2/e2_validation/post_final_exploratory/SUMMARY.md`; manuscript Limitation 14 |
| AlphaGenome exposure undetermined | n/a | `exposure_audit/EXPOSURE_AUDIT.md` §1 |

## Slides 10–11: Conclusions, next steps, questions
| Claim | Source |
|---|---|
| Framing B adopted by pre-registered rule | SD-024; `REVIEW_FINDINGS.md` RV-007 |
| Fine-tuning claim withdrawn (scoped to the head-only recipe) | SD-019, SD-020, SD-021 |
| Manuscript structure | SD-025; `manuscript/manuscript_v2.md` |
| Next steps (BL-007 author contact; independent validation with nested test; Fig. 1, S1, intro) | `research_management/NEXT_ACTIONS.md`; SD-027; manuscript §4 item 7 |
| Questions (title wording SD-028; repo data sharing BL-009) | `SCIENTIFIC_DECISIONS.md` SD-028; `BLOCKERS.md` BL-009 |

## Discrepancies investigated
- Annotation-count CI upper bound appears as 0.684 (abstract) and 0.6845 (Table 1); raw value 0.68449 — both are correct roundings; deck uses 0.684.
- Power: 0.5834 (pre-unblinding) vs 0.5847 (final-run recompute); not on slides.
- No other disagreement between raw outputs and the manuscript was found for the values used.

## Facts not independently verifiable from the repository
- BigWig download source (Cistrome DB style; source unrecorded) — not stated on slides.
- AlphaGenome training-data identity (manifest not obtainable).
- Enformer snapshot revision for the August oracle run (supplementary only; not on slides).
- Literature claims (Chen 2024, Law 2024) rely on `manuscript/references.tsv` entries verified in earlier sessions.

## Scientific review
Native scientific-reviewer agent (Opus), 2026-10-20 (agent-based review, not human peer review).
- Round 1: **REVISE**. No numerical errors were found; every value was recomputed or matched against the raw files. There were 4 major wording issues: the slide 5 title scope, the 152 vs 183 cell-line robustness note, the AlphaGenome score definition, and the post hoc clogit omission. There were also 12 minor issues: fold-overlap wording, "exploratory" tumour specificity, the slide 5 CI and recipe labels, framing-rule wording, ΔAUROC headers and the Holm floor, the P2 note, pre-registration scope, review wording, 5,739 of 5,741, the 152-track and tumour labels, the SD-024 qualifiers, and the title options. All were applied.
- Round 2: **PASS**. All 16 findings were verified fixed. One optional minor wording fix (GTEx vs epigenomic annotations, slide 9 note) was applied afterwards.
