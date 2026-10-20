# Phase 1 Research Recovery Report — CRC Computational Genomics

Date: 2026-10-19 · Branch: `research/crc-autonomous-study` · No GPU jobs or experiments were run.
Evidence base: `PHASE1_INVENTORY.md` (investigator, sonnet), strategist evaluation (opus), coordinator spot-checks.
Labels: **VERIFIED** = seen in files and/or re-checked by coordinator; **INFERRED**; **UNVERIFIED**.

## 1. Original research objectives
Borzoi transfer learning for CRC TF binding: a supervised MLP head on a frozen 4-replicate Borzoi trunk predicting
183 CRC TF ChIP-seq tracks (77 TFs; 10 named cell lines + 20 colon-tissue/unannotated tracks). The draft has two arms: (A) an augmentation ablation and (B) a
comparison against pretrained Borzoi best-matched per target. Variant-effect scoring was planned but not done (VERIFIED, draft).

## 2. Completed experiments (VERIFIED; do not repeat)
| ID | Experiment | Output |
|---|---|---|
| EXP-001 | Borzoi-TL training, Slurm 12212537 | `borzoi_ckpt_full/model_head_best.pt`, `results_full/` |
| EXP-002 | Filtered pretrained-Borzoi baseline, Slurm 13005163 | `results_baseline_filtered/` |
| EXP-003 | Four-model benchmark (oracle mapping) | `results_benchmark/` |
| EXP-000 | Qing deliverables | `reports/qing_deliverables/` |
| EXP-004 | AlphaGenome variant scoring, 10,148 rows | `alphagenome_variant_effects/output/` |

All results live on `ff88be2` (= `experimental_branch` = `research/crc-manuscript-study`). Other branches hold no
unique results. About 10 failed or cancelled Slurm jobs are catalogued in the inventory.

## 3. Qing's completed Borzoi evaluation (VERIFIED)
- Recomputed Borzoi-TL r, r² and prediction R².
- Re-ran pretrained Borzoi with per-tile statistics; it reproduces August to 3e-8.
- Flagged 24 LoVo GSE51142 tracks as training data of the pretrained models (GSM match). **RV-002 correction:** 4 more ENCODE colon tracks (145–148, ENCSR-identical) are also in both vocabularies → ≥28 contaminated.
- Analysed the 51 zero-signal tiles.
- Added TF-cluster and tile bootstraps.
- Two Opus reviews found no critical issues.
- Outstanding: Enformer/AlphaGenome non-oracle re-runs, a supervised baseline, backbone unfreezing.

## 4. Model checkpoints
- **Borzoi-TL** (VERIFIED): `model_head_best.pt`, MLP 1920→1024→183 with 2,154,679 parameters.
  - Epoch 5 of 8; best validation NLL 0.18523, confirmed in `history.json`.
  - `--balance-tracks` was not used and no seed was set.
- **Pretrained Borzoi**: HF `johahi/borzoi-replicate-{0..3}`, snapshots pinned.
- **Enformer**: HF `EleutherAI/enformer-official-rough`. Two snapshots are cached and the revision used is UNKNOWN.

## 5. Existing SNP analyses
- **AlphaGenome hosted API**: 10,148/10,148 SUCCESS, 19 recommended scorers, 1 Mb window. VERIFIED, coordinator re-checked.
- **REF alleles**: REF matches hg38 at 1-based position for all rows (coordinator: 0 mismatches). A 0-based read would give 7,757 mismatches.
- **Multi-allelic expansion (critical)**: the input has 5,739 unique positions expanded to 10,148 rows. VERIFIED.
  - 2,419 positions have 1 ALT, 2,231 have 2, and 1,089 have 3. That gives 4,409 surplus rows over one-allele-per-position. The GWAS-tested allele is unidentifiable at 3,320 positions (7,729 rows) without the source table.
  - ALTs are alphabetically ordered (row order carries no information); the transition fraction (0.766 single-ALT vs ~0.42 per multi-ALT rank) suggests a dbSNP multi-allelic expansion. The extra rows are valid variants scored correctly, just not necessarily the tested allele (RV-002 F4).
  - The GRCh37 origin and any GRCh37→hg38 remap are INFERRED/undocumented. Chen reports 5,741 CCVs vs 5,739 positions here (unreconciled).
- **Variant set provenance**: very likely Chen et al., *Nat Commun* 15:3557 (2024), Supplementary Data 6 (5,741 CRC credible causal variants).
  - Matches: position count, about 143 regions, and the known CRC lead SNPs (rs6983267, rs1800734, …).
  - UNVERIFIED: the supplementary file itself has not been opened.
- **Gaps**: 242 rows (147 positions; chr19 133, chr21 93, chr17 16 rows) have windows shifted at chromosome q-ter; variants remain inside the window. There is no downstream analysis, the local-weights backend was never run, and no Borzoi/Enformer/Borzoi-TL variant scorer exists (VERIFIED by grep).

## 6. Verified numerical results (test chr9+chr22, 359 tiles, 183 tracks; hg38)
| Quantity | Value | Metric | Source |
|---|---|---|---|
| Borzoi-TL, native 32 bp | 0.6634 | Pearson r | log 12212537; `results_comparison_borzoi/summary_stats_test.csv` |
| Borzoi-TL, shared 128 bp | 0.682 | Pearson r | `results_benchmark/aggregate_metrics_test.csv`; `summary_numbers.json` |
| Borzoi-TL, shared 128 bp | 0.4866 | squared Pearson r | `summary_numbers.json` |
| Borzoi-TL, shared 128 bp | 0.4810 | prediction R² | `summary_numbers.json` (0.480966; RV-002 re-derived) |
| Borzoi-TL, 51 zero-signal tiles masked | 0.663 / 0.459 | r / R² | `summary_numbers_followup.json` |
| Pretrained Borzoi, oracle | 0.664 | r | `aggregate_metrics_test.csv` |
| Pretrained Borzoi, validation-selected | 0.661 / 0.444 | r / validation-calibrated R² | Qing §7.4 |
| Enformer, oracle (August, not re-run) | 0.644 | r | `aggregate_metrics_test.csv` |
| AlphaGenome, oracle (August, not re-run) | 0.661 | r | `aggregate_metrics_test.csv` |
| Filtered Borzoi, native 32 bp | 0.6456 | r | log 13005163 |
| Δr TL − Borzoi, shared grid | +0.018 [0.009, 0.026] | r | `results_benchmark/paired_stats_test.csv` |
| Δr, 159 uncontaminated / 24 contaminated tracks | +0.030 / −0.061 | r | Qing |
| Δr, 155 uncontaminated / 28 contaminated tracks | +0.030 / −0.051 | r | RV-002 recomputation (not yet a registered artifact) |
| Δr, TF-cluster bootstrap CI | [+0.004, +0.028] | r | Qing |
| Δr, cell-line-cluster bootstrap CI (includes 0) | [−0.021, +0.038] | r | `evaluation_evidence/cluster_robust_stats.csv` (n_clusters=10; handling of 20 unannotated tracks unchecked) |

**Do not quote these numbers:**
- The 0.6104 / 0.5022 "no augmentation" ablation and the derived +28% / +9%. They have no CSV, JSON or log source (coordinator grep confirmed).
- The validation filtered-Borzoi statistics (r 0.625 etc.). Their source directory is missing.
- "42 same-experiment tracks" (description-based) or "24 contaminated". Use "≥28 (24 GSM + 4 ENCSR)".
- Never mix native-grid 0.663 with shared-grid 0.682.

## 7. Unfinished experiments
- No variant-effect analysis of any kind.
- No supervised (ridge or linear-probe) baseline.
- No ablation of any recipe component.
- No seeds or folds.
- Enformer/AlphaGenome never re-run with a non-oracle mapping.
- `compare_transfer_*.py` and `pi_update_presentation.py` point to missing directories (`results_full_2`, `results_comparison*`), so they cannot be re-run as committed.

## 8. Scientific limitations
- **Leakage**:
  - ≥28 contaminated tracks (24 LoVo GSM + 4 ENCODE colon ENCSR; lower bound pending ENCODE accession lookup, Qing A19).
  - Backbone sequence exposure on chr9/22: the pretrained models' held-out-chromosome conventions are UNVERIFIED.
  - Test results were viewed during development.
- **Design**: oracle track selection; asymmetric calibration; a single split and seed.
- **Non-independence**: 77 TFs across 10 named cell lines + 20 unannotated colon tracks.
- **Mismatch**: cell types; resolution and context window.
- **Metrics**: the 51 zero-signal tiles inflate scores (contrary to `results_benchmark/README.md`); pooled Pearson.
- **SNP arm**: wrong-allele rows, LD between variants in a locus, and multiple testing across scorers × tracks.

## 9. Manuscript progress
`research_paper_draft.md` / `manuscript.md` are a full skeleton.
- **Supported**: the test-split TL-vs-filtered-Borzoi statistics. The 149-target subset was recomputed and matches, but it is outcome-conditioned.
- **Unsupported or wrong**:
  - the augmentation ablation;
  - "inverse-signal loss weighting" (not used);
  - "three cell lines" (it is 10 named + 20 unannotated);
  - low-variance tracks "automatically down-weighted" (`research_paper_draft.md:95`; `low_variance_flag` False for all 183, nothing down-weighted);
  - "outperforms … on held-out validation and test" (`manuscript.md:55`; validation baseline untraceable);
  - "consistent, converging evidence" from the internal ablation (`manuscript.md:107`; ablation untraceable);
  - trunk-tap choice implied to be performance-driven (`manuscript.md:32`; no tap ablation);
  - validation paired t 0.00017 / Wilcoxon 1.1e-9 (`build_presentation.py:517`; no source);
  - `results_benchmark/README.md` claim that zero-signal tiles "deflate" scores (they inflate);
  - stale Enformer text;
  - REST Δ −0.31 is not in the CSV.
- **Missing**: leakage analysis, R² work, limitations.

## 10. Strategy: most important remaining experiments (strategist, ranked)
1. **E1 Variant identity (gate)**: recover GWAS-reported alleles, risk allele and signal from Chen 2024 Supp Data 6. CPU, <1 h.
2. **E2 MPRA validation**: AlphaGenome pre-specified score vs Law et al., *Nat Genet* 56:2104 (2024) MPRA allelic hits.
   - Tests are within locus, with locus-clustered bootstrap and a direction sign test. CPU, existing cache.
   - If null, drop the variant-prioritisation claim.
3. **E3 Borzoi-TL / pretrained-Borzoi / Enformer variant scorer**, then E2 on those scores. About 6–10 A100-h.
4. **E4 Supervised ridge / linear-probe baseline** on pretrained outputs. If ridge ≥ TL, the "transfer learning helps" claim is withdrawn.
5. **E5 Robustness**: cache trunk embeddings, then ≥3 seeds, chromosome-fold CV and a real no-RC ablation (replaces the untraceable ablation). About 8 A100-h; check disk first.
6. **E6 GTEx colon eQTL direction concordance** (supporting evidence). CPU.
7. **E7 Non-oracle Enformer / AlphaGenome baselines**, required before any ranking claim. Minutes.
8. Optional **E8**: cross-cell-line transfer.

### Methodological risks (most severe first)
1. Wrong allele / wrong direction in the SNP input.
2. LD confounding (only about 143 independent regions).
3. Vocabulary / track-identity confounding of the TL gain.
4. Test-set exposure and a single seed.
5. Multiple testing (19 scorers × about 5,000 tracks).
6. Validation data overlapping pretrained-model training data (low risk for MPRA/eQTL, but must be stated).
7. Cell-type mismatch and tiny RNA effects.
8. Oracle baselines and pooled Pearson.

### Novelty and manuscript direction
- The method has limited novelty. Li et al., *PLOS Genet* (2026), doi 10.1371/journal.pgen.1012145, already fine-tuned Enformer on breast/prostate TF ChIP.
- **Recommended Framing B, conditional on E2**: multi-model functional annotation of CRC credible causal variants, validated against colon MPRA and eQTL, with Borzoi-TL as one benchmarked component.
- **Fallback Framing A**: a leakage-aware benchmark of transfer learning for CRC TF binding, with modest, vocabulary-dependent gains.
- Proposed decisions SD-003 to SD-010 are in `SCIENTIFIC_DECISIONS.md` (status PROPOSED).

### Sources located by agents
Not independently re-opened by the coordinator.
- Chen 2024 (doi 10.1038/s41467-024-47399-x, PMC11053150)
- Law 2024 (doi 10.1038/s41588-024-01900-w, PMC11525171)
- Li 2026 (doi 10.1371/journal.pgen.1012145)
- AlphaGenome (doi 10.1038/s41586-025-10014-0)
- Borzoi (doi 10.1038/s41588-024-02053-6)
- From search results only: the Borzoi PEFT paper (doi 10.1186/s13059-026-03935-0) and Sasse 2023 (doi 10.1038/s41588-023-01574-w).

## Open items before Phase 2
- Scientific-reviewer gate: RV-002 REVISE → corrections applied; RV-003 PASS (2026-10-19). Phase 1 accepted.
- User decision on SD-003 to SD-010 and the manuscript framing.
- BL-001 / BL-002: retrieve Chen Supp Data 6 and the Law MPRA table.
- BL-003: GPU authorization.
- BL-004 ENCODE accession lookup (final contamination count); BL-005 Borzoi/Enformer fold BEDs; BL-006 GTEx v8 colon eQTL access.
