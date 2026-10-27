# Project State

Last updated: 2026-10-20 ~12:40 CDT (overnight autonomous session; Slurm job 14717985, A100, ends ~22:00 CDT)

## Update 2026-10-26 ~19:40 CDT — final VM sprint (see research_management/FINAL_VM_HANDOFF.md)
- **EXP-019 (locked)**: the primary 8-member FT ensemble reaches 0.7578 vs 0.7406 for the E4 ridge (152 tracks), Δ +0.0173, TF [0.0123, 0.0252], two-way [0.0111, 0.0263]. Audit EXP-022 found no leakage. The partial-4 arm changed its trunk weights; the LoRA arm left trunk weights frozen and trained adapters only. All 8 checkpoints match their SHA records.
- **EXP-020 (post hoc ablation)**: selection was locked in 9aa33aa before test inference.
  - 7/8 single members beat the ridge.
  - Validation-selected 3–6-member ensembles without FT-L2 reach 0.7607–0.7610; the 2-member ensemble reaches 0.7586.
  - **FT-L2 fails on test chr9:20–32 Mb** (0.6297 alone), with no validation signal.
- **EXP-021**: gains concentrate in tracks with a low ridge baseline or sparse signal (post hoc).
- Manuscript v2 was updated: terminology, 3.7.6, 3.7.7, Discussion 3a(v), Limitations 28–29, Figs 6–8 in `figures_ft_sprint/`.
- Reviews: RV-FT-04 REVISE → corrected → PASS.

## Current phase
**All five phases ACCEPTED 2026-10-20** (Phase 1 RV-003; Phase 2/3 RV-010; Phase 4 RV-008 + strategist PASS; Phase 5 RV-009).
Framing B adopted by the pre-registered rule (SD-024). Open user decisions: BL-007 (MPRA sign convention), BL-009 (repo hygiene), SD-028 (title term).
Pre-submission items: Fig. 1 schematic, Supplementary Methods S1 (BigWig processing), Introduction prose. Manuscript not shared externally.

## Active jobs
None. All GPU work finished 2026-10-20 10:27. Trunk cache (160 GB, regenerable) remains on /nobackup/user/yejz1/crc_phase2/ (symlinked).
Storage: home 174.0/186.3 GB soft. Check `accre_storage`.

## Verified results (test chr9+chr22, 359 tiles, 183 tracks, shared 128 bp grid, hg38; Pearson r unless stated)
- Borzoi-TL original ckpt 0.6820 (r² 0.4866, prediction R² 0.4810); cache-recipe seeds 1–5 0.6819–0.6837 (SD 0.0006 over 1–3).
- Pretrained Borzoi val-selected 0.6609 (validation-calibrated R² 0.444); oracle 0.6640.
- **E4 (EXP-013/014, RV-006 REVISE → corrected):** ridge[pre_raw:union1888] 0.7426, trunk linear probe 0.7223. Primary (SD-006, TF-cluster,
  152 non-identical): TL − ridge −0.0555 [−0.0756, −0.0428]; TL − probe −0.0378 [−0.0525, −0.0289]. Robust over seeds and folds F2/F3/F4
  (TL − ridge −0.062…−0.083 TF-cluster CIs exclude 0), exposure-controlled ridge (0.7420), Borzoi-held-out-fold tiles (−0.068).
  **SD-019: SD-008 rule triggered → no TL-superiority / CRC-adaptation claim.**
- TL − pretrained valsel +0.021 (all183) — cell-line-cluster CI [−0.018, 0.041] includes 0.
- E7 non-oracle: Enformer 0.6406, AlphaGenome 0.6569 (EXP-010).
- E8 (EXP-016): held-out cell lines GP5d/LS174T/LS180/DLD-1 (17 tracks): TL same-TF transfer 0.465 vs pretrained valsel 0.514 → "no transfer advantage" (≈3 donor clusters; descriptive).
- Exposure audit (EXP-005): ≥31 identical experiments; test tiles 43.6% bp in Borzoi training folds.
- E1 (EXP-007): 5,729 allele-matched CCVs (GWAS Catalog GCST90129505 alleles; SD-015).
- **E2 FINAL (EXP-017, RV-007 PASS):** within-locus AUROC Enformer 0.674 [0.607, 0.756], AlphaGenome 0.666 [0.590, 0.759], Borzoi-TL 0.661
  [0.584, 0.768], Borzoi 0.640 [0.554, 0.744]; Holm p 0.0004 (permutation floor); distance 0.515; annotation count 0.618 (not beaten; no model
  differences). P2 0.27–0.36, uninterpretable (SD-017, BL-007).
- EXP-012 (label-free): AlphaGenome vs Enformer Spearman 0.577 [0.554, 0.599] (matched set).

## Framing
**Framing B** (SD-024, §7 rule met by all 4 models). Track benchmark reported as a negative result (SD-019). Manuscript structure SD-025.

## Reviews
RV-004 (E2 design) REVISE → A1; RV-005 REVISE → A3; RV-006 (Priority D) REVISE → corrected; RV-007 PASS (E2 final); RV-008 Phase 4 PASS / Phase 5 REVISE → RV-009 PASS; RV-010 Phase 2/3 PASS; strategist Phase 4 PASS.

## Do not quote (untraceable / conflicting)
- 0.6104 / 0.5022 "no augmentation" ablation and derived +28%/+9%; validation filtered-Borzoi stats (r 0.625); "42 same-experiment" / "24 contaminated" (use ≥31).
- Never mix native-grid 0.663 with shared-grid 0.682.

## Provenance status (REPRODUCE.md)
hg38.ml.fa md5 c2cc5a20… = Calico basenji_barnyard2 hg38.ml.fa.gz (verified 2026-10-20); Enformer targets byte-identical to calico/basenji
cross2020; Borzoi targets row-identical to seqnn-share; bigwigs Cistrome-DB style, download source UNRECORDED (ask PI/Qing).
