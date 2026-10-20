# Project State

Last updated: 2026-10-20 08:15 CDT (overnight autonomous session; Slurm job 14717985, A100, ends ~22:00 CDT)

## Current phase
**Phase 2 — IN PROGRESS** (E2 final 4-model MPRA run pending Borzoi/TL variant scores). Phases 3–5 partially done in parallel
(figures 1–6, ST1–ST4, manuscript v2 track-benchmark sections, REPRODUCE.md + CPU clean room).
Phase 1 — ACCEPTED 2026-10-19 (RV-003 PASS). Full evidence: `PHASE1_INVENTORY.md`.

## Active jobs (check with `ps`; all resumable / skip-if-done)
- EXP-008 Borzoi + Borzoi-TL variant scoring: `variant_scoring/score_borzoi.py` (PID 3611891), chunks in
  `results_phase2/variant_scores/borzoi/chunks/` (12/20 at 08:10; ~36 min/chunk). Resume: `nohup ./torch-venv/bin/python variant_scoring/score_borzoi.py`.
  When `results_phase2/variant_scores/borzoi/summary.tsv` exists → `./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --out results_phase2/e2_validation/final`.
- `analysis/priorityD/queue_downstream2.sh` (PID 3621911): fold-seed extras (F2–F4 seeds 2–3); regenerates summaries at end.
- Storage: home 172.6/186.3 GB soft; nobackup 160.5 GB (trunk cache 160 GB + e7 outputs symlinked from `results_phase2/`). Check `accre_storage`.

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
- E2 interim (EXP-011; unadjusted, Holm deferred): P1 within-locus AUROC AlphaGenome 0.666 [0.590, 0.759], Enformer 0.674 [0.607, 0.756];
  distance baseline 0.515; annotation count 0.618. P2 direction uninterpretable (SD-017, BL-007).
- EXP-012 (label-free): AlphaGenome vs Enformer Spearman 0.577 [0.554, 0.599] (matched set).

## Framing
Decided by `variant_scoring/E2_ANALYSIS_PLAN.md` §7 at the E2 final run (SD-011, SD-020). Both titles fixed in SD-020. Track benchmark is a
negative result under either framing.

## Reviews
RV-004 (E2 design) REVISE → A1; RV-005 REVISE → A3; RV-006 (Priority D) REVISE → corrected (re-review folded into E2-final review).

## Do not quote (untraceable / conflicting)
- 0.6104 / 0.5022 "no augmentation" ablation and derived +28%/+9%; validation filtered-Borzoi stats (r 0.625); "42 same-experiment" / "24 contaminated" (use ≥31).
- Never mix native-grid 0.663 with shared-grid 0.682.

## Provenance status (REPRODUCE.md)
hg38.ml.fa md5 c2cc5a20… = Calico basenji_barnyard2 hg38.ml.fa.gz (verified 2026-10-20); Enformer targets byte-identical to calico/basenji
cross2020; Borzoi targets row-identical to seqnn-share; bigwigs Cistrome-DB style, download source UNRECORDED (ask PI/Qing).
