# Execution Log

Chronological log of sessions, commands run, and outcomes. Newest at the bottom.

## 2026-10-19 · Setup
- Created branch `research/crc-autonomous-study` from `research/crc-manuscript-study` (`ff88be2`).
- Created `.claude/settings.json` (model opus, empty attribution, force-push/main-push deny rules).
- Created 4 subagents in `.claude/agents/`, `CLAUDE.md`, and the `research_management/` state files.
- Verified: Enterprise OAuth account (Stanford University, claude_enterprise), no API-key env vars,
  no managed model restrictions.
- No experiments or GPU jobs run.

## 2026-10-19 · Phase 1 — discovery and inventory
- Native subagents not loadable in this session (created after session start); used general-purpose agents
  with the agent definitions verbatim and the same models (investigator → sonnet, strategist → opus).
- research-investigator wrote `PHASE1_INVENTORY.md` (CPU reads only; no code/results modified).
- Coordinator spot-checks (all passed): experimental_branch == research/crc-manuscript-study == ff88be2;
  0.6104 absent from all CSV/JSON/.out files (present only in build_presentation.py / draft);
  variant_run_status.tsv 10,148 SUCCESS and ref == hg38_ref for all rows; history.json val-loss minimum at
  epoch 5 (0.185226).
- Registry: EXP-001..EXP-004 imported from pre-existing work.
- No GPU jobs, no experiments run.
- 2026-10-19 · RV-002 scientific-reviewer (opus) Phase 1 gate: CPU-only read-only recomputation (scratch scripts refchk.py,
  recomp.py, delta.py in session scratchpad, not committed). No GPU, no new experiments. Verdict REVISE; doc corrections applied.

## 2026-10-20 · Overnight autonomous session (Phase 2 start)
- Branch verified `research/crc-autonomous-study` at c91806f. Slurm job 14717985 (A100-80GB, 8 CPU, 64G), ~22 h left at start.
- User approved SD-003..SD-010 (with RV-002 modifications), SD-011 framing B→A fallback, SD-012 E8, SD-013 CLAUDE.md note; BL-003 resolved.
- Dispatched (native subagents, parallel, independent): research-investigator ×3 (E1 provenance; validation availability BL-002/006;
  training-exposure audit BL-004/005/E8); computational-engineer (E3 scoring spec + Borzoi/TL/Enformer scorers + GPU run).
- 2026-10-20 01:04 · E2 UNBLIND look #1 (interim, AlphaGenome only): ./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --models alphagenome_quantile --out results_phase2/e2_validation/interim_alphagenome (RV-005 permitted)
  Interim AlphaGenome result (unadjusted, Holm deferred): P1 stratified AUROC 0.666 [0.590, 0.759], perm p 1e-4 (58 regions);
  distance baseline 0.515; Chen annotation-count baseline 0.618. P2 direction concordance 0.311 [0.232, 0.395] (n=132) — significantly
  BELOW 0.5. Post-unblinding observation: the ST4 logFC sign convention is ambiguous (MPRAnalyze factor-level ordering; rs6983267 ST4
  REF G/ALT T logFC +0.16). P2 will NOT be re-signed post hoc; a model-independent convention check was dispatched
  (external_data/validation/MPRA_SIGN_CONVENTION.md). Any re-interpretation will be labelled post hoc.
- 2026-10-20 02:36 · E2 UNBLIND look #2 (interim, Enformer only; frozen run finalised): run_e2.py --unblind --models enformer --out results_phase2/e2_validation/interim_enformer
  Interim Enformer (unadjusted, Holm deferred): P1 0.674 [0.607, 0.756] perm p 1e-4; P2 0.364 [0.276, 0.454] (n=132), again below 0.5.
  Post-hoc observation (not a test): two models independently anti-concordant with ST4 logFC under the ALT/REF reading — consistent with
  logFC = log(REF/ALT) but does not resolve BL-007 (model-derived evidence; SD-017 unchanged). Held-out-chromosome subset P1 ~0.55 (10 regions, descriptive).
- 2026-10-20 ~03:15 · Home quota exceeded → Borzoi scoring (PID 3442888, 4/20 chunks saved) and cache build died. 03:45 moved
  results_phase2/{trunk_cache,e7_nonoracle} to /nobackup/user/yejz1/crc_phase2/ (symlinks). 03:48 Borzoi scoring resumed (PID 3611891).
- 2026-10-20 ~05:00 · EXP-012 label-free concordance (engineer, CPU, no labels read): AG quantile vs Enformer Spearman primary 0.577
  [0.554, 0.599] (matched 5,729; region bootstrap 2,000); sign agreement 0.740. Borzoi to be added with `--models` after summary.tsv.
- 2026-10-20 ~05:05 · Provenance (investigator): Enformer targets byte-identical to calico/basenji cross2020 targets_human.txt; Borzoi
  targets row-identical to seqnn-share/borzoi/hg38/targets.txt; bigwigs Cistrome-DB style (DCid), source unrecorded; REPRODUCE.md updated.
  hg38.ml.fa md5 check vs basenji_barnyard2/hg38.ml.fa.gz (streamed, nobackup/crc_phase2/provcheck) running.
- 2026-10-20 06:07 CDT · Slim trunk cache complete (160.5 GB nobackup). Downstream queue started. Gate: original TL checkpoint re-evaluated
  from the fp16 slim cache gives test r 0.68199 (= 0.6820); pretrained val-selected 0.6609 (= EXP-006). E4 (EXP-013): ridge on pretrained
  outputs pre_raw union1888 test r 0.7426 (λ=1e-4 from val), probe on frozen trunk 0.7223; TL − ridge −0.061 [−0.066, −0.053] (tile-block).
  Pre-registered SD-008 rule triggered: CRC-specific adaptation claim not supported beyond a supervised linear readout. E5 seeds running.
- 2026-10-20 08:35 CDT · queue_downstream2.sh finished (all E4/E5/E8 + fold seeds 2–3). Final summaries: 9 fold×seed heads; TL − ridge
  (152 primary) F2 −0.062/−0.062/−0.062, F3 −0.073/−0.072/−0.077, F4 −0.063/−0.062/−0.062 (tile-block CIs exclude 0). Fig 6 regenerated.
- 2026-10-20 10:55 CDT · EXP-008 Borzoi/TL scoring finalised (10,148 rows, 0 NaN, ref_match 10,148/10,148). E2 FINAL UNBLIND look (all 4 models; Borzoi pretrained + TL unblinded for the first time): ./torch-venv/bin/python analysis/e2_validation/run_e2.py --unblind --out results_phase2/e2_validation/final
  E2 FINAL (EXP-017): P1 AUROC AG 0.666 [0.590, 0.759], Borzoi 0.640 [0.554, 0.744], TL 0.661 [0.584, 0.768], Enformer 0.674 [0.607, 0.756];
  Holm p 0.0004 for all; every model beats distance baseline (Δ CI excludes 0). §7 → Framing B. Not better than annotation count (Δ CIs include 0).
  P2 0.27–0.36 (all below 0.5; SD-017 uninterpretable). Fig 5 regenerated (FINAL); EXP-012 re-run with Borzoi models
  (Enformer–Borzoi pretrained Spearman 0.80; TL vs others 0.50–0.58).

## 2026-10-20 ~21:15 CDT — PI update deck (2026-10-21 meeting)
- Built 11-slide deck + speaker script + source audit from verified outputs (no new analyses, no GPU). python-pptx (scratch venv); PDF via LibreOffice in an apptainer image (scratch).
- scientific-reviewer (Opus): REVISE (0 numeric errors; 4 major + 12 minor wording) → all applied → PASS. Tracked copies in `presentations/2026-10-21_PI_update/` (reports/ is gitignored).

## 2026-10-23 22:00 – 2026-10-25 CDT — Borzoi performance-improvement program (FT; details: research_management/FINETUNING_PROGRESS.md, finetuning/FT_REGISTRY.csv)
- Infra: checkpoint/resume library (atomic, 2 recent + best, 30-min wall-clock), deadline supervisor independent of the Claude session (setsid, PID in finetuning/supervisor.pid), run queue. Profiling FT-P01..P12.
- Cache experiments FT-C01..C85 (validation only): best ridge FT-C61 val 0.7392 / 0.7370; heads up to 0.7404.
- Fine-tuning runs (replicate-specific Borzoi trunks; validation only): frozen S01/S07, partial S02/S03/S10/S11/S12, LoRA S04, full S05, replicates R1-R3 (partial) and L1-L3 (LoRA), exploration E1-E5. Blends B01..B32.
- Review RV-FT-01 (Opus) REVISE → corrections → TEST_LOCK.csv (6 configs) committed at dc0bcd5 BEFORE any FT test evaluation.
- Locked test evaluations (one each; FT-T01..T06): predict_ft.py --split test --config-id <id>; test_eval_cache.py --config-id FT-C61.
  PRIMARY ENS 8 fine-tuned members (partial-4 + LoRA-r16, replicates 0-3), RC TTA: test r 0.7586 (183) / 0.7578 (152) vs E4 ridge 0.7426 / 0.7406;
  delta (152) +0.0173, TF-cluster 95% CI [0.0123, 0.0252], two-way tile-block x TF [0.0111, 0.0263] → pre-registered claim rule met.

## 2026-10-26 19:10 – 19:45 CDT — Final VM sprint (EXP-020/021/022; no training)
- Phase 1 audit (EXP-022): weight_change_check.py (partial-4 trunk tensors all changed, median rel 0.02; LoRA 53 non-zero lora_B), verify_ckpts.py (8/8 sha match), lock timing dc0bcd5 23:22:50 < first test 23:31. Report reports/FINETUNING_VALIDATION_AUDIT.md.
- Phase 2 ablation (EXP-020): `ablation.py select` (validation only) → ABLATION_LOCK.csv committed 9aa33aa (19:16) → `ablation.py test-preds` (GPU, 8 members x ~60 s, RC TTA) → `ablation.py test-eval` (17 configs, one each; FT-R1 single re-run after an interrupted, empty first attempt) → ablation_report.py. Primary reproduced exactly (0.7578 / +0.0173). FT-L2 alone 0.6297 on test: over-prediction on chr9:20-32 Mb (ftl2_diag.py).
- Phase 3 (EXP-021): bio_analysis.py (computational-engineer subagent), reports/FINETUNING_BIOLOGICAL_ANALYSIS.md, reports/ft_bio/.
- Phase 4 figures: figures_ft_sprint/ (5 figures, PNG 300 dpi + PDF; plotted values in figures_ft_sprint/data/).
- Phase 5: manuscript_v2 terminology bullet (2.8c), 3.7.6, 3.7.7, Discussion 3a(v), Limitations 28-29, Fig. 6-8 entries. Review RV-FT-04 (Opus).
- Review RV-FT-04 (Opus): REVISE (1 major: 2-member-ensemble overclaim; 12 minor) → corrected → PASS. Checkpoints verified (8/8 sha). Home 147.6/186.3 GB.
