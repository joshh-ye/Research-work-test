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
