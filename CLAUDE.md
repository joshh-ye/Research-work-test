# CRC Genomics Research — Project Instructions

Colorectal cancer (CRC) genomics: pretrained Borzoi, fine-tuned Borzoi (Borzoi-TL, CRC TF ChIP tracks),
Enformer, AlphaGenome, and SNP/variant-effect analysis.

## Session start (every session)
1. `git branch --show-current` must print `research/crc-autonomous-study`. If not, stop and switch
   (`git switch research/crc-autonomous-study`) before modifying any file.
2. Read `research_management/PROJECT_STATE.md`, `NEXT_ACTIONS.md`, and `BLOCKERS.md`. Read other
   state files only when the task needs them.
3. Continue from `NEXT_ACTIONS.md`; do not re-derive what the state files already record.

## Account and model constraints
- Use only the signed-in university Claude Enterprise account. Do not switch accounts, do not set or use
  external API keys (`ANTHROPIC_API_KEY`, Bedrock, Vertex), and do not use Fable models.
- Do not attempt to bypass Enterprise usage limits.
- The main session is the **Opus coordinator**. Subagents (`.claude/agents/`):
  | Agent | Model | Role |
  |---|---|---|
  | research-investigator | sonnet | history, provenance, existing results, literature — evidence only |
  | computational-engineer | sonnet | code, inference, SNP analysis, tests, stats, figures, Slurm |
  | scientific-reviewer | opus | independent review → PASS / REVISE / BLOCKED (read-only) |
  | research-strategist | opus | novelty, design, experiment priority, manuscript direction (read-only) |

## Usage management
- Routine work → Sonnet subagents. Opus subagents only at major checkpoints and phase boundaries;
  never for trivial changes (typos, formatting, state-file updates, small refactors).
- Avoid unnecessary parallel subagents; run in parallel only when tasks are truly independent.
- Avoid redundant file reads and repeated experiments: check `EXPERIMENT_REGISTRY.csv` before running anything.
- Do not repeat Qing's completed analysis (`reports/qing_deliverables/`) unless the justification is
  recorded in `SCIENTIFIC_DECISIONS.md`.

## Git rules
- All work happens on `research/crc-autonomous-study`. Never push to `main`/`master`. Never force-push.
- Preserve the configured Git author identity; do not change `user.name`/`user.email`.
- Commit messages contain **no** AI co-author trailers (no `Co-Authored-By: Claude` or equivalent).
  `.claude/settings.json` sets `attribution.commit`/`attribution.pr` to empty strings.
- Commit verified milestones only, then `git push origin research/crc-autonomous-study`.
- Never stage large data: `alphagenome_variant_effects/` (≈26 GB untracked), `*.tar*`, checkpoints,
  `results*/`, `reports/`, venvs. Stage files explicitly by path, never `git add -A`.

## Scientific requirements (non-negotiable)
- Correct, explicit genome build (hg19 vs hg38) end to end; liftover documented if used.
- SNP REF/ALT alleles verified against the reference FASTA; 0- vs 1-based coordinates and strand handled explicitly.
- Consistent genomic coordinates across variants, tracks, model windows, and bins.
- Correct model output → track/assay/cell-type mappings, with the mapping file recorded.
- Training/test leakage assessed, including test experiments that were training data of pretrained
  models (≥31 of 183 tracks are identical Borzoi/Enformer training experiments: 24 LoVo GSE51142 by GSM (idx 13–36),
  3 HCT-116 ENCODE (idx 78/81/93: ENCSR000DTO/BML/BSA) and 4 colon-tissue ENCSR (idx 145–148); 0-based; lower bound —
  only HCT116/Caco-2 ENCODE xrefs searched; AlphaGenome identity unverifiable). Tracks = 10 resolved cell lines
  (HCT-116 105, LoVo 41, …) + 4 colon tissue + 1 tumour; see `exposure_audit/track_exposure.tsv`. Project test
  chromosomes chr9/chr22 are largely in Borzoi/Enformer *training* folds (`exposure_audit/EXPOSURE_AUDIT.md`).
- Statistical uncertainty reported (CIs, bootstrap over appropriate units, paired tests, multiple-testing control).
- Independent validation where a claim requires it.
- Always distinguish **Pearson r**, **squared Pearson r (r²)**, and **prediction R² = 1 − SS_res/SS_tot**.
  Baseline R² from calibrated pretrained outputs must be labelled "validation-calibrated R²".
- No fabricated results. No invented references. Untraceable numbers are not quoted.
- Limitations are documented explicitly.

## Research phases
1. Repository discovery and research inventory
2. SNP experiments and model analysis
3. Statistical analysis and publication figures
4. Manuscript preparation
5. Reproducibility and GitHub consolidation

Acceptance criteria per phase: `research_management/PHASE_ACCEPTANCE.md`.

## Checkpoint protocol (major checkpoints)
Implementation → Testing → Independent Opus review (scientific-reviewer) → Corrections →
Verification → Documentation (state files) → Git commit → Next task.

- Record every review verdict in `REVIEW_FINDINGS.md`; a phase closes only after a PASS.
- BLOCKED → record in `BLOCKERS.md` and ask the user.
- Log every executed run in `EXECUTION_LOG.md` and `EXPERIMENT_REGISTRY.csv`.

## Environment notes
- `./torch-venv/bin/python` for PyTorch code; install with `./torch-venv/bin/python -m pip` (never bare `pip`).
- `./tf-venv` for TensorFlow/Enformer; `ag-venv` (AlphaGenome hosted API, needs key file) or
  `alphagenome_variant_effects/ag-local-venv` (local weights, needs HF token). See
  `alphagenome_variant_effects/README.md`.
- The interactive session already holds an A100: run `.slurm` scripts with `bash`, not `sbatch`.
- GPU runs require an explicit go-ahead in the current task.
- Authoritative Borzoi-TL artifacts: `borzoi_ckpt_full/model_head_best.pt`, `results_full/test_{preds,targets}.npy`,
  `results_benchmark/benchmark_comparison_test.csv`, `reports/qing_deliverables/`.
