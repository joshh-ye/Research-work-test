# Phase Acceptance Criteria

A phase is accepted only when every box is checked, a scientific-reviewer PASS is recorded in
`REVIEW_FINDINGS.md`, and the milestone is committed and pushed to `research/crc-autonomous-study`.

## Phase 1 — Repository discovery and research inventory · status: **ACCEPTED 2026-10-19** (RV-002 REVISE → corrections → RV-003 PASS)
- [x] Every existing result file mapped to its producing script and commit (or flagged untraceable). (Enformer/AG producing runs flagged unrecorded)
- [x] Genome build confirmed for each dataset, model, and the SNP inputs. (hg38 verified: 0 REF mismatches 1-based; GRCh37 origin of SNP input INFERRED, remap undocumented — RV-002 F5)
- [x] Train/val/test splits and pretrained-model training overlap documented. (contamination corrected to ≥28 per RV-002 F1; pretrained held-out-chromosome conventions UNVERIFIED — fold BEDs not in repo)
- [x] Unfinished work and inconsistencies listed.
- [x] `PROJECT_STATE.md` and `EXPERIMENT_REGISTRY.csv` updated.
- [x] Strategist recommendations recorded in `SCIENTIFIC_DECISIONS.md` (as PROPOSED).
- [x] Reviewer PASS. (RV-003, 2026-10-19)

## Phase 2 — SNP experiments and model analysis · status: NOT STARTED
- [ ] REF alleles verified against the reference FASTA; mismatch counts reported.
- [ ] Coordinate convention (0/1-based) and variant centering tested.
- [ ] Track mappings recorded for every model used.
- [ ] Every run logged in the registry with command and output path.
- [ ] Tests for allele/coordinate/metric handling pass.
- [ ] Reviewer PASS per experiment.

## Phase 3 — Statistical analysis and publication figures · status: NOT STARTED
- [ ] Uncertainty (CI/bootstrap over appropriate units) for every headline number.
- [ ] Paired model comparisons; multiple-testing control where applicable.
- [ ] Contaminated vs uncontaminated tracks stratified.
- [ ] r / r² / R² labelled correctly in all tables and figures.
- [ ] Each figure has a script, input data, and saved plotted values.
- [ ] Reviewer PASS.

## Phase 4 — Manuscript preparation · status: ACCEPTED 2026-10-20 (RV-008 PASS; strategist REVISE → SD-025..028 → strategist PASS; pre-submission items open: Fig. 1, Suppl. Methods S1, intro prose, title wording)
- [ ] Every numeric claim traces to a registry entry.
- [ ] Every reference verified (DOI/PMID resolved).
- [ ] Limitations section covers leakage, calibration, cell-type mismatch, validation gaps.
- [ ] Independent-validation status stated for each key claim.
- [ ] Strategist and reviewer PASS.

## Phase 5 — Reproducibility and GitHub consolidation · status: ACCEPTED 2026-10-20 (RV-008 REVISE → RV-009 PASS; BL-009 user decisions open)
- [ ] Environment specs (requirements/lockfiles) for each venv.
- [ ] One-command (or documented) reproduction of headline results from saved artifacts.
- [ ] Clean-room rerun of CPU-level analyses matches recorded numbers.
- [ ] README updated; no large data or secrets in git.
- [ ] Reviewer PASS; final push to the dedicated branch.
