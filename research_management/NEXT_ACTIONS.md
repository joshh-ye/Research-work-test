# Next Actions

Ordered; the coordinator picks from the top.

## Phase 1 — Repository discovery and research inventory
1. research-investigator: inventory all scripts, Slurm logs, result dirs, and branches
   (`main`, `stable-version`, `experimental_branch`, `experimentational_fix`) → map each result file to
   the script/commit that produced it.
2. research-investigator: dataset provenance — accessions, genome builds, train/val/test chromosomes,
   overlap with pretrained Borzoi/Enformer/AlphaGenome training data.
3. research-investigator: audit `alphagenome_variant_effects/` (inputs, genome build, REF/ALT checks,
   completion status of the S6_CCV batch, outputs).
4. research-investigator: review `research_paper_draft.md` and `research_paper_prep/`; list claims and
   whether each is supported by a traceable result.
5. Coordinator: consolidate into `PROJECT_STATE.md` and `EXPERIMENT_REGISTRY.csv`.
6. research-strategist: propose refined questions and ranked Phase 2 experiments.
7. scientific-reviewer: review the Phase 1 inventory → PASS before Phase 2.
