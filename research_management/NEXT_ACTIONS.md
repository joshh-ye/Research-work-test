# Next Actions

Ordered; the coordinator picks from the top. Source: strategist report in `PHASE1_RECOVERY_REPORT.md`.

## Close Phase 1
0. ~~scientific-reviewer gate~~ RV-002 REVISE → RV-003 PASS. **Phase 1 accepted 2026-10-19.**
0b. User: approve/modify proposed SD-003..SD-010 and the Framing B (conditional) direction. RV-002 recommendations per SD are in `SCIENTIFIC_DECISIONS.md`.
0c. User: decide whether E8 (held-out-vocabulary generalisation test) becomes required; approve the CLAUDE.md note change
    "24 LoVo GSE51142 tracks" → "≥28 (24 GSM + 4 ENCSR)" (coordinator does not edit CLAUDE.md without user approval).
0d. Phase 2 prerequisites (CPU/data): BL-004 ENCODE accession lookup; BL-005 fold BEDs; BL-006 GTEx eQTL access; check the
    cell-line-cluster bootstrap handling of the 20 unannotated tracks.

## Phase 2 — SNP experiments and model analysis (none started; GPU not authorized yet)
1. **E1** research-investigator: obtain Chen 2024 Supp Data 6 and Law 2024 Supp Table 4; map rsID/position
   (GRCh37→hg38), GWAS-reported allele, risk allele, signal/lead SNP; confirm S6_CCV identity; MPRA overlap. CPU <1 h. **Gate for all SNP work.**
2. **E2 + E6** computational-engineer: freeze primary score (SD-005), then AlphaGenome scores vs MPRA allelic hits
   (within-locus AUPRC/rank, locus-clustered bootstrap, direction sign test, distance-to-TSS control) and GTEx colon eQTL direction. CPU, existing cache.
3. scientific-reviewer: gate review of E1/E2 design and pre-specification.
4. **E3** computational-engineer: Borzoi-TL + pretrained-Borzoi (+ Enformer) variant scorer (RC-averaged); validate on
   smoke variant and rs6983267; apply E2. GPU ~6–10 A100-h.
5. **E5 → E4** computational-engineer: cache trunk embeddings (check disk, ~20–90 GB) → ridge/linear-probe baseline
   on pretrained outputs → ≥3 seeds, chromosome folds, no-RC ablation. GPU ~8–11 h.
6. **E7** computational-engineer: non-oracle Enformer / AlphaGenome baselines with per-tile stats.
7. scientific-reviewer: stats review of E2–E7; strategist + coordinator choose Framing A vs B
   (rule: E2 locus-clustered enrichment CI excludes null → B).
Optional E8: cross-cell-line transfer test after E5.
