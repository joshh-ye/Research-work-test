# Blockers

Format: ID · date opened · description · needed from (user / data / compute) · status.

- **BL-001 · 2026-10-19** · Chen 2024 Supplementary Data 6 not yet retrieved (one mirror returned 403); needed to
  recover GWAS-reported alleles for the 10,148-row variant input. Needed from: data (Nature/PMC supplement download, or user copy). Status: open — blocks SNP analysis (E2, E3).
- **BL-002 · 2026-10-19** · Law 2024 MPRA per-variant table (Supp Table 4) public availability unconfirmed; raw data
  on EGA (controlled access). Needed from: data / possibly user. Status: open — blocks E2 primary validation.
- **BL-003 · 2026-10-19** · Phase 2 GPU work (E3, E4/E5, E7) requires explicit user authorization. Status: open.

## Notes (not blocking)
- AlphaGenome hosted-API runs need the existing key file; local runs need an HF token. No new credentials will be
  created or requested by agents.
- Disk space for the trunk-embedding cache (E5, ~20–90 GB) to be checked before running.
