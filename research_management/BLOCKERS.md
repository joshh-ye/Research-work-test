# Blockers

Format: ID · date opened · description · needed from (user / data / compute) · status.

- **BL-001 · 2026-10-19** · Chen 2024 Supplementary Data 6 not yet retrieved (one mirror returned 403); needed to
  recover GWAS-reported alleles for the 10,148-row variant input. Needed from: data (Nature/PMC supplement download, or user copy). Status: open — blocks SNP analysis (E2, E3).
- **BL-002 · 2026-10-19** · Law 2024 MPRA per-variant table (Supp Table 4) public availability unconfirmed; raw data
  on EGA (controlled access). Needed from: data / possibly user. Status: **partly resolved 2026-10-20** — Law 2024 ST2 (8,880 tested variants, hg19+hg38, per-cell-line 0/1/2 flags, no alleles/effects) and ST4 (275 FDR<1e-3 hits with REF/ALT, logFC vs REF) are open (external_data/validation/law2024_nat_genet/); non-hit effect sizes only on EGA (EGAD50000000596, controlled) → SD-004 hit-vs-non-hit fallback applies. 4,323/5,739 CCV positions tested; 134 in ST4; ST2 flags 433 vs ST4 275 — hit definition to be pre-specified.
- **BL-003 · 2026-10-19** · Phase 2 GPU work requires explicit user authorization. Status: **resolved 2026-10-20** (user authorized A100 use).

- **BL-004 · 2026-10-19** · ENCODE accession lookup (network) needed to finalise the contamination count (Qing A19:
  HCT-116 ENCODE tracks may share experiments with GSM-labelled targets). Current count ≥28 is a lower bound. Needed from: data. Status: **resolved 2026-10-20 (lower bound ≥31)** — 3 HCT-116 GSMs map via ENCODE dbxrefs to ENCSR000DTO/BML/BSA present in both vocabularies (coordinator-verified in targets_human.csv). Remaining: other cell lines' ENCODE xrefs, AlphaGenome manifest (not obtained).
- **BL-005 · 2026-10-19** · Published Borzoi/Enformer fold BEDs are not in the repo, so chr9/chr22 test-tile exposure
  to backbone training cannot be quantified. Status: **resolved 2026-10-20** — test tiles bp fraction in Borzoi train folds 0.436 (test 0.304), Enformer train 0.730 (HF third-party fold file); see exposure_audit/.
- **BL-006 · 2026-10-19** · GTEx colon eQTL access/version for E6. Status: **resolved 2026-10-20** — GTEx v8 + v10 colon transverse/sigmoid significant pairs downloaded (hg38, slope = ALT effect); CEDAR colon credible sets (eQTL Catalogue FTP). Not independent of AlphaGenome/Borzoi training (GTEx RNA-seq).
- **Validation gap (not a blocker, decision needed):** no independent validation is planned for the TL accuracy claim
  (held-out tracks/cell lines absent from pretrained vocabularies, orthogonal ChIP replicates). RV-002 recommends making E8
  required for any generalisation claim, and adding allele-specific binding or colon caQTL as a second readout for Framing B.

## Notes (not blocking)
- AlphaGenome hosted-API runs need the existing key file; local runs need an HF token. No new credentials will be
  created or requested by agents.
- Disk space for the trunk-embedding cache (E5, ~20–90 GB) to be checked before running.
- **BL-007 · 2026-10-20** · Law 2024 MPRA ST4 logFC sign convention (ALT/REF vs REF/ALT) unresolved from public material
  (`external_data/validation/MPRA_SIGN_CONVENTION.md`). Needed from: user — contact the authors (Law/Houlston) or EGA
  EGAD50000000596 documentation (controlled access). Blocks interpretation of the P2 direction test only (SD-017). Status: open.
- **BL-008 · 2026-10-20 ~03:15** · Home quota (232.8 GB hard) hit by the trunk cache (52 GB of a planned ~330 GB) → cache build and Borzoi
  variant scoring died (writes failed). Mitigation: cache + E7 outputs moved to /nobackup/user/yejz1/crc_phase2/ (symlinked); Borzoi
  scoring resumed from 4 saved chunks; cache redesigned to fit (SD-018). Status: mitigated; user should review storage (home ~171 GB used).
