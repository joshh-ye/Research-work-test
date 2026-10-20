# E1 Variant provenance - S6_CCV input vs Chen et al. 2024 (generated 2026-10-20)

Script: `variant_provenance/build_provenance.py` (CPU only; log: `build_provenance.log`). Re-run: `./torch-venv/bin/python -I variant_provenance/build_provenance.py`.

## Sources (raw files in `external_data/chen2024/`, nothing executed from there)
| File | URL | md5 |
|---|---|---|
| `art.html` (article page) | https://www.nature.com/articles/s41467-024-47399-x (DOI 10.1038/s41467-024-47399-x) | not recorded (page, volatile) |
| `MOESM4_ESM.xlsx` = Supplementary Data 1-22 | https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-47399-x/MediaObjects/41467_2024_47399_MOESM4_ESM.xlsx | 16054cae9fe3b269df5fcc8dd223cb1e |
| `MOESM1/2/3/5_ESM.pdf` (Supp Info, peer review, reporting summary, source file; not parsed) | same path, `..._MOESM{1,2,3,5}_ESM.pdf` | 30b55fb9..., 972e6b4e..., 6e244cee..., e701a1bb... (full md5 via `md5sum`) |
| GWAS Catalog GCST90129505 harmonised (GRCh38) | https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90129001-GCST90130000/GCST90129505/harmonised/GCST90129505.h.tsv.gz | 35e40a4c91d58960666616c932a9fd7f (matches EBI md5sum.txt) |
| GWAS Catalog GCST90129505 original GRCh37 | .../GCST90129505/GCST90129505_buildGRCh37.tsv | 1e212d4eb0bba5cb5d3fd2fcd279e3a3 (matches EBI md5sum.txt) |
| UCSC liftover chain | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz | 35887f73fe5e2231656504d1f6430900 |
| Ensembl REST `POST /variation/homo_sapiens` (release 116 reported by /info/data, GRCh38), 29 batches of <=200 ids | https://rest.ensembl.org | cached JSON in `ensembl_rest/` |
| Chen code repo (not used for data) | https://github.com/zhishanchen/CRC_Finemapping | README only |
All downloaded 2026-10-20. The Nature static-content URL returned HTTP 200 on the first attempt (no 403).

Data-availability statement of the paper (verified in article page): GWAS summary statistics at GWAS Catalog GCST90129505. Its metadata lists sample sizes 69,175 + 185,616 = 254,791 = 100,204 cases + 154,587 controls, i.e. the trans-ancestry meta-analysis used for fine-mapping. License: academic, non-commercial (GWAS Catalog README).

## What the source contains (verified)
- Supplementary Data 6 (sheet "Supplementary Data 6", header on row 2): "List of 5,741 CCVs". **Columns that matter: Fine_mapping region, Lead variant of independent risk signal, CCV (rsID), Chr (hg19), Position (hg19)**, then 0/1 flags for epigenomic features (accessible chromatin, actively transcribed genes, distal regulatory elements, active promoters, heterochromatin, repressed regulatory elements, TF binding sites). **No alleles, no PIP, no p-value, no beta in S6.**
- Build: all Chen supplementary tables are labelled "Chr (hg19)"; GWAS Catalog original file is GRCh37 (Variant_id e.g. `10:61331_G/A` described as build 37). Source positions are therefore **hg19/GRCh37**; the S6_CCV input is hg38 (lifted/looked up upstream - method of that upstream conversion is unknown).
- CCV definition (Methods/Results, article page): variants with conditional P within two orders of magnitude of the most significant variant in each independent signal (238 signals, 142 regions). **There is no posterior inclusion probability**; do not describe any column as PIP.
- Allele-bearing tables in the supplement: S3-S5 (lead variants only: effective/reference allele, frequency, beta), S15-S19 (colocalised CCVs only: effect/reference allele, GWAS beta). Effect alleles there are conditional-analysis labels, not risk alleles; they agree with GWAS Catalog effect allele only ~54% (arbitrary orientation), so they are used for consistency checks, not as the allele source.
- Therefore the **allele source for the allele-matched set is GWAS Catalog GCST90129505** (the deposited summary statistics of the same GWAS), not S6. This is a deviation from the "S6 or equivalent" wording of SD-003 and should be accepted explicitly.

## Counts at each step (all from `build_provenance.log`)
| Step | Count |
|---|---|
| S6 rows / unique rsIDs / unique hg19 positions / signals (region, lead) / regions | 5,741 / 5,741 / 5,741 / 238 / 142 |
| S6_CCV input rows / unique rsIDs / unique hg38 positions | 10,148 / 5,739 / 5,739 |
| ALTs per position (1 / 2 / 3) | 2,419 / 2,231 / 1,089 positions (all SNVs) |
| Duplicate keys; positions with >1 rsID; rsIDs with >1 position | 0; 0; 0 |
| Input REF != hg38 FASTA (`borzoi_data/hg38/hg38.ml.fa`) | 0 |
| Input rsIDs not in S6 | 0 |
| S6 rsIDs present in input | 5,739 |
| Ensembl GRCh38 position equals input position (rsID-matched) | 5,739 / 5,739 |
| UCSC liftover(hg19 S6) equals input position | 5,737 / 5,739; the 2 failures (rs35204860, rs34403018, chr17) are *unmapped* by the chain, not discordant; Ensembl confirms the input positions |

### Reconciliation 5,741 vs 5,739 (verified)
S6 has 5,741 distinct CCVs; the input has 5,739. The 2 missing:
- **rs149685934** (S6: chr1:222148219 hg19, GWAS Catalog alleles G/A). Ensembl now lists it as a *synonym of rs12024666* (GRCh38 1:221974877, G/A, hg38 REF G). It is a merged rsID: the input pipeline (rsID-keyed) dropped it. It is an SNV that could be added back using the new rsID.
- **rs1312157506** (S6: chr9:33952629 hg19, GWAS Catalog alleles T/C). Ensembl now maps this rsID to **chrX:33934512** (rsID reassigned), so an rsID-keyed lookup cannot place it on chr9. UCSC liftover of the hg19 position gives 9:33952631 (REF T).
No duplicates exist on either side; the difference is entirely rsID drift (merge/reassignment). Both are listed in `unmatched_or_excluded.tsv` (status `S6_CCV_absent_from_input`).

## Allele matching (method)
For every input row: link to S6 by rsID; link S6 to GWAS Catalog by rsID **and** Chen hg19 position (harmonised file `Variant_id` carries the original build-37 position); require GWAS Catalog GRCh38 position == input position; the row is `is_reported_allele=True` iff {input REF, input ALT} == {GWAS effect_allele, other_allele} (or its complement; complemented count = 0 so no strand flip was needed). Rows at a position whose ALT is not the reported counter-allele get `False`; positions with no GWAS allele data get `NA`. For 4 CCVs absent from the harmonised file, alleles came from the original GRCh37 file by exact hg19 position, accepted only because liftover or Ensembl corroborates the input hg38 position (method label `gwascat_grch37_hg19pos_...`).

| Result | Count |
|---|---|
| Allele-matched input rows = positions (**one ALT per position**) | 5,729 |
| of which palindromic (A/T or C/G) | 822 |
| strand-flip required | 0 |
| via harmonised rsID+position / via original-file fallback | 5,725 / 4 |
| input rows `is_reported_allele=False` (other ALTs at matched positions) | 4,396 |
| input rows `NA` (no allele info), rows / rsIDs | 23 / 10 (rs1294438, rs17823954, rs2742323, rs3798847, rs6793295, rs6920820, rs787334, rs9537756, rs954016, rs9637173) |
| S6 CCVs with no allele-matched row (10 no allele info + 2 absent from input) | 12 |
Check: 5,729 + 12 = 5,741. Rows: 5,729 + 4,396 + 23 = 10,148.
- ALT = GWAS effect allele in 5,727/5,729 matched rows (the effect allele is almost always the non-REF allele).
- Consistency with Chen supplement alleles: 3,149 matched CCVs have alleles in S3-S5/S15-S19; 3,149/3,149 allele sets are within the tested pair.
- Orientation check (independent of GWAS Catalog harmonisation): for 1,893 CCV rows of S15 the sign of (Chen effect-allele beta, re-expressed on the GWAS Catalog effect allele) equals the GWAS Catalog beta sign in 1,893/1,893, including 278/278 palindromic variants. This supports the forward-strand assumption for palindromic SNPs (harmonisation log: palindromic mode forward, consensus ratio 0.9989). Palindromic variants outside S15 (544) rest on the harmonisation assumption alone.
- Which allele is "risk": `gwascat_risk_allele_by_beta_sign` = effect allele if beta>0 else other allele (GWAS Catalog marginal beta, not Chen's conditional beta). The tested ALT is the risk allele only where `alt == that column`.

## Output files (`variant_provenance/`)
- `allele_matched_variants.tsv` (5,729 rows; key `S6_CCV_input_key` = `chr:pos:ref:alt` of the AlphaGenome input; `PIP` column is a text note that none exists).
- `unmatched_or_excluded.tsv` (23 no-allele rows, 4,396 non-reported ALT rows, 2 absent S6 CCVs, with reasons).
- `exploratory_all_rows.tsv` (all 10,148 input rows; `is_reported_allele` True/False/NA).

## Limitations
- "Matched" certifies the allele pair tested in the GWAS at the position. The 4,396 additional ALTs at matched positions (and the 5,729 matched alleles' relation to the fine-mapped causal allele) are exploratory/unvalidated beyond that.
- Allele source is the marginal trans-ancestry GWAS summary statistics, not Chen's conditional analyses; S6 itself contains no alleles.
- The upstream hg19->hg38 step that created the input is undocumented; verified here only by liftover + Ensembl (release 116, current dbSNP rsID mapping), 5,739/5,739 agree.
- GWAS Catalog harmonisation used Ensembl 95/dbSNP 151 (not re-derived here); all matched positions were independently checked against hg38 REF.
- 544 palindromic matched variants not covered by S15 rely on the forward-strand harmonisation assumption (no allele frequency in the file).
- 10 CCVs (23 rows) are absent from the GWAS Catalog file; they are excluded, not guessed.
- Licence of GWAS Catalog data: academic non-commercial.

## Functional / validation data in Chen 2024 supplement (reported, not analysed)
- **No MPRA and no allele-specific-binding (ASB) data** in any of the 22 supplementary sheets (sheet titles listed in log). Fine-mapping uses the paper's own colocalisation/QTL data.
- S13 (cis-eQTL for lead variants: UVA, ACCC, GTEx, Colonomics betas), S14 (cis-mQTL), S15-S17 (eQTL/SMR colocalisation per CCV: p, effect allele, TSMR p), S18-S19 (mQTL colocalisation per CCV), S20 (288 putative target genes), S21 (cell-type expression), S6 binary epigenomic overlaps (incl. TF binding sites). Gene-level eQTL colocalisation uses GTEx (partly overlapping the training data of Borzoi/AlphaGenome) plus colon-tissue eQTL (UVA/ACCC/Colonomics), so only partially independent. Concordance validation would be eQTL/mQTL-direction per CCV (S15-S19: effect allele + beta available for colocalised CCVs only).
