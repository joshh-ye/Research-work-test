# Training-exposure audit (BL-004, BL-005, SD-012/E8 prerequisites)
Date 2026-10-20. Scripts: `01_fetch_metadata.py` (GEO + ENCODE REST; raw responses in `cache/`), `02_tile_fold_overlap.py`, `03_track_exposure.py`. Run with `./torch-venv/bin/python -I`, CPU only.
Outputs: `track_exposure.tsv` (183 rows), `tile_fold_overlap.tsv` (16 rows).

## 1. Track identity (BL-004)
- 183 BigWigs ordered by `borzoi_ckpt_full/bw_files.json` (track_idx = list position; DCid = file stem). 179 tracks have a GSM; 4 have ENCSR (DCid 62707/62737/64170/64457, `GSMID` column `ENCSR..._1`).
- GEO text (`acc.cgi?targ=self&form=text&view=brief`) fetched for all 179 GSM (0 errors). ENCODE JSON for 4 ENCSR (all TF ChIP-seq, transverse colon tissue, Snyder lab, GEO:GSE105270/105277/105623/106004).
- **CSV Cell_line is NaN for 20 tracks**: 4 ENCSR (transverse colon tissue) + 15 GSM that GEO text identifies as HCT116 (GSE38258, GSE43539/43542, GSE63314) + 1 (OGT, GSM2459951, GSE93656, "colon tumor cells, isolated from primary tumor"). Resolved cell-line counts: HCT-116 105, LoVo 41, Caco-2 7, GP5d 6, LS174T 5, LS180 5, SW480 4, RKO 3, DLD-1 1, HT-29 1, transverse-colon tissue 4, primary tumor 1. (Previous "HCT-116 90" counts only CSV-annotated rows.) GEO text agreed with the CSV cell line for all non-NaN GSM (0 conflicts).
- **A19 resolved**: 21 project HCT-116 GSMs are ENCODE experiments (ENCODE search `dbxrefs` GEO:GSM -> ENCSR; bulk query HCT116 + Caco-2 only; other cell lines returned 404 = zero ENCODE ChIP hits). 3 of those ENCSR are in Borzoi AND Enformer vocab file paths: ENCSR000DTO CTCF (GSM1022652), ENCSR000BML POLR2A (GSM803474; vocab label is POLR2AphosphoS5), ENCSR000BSA JUND (GSM1010847). So **identical-experiment tracks = 24 LoVo + 4 colon ENCSR + 3 HCT-116 = 31 (Borzoi and Enformer, same set)**, still a lower bound (only HCT116/Caco-2 ENCODE xrefs searched; vocab only matched by GSM/ENCSR).
- Caveat: ENCODE dbxref coverage of GSM is incomplete in principle; a GEO-only sample reprocessed elsewhere would be missed.

## 2. Exposure classes (hierarchy: identical > same cell line+target > same cell line other target (any assay) > same target other cell line > unexposed)
Vocab parsing: Borzoi `borzoi_data/targets_human.csv` (7,611), Enformer `enformer_data/targets_human.txt` (5,313). Cell line = regex over description of all assays (CAGE/DNase/ATAC/ChIP/RNA). Enformer LoVo GEO rows have target "." and TF is inside the batch title (`batch1_chrom1_LoVo_<TF>_...`); parsed accordingly (initial run without this under-counted).
Counts (tracks, n=183):
| class | Borzoi | Enformer | AlphaGenome |
|---|---|---|---|
| identical_experiment | 31 | 31 | not determinable (0 verified; 25 ENCODE-linked tracks possible) |
| same_cellline_same_target | 22 | 22 | 38 |
| same_cellline_other_target | 112 | 112 | 74 |
| same_target_other_cellline | 13 | 11 | 45 |
| unexposed | 5 | 7 | 0 |
| unknown | 0 | 0 | 26 |
Note "same_cellline_*" for 4 tissue tracks means tissue-level, not a cell line.
AlphaGenome: only the ChIP-TF head metadata is on disk (`results_baseline_alphagenome/alphagenome_tracks_test.csv`, 1,617 tracks, all `data_source=encode`, no accessions; cell lines present: HCT116 25 tracks, Caco-2 1, plus colon tissues). Other heads (DNase, CAGE, RNA...) and the training manifest (paper Supplementary Table 2, Avsec et al., Nature, doi 10.1038/s41586-025-10014-0, PMC12851941 - opened; main text states accession manifest is in Supp. Table 2) were NOT obtained, so AG cell-line exposure from non-ChIP assays is **unknown**. LoVo/GEO samples cannot be AG training experiments for ChIP-TF (ENCODE only), but AG exposure to LoVo through other assays is unknown.

## 3. SD-012/E8 candidates: no cell-line exposure in Borzoi or Enformer vocab (any assay)
GP5d 6 tracks (CTCF, HNF4A, NIPBL, RAD21 x2, SMC3), LS174T 5 (GATA6 x3, ZBTB17 x2), LS180 5 (VDR, CEBPB x2, CDX2 x2), DLD-1 1 (REPIN1) = **17 cell-line tracks**, plus 1 primary-tumor OGT track (tissue, not cell line) = 18. Within these, same-target-in-other-cell-line: Borzoi 13 / Enformer 11; fully unexposed target+cell: Borzoi 5 / Enformer 7. Cell-line-clustered inference would have 4 clusters only (low power). Cell lines with only DNase exposure (RKO, SW480; 3+4 tracks) are partial candidates. For AlphaGenome these cell lines are not verifiably unexposed (other heads unknown).
Caveat: "colon epithelial cell line" DNase track (ENCSR038XTK) has an unresolved biosample (not looked up) and could be one of the lines; description-regex also cannot see cell lines named only by ontology term.

## 4. Fold overlap (BL-005)
Downloads (`external_data/folds/`, md5):
- `borzoi_sequences_human.bed.gz` <- https://raw.githubusercontent.com/calico/borzoi/main/data/sequences_human.bed.gz (git blob 1a4b80d8..., md5 a32f2b21391403252e864e2f3543639a). 55,497 rows: fold0-7 (7090/6713/6526/6888/6910/6994/7363/7013), width 196,608. Borzoi README: 4 replicates with identical splits, **test=fold3, valid=fold4**, rest train (README lines 66-71; unverified beyond the README).
- `basenji2_barnyard2_sequences_human.bed` <- https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed (md5 2c256f1c667fe3318865303248e70659; train 34021/valid 2213/test 1937, 131,072 bp). (basenji_barnyard/data is requester-pays; not accessible.)
- `enformer_genentech_human_intervals.tsv` <- https://huggingface.co/datasets/Genentech/enformer-data/resolve/main/human_intervals.tsv (md5 fd07a3b0a56af5db4e4ffc2833113744; same 34021/2213/1937 split, 196,608 bp = Basenji2 extended; dataset card describes this derivation; this is a third-party reproduction of the Enformer intervals, not the DeepMind file).
Project tiles: reconstructed with `borzoi_code/genome_tiler.py` logic from `hg38.ml.fa.fai` (524,288-bp non-overlapping windows from 0; val chr1+chr8, test chr9+chr22) - not saved anywhere; yields 4659 train / **750 val / 359 test** (359 matches the known test count). Targets cover the central 131,072 bp (`borzoi_code/dataset.py`). Borzoi input windows = bed +/-163,840 (assumption: bed lists the 196,608 cropped region of 524,288 inputs; reported both ways).
Results (bp fractions; full table `tile_fold_overlap.tsv`; fractions need not sum to 1 because bed windows do not tile the genome - uncovered fraction given):
| project split / region | Borzoi train | valid | test | uncov | Enformer train | valid | test | uncov |
|---|---|---|---|---|---|---|---|---|
| test chr9+22, tile 524,288 (bed as given / Enformer 196,608) | 0.436 | 0.010 | 0.304 | 0.250 | 0.730 | 0.000 | 0.017 | 0.253 |
| test, centre 131,072 | 0.437 | 0.011 | 0.304 | 0.248 | 0.733 | 0.000 | 0.017 | 0.251 |
| val chr1+8, tile 524,288 | 0.657 | 0.169 | 0.089 | 0.085 | 0.897 | 0.005 | 0.008 | 0.090 |
Interpretation: project test tiles are **not** held out from either model: ~44% (Borzoi) and ~73% (Enformer) of test bp lie in the pretrained model's training folds (Borzoi input-extended: 0.442); only ~30% falls in Borzoi's test fold (Enformer 1.7%). Project chr9/chr22 holdout is a project choice, independent of the pretrained folds. Caveat: Borzoi `johahi/borzoi-replicate-*` HF weights are ports of the original replicates; if replicates truly use different folds the train fraction per replicate changes (README says identical).

## Limitations
GSM/ENCSR identity only; reprocessed or pooled duplicates of the same biological sample are not detected. Vocab cell-line detection by regex. AlphaGenome training manifest not obtained. Fold overlap is bp-level at the bed window level (ignoring Borzoi's per-track blacklists/gaps). No statistical analysis performed.
