# E7 selection rule (pre-specified 2026-10-20, BEFORE any E7 validation or test statistic was computed)

Applies SD-007. Written before running any E7 inference.

1. Geometry (unchanged from the August benchmark, so numbers are comparable): hg38; tiles = `tile_genome(borzoi_data/hg38/hg38.ml.fa)[split]`
   (val = chr1+chr8, 750 tiles; test = chr9+chr22, 359 tiles); shared grid = central 114,688 bp at 128 bp (896 bins).
   Enformer input = central 196,608 bp of each 524,288 bp tile (no RC averaging, no shifts, fp32), output 896x128 bp.
   AlphaGenome input = full 524,288 bp tile (hosted API, CHIP_TF, all ontologies, ontology_terms=None), output cropped to the central
   114,688 bp and mean-pooled to 128 bp. Targets = `results_full/{split}_targets.npy` via `targets_to_enformer_grid`.
2. Candidate vocabulary (identical to the oracle's): Enformer = 1,990 non-histone CHIP tracks (`results_baseline_enformer/included_enformer_tracks.csv`);
   AlphaGenome = all 1,617 CHIP_TF tracks (`results_baseline_alphagenome/alphagenome_tracks_test.csv`).
3. PRIMARY non-oracle selection ("valsel"): for each of the 183 targets, the candidate with the highest validation Pearson r, where r is
   computed over all 750 validation tiles x 896 bins flattened (same definition as the oracle, same zero-signal tiles kept). Ties -> lowest index.
   Selection uses chr1/chr8 only; test statistics are never used. Selections are written to `selection_valsel.csv` before test inference starts.
4. ORACLE (upper bound, comparison only): per target argmax of test Pearson r over the same vocabulary. The August oracle values are re-derived
   from the new test statistics and must reproduce the benchmark CSV (Enformer 0.6443, AlphaGenome 0.6612) or the run is flagged.
5. SD-007 sensitivity "valsel_excl_identical": same rule with identical-experiment candidates removed from the vocabulary.
   - Enformer: candidates whose vocab `identifier` is a GSM id of a project track, or whose `file` path contains an ENCSR of a project track
     (`exposure_audit/track_exposure.tsv` GSM/ENCSR columns; the same matching rule as exposure_audit/03_track_exposure.py). The removal is
     set-level (candidate is a project experiment), applied to all targets (conservative).
   - AlphaGenome: the hosted metadata has no accession, so identity cannot be checked. Proxy (conservative): for targets flagged
     `AlphaGenome_possible_identical_ENCODE` in track_exposure.tsv (ENCSR present), remove candidates with the same TF and a biosample in
     {HCT116, Caco-2, transverse/sigmoid colon, colonic mucosa, mucosa of descending colon}. Documented as a proxy, not an identity proof.
6. Metrics: Pearson r (primary), r^2 = squared Pearson (reported separately), prediction R^2 = 1-SSE/SST with a linear calibration (a, c) fitted on
   validation for the selected pair, labelled "validation-calibrated R^2". Pearson r flattened over tiles x bins per target.
7. Inference: Delta r = TL - model, per target; mean over targets; subsets all-183, primary-152 (non-identical by Borzoi class), identical-31.
   Bootstraps (10,000, percentile 95% CI, seed 20261020): TF-cluster, cell-line-cluster, tile-block (10 contiguous tiles), and two-way
   tile-block x cluster, reusing analysis/exposure_stratified/estats.py.
