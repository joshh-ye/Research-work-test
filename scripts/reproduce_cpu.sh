#!/usr/bin/env bash
# Phase 5 clean-room check: re-run every CPU-level analysis from saved artifacts into a SCRATCH tree and diff against recorded outputs.
# Usage: bash scripts/reproduce_cpu.sh [--scratch DIR] [--with-provenance] [--skip-e2] [--n-boot N]
#   --with-provenance : also rebuild variant_provenance (uses cached Ensembl JSON; falls back to the network only for cache misses)
#   --skip-e2         : skip the E2 interim AlphaGenome run (10,000 bootstraps + 10,000 permutations; slowest step)
# No GPU, no network (unless --with-provenance hits a cache miss), no write to the recorded outputs: each analysis script has an absolute
# repo-root path hard-coded, so the code is COPIED into the scratch tree with that path rewritten (sed) and inputs are symlinked read-only.
# E2: only `--unblind --models alphagenome_quantile` is run (the interim single-model run); other models are never unblinded here.
set -euo pipefail
REPO=/home/yejz1/experimental_research
PY=$REPO/torch-venv/bin/python
SCR=$REPO/results_cleanroom_cpu; WITH_PROV=0; SKIP_E2=0; NBOOT=10000
while [ $# -gt 0 ]; do case $1 in
  --scratch) SCR=$2; shift 2;; --with-provenance) WITH_PROV=1; shift;; --skip-e2) SKIP_E2=1; shift;;
  --n-boot) NBOOT=$2; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
cd "$REPO"; SCR=$(readlink -m "$SCR")
rm -rf "$SCR"; mkdir -p "$SCR/results_phase2" "$SCR/logs"
# ---- code copies with rewritten ROOT -------------------------------------------------------------------------------------------------
for d in analysis/exposure_stratified analysis/e7_nonoracle analysis/e2_validation variant_scoring variant_provenance; do
  mkdir -p "$SCR/$d"; rsync -a --exclude __pycache__ --exclude tests --exclude '*.tsv' --exclude '*.log' --exclude '*.md' "$REPO/$d/" "$SCR/$d/"
done
find "$SCR/analysis" "$SCR/variant_scoring" "$SCR/variant_provenance" -name '*.py' -exec sed -i \
  -e "s#/panfs/accrepfs.vampire/home/yejz1/experimental_research#@@ROOT@@#g" -e "s#/home/yejz1/experimental_research#@@ROOT@@#g" -e "s#@@ROOT@@#$SCR#g" {} +
# gitignored/real-data inputs (read-only symlinks); NOTE variant_scoring/track_lists is an input too
ln -s "$REPO/variant_scoring/track_lists" "$SCR/variant_scoring/track_lists" 2>/dev/null || { rm -rf "$SCR/variant_scoring/track_lists"; ln -s "$REPO/variant_scoring/track_lists" "$SCR/variant_scoring/track_lists"; }
for e in "$REPO"/*; do b=$(basename "$e"); case $b in results_phase2|analysis|variant_scoring|variant_provenance|results_cleanroom_cpu|scripts|env|*venv|hf_cache|*.tar|*.tar.gz) continue;; esac
  [ -e "$SCR/$b" ] || ln -s "$e" "$SCR/$b"; done   # every other top-level input (read-only symlinks)
# saved model-derived artifacts consumed by the CPU steps
R=results_phase2
mkdir -p "$SCR/$R/e2_validation/labels" "$SCR/$R/e7_nonoracle/enformer" "$SCR/$R/e7_nonoracle/alphagenome"
ln -s "$REPO/$R/e2_validation/scores" "$SCR/$R/e2_validation/scores"
ln -s "$REPO/$R/e2_validation/labels/cache" "$SCR/$R/e2_validation/labels/cache"
ln -s "$REPO/$R/e2_validation/tss_protein_coding_gencode_v46_hg38.tsv" "$SCR/$R/e2_validation/"
ln -s "$REPO/$R/variant_scores" "$SCR/$R/variant_scores"
for m in enformer alphagenome; do for f in tile_stats_val.npz tile_stats_test.npz selection_valsel.csv; do
  ln -s "$REPO/$R/e7_nonoracle/$m/$f" "$SCR/$R/e7_nonoracle/$m/$f"; done; done
ln -s "$REPO/$R/e7_nonoracle/alphagenome/val_pearson_matrix.npy" "$SCR/$R/e7_nonoracle/alphagenome/" 2>/dev/null || true
ln -s "$REPO/$R/e7_nonoracle/enformer/val_pearson_matrix.npy" "$SCR/$R/e7_nonoracle/enformer/" 2>/dev/null || true
ln -s "$REPO/$R/e7_nonoracle/alphagenome/tiles_test" "$SCR/$R/e7_nonoracle/alphagenome/tiles_test" 2>/dev/null || true
ln -s "$REPO/$R/e7_nonoracle/alphagenome/tiles_val" "$SCR/$R/e7_nonoracle/alphagenome/tiles_val" 2>/dev/null || true
# provenance: rebuild (optional) or reuse the recorded allele-matched table as an input
if [ $WITH_PROV = 1 ]; then
  echo "[1/5] variant provenance rebuild"; ( cd "$SCR" && $PY -I variant_provenance/build_provenance.py ) > "$SCR/logs/provenance.log" 2>&1
else
  ln -s "$REPO/variant_provenance/allele_matched_variants.tsv" "$SCR/variant_provenance/allele_matched_variants.tsv"
  echo "[1/5] provenance rebuild skipped (use --with-provenance); recorded allele_matched_variants.tsv used as input"
fi
echo "[2/5] exposure-stratified stats";  ( cd "$SCR" && $PY analysis/exposure_stratified/run_exposure_stratified.py --n-boot $NBOOT ) > "$SCR/logs/exposure_stratified.log" 2>&1
echo "[3/5] E7 analyze (saved per-tile npz, CPU)"
for m in enformer alphagenome; do ( cd "$SCR" && $PY analysis/e7_nonoracle/analyze.py $m --n-boot $NBOOT ) > "$SCR/logs/e7_$m.log" 2>&1; done
echo "[4/5] E2 labels + power (label-only)"; ( cd "$SCR" && $PY analysis/e2_validation/labels.py ) > "$SCR/logs/e2_labels.log" 2>&1
if [ $SKIP_E2 = 0 ]; then
  echo "[5/5] E2 interim, alphagenome_quantile only"
  ( cd "$SCR" && $PY analysis/e2_validation/run_e2.py --unblind --models alphagenome_quantile --out "$SCR/$R/e2_validation/interim_alphagenome" --boot $NBOOT --perm $NBOOT ) > "$SCR/logs/e2_interim.log" 2>&1
else echo "[5/5] E2 interim skipped"; fi
echo "== compare =="
$PY "$REPO/scripts/compare_outputs.py" --scratch "$SCR" $( [ $SKIP_E2 = 1 ] && echo --skip-e2 ) $( [ $WITH_PROV = 1 ] && echo --with-provenance ) | tee "$SCR/compare_report.txt"
