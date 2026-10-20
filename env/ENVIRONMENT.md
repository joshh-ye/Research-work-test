# Environment specification (captured 2026-10-20)

Host: ACCRE (Vanderbilt), Rocky Linux 9.8 (Blue Onyx), kernel 5.14.0-687.52.1.el9_8.x86_64. Work is run from an interactive GPU node
(Slurm account `accre_dgx_iacc`, partition `interactive_gpu`, qos `dgx_iacc`, 1x `nvidia_a100-sxm4-40gb`).
GPU/driver (header of `nvidia-smi` only): NVIDIA-SMI 580.178.04, Driver 580.178.04, CUDA (driver) 13.0. The CUDA toolkit used by the
Slurm scripts is the ACCRE module `cuda/12.6` (the torch wheel is `torch 2.11.0+computecanada`, i.e. the ComputeCanada/ACCRE wheelhouse build).

## Virtual environments
| venv | python | freeze | used for |
|---|---|---|---|
| `torch-venv` | see `torch-venv.python_version.txt` | `torch-venv.requirements.txt` (63 pkgs) | training, Borzoi/Enformer inference (`borzoi-pytorch 0.4.4`, `enformer-pytorch 0.8.11`), all CPU analyses |
| `tf-venv` | `tf-venv.python_version.txt` | `tf-venv.requirements.txt` (78 pkgs) | TensorFlow stack (`tensorflow 2.19.1`); no registered experiment currently depends on it |
| `ag-venv` | `ag-venv.python_version.txt` | `ag-venv.requirements.txt` (47 pkgs) | AlphaGenome hosted API client (`alphagenome 0.8.0`; pyarrow for parquet conversion) |
| `alphagenome_variant_effects/ag-local-venv` | `ag-local-venv.python_version.txt` | `ag-local-venv.requirements.txt` (130 pkgs) | AlphaGenome local weights (`alphagenome 0.9.0`, `tensorflow 2.21.0`, `alphagenome_research` pinned to git `0db53bd4`, an editable VCS line in the freeze) |

Caveats (do not "fix" silently):
- `+computecanada` local-version suffixes (torch, numpy, scipy, pandas ...) refer to the ACCRE wheelhouse; a plain `pip install -r` from PyPI will not find
  those exact versions. Strip the suffix to reproduce on another cluster (versions may then differ in build, not API).
- The torch-venv was copied from an older project: `bin/pip` has a stale shebang. Always use `./torch-venv/bin/python -m pip`.
- `transformers` must stay `<4.51.0,>=4.34.1` for `borzoi-pytorch` (see `baseline_enformer.slurm` comment).
- tf-venv contains `torch 2.12.0+computecanada` (differs from torch-venv 2.11.0). Pin per venv, never share.
- Freezes capture the venv state at the date above; the venvs were mutated over time (e.g. enformer-pytorch 0.8.12 installed 2026-08-10, torch-venv now reports 0.8.11).
  The Aug-2026 benchmark was produced with `enformer-pytorch 0.8.11` per `results_benchmark/benchmark_config_test.json`.

## Module loads used in the Slurm scripts
| script | module lines |
|---|---|
| `training_run.slurm`, `setup_accre_torch_gpu.slurm`, `baseline_borzoi.slurm`, `baseline_borzoi_filtered.slurm`, `baseline_enformer.slurm` | `module purge; module load python/3.12.4 cuda/12.6` then `source torch-venv/bin/activate` |
| `alphagenome_variant_effects/run_alphagenome_local_a100.slurm` | `module purge; module load python/3.12.4 cuda/12.6` (errors ignored) then `./ag-local-venv/bin/python` |
| `alphagenome_variant_effects/run_alphagenome_variant_effects.slurm` | `module purge; module load python/3.12.4` then `../ag-venv/bin/python` (hosted API; no CUDA needed) |
Slurm resources: training `--gres=gpu:nvidia_a100-sxm4-40gb:1 --cpus-per-task=8 --mem=32G --time=7-00:00:00`; Enformer baseline `--cpus-per-task=1 --mem=32G --time=00:25:00`.
Interactive Claude Code sessions already hold an A100: run `.slurm` files with `bash`, not `sbatch`.

## Secrets (never committed)
- AlphaGenome hosted API: key via `$ALPHAGENOME_API_KEY` or the git-ignored `alphagenome_variant_effects/.alphagenome_api_key` (mode 0600). Contents are never logged or recorded here.
- AlphaGenome local weights: Hugging Face token (gated model). Not recorded.
