# variant_scoring RUNS (all 2026-10-20, CDT; coordinator to transfer to EXPERIMENT_REGISTRY)

| # | command (cwd /home/yejz1/experimental_research) | output | notes |
|---|---|---|---|
| 1 | `./torch-venv/bin/python variant_scoring/make_track_lists.py` | `variant_scoring/track_lists/*.tsv` | CPU; metadata only; TL primary=152 via exposure_audit/track_exposure.tsv |
| 2 | `./torch-venv/bin/python -m pytest variant_scoring/tests -q` | stdout | 15 passed |
| 3 | `./torch-venv/bin/python variant_scoring/score_borzoi.py --selftest` | stdout | TL vs results_full/test_preds.npy[60] max abs diff 1.9e-6; pretrained vs full forward 0 |
| 4 | `./torch-venv/bin/python variant_scoring/score_borzoi.py --smoke` | `results_phase2/variant_scores/borzoi_smoke/` | 18 rows incl. rs6983267, REF==ALT control = 0, 74 s for 53 sequence-units |
| 5 | `./torch-venv/bin/python variant_scoring/score_enformer.py --smoke` | `results_phase2/variant_scores/enformer_smoke/` | 27.6 s (GPU shared) |
| 6 | `nohup ./torch-venv/bin/python variant_scoring/score_borzoi.py` (full 10,148 rows) | `results_phase2/variant_scores/borzoi/`, log `results_phase2/variant_scores/logs/borzoi_full_20261020_0014.log` | resumable (rerun same command) |
| 7 | `nohup ./torch-venv/bin/python variant_scoring/score_enformer.py` (full) | `results_phase2/variant_scores/enformer/`, log `.../logs/enformer_full_*.log` | resumable |
| 6b | resume after home-quota kill (2026-10-20 ~03:15): same command, log results_phase2/variant_scores/logs/borzoi_full_resume_20261020_0346.log | | coordinator |
