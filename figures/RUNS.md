# Figure runs (CPU only, 2026-10-20; cwd /home/yejz1/experimental_research)
```
./torch-venv/bin/python figures/scripts/fig1_model_comparison.py
./torch-venv/bin/python figures/scripts/fig2_exposure.py
./torch-venv/bin/python figures/scripts/fig3_track_scatter.py
./torch-venv/bin/python figures/scripts/fig4_variant_flow.py
./torch-venv/bin/python figures/scripts/fig5_e2_interim.py     # also writes fig5S_e2_p2_caveat
./torch-venv/bin/python figures/scripts/make_supp_tables.py
```
Each script asserts that plotted means/counts equal the source files (all passed). Fig 5 final-mode (results_phase2/e2_validation/final/tables/) was exercised on a synthetic copy in the session scratchpad only; no final run exists yet.
