# MORNING status — capstone follow-up

Steps A–G are complete. Each step was committed and pushed to `origin/master`; no files were deleted, `data/` and scripts 01–07 were not edited, and no packages were installed.

| Verified run | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|
| 03b LightGBM | 0.7046 | 0.5698 |
| 05b GraphSAGE | 0.9686 | 0.4246 |
| Fusion tabular-only | — | 0.7906 |
| Fusion gated | — | 0.7435 |
| SMS cleaned-text model | — | 0.9780 |
| URL cleaned-text model | — | 0.9574 |

Values come from the run CSVs under `notebooks/`. Tabular-only had the highest fusion test PR-AUC; the gated variant remains the API's graph-aware output. The API endpoints were exercised, and latency measurements over the recorded request set are in `docs/latency.md` with CSV source data. `pytest -q` and Ruff pass locally. Report tables and 300-DPI figures are in `docs/report_assets/`.

The protected `scripts/07_explainability.py` was left unchanged; `scripts/07b_explainability_new_winners.py` ran against the new LightGBM and GraphSAGE bundles. Docker was skipped as requested. No Colab/GPU or approval blocker remains. Nothing needs your approval.
