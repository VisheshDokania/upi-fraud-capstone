# Code map for the current upi-synthetic branch

This file replaces the older "copy new files into vscode-project" packaging note. The repository is already integrated. Use [README.md](README.md) as the main setup and results reference; do not copy these files into a second tree.

| Area | Current purpose |
|---|---|
| `scripts/common_eval.py`, `03b_tabular_timesplit.py` | Validation-selected, time-split IEEE-CIS card-fraud evaluation and saved tabular bundle. |
| `scripts/05b_graph_ablation_valsplit.py`, `gnn_models.py` | Elliptic graph models, validation model/epoch selection and later fixed test. |
| `scripts/08_fusion_elliptic.py` | Seven within-Elliptic fusion variants. Tabular-only leads held-out PR-AUC; the degree-aware gate supplies the saved graph-aware demo output. |
| `scripts/06b_export_nlp_models.py`, `app/text_features.py` | Clean and export independent SMS and phishing-URL classifiers. |
| `app/api.py` | Local `/health`, `/score/transaction`, `/score/text` and `/graph/node/{idx}` endpoints. No synthetic-UPI API route. |
| `dashboard/streamlit_app.py`, `.streamlit/config.toml` | Dark-theme, **five-tab** dashboard: Transaction workbench, SMS / URL screening, Graph risk, Model & evaluation, UPI Live Scoring. |
| `upi_synthetic/generate_data.py`, `train_model.py` | Deliberately generated collect-request, QR-code and velocity scenarios; separate logistic-regression model with validation-tuned threshold. Directly loaded by the fifth Streamlit tab. |
| `scripts/07b_explainability_new_winners.py`, `09_make_report_assets.py`, `10_measure_latency.py` | Explanations, exported report assets and local request timings. |
| `tests/test_pipeline.py`, `.github/workflows/ci.yml` | Lightweight checks, not a real-data benchmark or production readiness test. |

Raw data, generated reports and model files are ignored. Run the branches and synthetic trainer before expecting all dashboard tabs to be populated. `docs/report_assets/` holds committed figures/tables, not the raw source datasets. For a walkthrough see [DEMO.md](DEMO.md). Older prompts in `docs/PROMPTS.md` describe prior phases and are not the current execution plan.
