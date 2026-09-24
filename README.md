# Real-time digital payment fraud detection capstone

This project demonstrates fraud-risk scoring with tabular, transaction-graph, and text models, plus explainability and a local API/dashboard demo. It does **not** measure fraud performance on UPI transactions: public transaction-level UPI fraud data is unavailable. IEEE-CIS, Elliptic, ULB credit-card data, SMS Spam Collection, and phishing URLs are real datasets; PaySim is synthetic and is used only as a baseline.

## Verified results

Every metric below is copied from the referenced CSV. Smoke-test outputs are excluded.

### Tabular models

| Evaluation | Model | PR-AUC | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Week 3 shuffled CV mean | XGBoost | Not reported | 0.9610 | 0.6090 |
| 03b time-split validation | LightGBM | 0.7046 | 0.9367 | 0.6687 |
| 03b time-split test | LightGBM | 0.5698 | 0.9084 | 0.5520 |
| 03b time-split validation | XGBoost | 0.5951 | 0.9214 | 0.6104 |
| 03b time-split test | XGBoost | 0.3264 | 0.8699 | 0.3639 |

Source: `notebooks/ablation_report/ablation_summary.csv` and `notebooks/tabular_timesplit_report/timesplit_summary.csv`. LightGBM is selected by validation PR-AUC. The shuffled-CV and time-split results use different protocols; Week 3's CSV does not report PR-AUC. See [COMPARISON.md](COMPARISON.md) for the period-drift diagnostics and curves.

### Graph models

| Evaluation | Model | PR-AUC | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Week 5 reported result | GraphSAGE | Not reported | 0.9002 | 0.5739 |
| 05b validation | GraphSAGE | 0.9686 | 0.9861 | 0.9096 |
| 05b test | GraphSAGE | 0.4246 | 0.8605 | 0.4959 |

Source: `notebooks/graph_ablation_report/graph_ablation_summary.csv` and `notebooks/graph_valsplit_report/graph_valsplit_summary.csv`. 05b selects its epoch/model on validation steps 30–32 and tunes its threshold on steps 33–34. Week 5's CSV has no PR-AUC and its earlier protocol used test F1 for epoch selection.

### Fusion

| Variant | Test PR-AUC | Test ROC-AUC | F1 |
|---|---:|---:|---:|
| Tabular only | 0.7906 | 0.9260 | 0.7701 |
| Weighted | 0.7822 | 0.9176 | 0.7701 |
| Stacked | 0.7776 | 0.9173 | 0.7944 |
| Tabular local only | 0.7664 | 0.8924 | 0.7324 |
| Average | 0.7569 | 0.9157 | 0.7379 |
| Gated | 0.7435 | 0.9048 | 0.7794 |
| Graph only | 0.4246 | 0.8605 | 0.4959 |

Source: `notebooks/fusion_report/fusion_summary.csv`. Fusers fit on steps 30–32, thresholds tune on steps 33–34, and test metrics use steps 35–49. Tabular-only has the highest test PR-AUC in this run; this is descriptive, not test-based model selection. The gated score is kept as the graph-aware demo output. It combines tabular risk, graph risk, node log-degree, and risk-by-degree interactions so that the blend can vary with graph connectivity.

### Text models

| Model | Test PR-AUC | F1 at 0.5 |
|---|---:|---:|
| SMS | 0.9780 | 0.9278 |
| URL | 0.9574 | 0.8855 |

Source: `notebooks/nlp_module_report/nlp_export_summary.csv`. Texts are deduplicated after cleaning; cleaned texts with conflicting labels are removed before splitting.

## Setup on Windows

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
```

The data, model artifacts, virtual environment, and generated notebook reports are ignored by Git. Place the downloaded datasets under `data/` using the directory names expected by the scripts. Never commit raw datasets or credentials.

## Run the workflow

Run each task from the project root:

```powershell
.\tasks.ps1 test
.\tasks.ps1 tabular
.\tasks.ps1 graph
.\tasks.ps1 fusion
.\tasks.ps1 nlp
```

For current-winner explanations, run `python scripts/07b_explainability_new_winners.py`. The protected Week 7 script remains unchanged. The graph task uses PyTorch Geometric; the tested local run used CPU.

## Local demo

Start the API and dashboard in separate PowerShell terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

The API provides `/health`, `/score/transaction`, `/score/text`, and `/graph/node/{idx}`. Transaction explanations use the cached SHAP explainer; an explanation failure is returned in `shap_error`. The local request measurements are documented in [docs/latency.md](docs/latency.md), and [DEMO.md](DEMO.md) contains the presentation script.

## Checks and CI

Run lint and tests locally:

```powershell
.\.venv\Scripts\ruff.exe check
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions runs Ruff and pytest on pushes and pull requests. The CI job installs only the test/runtime packages needed for these checks; graph-training packages are not downloaded there.

## Explainability and report assets

The new-winner runner writes tabular SHAP and graph GNNExplainer artifacts under `notebooks/explainability_report/`. Rebuild the shareable report figures and tables with `python scripts/09_make_report_assets.py`; exported assets go to `docs/report_assets/`.
