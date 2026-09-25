# Real Time UPI Fraud Detection

A research capstone and local demonstration of fraud-risk scoring. The project evaluates **separate** tabular, graph and text models on public proxy datasets, then adds a **synthetic UPI-like scenario model** for an interactive teaching demo. It does not measure fraud accuracy on real UPI transactions: no permissioned, labelled UPI transaction feed is used. Do not compare the scores across datasets as if they came from one joined payment stream.

**What runs today:** a FastAPI service scores IEEE-CIS transactions, SMS/URLs and saved Elliptic graph nodes; a five-tab Streamlit dashboard presents those outputs. The fifth tab, UPI Live Scoring, loads a separate saved synthetic model **directly in Streamlit**, not through FastAPI. “Live” means interactive local scoring, not a bank connection, streaming feed, auto-blocking or a production SLA.

## Get started on Windows

Use PowerShell from the repository root. The current work is on the `upi-synthetic` branch, not necessarily on the default branch.

```powershell
git switch upi-synthetic
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
```

The checked-in source does **not** include raw datasets, fitted models or generated `notebooks/*_report/` outputs. Put authorized downloaded datasets under `data/` in the locations expected by the acquisition/training scripts. See the historical acquisition guide for dataset names, but rely on the current scripts for exact paths. To reproduce the benchmark branches, run them from the project root in order; these require their corresponding datasets and, for graph training, PyTorch Geometric:

```powershell
.\tasks.ps1 test
.\tasks.ps1 tabular
.\tasks.ps1 graph
.\tasks.ps1 fusion
.\tasks.ps1 nlp
```

See [VSCODE_SETUP.md](VSCODE_SETUP.md) for local setup and data prerequisites. `tasks.ps1` supports `test`, `tabular`, `graph`, `fusion`, `nlp`, `api` and `dashboard`; it does not run the synthetic generator. Do not commit raw data, model bundles or credentials. Only load `.pkl` model files you trust, since pickle loading can execute code.

Generate and fit the separate UPI-like scenario model:

```powershell
.\.venv\Scripts\python.exe -m upi_synthetic.generate_data
.\.venv\Scripts\python.exe -m upi_synthetic.train_model
```

This writes generated rows and evaluation/coefficients to `notebooks/upi_synthetic_report/` and a fitted model to `models/upi_synthetic_model.pkl`; those paths are ignored by Git. Run these commands before using the fifth dashboard tab. The model uses a chronological 70/15/15 train/validation/test split and picks its F1 alert threshold on validation only. A local run reported synthetic test PR-AUC **0.9980**, but labels and strong signals were deliberately constructed together. This is **not** real UPI performance.

## Run the local demo

With the needed model/report artifacts present, open two PowerShell terminals in the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

The API offers `GET /health`, `POST /score/transaction`, `POST /score/text` and `GET /graph/node/{idx}`. The API `/docs` page is available locally at `http://127.0.0.1:8000/docs`. The dashboard is set to a dark theme in [`.streamlit/config.toml`](.streamlit/config.toml), with contrast overrides in [`dashboard/streamlit_app.py`](dashboard/streamlit_app.py).

| Dashboard tab | What it shows | Boundary |
|---|---|---|
| **Transaction workbench** | Held-out IEEE-CIS demo rows, editable supported fields, LightGBM score/alert threshold, top-five signed TreeSHAP contributions and session history | Card-fraud proxy, not UPI. The session log lasts only in this browser session; no payment is blocked. |
| **SMS / URL screening** | Separate TF-IDF classifiers through the API | The fixed **0.50** text risk threshold is advisory, not the transaction model's validation-selected threshold. No per-token explanation is returned. |
| **Graph risk** | Elliptic node's saved `fused_risk` and neighboring node scores via the API | Elliptic is a Bitcoin transaction graph, not UPI. Neighbor scores are not causal explanations; per-node GNNExplainer is not served. |
| **Model & evaluation** | Tabular model card, validation versus later test metrics, graph/fusion/text result tables and explicit task limits | Results are read from locally generated summary CSVs and may be unavailable until the pipelines run. |
| **UPI Live Scoring** | Interactive collect-request, QR-spoofing and velocity presets, UPI-like fields, synthetic logistic-regression score and coefficient contributions | Loads the separate saved model inside Streamlit. Synthetic scenarios are not an observed UPI fraud test. |

The synthetic model uses transaction type, initiation mode, merchant MCC, status, failed PIN attempts, receiver/merchant age and velocity signals, plus VPA-provider categories such as `okaxis` and `ybl`. Raw sender/receiver VPAs are **excluded** from the fitted features to avoid memorizing account identifiers. Positive/negative coefficient terms are local model contributions, not causal explanations. The three scam scenarios are deliberately encoded: a high-value collect request to a new receiver, a QR-code merchant burst, and small failed probes before a larger payment.

For transaction scoring, the API aligns IEEE-CIS fields to the saved model's schema and returns a probability, risk band, threshold decision and SHAP contributions. A SHAP failure is reported in `shap_error`. The demo row contains 431 model input fields; `TransactionDT` is displayed as context but not fed to that model. The transaction alert threshold was chosen for validation F1 (about **14.8%** in the saved run), not tuned on the held-out test period.

## Evaluation results and what they mean

These are offline results from **different tasks**; PR-AUC is average precision. Public report assets under [`docs/report_assets/`](docs/report_assets/) mirror the local run summaries; [COMPARISON.md](COMPARISON.md) discusses protocol and drift. Do not treat descriptive test leaders as models chosen after seeing test labels.

| Task and model | Validation PR-AUC | Later test PR-AUC | Notes |
|---|---:|---:|---|
| IEEE-CIS card transactions, LightGBM | 0.7046 | 0.5698 | Chronological 70/10/20 split; selected by validation PR-AUC. Test ROC-AUC 0.9084, F1 0.5520. |
| IEEE-CIS card transactions, XGBoost | 0.5951 | 0.3264 | Same later-period comparison; test ROC-AUC 0.8699. |
| Elliptic Bitcoin graph, GraphSAGE | 0.9686 | 0.4246 | Train steps 1–29; model/epoch on steps 30–32; threshold on 33–34; test on 35–49. Test ROC-AUC 0.8605. |
| Elliptic fusion, tabular-only | - | **0.7906** | Highest test PR-AUC among seven variants in this run; this is an observed comparison, not retroactive selection. |
| Elliptic fusion, degree-aware gate | - | 0.7435 | Graph-aware output retained in the node demo, not the test-PR-AUC winner. |
| SMS text classifier | - | 0.9780 | Separate cleaned/deduplicated text task; test F1 at 0.50 is 0.9278. |
| Phishing URL text classifier | - | 0.9574 | Separate cleaned/deduplicated text task; test F1 at 0.50 is 0.8855. |

Source tables: [`docs/report_assets/tabular_timesplit.csv`](docs/report_assets/tabular_timesplit.csv), [`graph_valsplit.csv`](docs/report_assets/graph_valsplit.csv), [`fusion_test.csv`](docs/report_assets/fusion_test.csv), [`nlp_cleaned_text.csv`](docs/report_assets/nlp_cleaned_text.csv). Graph fusion and base-model fitting reused labelled steps 30–32 for development; thresholds use 33–34. No record-level fusion crosses IEEE-CIS, Elliptic, SMS and URL datasets. SMS and URL are split 80/20 separately, **not** claimed to be time-ordered. The synthetic UPI-like result is not in this cross-task table because it is built from generated labels.

Local request latency is recorded in [`docs/latency.md`](docs/latency.md) and [`docs/latency_requests.csv`](docs/latency_requests.csv). Those single-setup measurements are **not** production throughput or a bank integration. Figures and tables can be regenerated with `python scripts/09_make_report_assets.py` once the required local summaries exist. For new-winner explanations use `python scripts/07b_explainability_new_winners.py`; it produces SHAP and graph GNNExplainer artifacts under an ignored report directory. The graph API does not serve the latter per-node artifact.

## Code map and checks

- [`app/api.py`](app/api.py): four local endpoints and model-bundle loading.
- [`dashboard/streamlit_app.py`](dashboard/streamlit_app.py): five-tab Streamlit UI and direct synthetic-model scoring.
- [`upi_synthetic/`](upi_synthetic/): reproducible scenario generator and standalone logistic-regression trainer.
- [`scripts/03b_tabular_timesplit.py`](scripts/03b_tabular_timesplit.py), [`05b_graph_ablation_valsplit.py`](scripts/05b_graph_ablation_valsplit.py), [`08_fusion_elliptic.py`](scripts/08_fusion_elliptic.py), [`06b_export_nlp_models.py`](scripts/06b_export_nlp_models.py): proxy evaluations and exports.
- [`tests/test_pipeline.py`](tests/test_pipeline.py): pipeline/API-oriented checks.

Run locally:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

CI runs Ruff and pytest on pushes and pull requests; it does not download raw datasets, train the graph or attest to live UPI performance. The checked-in figures and metric tables are derived report assets, not the raw training data or fitted model binaries.
