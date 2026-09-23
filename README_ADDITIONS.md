# Capstone additions (Weeks 8-10 + fixes)

These are NEW files only. Nothing in your existing `vscode-project/` is changed.
Copy the folders into `vscode-project/` (merge `scripts/`, add `app/`, `dashboard/`,
`tests/`, `docs/`). If a file name already exists, stop - none should.

## What's here

| File | Week | What it does |
|---|---|---|
| scripts/common_eval.py | all | PR-AUC, ROC-AUC, F1 at a validation-tuned threshold, recall at 1% FPR |
| scripts/03b_tabular_timesplit.py | 3 fix | IEEE-CIS re-run with a time-based split, identity table merged, threshold tuned on validation, saves `models/tabular_winner.pkl` |
| scripts/05b_graph_ablation_valsplit.py | 5 fix | GNN ablation with a validation period (steps 30-34) instead of picking epochs on the test set; saves node probabilities for fusion |
| scripts/06b_export_nlp_models.py | 6 fix | removes duplicate texts, re-scores honestly, saves SMS + URL models for the API |
| scripts/08_fusion_elliptic.py | 8 | tabular (XGBoost on node features) + graph (GNN) fusion: avg / weighted / stacked / gated; F1 per time step |
| app/api.py | 9 | FastAPI: /health, /score/transaction (+SHAP drivers), /score/text, /graph/node/{idx} |
| app/text_features.py | 9 | picklable text cleaners (fixes a Week 6 bug that deleted the link token) |
| dashboard/streamlit_app.py | 9 | 4-tab demo: transaction risk, SMS/URL check, graph view, results |
| tests/test_pipeline.py | 10 | pytest: cleaners, metrics, smoke runs of 03b and 08, API health |
| docs/REVIEW_FINDINGS.md | 10 | issues found in Weeks 3-7 and how each is handled |
| docs/REPORT_OUTLINE.md | 10 | capstone report chapters + figures/tables list + viva questions |

## Run order (from the project root, venv active)

```bash
pip install -r requirements.txt -r requirements-additions.txt
pytest -q                                   # ~10 s, no real data needed

python scripts/03b_tabular_timesplit.py     # 20-60 min on a laptop (IEEE-CIS is big)
python scripts/05b_graph_ablation_valsplit.py   # needs torch + torch_geometric (Colab is fine)
python scripts/08_fusion_elliptic.py
python scripts/06b_export_nlp_models.py

uvicorn app.api:app --port 8000             # terminal 1  -> http://127.0.0.1:8000/docs
streamlit run dashboard/streamlit_app.py    # terminal 2
```

Every new script has `--smoke-test` (03b, 08) or runs on tiny data in tests. Smoke
outputs go to `.../smoke/` folders and are labelled NOT A RESULT - never put
those numbers in the report.

## What was actually tested

Tested here on synthetic data: common_eval, 03b (XGBoost path), 08 fusion, API
(/health, /score/transaction, /score/text), text cleaners. All 6 pytest checks pass.
NOT run here (no real data / no torch in this sandbox): 05b, 06b on the real CSVs,
the LightGBM/CatBoost paths of 03b, the Streamlit UI (syntax-checked only).
Real numbers only exist once you run these on your machine or Colab.
