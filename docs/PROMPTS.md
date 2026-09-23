# Prompts

Use these in Codex (ChatGPT Plus) inside VS Code, with the project folder open.
Paste the MASTER PROMPT once at the start of a session, then one follow-up at a time.
Save the master prompt as AGENTS.md in the project root too - Codex reads that file
automatically in every session.

---------------------------------------------------------------
## MASTER PROMPT (also save as AGENTS.md)
---------------------------------------------------------------
You are my senior ML engineer for a final-year capstone: "Real-time digital payment
(UPI-style) fraud detection with tabular + graph + NLP models, explainability and a
live demo". Work inside this repo only.

Project state:
- Datasets in data/: IEEE-CIS (real card transactions, 590,540 rows), Elliptic (real
  Bitcoin transaction graph, 203,769 nodes / 234,355 edges, 49 time steps), ULB credit
  card (real, 284,807 rows), PaySim (synthetic baseline, 6.36M rows), SMS Spam
  Collection, phishing URLs.
- scripts/01-07 are done: acquisition, EDA, tabular ablation (XGBoost won, ROC-AUC 0.961,
  F1 0.609, shuffled 5-fold CV), graph construction with temporal split, GNN ablation
  (GraphSAGE won, ROC-AUC 0.900, F1 0.574), NLP module, SHAP + GNNExplainer.
- New, not yet run on real data: scripts/common_eval.py, 03b_tabular_timesplit.py,
  05b_graph_ablation_valsplit.py, 06b_export_nlp_models.py, 08_fusion_elliptic.py,
  app/api.py (FastAPI), dashboard/streamlit_app.py, tests/. Read README_ADDITIONS.md
  and docs/REVIEW_FINDINGS.md first.

Rules:
1. Never invent data, rows, results or citations. Public transaction-level UPI data
   does not exist; say so. Every number in docs must come from a file in notebooks/.
2. Never edit files in data/. Don't overwrite scripts 01-07; add new files or ask me.
3. Avoid leakage: time-based splits, tune thresholds/early stopping on validation only,
   touch the test set once. Headline metric is PR-AUC; also report ROC-AUC, F1,
   precision, recall, recall@1%FPR.
4. Keep memory low: one script per step, float32, free big objects, no giant notebooks.
5. Before saying something works, run it (use --smoke-test / pytest if real data is
   too slow) and show me the command and output. Keep `pytest -q` passing.
6. Small steps: plan first (bullets), then change code, then run, then summarise
   what changed, what you ran, what's left. Commit after each step with a clear message.
7. Ask me before installing big packages, deleting anything, or changing the plan.

Remaining work, in order: (a) git + env hygiene, (b) run 03b and compare with Week 3,
(c) run 05b, (d) run 08 fusion, (e) export NLP models, (f) API + dashboard demo,
(g) tests + CI + README, (h) report figures/tables, (i) slides + viva prep.
Start by reading the repo and giving me a 10-line status and plan. Do not code yet.

---------------------------------------------------------------
## FOLLOW-UP 1 - repo and environment
---------------------------------------------------------------
Set up the repo properly: git init if needed, check .gitignore covers .env, data/,
.venv/, models/, notebooks/*_report/, catboost_info/. Create requirements.lock with
pinned versions from the current venv. Add a Makefile (or tasks.ps1 on Windows) with
targets: test, tabular, graph, fusion, nlp, api, dashboard. Add ruff config and fix
lint only in the new files. Run pytest -q and show output. Commit.

---------------------------------------------------------------
## FOLLOW-UP 2 - tabular time split
---------------------------------------------------------------
Run python scripts/03b_tabular_timesplit.py on the real data (if RAM is tight, add
--models xgboost,lightgbm first). Then write notebooks/tabular_timesplit_report/
COMPARISON.md: a table of Week 3 (shuffled CV) vs 03b (time split) for each model, and
3-4 sentences explaining the difference (leakage, threshold tuning). Numbers only from
the CSVs. Also make a PR-curve PNG for the winner on the test set.

---------------------------------------------------------------
## FOLLOW-UP 3 - graph with validation split (run on Colab if no GPU)
---------------------------------------------------------------
Help me run scripts/05b_graph_ablation_valsplit.py. If torch_geometric fails to install
locally, write colab/run_05b.ipynb that mounts Drive, installs torch_geometric, uploads
only notebooks/graph_pipeline/*.npy + train/test ids + scripts/gnn_models.py +
common_eval.py + 05b, runs it, and zips notebooks/graph_valsplit_report/ for download.
Then compare with Week 5 in a COMPARISON.md like before.

---------------------------------------------------------------
## FOLLOW-UP 4 - fusion
---------------------------------------------------------------
Run scripts/08_fusion_elliptic.py. Read fusion_summary.csv, f1_by_time_step.csv and
fusion_params.json. Tell me honestly: does graph-only or fusion beat tabular-only on
PR-AUC? If tabular-only wins, say so and suggest one principled improvement (for
example adding GNN embeddings as XGBoost features), implement it as 08b, rerun, and
compare. Explain the gate coefficients in plain words for the report.

---------------------------------------------------------------
## FOLLOW-UP 5 - demo
---------------------------------------------------------------
Run 06b, then start the API and dashboard. Check every endpoint in /docs works with
real models, measure average latency of /score/transaction over 200 requests and save
it to docs/latency.md. Fix any bugs. Add a Dockerfile + docker-compose.yml (api +
dashboard) and a DEMO.md with a 3-minute demo script (which transaction, which SMS,
which node to show).

---------------------------------------------------------------
## FOLLOW-UP 6 - tests, CI, README
---------------------------------------------------------------
Add tests for the API with a tiny fake model, for 05b's mask logic (no torch needed
for mask building - refactor into a small function if required), and for fusion
threshold selection. Add .github/workflows/ci.yml running ruff + pytest on push.
Rewrite README.md: overview, architecture diagram (mermaid), datasets and honesty
note, setup, run order, results table (from CSVs), demo, limitations.

---------------------------------------------------------------
## FOLLOW-UP 7 - report figures and tables
---------------------------------------------------------------
Write scripts/09_make_report_assets.py that reads every *_summary.csv and writes
docs/report_assets/: LaTeX + Markdown tables (Tables 1-4 in docs/REPORT_OUTLINE.md)
and clean 300-dpi PNGs (ablation bars, PR curves, F1-by-time-step, SHAP, GNNExplainer).
Same colours and fonts on every figure. Run it and list the outputs.

---------------------------------------------------------------
## GEMINI (AI Pro) - literature review, use Deep Research
---------------------------------------------------------------
Research published work (2019-2026) for my capstone on fraud detection in digital
payments using gradient-boosted trees, graph neural networks (GraphSAGE, GAT, GCN) on
the Elliptic dataset, tabular+graph fusion, SHAP / GNNExplainer, and SMS/URL phishing
detection. Give 15 papers with full citation, DOI or arXiv link, dataset, method, best
metric, and the gap each leaves. Only include papers you can link to. Then write a
1,200-word literature review in IEEE citation style that leads to my approach:
time-aware evaluation, PR-AUC, and gated fusion of tabular and graph scores.
(Open every link yourself before using it in the report.)

---------------------------------------------------------------
## GEMINI - report chapters (paste your results CSVs + REPORT_OUTLINE.md)
---------------------------------------------------------------
Using ONLY the attached results files and outline, draft chapters 5-8 of my capstone
report. Every number must match the files exactly; if a number is missing, write
[MISSING: what] instead of guessing. Formal academic tone, short paragraphs, refer to
tables/figures by number.

---------------------------------------------------------------
## VIVA PREP (either tool)
---------------------------------------------------------------
Act as a strict capstone panel. Using my README, results CSVs and report draft, ask me
15 hard questions one at a time (data realism, leakage, metrics, why graphs, drift after
time step 43, real-time design, ethics). After each answer, score it 1-5 and show a
better answer.
