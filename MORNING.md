# Morning status

- Reviewed `03b`, `common_eval.py`, and tests; fixed validation-only selection, model memory retention, and CSV dtype loading in new code. Each fix is locally committed.
- Smoke command: `& .\.venv\Scripts\python.exe scripts\03b_tabular_timesplit.py --smoke-test --models xgboost`; passed (smoke-only, not a result).
- Real command: `& .\.venv\Scripts\python.exe scripts\03b_tabular_timesplit.py --models xgboost,lightgbm`; completed. XGBoost selected on validation PR-AUC 0.5951; test PR-AUC 0.3264. Full log: `notebooks/tabular_timesplit_report/run.log`.
- Wrote `COMPARISON.md` and `tabular_timesplit_pr_curve.png`; comparison values come from the Week 3 and 03b CSVs. Week 3 CSV has no PR-AUC. Test evaluated once.
- RAM was 0.71 GiB available before the run, fell as low as 0.02 GiB during it, then recovered; C: had 514.40 GiB free before the run. No OOM occurred. Nonfatal LightGBM deprecation, matplotlib cache permission, and pandas fragmentation warnings appeared.
- No files deleted, no `data/` or scripts 01–07 edited, no packages installed, and nothing pushed. No approval items waiting. Five local commits; branch is five commits ahead of `origin/master`.

Next command (Follow-up 2 graph run):

```powershell
& .\.venv\Scripts\python.exe scripts\05b_graph_ablation_valsplit.py
```
