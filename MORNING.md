# Morning status

- The six previous local commits were pushed to `origin/master` before this follow-up.
- Corrected LightGBM to early-stop on average precision; updated CSV numeric loading, retained only a small demo sample, released split frames, and added per-model exception handling, metrics, and model artifacts. Each code fix has its own commit.
- Revised smoke command passed: `& .\.venv\Scripts\python.exe scripts\03b_tabular_timesplit.py --smoke-test --models xgboost,lightgbm` (smoke-only).
- Real run completed with `& .\.venv\Scripts\python.exe -u scripts\03b_tabular_timesplit.py --models xgboost,lightgbm`; see `notebooks/tabular_timesplit_report/run2.log`.
- Validation/test PR-AUC: XGBoost 0.5951/0.3264; LightGBM 0.7046/0.5698. LightGBM is the new validation winner. `COMPARISON.md` summarizes the results and drift evidence.
- Validation/test fraud rates were similar, while the numeric feature drift CSV shows distribution changes. RAM remained tight during training; the run completed. Nonfatal LightGBM deprecation and matplotlib cache permission warnings appeared.
- No data or scripts 01–07 were edited; no packages installed; no files deleted.

Next command (Follow-up 2 graph run):

```powershell
& .\.venv\Scripts\python.exe scripts\05b_graph_ablation_valsplit.py
```
