# Week 3 vs. temporal split

Metrics below are from `notebooks/ablation_report/ablation_summary.csv` and `notebooks/tabular_timesplit_report/timesplit_summary.csv`.

| Evaluation | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Week 3 shuffled CV mean | XGBoost | Not reported in CSV | 0.9610 | 0.6090 | 0.4888 | 0.8079 |
| 03b validation time split | XGBoost | 0.5951 | 0.9214 | 0.6104 | 0.7076 | 0.5366 |
| 03b test time split | XGBoost | 0.3264 | 0.8699 | 0.3639 | 0.2964 | 0.4712 |
| 03b validation time split | LightGBM | 0.7046 | 0.9367 | 0.6687 | 0.7972 | 0.5759 |
| 03b test time split | LightGBM | 0.5698 | 0.9084 | 0.5520 | 0.6795 | 0.4648 |

LightGBM is now the validation PR-AUC winner; the winner changed from XGBoost. Thresholds and early stopping use validation data. Both models were evaluated once on the fixed test period; test scores did not select the winner.

The Week 3 CSV does not include PR-AUC, and its shuffled CV results are not directly comparable to 03b's later-period test results. The temporal test gap is not explained by fraud prevalence alone: `period_summary.csv` reports validation and test fraud rates of 0.0349003 and 0.0344092. `feature_drift.csv` shows changes in numeric feature distributions; the largest absolute standardized mean difference is 0.5241 for `id_13`, with several `V` features also near that level. This is consistent with temporal covariate drift contributing to the PR-AUC drop, though the diagnostics do not establish which feature changes caused it.

PR curves for both models: [tabular_timesplit_pr_curve.png](tabular_timesplit_pr_curve.png). Per-model scores and per-period/per-feature diagnostics are in the cited CSVs.
