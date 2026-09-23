# Week 3 vs. temporal split

Values below are copied from `notebooks/ablation_report/ablation_summary.csv` and `notebooks/tabular_timesplit_report/timesplit_summary.csv`.

| Evaluation | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Week 3 shuffled 5-fold mean | XGBoost | Not reported in CSV | 0.9610 | 0.6090 | 0.4888 | 0.8079 |
| 03b validation time split | XGBoost | 0.5951 | 0.9214 | 0.6104 | 0.7076 | 0.5366 |
| 03b test time split | XGBoost | 0.3264 | 0.8699 | 0.3639 | 0.2964 | 0.4712 |

The Week 3 CSV does not include PR-AUC. Its shuffled cross-validation results and 03b's validation/test results use different evaluation protocols, so they are not directly comparable. The 03b model was selected by validation PR-AUC; test metrics were calculated once for the selected XGBoost model.

PR curve: [tabular_timesplit_pr_curve.png](tabular_timesplit_pr_curve.png). The curve was generated from the 03b test predictions.
