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

## Week 5 vs. Follow-up 3 graph results

Values are from `notebooks/graph_ablation_report/graph_ablation_summary.csv` (Week 5) and `notebooks/graph_valsplit_report/graph_valsplit_summary.csv` (05b).

| Evaluation | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Week 5 reported result | GraphSAGE | Not reported in CSV | 0.9002 | 0.5739 | 0.5365 | 0.6168 |
| 05b validation period | GraphSAGE | 0.9561 | 0.9861 | 0.9049 | 0.9080 | 0.9019 |
| 05b test period | GraphSAGE | 0.4550 | 0.8700 | 0.5300 | 0.5033 | 0.5596 |
| 05b validation period | GAT | 0.8971 | 0.9650 | 0.8369 | 0.8790 | 0.7986 |
| 05b test period | GAT | 0.3397 | 0.8308 | 0.4503 | 0.4646 | 0.4367 |
| 05b validation period | GCN | 0.8401 | 0.9609 | 0.8137 | 0.7867 | 0.8426 |
| 05b test period | GCN | 0.4866 | 0.8756 | 0.5281 | 0.5356 | 0.5208 |

05b selects GraphSAGE by validation PR-AUC. GCN has the highest 05b test PR-AUC, but test scores do not select the winner. Week 5's CSV has no PR-AUC and its original protocol selected using test F1, so this comparison is descriptive rather than like-for-like.
