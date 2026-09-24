# Capstone model comparisons

## Week 3 tabular results vs. 03b time split

Metrics are copied from `notebooks/ablation_report/ablation_summary.csv` and `notebooks/tabular_timesplit_report/timesplit_summary.csv`.

| Evaluation | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Week 3 shuffled CV mean | XGBoost | Not reported in CSV | 0.9610 | 0.6090 | 0.4888 | 0.8079 |
| 03b validation time split | XGBoost | 0.5951 | 0.9214 | 0.6104 | 0.7076 | 0.5366 |
| 03b test time split | XGBoost | 0.3264 | 0.8699 | 0.3639 | 0.2964 | 0.4712 |
| 03b validation time split | LightGBM | 0.7046 | 0.9367 | 0.6687 | 0.7972 | 0.5759 |
| 03b test time split | LightGBM | 0.5698 | 0.9084 | 0.5520 | 0.6795 | 0.4648 |

LightGBM is the validation PR-AUC winner. Both models were evaluated once on the fixed test period; test scores did not select the winner. The Week 3 CSV does not include PR-AUC, and shuffled CV is not directly comparable to a later-period test. Validation and test fraud rates were similar in `notebooks/tabular_timesplit_report/period_summary.csv`; `feature_drift.csv` shows numeric feature distribution shifts. This is consistent with temporal covariate drift contributing to the PR-AUC gap, but does not identify a causal feature.

PR curves: [tabular_timesplit_pr_curve.png](tabular_timesplit_pr_curve.png).

## Week 5 graph results vs. 05b

Metrics are copied from `notebooks/graph_ablation_report/graph_ablation_summary.csv` and `notebooks/graph_valsplit_report/graph_valsplit_summary.csv`.

| Evaluation | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Week 5 reported result | GraphSAGE | Not reported in CSV | 0.9002 | 0.5739 | 0.5365 | 0.6168 |
| 05b validation steps 30-32 | GraphSAGE | 0.9686 | 0.9861 | 0.9096 | 0.9398 | 0.8814 |
| 05b test steps 35-49 | GraphSAGE | 0.4246 | 0.8605 | 0.4959 | 0.5128 | 0.4801 |
| 05b validation steps 30-32 | GAT | 0.9265 | 0.9657 | 0.8437 | 0.7866 | 0.9096 |
| 05b test steps 35-49 | GAT | 0.3081 | 0.8236 | 0.4406 | 0.3869 | 0.5115 |
| 05b validation steps 30-32 | GCN | 0.8746 | 0.9601 | 0.8074 | 0.8909 | 0.7382 |
| 05b test steps 35-49 | GCN | 0.4528 | 0.8708 | 0.4784 | 0.5671 | 0.4137 |

05b selects GraphSAGE by validation PR-AUC. GCN has the highest 05b test PR-AUC, but test scores do not select the model. Week 5's CSV has no PR-AUC and its original protocol selected using test F1, so the comparison is descriptive rather than like-for-like.

The GNN's epoch selection used steps 30-32, the same steps used to fit the fusers; final test metrics on steps 35-49 are unaffected.

## Follow-up 4 Elliptic fusion

Metrics are copied from `notebooks/fusion_report/fusion_summary.csv`. Base tabular predictions and learned fusers use labelled steps 30-32; thresholds are tuned on steps 33-34; final metrics use steps 35-49.

| Variant | Test PR-AUC | Test ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Tabular only | 0.7906 | 0.9260 | 0.7701 | 0.8189 | 0.7267 |
| Weighted | 0.7822 | 0.9176 | 0.7701 | 0.8168 | 0.7285 |
| Stacked | 0.7776 | 0.9173 | 0.7944 | 0.8694 | 0.7313 |
| Tabular local only (f0-f92) | 0.7664 | 0.8924 | 0.7324 | 0.7519 | 0.7138 |
| Average | 0.7569 | 0.9157 | 0.7379 | 0.8424 | 0.6565 |
| Gated | 0.7435 | 0.9048 | 0.7794 | 0.8330 | 0.7322 |
| Graph only | 0.4246 | 0.8605 | 0.4959 | 0.5128 | 0.4801 |

Tabular only has the highest test PR-AUC in this run; the gate does not win. This is a descriptive test comparison, not a test-based selection rule. The gate is a small logistic model that combines the tabular risk score, graph risk score, node log-degree, and risk-by-degree interactions. In plain terms, it can adjust how much it trusts each score based on how connected the node is. `fused_node_proba.npy` contains the gated scores for the API; the comparison table keeps every variant visible.
