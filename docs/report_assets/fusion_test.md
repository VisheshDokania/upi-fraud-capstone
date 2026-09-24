# Fusion Test

Source CSV: `notebooks/fusion_report/fusion_summary.csv`.

| model | test_pr_auc | test_roc_auc | test_f1 | test_precision | test_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| tabular_only | 0.7906 | 0.9260 | 0.7701 | 0.8189 | 0.7267 | 0.5925 |
| weighted | 0.7822 | 0.9176 | 0.7701 | 0.8168 | 0.7285 | 0.5257 |
| stacked | 0.7776 | 0.9173 | 0.7944 | 0.8694 | 0.7313 | 0.4152 |
| tabular_local_only | 0.7664 | 0.8924 | 0.7324 | 0.7519 | 0.7138 | 0.7113 |
| avg | 0.7569 | 0.9157 | 0.7379 | 0.8424 | 0.6565 | 0.5218 |
| gated | 0.7435 | 0.9048 | 0.7794 | 0.8330 | 0.7322 | 0.3638 |
| graph_only | 0.4246 | 0.8605 | 0.4959 | 0.5128 | 0.4801 | 0.9111 |
