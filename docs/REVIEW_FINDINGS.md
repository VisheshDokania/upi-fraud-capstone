# Review of Weeks 1-7 (what a panel could challenge, and the fix)

Your current results (from the notebooks/ folder in the zip):
- Tabular, IEEE-CIS, 5-fold shuffled CV: XGBoost ROC-AUC 0.961, F1 0.609 (LightGBM 0.957 / 0.483, CatBoost 0.953 / 0.455)
- Graph, Elliptic temporal split: GraphSAGE ROC-AUC 0.900, F1 0.574 (GAT 0.884 / 0.483, GCN 0.843 / 0.471)
- NLP: SMS spam ROC-AUC 0.990, F1 0.932; phishing URLs ROC-AUC 0.979, F1 0.839
- Explainability: SHAP top drivers V149, V258, V198, V187, C14; GNNExplainer on node 21458 (degree 233)

| # | Issue | Why it matters | Fix (new file) |
|---|---|---|---|
| 1 | Week 3 uses shuffled StratifiedKFold on IEEE-CIS | Data is time-ordered; shuffling leaks the future, scores are optimistic | 03b: train on first 70% by TransactionDT, validate on next 10%, test on last 20% |
| 2 | Week 3 early-stops on the same fold it reports | Reported fold score is optimistic | 03b: early stopping + threshold on validation only |
| 3 | Models compared at fixed threshold 0.5 | Unfair across class-weighted models; explains LightGBM/CatBoost's low precision | common_eval: threshold tuned on validation; PR-AUC as headline |
| 4 | train_identity.csv downloaded but never used | Loses device/browser features | 03b merges it |
| 5 | Week 5 picks the best epoch by TEST F1 | Test-set leakage; GNN numbers are optimistic | 05b: validation period steps 30-34 |
| 6 | Week 5 uses one-direction edges, raw features | Money-flow graphs usually work better with both directions + scaled features | 05b: adds reverse edges, standardises on train stats |
| 7 | No tabular baseline on Elliptic | Published work (Weber et al., 2019) found a Random Forest on node features beats GCN on Elliptic; a panel may ask "does the graph even help?" | 08: XGBoost-on-features vs GNN vs fusion, same split |
| 8 | "Gated ensemble" across IEEE-CIS + Elliptic is not possible | The datasets share no transaction key, so scores can't be fused per transaction | 08: fusion done on Elliptic, which has both features and a graph |
| 9 | Week 6 clean_text replaces links with "URL" then deletes uppercase letters | The link signal is silently lost | app/text_features.py uses "urltoken" |
| 10 | Phishing URL set has duplicate rows; random split | Same URL in train and test inflates scores | 06b drops duplicates before splitting |
| 11 | Week 6 models are never saved | No way to serve them | 06b saves pipelines |
| 12 | The Drive zip includes .env (Kaggle key), .venv (46k files) and raw data (1.1 GB) | Anyone with the link can download the key | Rotate the Kaggle token; share code only (see below) |

## Security note (do this first)

The shared zip contains `vscode-project/.env` with a Kaggle username and key filled in,
and the Drive link downloads without signing in. Priyanshu should:
1. kaggle.com/settings -> API -> "Expire token", then create a new one
2. share a code-only zip or, better, a private GitHub repo (the .gitignore already excludes .env, data/, .venv/)

## Honesty rules to keep (the earlier plan already set these, keep them)
- No invented "Indian UPI fraud" rows. Say clearly: public transaction-level UPI data does not exist; IEEE-CIS, Elliptic and ULB credit-card data are real; PaySim is synthetic and used as a baseline.
- Report numbers only from real runs. Smoke-test numbers never go in the report.
- Numbers from the shuffled-CV run can go in an appendix labelled "random split (optimistic)" next to the time-split numbers. Showing both is a strength.
