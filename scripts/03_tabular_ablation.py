"""
Week 3 — Tabular Ablation: LightGBM vs XGBoost vs CatBoost (local version)
=============================================================================
Run:
    cd scripts
    python 03_tabular_ablation.py
"""

import gc
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "ablation_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


def load_ieee_cis():
    fpath = DATA_DIR / "ieee_cis" / "train_transaction.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found — run 01_data_acquisition.py first.")
    df = pd.read_csv(fpath)
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, y, cat_cols


def run_lightgbm(X_tr, y_tr, X_val, y_val, cat_cols):
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set)
    params = dict(objective="binary", metric="auc", verbosity=-1,
                  scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                  seed=RANDOM_STATE)
    model = lgb.train(params, train_set, num_boost_round=500, valid_sets=[val_set],
                       callbacks=[lgb.early_stopping(30, verbose=False)])
    proba = model.predict(X_val, num_iteration=model.best_iteration)
    return proba, model


def run_xgboost(X_tr, y_tr, X_val, y_val, cat_cols):
    X_tr2, X_val2 = X_tr.copy(), X_val.copy()
    for c in cat_cols:
        X_tr2[c] = X_tr2[c].cat.codes
        X_val2[c] = X_val2[c].cat.codes
    dtrain = xgb.DMatrix(X_tr2, label=y_tr)
    dval = xgb.DMatrix(X_val2, label=y_val)
    params = dict(objective="binary:logistic", eval_metric="auc",
                  scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                  seed=RANDOM_STATE, tree_method="hist")
    model = xgb.train(params, dtrain, num_boost_round=500,
                       evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
    proba = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    del X_tr2, X_val2, dtrain, dval
    return proba, model


def run_catboost(X_tr, y_tr, X_val, y_val, cat_cols):
    X_tr2, X_val2 = X_tr.copy(), X_val.copy()
    for c in cat_cols:
        # Fill NaN BEFORE casting to str -- .astype(str) on a pandas
        # Categorical with missing values leaves NaN as float NaN, not
        # the string "nan", so CatBoost crashes on the float it still sees.
        X_tr2[c] = X_tr2[c].astype(object).fillna("missing").astype(str)
        X_val2[c] = X_val2[c].astype(object).fillna("missing").astype(str)
    model = CatBoostClassifier(
        iterations=500, eval_metric="AUC", loss_function="Logloss",
        cat_features=cat_cols, random_seed=RANDOM_STATE, verbose=False,
        early_stopping_rounds=30,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
    )
    model.fit(X_tr2, y_tr, eval_set=(X_val2, y_val), use_best_model=True)
    proba = model.predict_proba(X_val2)[:, 1]
    del X_tr2, X_val2
    return proba, model


def score(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y_true, proba), "f1": f1_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0)}


def run_ablation(X, y, cat_cols):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {m: [] for m in ["lightgbm", "xgboost", "catboost"]}
    # keep only fold-0's model per algorithm for Week 7 SHAP -- keeping all
    # 15 trained models (3 algos x 5 folds) in memory is wasteful and
    # unnecessary, since Week 7 only explains one representative model
    fold0_models = {}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        for name, fn in [("lightgbm", run_lightgbm), ("xgboost", run_xgboost), ("catboost", run_catboost)]:
            t0 = time.time()
            proba, model = fn(X_tr, y_tr, X_val, y_val, cat_cols)
            elapsed = time.time() - t0
            metrics = score(y_val, proba)
            metrics["fold"] = fold
            metrics["train_seconds"] = round(elapsed, 2)
            results[name].append(metrics)
            if fold == 0:
                fold0_models[name] = model
            else:
                del model  # only fold 0's model is kept
            auc_v, f1_v = metrics["roc_auc"], metrics["f1"]
            print(f"fold {fold} {name}: AUC={auc_v:.4f} F1={f1_v:.4f} ({elapsed:.1f}s)")
        del X_tr, X_val, y_tr, y_val
        gc.collect()

    summary = {}
    for name, folds in results.items():
        df = pd.DataFrame(folds)
        summary[name] = {"mean_roc_auc": df.roc_auc.mean(), "std_roc_auc": df.roc_auc.std(),
                          "mean_f1": df.f1.mean(), "std_f1": df.f1.std(),
                          "mean_precision": df.precision.mean(), "mean_recall": df.recall.mean(),
                          "mean_train_seconds": df.train_seconds.mean()}
        df.to_csv(OUT_DIR / f"{name}_folds.csv", index=False)

    summary_df = pd.DataFrame(summary).T.sort_values("mean_f1", ascending=False)
    summary_df.to_csv(OUT_DIR / "ablation_summary.csv")
    winner = summary_df.index[0]

    print("\n" + "=" * 60)
    print("TABULAR ABLATION — SUMMARY (5-fold CV, identical folds per model)")
    print("=" * 60)
    print(summary_df.round(4).to_string())
    print(f"\nWinner (highest mean fraud-class F1): {winner}")

    with open(OUT_DIR / "ablation_winner.json", "w") as f:
        json.dump({"winner": winner, "summary": summary}, f, indent=2, default=str)

    # persist the winning model to disk so Week 7 can load it without
    # needing this whole script's variables still in memory
    import pickle
    with open(OUT_DIR / "winner_model.pkl", "wb") as f:
        pickle.dump({"name": winner, "model": fold0_models[winner]}, f)
    print(f"Winning model saved to {OUT_DIR}/winner_model.pkl for Week 7")

    return summary_df, winner


if __name__ == "__main__":
    X, y, cat_cols = load_ieee_cis()
    print(f"Loaded real IEEE-CIS: {X.shape[0]:,} rows, {X.shape[1]} features, "
          f"fraud rate {y.mean()*100:.3f}%")
    run_ablation(X, y, cat_cols)
    print("\nScript exiting now — all training memory released.")
