"""
Week 3b - Tabular ablation, re-run with a TIME-BASED split (new file).
========================================================================
Why: 03_tabular_ablation.py used shuffled StratifiedKFold on IEEE-CIS. IEEE-CIS is
time-ordered (TransactionDT), and the same cards/users appear across time, so a
shuffled split leaks the future into training and inflates scores. It also
early-stopped on the same fold it reported, and compared models at a fixed 0.5
threshold. A viva panel can catch all three.

This script (does NOT overwrite Week 3 outputs):
  1. merges train_transaction + train_identity (identity was never used before)
  2. sorts by TransactionDT: first 70% train, next 10% validation, last 20% test
  3. early-stops and tunes the decision threshold on validation only
  4. reports PR-AUC, ROC-AUC, F1/precision/recall, recall@1%FPR on the untouched test
  5. saves the winner + feature list for the API (app/api.py)

Run:
    cd scripts
    python 03b_tabular_timesplit.py            # real data
    python 03b_tabular_timesplit.py --smoke-test   # tiny synthetic data, NOT a result
"""
import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common_eval import best_f1_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "notebooks" / "tabular_timesplit_report"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42


def load_ieee(smoke=False):
    if smoke:
        rng = np.random.default_rng(0)
        n = 6000
        df = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.sort(rng.integers(0, 10**6, n)),
                           "TransactionAmt": rng.gamma(2, 50, n), "ProductCD": rng.choice(list("WHCSR"), n),
                           "card4": rng.choice(["visa", "mastercard", None], n)})
        for i in range(1, 11):
            df[f"V{i}"] = rng.normal(size=n)
        logit = 1.5 * df["V1"] - 1.0 * df["V2"] - 3.5
        df["isFraud"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return df
    tr_path = DATA_DIR / "ieee_cis" / "train_transaction.csv"
    tr_dtypes = _compact_dtypes(tr_path, {"TransactionID", "TransactionDT", "isFraud"})
    tr = pd.read_csv(tr_path, dtype=tr_dtypes)
    idp = DATA_DIR / "ieee_cis" / "train_identity.csv"
    if idp.exists():
        id_dtypes = _compact_dtypes(idp, {"TransactionID"})
        identity = pd.read_csv(idp, dtype=id_dtypes)
        tr = tr.merge(identity, on="TransactionID", how="left")
        del identity
        gc.collect()
    # Some columns inferred as integers in the sample may contain missing values later.
    fcols = tr.select_dtypes("float64").columns
    tr[fcols] = tr[fcols].astype("float32")
    return tr


def _compact_dtypes(path, preserve):
    """Infer low-memory read types from a small sample, preserving join/time keys."""
    sample = pd.read_csv(path, nrows=1000)
    dtypes = {}
    for column, dtype in sample.dtypes.items():
        if column in preserve:
            continue
        if pd.api.types.is_float_dtype(dtype):
            dtypes[column] = "float32"
        elif (pd.api.types.is_object_dtype(dtype)
              or pd.api.types.is_string_dtype(dtype)):
            dtypes[column] = "category"
    del sample
    gc.collect()
    return dtypes


def time_split(df):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.7), int(n * 0.8)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def prep(df, cat_cols, levels=None):
    """levels: category lists learned on TRAIN, so val/test/API rows use the same codes."""
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=levels[c]) if levels else X[c].astype("category")
    return X, df["isFraud"].values


def fit_xgb(Xtr, ytr, Xva, yva, cat_cols):
    import xgboost as xgb
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
                          colsample_bytree=0.6, tree_method="hist", enable_categorical=True,
                          eval_metric="aucpr", early_stopping_rounds=100, scale_pos_weight=spw,
                          random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lgb(Xtr, ytr, Xva, yva, cat_cols):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=256, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, is_unbalance=True,
                           random_state=RANDOM_STATE, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def fit_cat(Xtr, ytr, Xva, yva, cat_cols):
    from catboost import CatBoostClassifier
    def fix(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(object).fillna("missing").astype(str)
        return X
    m = CatBoostClassifier(iterations=2000, learning_rate=0.08, depth=8, eval_metric="PRAUC",
                           auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False,
                           early_stopping_rounds=100, cat_features=cat_cols)
    m.fit(fix(Xtr), ytr, eval_set=(fix(Xva), yva), use_best_model=True)
    m._fix = fix  # used below for predict
    return m


def predict(m, X):
    if hasattr(m, "_fix"):
        X = m._fix(X)
    return m.predict_proba(X)[:, 1]


def main(smoke, models):
    out_dir = OUT_DIR / "smoke" if smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_ieee(smoke)
    cat_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c].dtype)
        or pd.api.types.is_string_dtype(df[c].dtype)
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    tr, va, te = time_split(df)
    del df
    gc.collect()
    Xtr, ytr = prep(tr, cat_cols)
    levels = {c: list(Xtr[c].cat.categories) for c in cat_cols}
    Xva, yva = prep(va, cat_cols, levels)
    Xte, yte = prep(te, cat_cols, levels)
    features = list(Xtr.columns)
    cat_levels = {c: list(Xtr[c].cat.categories) for c in cat_cols}
    print(f"train {len(ytr):,} ({ytr.mean():.3%} fraud) | val {len(yva):,} | test {len(yte):,} ({yte.mean():.3%})")

    fitters = {"xgboost": fit_xgb, "lightgbm": fit_lgb, "catboost": fit_cat}
    rows = []
    winner = None
    winner_model = None
    winner_score = -np.inf
    threshold = 0.5
    for name in models:
        try:
            t0 = time.time()
            m = fitters[name](Xtr, ytr, Xva, yva, cat_cols)
            val_p = predict(m, Xva)
            thr = best_f1_threshold(yva, val_p)
            val_results = evaluate(yva, val_p, thr)
            row = {
                "model": name,
                "val_pr_auc": val_results["pr_auc"],
                "val_roc_auc": val_results["roc_auc"],
                "val_f1": val_results["f1"],
                "val_precision": val_results["precision"],
                "val_recall": val_results["recall"],
                "val_recall_at_1pct_fpr": val_results["recall_at_1pct_fpr"],
                "threshold": thr,
                "train_seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            if np.isfinite(row["val_pr_auc"]) and row["val_pr_auc"] > winner_score:
                winner = name
                winner_model = m
                winner_score = row["val_pr_auc"]
                threshold = thr
            print(
                f"{name}: validation PR-AUC={row['val_pr_auc']:.4f} "
                f"ROC-AUC={row['val_roc_auc']:.4f} F1={row['val_f1']:.4f} "
                f"(threshold {thr:.3f})"
            )
            del m, val_p, val_results
            gc.collect()
        except ImportError as e:
            print(f"skipping {name}: {e}")

    if winner_model is None:
        raise RuntimeError("No requested model produced a finite validation PR-AUC.")

    summary = pd.DataFrame(rows).set_index("model").sort_values("val_pr_auc", ascending=False)
    del Xtr, Xva, ytr, yva, tr, va
    gc.collect()
    test_p = predict(winner_model, Xte)
    test_results = evaluate(yte, test_p, threshold)
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(yte, test_p)
    curve = pd.DataFrame({"recall": recall, "precision": precision})
    curve.to_csv(out_dir / "pr_curve.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"{winner} (test PR-AUC={test_results['pr_auc']:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Time-split test precision-recall curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close(fig)
    del test_p, Xte, yte
    gc.collect()
    for metric, value in test_results.items():
        summary.loc[winner, f"test_{metric}"] = value
    summary.to_csv(out_dir / "timesplit_summary.csv")
    print("\nValidation model comparison (winner selected by validation PR-AUC):")
    print(summary.drop(columns=[c for c in summary.columns if c.startswith("test_")]).round(4).to_string())
    print(f"\nSelected model: {winner}")
    print("Test metrics for the selected model (test evaluated once):")
    print(pd.Series(test_results).round(4).to_string())
    if smoke:
        print("[SMOKE TEST - NOT A RESULT]")

    # a few real test rows the dashboard can use as demo inputs
    demo = te.drop(columns=["isFraud"]).groupby(te["isFraud"]).head(10)
    demo.assign(isFraud=te.loc[demo.index, "isFraud"]).to_csv(out_dir / "demo_transactions.csv", index=False)
    bundle = {"name": winner, "model": winner_model, "features": features,
              "cat_cols": cat_cols,
              "cat_levels": cat_levels,
              "threshold": threshold,
              "smoke_test": smoke}
    target = MODEL_DIR / ("tabular_winner_SMOKE.pkl" if smoke else "tabular_winner.pkl")
    with open(target, "wb") as f:
        pickle.dump(bundle, f)
    with open(out_dir / "winner.json", "w") as f:
        json.dump({"winner": winner, "threshold": bundle["threshold"], "smoke_test": smoke}, f, indent=2)
    print(f"Saved {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--models", default="xgboost,lightgbm,catboost")
    a = ap.parse_args()
    main(a.smoke_test, a.models.split(","))
