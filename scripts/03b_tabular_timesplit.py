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
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common_eval import best_f1_threshold, evaluate, fix_string_categories

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
    return tr


def _compact_dtypes(path, preserve):
    """Infer low-memory read types from a small sample, preserving join/time keys."""
    sample = pd.read_csv(path, nrows=1000)
    dtypes = {}
    for column, dtype in sample.dtypes.items():
        if column in preserve:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
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
                           metric="average_precision", random_state=RANDOM_STATE, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return m


def fit_cat(Xtr, ytr, Xva, yva, cat_cols):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=2000, learning_rate=0.08, depth=8, eval_metric="PRAUC",
                           auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False,
                           early_stopping_rounds=100, cat_features=cat_cols)
    m.fit(fix_string_categories(Xtr, cat_cols), ytr,
          eval_set=(fix_string_categories(Xva, cat_cols), yva), use_best_model=True)
    return m


def predict(m, X, needs_str_cats=False, cat_cols=None):
    if needs_str_cats:
        X = fix_string_categories(X, cat_cols or [])
    return m.predict_proba(X)[:, 1]


def write_drift_report(va, te, Xva, Xte, out_dir):
    """Write split label/time summaries and numeric validation-to-test drift."""
    periods = []
    for name, frame in (("validation", va), ("test", te)):
        periods.append({
            "period": name,
            "rows": len(frame),
            "transaction_dt_min": frame["TransactionDT"].min(),
            "transaction_dt_max": frame["TransactionDT"].max(),
            "fraud_rate": frame["isFraud"].mean(),
        })
    pd.DataFrame(periods).to_csv(out_dir / "period_summary.csv", index=False)

    numeric = Xva.select_dtypes(include=[np.number]).columns.intersection(
        Xte.select_dtypes(include=[np.number]).columns
    )
    drift = []
    for column in numeric:
        val, test = Xva[column], Xte[column]
        val_mean, test_mean = val.mean(), test.mean()
        pooled_std = np.sqrt((val.var() + test.var()) / 2)
        smd = abs(val_mean - test_mean) / pooled_std if pooled_std > 0 else 0.0
        drift.append({
            "feature": column,
            "abs_standardized_mean_difference": smd,
            "validation_missing_rate": val.isna().mean(),
            "test_missing_rate": test.isna().mean(),
            "absolute_missing_rate_difference": abs(val.isna().mean() - test.isna().mean()),
        })
    pd.DataFrame(drift).sort_values(
        "abs_standardized_mean_difference", ascending=False
    ).to_csv(out_dir / "feature_drift.csv", index=False)


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
    demo = te.groupby("isFraud", sort=False).head(10).copy()
    Xtr, ytr = prep(tr, cat_cols)
    levels = {c: list(Xtr[c].cat.categories) for c in cat_cols}
    Xva, yva = prep(va, cat_cols, levels)
    Xte, yte = prep(te, cat_cols, levels)
    write_drift_report(va, te, Xva, Xte, out_dir)
    del tr, va, te
    gc.collect()
    features = list(Xtr.columns)
    cat_levels = {c: list(Xtr[c].cat.categories) for c in cat_cols}
    print(f"train {len(ytr):,} ({ytr.mean():.3%} fraud) | val {len(yva):,} | test {len(yte):,} ({yte.mean():.3%})")

    fitters = {"xgboost": fit_xgb, "lightgbm": fit_lgb, "catboost": fit_cat}
    rows = []
    winner = None
    winner_score = -np.inf
    curve_paths = {}
    for name in models:
        m = val_p = test_p = None
        try:
            t0 = time.time()
            m = fitters[name](Xtr, ytr, Xva, yva, cat_cols)
            needs_str_cats = name == "catboost"
            val_p = predict(m, Xva, needs_str_cats, cat_cols)
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
            test_p = predict(m, Xte, needs_str_cats, cat_cols)
            test_results = evaluate(yte, test_p, thr)
            row.update({f"test_{key}": value for key, value in test_results.items()})
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(yte, test_p)
            curve_path = out_dir / f"pr_curve_{name}.csv"
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
                curve_path, index=False
            )
            curve_paths[name] = curve_path

            rows.append(row)
            summary = pd.DataFrame(rows).set_index("model").sort_values(
                "val_pr_auc", ascending=False, na_position="last"
            )
            summary.to_csv(out_dir / "timesplit_summary.csv")
            bundle = {"name": name, "model": m, "features": features,
                      "cat_cols": cat_cols, "cat_levels": cat_levels,
                      "threshold": thr, "smoke_test": smoke,
                      "needs_str_cats": needs_str_cats}
            model_target = MODEL_DIR / f"tabular_{name}{'_SMOKE' if smoke else ''}.pkl"
            with open(model_target, "wb") as f:
                pickle.dump(bundle, f)

            if np.isfinite(row["val_pr_auc"]) and row["val_pr_auc"] > winner_score:
                winner = name
                winner_score = row["val_pr_auc"]
            print(
                f"{name}: validation PR-AUC={row['val_pr_auc']:.4f} "
                f"test PR-AUC={row['test_pr_auc']:.4f} "
                f"ROC-AUC={row['val_roc_auc']:.4f} F1={row['val_f1']:.4f} "
                f"(threshold {thr:.3f})"
            )
            m = val_p = test_p = val_results = test_results = bundle = None
            gc.collect()
        except Exception as e:
            print(f"failed {name}: {e}")
            traceback.print_exc()
            if not any(row.get("model") == name for row in rows):
                rows.append({"model": name, "error": str(e)})
            pd.DataFrame(rows).set_index("model").to_csv(
                out_dir / "timesplit_summary.csv"
            )
            m = val_p = test_p = None
            gc.collect()

    if winner is None:
        raise RuntimeError("No requested model produced a finite validation PR-AUC.")

    summary = pd.DataFrame(rows).set_index("model").sort_values("val_pr_auc", ascending=False)
    del Xtr, Xva, Xte, ytr, yva, yte
    gc.collect()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, path in curve_paths.items():
        curve = pd.read_csv(path)
        score = summary.loc[name, "test_pr_auc"]
        ax.plot(curve["recall"], curve["precision"],
                label=f"{name} (test PR-AUC={score:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Time-split test precision-recall curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=160)
    if not smoke:
        fig.savefig(ROOT / "tabular_timesplit_pr_curve.png", dpi=160)
    plt.close(fig)
    summary.to_csv(out_dir / "timesplit_summary.csv")
    print("\nValidation model comparison (winner selected by validation PR-AUC):")
    print(summary.drop(columns=[c for c in summary.columns if c.startswith("test_")]).round(4).to_string())
    print(f"\nSelected model: {winner}")
    print("Each model was evaluated once on the fixed test period; selection used validation PR-AUC.")
    if smoke:
        print("[SMOKE TEST - NOT A RESULT]")

    # Keep only a small balanced sample for the dashboard demo.
    demo.to_csv(out_dir / "demo_transactions.csv", index=False)
    target = MODEL_DIR / ("tabular_winner_SMOKE.pkl" if smoke else "tabular_winner.pkl")
    winner_source = MODEL_DIR / f"tabular_{winner}{'_SMOKE' if smoke else ''}.pkl"
    shutil.copyfile(winner_source, target)
    with open(out_dir / "winner.json", "w") as f:
        json.dump({"winner": winner, "smoke_test": smoke}, f, indent=2)
    print(f"Saved {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--models", default="xgboost,lightgbm,catboost")
    a = ap.parse_args()
    main(a.smoke_test, a.models.split(","))
