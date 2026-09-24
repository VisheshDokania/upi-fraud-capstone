"""
Shared evaluation helpers for the fraud models (new file - does not change Weeks 1-7).

Why this exists: fraud data is very imbalanced (IEEE-CIS 3.5% fraud, Elliptic ~10%
illicit in train, ~6.5% in test). ROC-AUC looks great even for weak models, and
F1 at a fixed 0.5 threshold is unfair between models that were trained with
different class weights. So every new script reports:
  - PR-AUC (average precision)  -> the headline metric for imbalanced data
  - ROC-AUC                     -> kept for comparison with the base paper
  - F1 / precision / recall at a threshold tuned on VALIDATION data only
  - recall at a fixed 1% false-positive rate (what a bank ops team cares about)
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def fix_string_categories(X, cat_cols):
    """Convert categorical model inputs to CatBoost-compatible strings."""
    X = X.copy()
    for column in cat_cols:
        X[column] = X[column].astype(object).fillna("missing").astype(str)
    return X


def make_time_masks(y, time_steps, periods):
    """Build disjoint labelled-node masks from explicit time-step groups."""
    y = np.asarray(y)
    time_steps = np.asarray(time_steps)
    labelled = y >= 0
    return {
        name: labelled & np.isin(time_steps, list(steps))
        for name, steps in periods.items()
    }


def make_graph_masks(y, time_steps):
    """05b labelled masks, with steps reserved for epoch and threshold tuning."""
    return make_time_masks(y, time_steps, {
        "train": range(1, 30),
        "val": range(30, 33),
        "threshold": range(33, 35),
        "test": range(35, 50),
    })


def best_f1_threshold(y_val, p_val):
    """Pick the threshold that maximises F1 on the validation set."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve returns one more (prec, rec) pair than thresholds
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def recall_at_fpr(y_true, p, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(y_true, p, threshold):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        "pr_auc": float(average_precision_score(y_true, p)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, p)) if has_both else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, p, 0.01) if has_both else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }
