"""
Week 8 - Tabular + Graph fusion (gated ensemble) on Elliptic (new file).
=========================================================================
Important design point for the viva:
IEEE-CIS (card transactions) and Elliptic (Bitcoin transactions) share NO common
key, so you cannot fuse a tabular score and a graph score for the same
transaction across those two datasets. The honest place to fuse is a dataset
that has BOTH per-transaction features AND a transaction graph: Elliptic.
  - tabular branch: XGBoost on each node's own 165 features (no graph)
  - graph branch  : the Week 5b GNN winner (uses neighbours)
  - fusion        : 4 variants, all learned on the validation period only
      avg          simple mean of the two probabilities
      weighted     w*p_tab + (1-w)*p_gnn, w tuned on validation PR-AUC
      stacked      logistic regression on [p_tab, p_gnn]
      gated        logistic regression on [p_tab, p_gnn, log-degree and
                   interactions] -> the model learns when to trust the graph
                   (well-connected nodes) vs the node's own features
Fusion/meta-models are fitted on labelled steps 30-32. Decision thresholds are
tuned on labelled steps 33-34, and final metrics are measured on steps 35-49.
The tabular-local baseline uses only Elliptic features f0-f92.

Run:
    cd scripts
    python 05b_graph_ablation_valsplit.py   # first, produces gnn_node_proba.npy
    python 08_fusion_elliptic.py
    python 08_fusion_elliptic.py --smoke-test   # synthetic, NOT a result
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from common_eval import best_f1_threshold, evaluate, make_time_masks

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "notebooks" / "graph_pipeline"
GNN_DIR = ROOT / "notebooks" / "graph_valsplit_report"
OUT_DIR = ROOT / "notebooks" / "fusion_report"
TRAIN_STEPS = range(1, 30)
FUSER_FIT_STEPS = range(30, 33)
THRESHOLD_STEPS = range(33, 35)
TEST_STEPS = range(35, 50)


def load(smoke):
    if smoke:
        rng = np.random.default_rng(1)
        n = 8000
        X = rng.normal(size=(n, 165)).astype("float32")
        ts = rng.integers(1, 50, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - 3)))).astype(int)
        y[rng.random(n) < 0.6] = -1  # unlabelled, like Elliptic
        src, dst = rng.integers(0, n, 12000), rng.integers(0, n, 12000)
        ei = np.vstack([src, dst])
        p_gnn = np.clip(0.5 * (y == 1) + rng.normal(0.2, 0.2, n), 0, 1)
        return X, y, ts, ei, p_gnn
    X = np.load(GRAPH_DIR / "elliptic_X.npy")
    y = np.load(GRAPH_DIR / "elliptic_y.npy")
    ts = np.load(GRAPH_DIR / "elliptic_timeSteps.npy")
    ei = np.load(GRAPH_DIR / "elliptic_edge_index.npy")
    p_path = GNN_DIR / "gnn_node_proba.npy"
    if not p_path.exists():
        raise FileNotFoundError(f"{p_path} missing - run 05b_graph_ablation_valsplit.py first")
    return X, y, ts, ei, np.load(p_path)


def masks(y, ts):
    return make_time_masks(y, ts, {
        "train": TRAIN_STEPS,
        "fuser_fit": FUSER_FIT_STEPS,
        "threshold": THRESHOLD_STEPS,
        "test": TEST_STEPS,
    })


def tune_threshold(y, scores, threshold_mask):
    """Tune F1 cutoff using only the reserved threshold-calibration rows."""
    return best_f1_threshold(np.asarray(y)[threshold_mask], np.asarray(scores)[threshold_mask])


def gate_features(p_tab, p_gnn, logdeg):
    return np.column_stack([p_tab, p_gnn, logdeg, p_tab * logdeg, p_gnn * logdeg])


def main(smoke):
    import xgboost as xgb
    out = OUT_DIR / "smoke" if smoke else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    X, y, ts, ei, p_gnn = load(smoke)
    m = masks(y, ts)
    deg = np.bincount(np.concatenate([ei[0], ei[1]]), minlength=len(y))
    logdeg = np.log1p(deg)

    fit_mask = m["fuser_fit"]
    threshold_mask = m["threshold"]
    spw = (y[m["train"]] == 0).sum() / max((y[m["train"]] == 1).sum(), 1)

    def fit_tabular(X_model):
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, eval_metric="aucpr", early_stopping_rounds=50,
            scale_pos_weight=spw, random_state=42, n_jobs=-1,
        )
        model.fit(X_model[m["train"]], y[m["train"]],
                  eval_set=[(X_model[fit_mask], y[fit_mask])], verbose=False)
        return model

    tab = fit_tabular(X)
    p_tab = tab.predict_proba(X)[:, 1]
    if X.shape[1] < 93:
        raise ValueError("Elliptic feature matrix must contain local features f0-f92.")
    X_local = X[:, :93]
    tab_local = fit_tabular(X_local)
    p_tab_local = tab_local.predict_proba(X_local)[:, 1]

    y_fit, yt = y[fit_mask], y[m["test"]]
    cand = {
        "tabular_only": p_tab,
        "tabular_local_only": p_tab_local,
        "graph_only": p_gnn,
        "avg": (p_tab + p_gnn) / 2,
    }

    ws = np.linspace(0, 1, 21)
    w = ws[int(np.argmax([
        average_precision_score(y_fit, (w_ * p_tab + (1 - w_) * p_gnn)[fit_mask])
        for w_ in ws
    ]))]
    cand["weighted"] = w * p_tab + (1 - w) * p_gnn

    stk = LogisticRegression(max_iter=1000).fit(
        np.column_stack([p_tab, p_gnn])[fit_mask], y_fit
    )
    cand["stacked"] = stk.predict_proba(np.column_stack([p_tab, p_gnn]))[:, 1]

    G = gate_features(p_tab, p_gnn, logdeg)
    gate = LogisticRegression(max_iter=1000).fit(G[fit_mask], y_fit)
    cand["gated"] = gate.predict_proba(G)[:, 1]

    rows, per_step = [], []
    for name, p in cand.items():
        thr = tune_threshold(y, p, threshold_mask)
        r = evaluate(yt, p[m["test"]], thr)
        row = {f"test_{key}": value for key, value in r.items()}
        row["model"] = name
        row["threshold"] = thr
        rows.append(row)
        for s in TEST_STEPS:
            sm = m["test"] & (ts == s)
            if sm.sum() and y[sm].sum():
                per_step.append({"model": name, "time_step": s,
                                 "f1": f1_score(y[sm], (p[sm] >= thr).astype(int), zero_division=0)})
    df = pd.DataFrame(rows).set_index("model").sort_values(
        "test_pr_auc", ascending=False
    )
    df.to_csv(out / "fusion_summary.csv")
    pd.DataFrame(per_step).pivot(index="time_step", columns="model", values="f1").to_csv(out / "f1_by_time_step.csv")
    json.dump({"weighted_w_tab": float(w), "gate_coefs": dict(zip(
        ["p_tab", "p_gnn", "logdeg", "p_tab*logdeg", "p_gnn*logdeg"], map(float, gate.coef_[0]))),
        "fuser_fit_steps": list(FUSER_FIT_STEPS),
        "threshold_steps": list(THRESHOLD_STEPS),
        "tabular_local_features": [f"f{i}" for i in range(93)],
        "thresholds": {row["model"]: row["threshold"] for row in rows},
        "smoke_test": smoke}, open(out / "fusion_params.json", "w"), indent=2)
    np.save(out / "fused_node_proba.npy", cand["gated"])
    np.save(out / "tabular_node_proba.npy", p_tab)
    np.save(out / "tabular_local_node_proba.npy", p_tab_local)
    print(df[["test_pr_auc", "test_roc_auc", "test_f1", "test_precision",
              "test_recall", "test_recall_at_1pct_fpr"]].round(4).to_string())
    print(f"weighted-avg tabular weight = {w:.2f}; highest test PR-AUC (descriptive only) = "
          f"{df.index[0]}")
    print("Fusers fit on steps 30-32; thresholds tuned on steps 33-34; test is steps 35-49.")
    if smoke:
        print("[SMOKE TEST - synthetic data, NOT a reported result]")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = pd.read_csv(out / "f1_by_time_step.csv", index_col=0)
        t.plot(figsize=(9, 4), marker="o")
        plt.title("Illicit-class F1 per test time step")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(out / "f1_by_time_step.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    main(ap.parse_args().smoke_test)
