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
Also reports F1 per test time step, because Elliptic has a known "dark market
shutdown" around step 43 after which every published model degrades - showing
that you know this is a strong viva point (Weber et al., 2019).

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

from common_eval import best_f1_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "notebooks" / "graph_pipeline"
GNN_DIR = ROOT / "notebooks" / "graph_valsplit_report"
OUT_DIR = ROOT / "notebooks" / "fusion_report"
TRAIN_STEPS, VAL_STEPS, TEST_STEPS = range(1, 30), range(30, 35), range(35, 50)


def load(smoke):
    if smoke:
        rng = np.random.default_rng(1)
        n = 8000
        X = rng.normal(size=(n, 20)).astype("float32")
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
    lab = y >= 0
    return {k: lab & np.isin(ts, list(r)) for k, r in
            [("train", TRAIN_STEPS), ("val", VAL_STEPS), ("test", TEST_STEPS)]}


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

    spw = (y[m["train"]] == 0).sum() / max((y[m["train"]] == 1).sum(), 1)
    tab = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.7, eval_metric="aucpr", early_stopping_rounds=50,
                            scale_pos_weight=spw, random_state=42, n_jobs=-1)
    tab.fit(X[m["train"]], y[m["train"]], eval_set=[(X[m["val"]], y[m["val"]])], verbose=False)
    p_tab = tab.predict_proba(X)[:, 1]

    yv, yt = y[m["val"]], y[m["test"]]
    cand = {"tabular_only": p_tab, "graph_only": p_gnn, "avg": (p_tab + p_gnn) / 2}

    ws = np.linspace(0, 1, 21)
    w = ws[int(np.argmax([average_precision_score(yv, (w_ * p_tab + (1 - w_) * p_gnn)[m["val"]]) for w_ in ws]))]
    cand["weighted"] = w * p_tab + (1 - w) * p_gnn

    stk = LogisticRegression(max_iter=1000).fit(np.column_stack([p_tab, p_gnn])[m["val"]], yv)
    cand["stacked"] = stk.predict_proba(np.column_stack([p_tab, p_gnn]))[:, 1]

    G = gate_features(p_tab, p_gnn, logdeg)
    gate = LogisticRegression(max_iter=1000).fit(G[m["val"]], yv)
    cand["gated"] = gate.predict_proba(G)[:, 1]
    # NOTE: stacked/gated are fitted on validation, so their thresholds are too.
    # This is standard for a small meta-learner; say so in the report.

    rows, per_step = [], []
    for name, p in cand.items():
        thr = best_f1_threshold(yv, p[m["val"]])
        r = evaluate(yt, p[m["test"]], thr)
        r["model"] = name
        rows.append(r)
        for s in TEST_STEPS:
            sm = m["test"] & (ts == s)
            if sm.sum() and y[sm].sum():
                per_step.append({"model": name, "time_step": s,
                                 "f1": f1_score(y[sm], (p[sm] >= thr).astype(int), zero_division=0)})
    df = pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
    df.to_csv(out / "fusion_summary.csv")
    pd.DataFrame(per_step).pivot(index="time_step", columns="model", values="f1").to_csv(out / "f1_by_time_step.csv")
    json.dump({"weighted_w_tab": float(w), "gate_coefs": dict(zip(
        ["p_tab", "p_gnn", "logdeg", "p_tab*logdeg", "p_gnn*logdeg"], map(float, gate.coef_[0]))),
        "smoke_test": smoke}, open(out / "fusion_params.json", "w"), indent=2)
    np.save(out / "fused_node_proba.npy", cand["gated"])
    np.save(out / "tabular_node_proba.npy", p_tab)
    print(df[["pr_auc", "roc_auc", "f1", "precision", "recall", "recall_at_1pct_fpr"]].round(4).to_string())
    print(f"weighted-avg tabular weight = {w:.2f}")
    if smoke:
        print("[SMOKE TEST - synthetic data, NOT a reported result]")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = pd.read_csv(out / "f1_by_time_step.csv", index_col=0)
        t.plot(figsize=(9, 4), marker="o")
        plt.axvline(43, ls="--", c="grey")
        plt.title("Illicit-class F1 per test time step (dashed: ~dark market shutdown)")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(out / "f1_by_time_step.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    main(ap.parse_args().smoke_test)
