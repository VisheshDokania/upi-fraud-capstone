"""Explain the current 03b tabular and 05b graph winner bundles.

This companion leaves the protected Week 7 script unchanged and writes outputs
to notebooks/explainability_report/.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "notebooks" / "explainability_report"
TABULAR_BUNDLE = ROOT / "models" / "tabular_winner.pkl"
GRAPH_BUNDLE = ROOT / "notebooks" / "graph_valsplit_report" / "winner_model.pkl"


def explain_tabular():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    demo_path = ROOT / "notebooks" / "tabular_timesplit_report" / "demo_transactions.csv"
    demo = pd.read_csv(demo_path)
    with TABULAR_BUNDLE.open("rb") as handle:
        bundle = pickle.load(handle)
    X = demo[bundle["features"]].copy()
    for column in bundle["cat_cols"]:
        X[column] = pd.Categorical(X[column], categories=bundle["cat_levels"][column])
    for column in X.columns.difference(bundle["cat_cols"]):
        X[column] = pd.to_numeric(X[column], errors="coerce")

    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT_DIR / "final_tabular_shap_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(values, X, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "final_tabular_shap.png", dpi=180, bbox_inches="tight")
    plt.close()

    probabilities = bundle["model"].predict_proba(X)[:, 1]
    fraud_indices = np.flatnonzero(demo["isFraud"].to_numpy() == 1)
    if len(fraud_indices):
        row_idx = int(fraud_indices[np.argmax(probabilities[fraud_indices])])
        case = pd.DataFrame({
            "feature": X.columns,
            "feature_value": X.iloc[row_idx].astype(str).to_numpy(),
            "shap_value": values[row_idx],
        }).assign(transaction_id=demo.iloc[row_idx].get("TransactionID"))
        case.sort_values("shap_value", key=np.abs, ascending=False).head(20).to_csv(
            OUT_DIR / "final_tabular_case_shap.csv", index=False
        )
    print(f"Tabular explanation saved for winner {bundle['name']}.")


def explain_graph(epochs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer
    import gnn_models  # noqa: F401 -- register pickled model classes

    graph_dir = ROOT / "notebooks" / "graph_pipeline"
    X = np.load(graph_dir / "elliptic_X.npy")
    y = np.load(graph_dir / "elliptic_y.npy")
    ts = np.load(graph_dir / "elliptic_timeSteps.npy")
    edge_index = np.load(graph_dir / "elliptic_edge_index.npy")
    train_rows = np.isin(ts, list(range(1, 30)))
    mean = X[train_rows].mean(axis=0)
    std = X[train_rows].std(axis=0) + 1e-6
    X = (X - mean) / std
    edge_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_tensor = torch.cat([edge_tensor, edge_tensor.flip(0)], dim=1)
    data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_tensor)

    with GRAPH_BUNDLE.open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["model"].cpu().eval()
    degree = np.bincount(np.concatenate([edge_index[0], edge_index[1]]), minlength=len(y))
    candidates = np.flatnonzero((y == 1) & np.isin(ts, list(range(35, 50))))
    if not len(candidates):
        raise RuntimeError("No illicit test-period node is available for graph explanation.")
    node_idx = int(candidates[np.argmax(degree[candidates])])

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    relevant = explanation.edge_mask > 1e-6
    edge_ids = relevant.nonzero(as_tuple=True)[0]
    scores = explanation.edge_mask[edge_ids]
    k = min(10, len(scores))
    records = []
    if k:
        top_scores, local_ids = scores.topk(k)
        top_edges = edge_ids[local_ids]
        graph = nx.DiGraph()
        graph.add_node(node_idx)
        for score, edge_id in zip(top_scores, top_edges):
            source, target = (int(v) for v in edge_tensor[:, edge_id].tolist())
            graph.add_edge(source, target, importance=float(score))
            records.append({"source_index": source, "target_index": target,
                            "importance": float(score)})

        pos = {node_idx: (0.0, 0.0)}
        neighbors = [node for node in graph if node != node_idx]
        for index, neighbor in enumerate(neighbors):
            angle = 2 * np.pi * index / max(len(neighbors), 1)
            pos[neighbor] = (np.cos(angle), np.sin(angle))
        widths = [1 + 4 * graph.edges[e]["importance"] / max(float(top_scores.max()), 1e-12)
                  for e in graph.edges]
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw_networkx(graph, pos, ax=ax, width=widths, node_size=650,
                         node_color=["#E74C3C" if n == node_idx else "#AED6F1"
                                     for n in graph.nodes])
        ax.set_title(f"GNNExplainer: {bundle['name']} on test node {node_idx}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "final_gnn_explainer.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(records, columns=["source_index", "target_index", "importance"]).to_csv(
        OUT_DIR / "final_gnn_explainer_edges.csv", index=False
    )
    with (OUT_DIR / "final_gnn_explainer_case.json").open("w", encoding="utf-8") as handle:
        json.dump({"winner": bundle["name"], "node_index": node_idx,
                   "nonzero_explanation_edges": int(relevant.sum())}, handle, indent=2)
    print(f"Graph explanation saved for winner {bundle['name']}.")


def main(epochs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    explain_tabular()
    explain_graph(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-explainer-epochs", type=int, default=100)
    main(parser.parse_args().gnn_explainer_epochs)
