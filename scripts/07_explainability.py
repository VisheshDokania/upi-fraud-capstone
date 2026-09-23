"""
Week 7 — Explainability (local version)
============================================
Loads the winning models from PICKLE FILES saved by Weeks 3 and 5, rather
than requiring those scripts' variables to still be in memory -- this is
the same "run, save to disk, exit" pattern used throughout, and it's what
lets you run Weeks 3-7 as separate terminal commands without ever holding
more than one week's data in RAM at a time.

Run:
    cd scripts
    python 07_explainability.py
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import shap
import networkx as nx

DATA_DIR = Path(__file__).parent.parent / "data"
ABLATION_DIR = Path(__file__).parent.parent / "notebooks" / "ablation_report"
GRAPH_DIR = Path(__file__).parent.parent / "notebooks" / "graph_pipeline"
GRAPH_ABLATION_DIR = Path(__file__).parent.parent / "notebooks" / "graph_ablation_report"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "explainability_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Part A — SHAP on the Week 3 winner
# ---------------------------------------------------------------------------
def run_shap_part():
    winner_path = ABLATION_DIR / "winner_model.pkl"
    if not winner_path.exists():
        raise FileNotFoundError(f"{winner_path} not found — run 03_tabular_ablation.py first.")
    with open(winner_path, "rb") as f:
        saved = pickle.load(f)
    winner_name, model = saved["name"], saved["model"]
    print(f"Explaining Week 3 winner: {winner_name}")

    fpath = DATA_DIR / "ieee_cis" / "train_transaction.csv"
    df = pd.read_csv(fpath)
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    del df

    # a manageable sample for the summary plot -- SHAP on the full 590K
    # rows is both slow and unnecessary for an explainability demo
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    y_sample = y.loc[X_sample.index]

    if winner_name == "lightgbm":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        proba_sample = model.predict(X_sample)
    elif winner_name == "catboost":
        # CatBoost needs its categoricals as strings, same fix as Week 3
        X_sample_cb = X_sample.copy()
        cat_cols = X_sample.select_dtypes(include=["category"]).columns
        for c in cat_cols:
            X_sample_cb[c] = X_sample_cb[c].astype(object).fillna("missing").astype(str)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_cb)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        proba_sample = model.predict_proba(X_sample_cb)[:, 1]
        X_sample = X_sample_cb
    else:  # xgboost
        X_sample_xgb = X_sample.copy()
        for c in X_sample.select_dtypes(include=["category"]).columns:
            X_sample_xgb[c] = X_sample_xgb[c].cat.codes
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_xgb)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        import xgboost as xgb
        proba_sample = model.predict(xgb.DMatrix(X_sample_xgb))
        X_sample = X_sample_xgb

    plt.figure()
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_global_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {OUT_DIR}/shap_global_importance.png")

    fraud_idx = np.where(y_sample.values == 1)[0]
    if len(fraud_idx) > 0:
        top_case = fraud_idx[np.argmax(proba_sample[fraud_idx])]
        row_shap = pd.Series(sv[top_case], index=X_sample.columns)
        top_drivers = row_shap.abs().sort_values(ascending=False).head(5)
        print(f"\nWorked example — flagged fraud case (score={proba_sample[top_case]:.3f}):")
        example = {"model_score": float(proba_sample[top_case]),
                    "top_5_driving_features": {f: float(row_shap[f]) for f in top_drivers.index}}
        for feat in top_drivers.index:
            print(f"  {feat}: SHAP={row_shap[feat]:.4f}")
        with open(OUT_DIR / "shap_worked_example.json", "w") as f:
            json.dump(example, f, indent=2)


# ---------------------------------------------------------------------------
# Part B — GNNExplainer on the Week 5 winner
# ---------------------------------------------------------------------------
def visualize_explanation(node_idx, top_edge_ids, top_vals, edge_index, out_path,
                           winner_name="GraphSAGE"):
    """
    Draws the explained subgraph -- the flagged node plus its most important
    neighbor connections -- as an actual image, instead of leaving this as
    a table of numbers only readable in the terminal. Edge color/thickness
    encodes importance, so the fan-in/fan-out pattern is visible at a glance.
    """
    G = nx.DiGraph()
    G.add_node(node_idx, is_center=True)
    weights = {}
    for score_val, edge_id in zip(top_vals, top_edge_ids):
        src, dst = edge_index[:, edge_id]
        src, dst = int(src.item()), int(dst.item())
        G.add_edge(src, dst, weight=float(score_val.item()))
        weights[(src, dst)] = float(score_val.item())

    # place the center node in the middle, neighbors in a circle around it --
    # more legible than a generic force-directed layout for a star/fan pattern
    neighbors = [n for n in G.nodes if n != node_idx]
    pos = {node_idx: (0, 0)}
    n = max(len(neighbors), 1)
    for i, nb in enumerate(neighbors):
        angle = 2 * np.pi * i / n
        pos[nb] = (np.cos(angle), np.sin(angle))

    fig, ax = plt.subplots(figsize=(9, 9))

    edge_weights = [weights[e] for e in G.edges]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [1 + 5 * (w / max_w) for w in edge_weights]
    edge_colors = plt.cm.Reds([0.3 + 0.6 * (w / max_w) for w in edge_weights])

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors,
                            arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors, ax=ax, node_color="#AED6F1",
                            node_size=600, edgecolors="#1B4F72")
    nx.draw_networkx_nodes(G, pos, nodelist=[node_idx], ax=ax, node_color="#E74C3C",
                            node_size=1000, edgecolors="#641E16")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(f"GNNExplainer: why {winner_name} flagged account {node_idx} as illicit\n"
                  f"(top {len(top_edge_ids)} contributing connections, red = higher importance)",
                  fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GNNExplainer visualization to {out_path}")


def run_gnn_explainer_part():
    winner_path = GRAPH_ABLATION_DIR / "winner_model.pkl"
    if not winner_path.exists():
        print(f"{winner_path} not found — skipping Part B. Run 05_graph_ablation.py first.")
        return

    import torch
    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer
    import gnn_models  # noqa: F401 -- unpickling winner_model.pkl needs this
    # module importable so pickle can resolve GraphSAGENet/GATNet/GCNNet by
    # their now-stable "gnn_models.<ClassName>" tag (see gnn_models.py's
    # docstring for why this matters).

    with open(winner_path, "rb") as f:
        saved = pickle.load(f)
    winner_name, model = saved["name"], saved["model"]
    print(f"\nExplaining Week 5 winner: {winner_name}")

    X = np.load(GRAPH_DIR / "elliptic_X.npy")
    y = np.load(GRAPH_DIR / "elliptic_y.npy")
    edge_index = np.load(GRAPH_DIR / "elliptic_edge_index.npy")
    data = Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y, dtype=torch.long))

    with open(GRAPH_DIR / "elliptic_id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    test_txids = set(int(i) for i in pd.read_csv(GRAPH_DIR / "test_ids.csv")["0"])
    test_illicit_indices = [id_to_idx[t] for t in test_txids
                             if t in id_to_idx and y[id_to_idx[t]] == 1]

    if not test_illicit_indices:
        print("No illicit nodes in test set to explain — check temporal split.")
        return

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=200),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    # Pick a higher-degree illicit node rather than just the first one --
    # a node with only 1-2 raw edges gives GNNExplainer almost nothing to
    # find, which is what happened on the first run (9 of 10 "top" edges
    # came back at exactly 0 importance because they weren't even part of
    # this node's neighborhood).
    def node_degree(idx):
        in_deg = (data.edge_index[1] == idx).sum().item()
        out_deg = (data.edge_index[0] == idx).sum().item()
        return in_deg + out_deg

    test_illicit_indices.sort(key=node_degree, reverse=True)
    node_idx = test_illicit_indices[0]
    print(f"Selected node {node_idx} (degree {node_degree(node_idx)}) "
          f"from {len(test_illicit_indices)} illicit test nodes, highest-degree first")

    explanation = explainer(data.x, data.edge_index, index=node_idx)

    # edge_mask covers ALL edges in the graph, not just ones touching
    # node_idx -- most of the graph is correctly irrelevant to any single
    # node's prediction and scores exactly 0. Filter to nonzero first, so
    # a naive top-k doesn't pad the list with unrelated zero-importance
    # edges once genuinely relevant ones run out.
    nonzero_mask = explanation.edge_mask > 1e-6
    n_relevant = nonzero_mask.sum().item()
    print(f"\nIllicit node {node_idx}: {n_relevant} edges with nonzero importance "
          f"out of {explanation.edge_mask.shape[0]:,} total graph edges")

    if n_relevant == 0:
        print("No edges scored as relevant for this node either -- Elliptic's "
              "graph is sparse enough that even high-degree nodes can come up "
              "empty; this is worth noting as a limitation in your report "
              "rather than a bug.")
    else:
        relevant_scores = explanation.edge_mask[nonzero_mask]
        relevant_edge_ids = nonzero_mask.nonzero(as_tuple=True)[0]
        k = min(10, n_relevant)
        top_vals, top_local_idx = relevant_scores.topk(k)
        top_edge_ids = relevant_edge_ids[top_local_idx]
        print(f"Top {k} contributing neighbor edges (mule-ring context):")
        for score_val, edge_id in zip(top_vals, top_edge_ids):
            src, dst = data.edge_index[:, edge_id]
            print(f"  edge {src.item()} -> {dst.item()}: importance {score_val.item():.4f}")

        visualize_explanation(node_idx, top_edge_ids, top_vals, data.edge_index,
                               OUT_DIR / "gnn_explainer_subgraph.png", winner_name=winner_name)

        edge_records = []
        for score_val, edge_id in zip(top_vals, top_edge_ids):
            src, dst = data.edge_index[:, edge_id]
            edge_records.append({"src": int(src.item()), "dst": int(dst.item()),
                                  "importance": float(score_val.item())})
        with open(OUT_DIR / "gnn_explainer_worked_example.json", "w") as f:
            json.dump({"explained_node": int(node_idx), "winner_model": winner_name,
                       "node_degree": int(n_relevant), "top_edges": edge_records}, f, indent=2)
        print(f"Saved underlying data to {OUT_DIR}/gnn_explainer_worked_example.json")


if __name__ == "__main__":
    print("=== Part A: SHAP (tabular) ===")
    run_shap_part()

    print("\n=== Part B: GNNExplainer (graph) ===")
    run_gnn_explainer_part()

    print("\nScript exiting now — all memory released.")
