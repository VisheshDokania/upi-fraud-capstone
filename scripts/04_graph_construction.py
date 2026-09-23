"""
Week 4 — Graph Construction & Temporal Split (Elliptic, local version)
==========================================================================
Run:
    cd scripts
    python 04_graph_construction.py
"""

import gc
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "graph_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Elliptic's Kaggle download extracts into a nested subfolder -- confirmed
# from an actual acquisition run, not assumed.
ELLIPTIC_DIR = DATA_DIR / "elliptic" / "elliptic_bitcoin_dataset"

TRAIN_TIME_STEPS = range(1, 35)
TEST_TIME_STEPS = range(35, 50)


def load_elliptic():
    feat_path = ELLIPTIC_DIR / "elliptic_txs_features.csv"
    class_path = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
    edge_path = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
    for p in (feat_path, class_path, edge_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run 01_data_acquisition.py first.")

    features = pd.read_csv(feat_path, header=None)
    n_feat_cols = features.shape[1] - 2
    features.columns = ["txId", "timeStep"] + [f"f{i}" for i in range(n_feat_cols)]
    classes = pd.read_csv(class_path)
    classes["class"] = classes["class"].astype(str)
    edges = pd.read_csv(edge_path)
    return features, classes, edges


def build_graph(features, classes, edges):
    merged = features.merge(classes, on="txId", how="left")
    merged["class"] = merged["class"].fillna("unknown")
    G = nx.DiGraph()
    for _, row in merged.iterrows():
        G.add_node(row["txId"], timeStep=row["timeStep"], label=row["class"])
    for _, row in edges.iterrows():
        if row["txId1"] in G.nodes and row["txId2"] in G.nodes:
            G.add_edge(row["txId1"], row["txId2"])
    return G, merged


def temporal_split(merged):
    labeled = merged[merged["class"] != "unknown"].copy()
    train_mask = labeled["timeStep"].isin(TRAIN_TIME_STEPS)
    test_mask = labeled["timeStep"].isin(TEST_TIME_STEPS)
    train_ids = set(labeled.loc[train_mask, "txId"])
    test_ids = set(labeled.loc[test_mask, "txId"])
    report = {
        "total_labeled_nodes": len(labeled), "train_nodes": len(train_ids), "test_nodes": len(test_ids),
        "train_illicit_rate": (labeled.loc[train_mask, "class"] == "1").mean(),
        "test_illicit_rate": (labeled.loc[test_mask, "class"] == "1").mean(),
    }
    return train_ids, test_ids, report


def to_pyg_format(G, merged, out_prefix):
    node_ids = sorted(G.nodes())
    id_to_idx = {tx_id: i for i, tx_id in enumerate(node_ids)}
    feature_cols = [c for c in merged.columns if c.startswith("f")]
    merged_sorted = merged.set_index("txId").loc[node_ids]
    X = merged_sorted[feature_cols].fillna(0).values.astype(np.float32)
    label_map = {"1": 1, "2": 0, "unknown": -1}
    y = merged_sorted["class"].map(label_map).values.astype(np.int64)
    edge_index = np.array([[id_to_idx[u], id_to_idx[v]] for u, v in G.edges()
                            if u in id_to_idx and v in id_to_idx]).T

    np.save(OUT_DIR / f"{out_prefix}_X.npy", X)
    np.save(OUT_DIR / f"{out_prefix}_y.npy", y)
    np.save(OUT_DIR / f"{out_prefix}_edge_index.npy", edge_index)
    np.save(OUT_DIR / f"{out_prefix}_timeSteps.npy", merged_sorted["timeStep"].values)
    with open(OUT_DIR / f"{out_prefix}_id_to_idx.json", "w") as f:
        json.dump({str(k): v for k, v in id_to_idx.items()}, f)

    print(f"Saved PyG-ready arrays: X{X.shape}, edge_index{edge_index.shape}, y{y.shape}")
    return X, y, edge_index


if __name__ == "__main__":
    features, classes, edges = load_elliptic()
    G, merged = build_graph(features, classes, edges)
    del features, classes, edges  # large intermediate objects, no longer needed
    gc.collect()

    print(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    train_ids, test_ids, report = temporal_split(merged)
    print("\nTemporal split report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    X, y, edge_index = to_pyg_format(G, merged, "elliptic")
    del G, merged
    gc.collect()

    pd.Series(list(train_ids)).to_csv(OUT_DIR / "train_ids.csv", index=False)
    pd.Series(list(test_ids)).to_csv(OUT_DIR / "test_ids.csv", index=False)
    pd.DataFrame([report]).to_csv(OUT_DIR / "temporal_split_report.csv", index=False)
    print(f"\nAll outputs written to {OUT_DIR}/")
    print("Script exiting now — all memory released.")
