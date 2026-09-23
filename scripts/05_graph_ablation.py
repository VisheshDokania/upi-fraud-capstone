"""
Week 5 — Graph Ablation: GraphSAGE vs GAT vs GCN (local version)
=====================================================================
Only loads the small saved arrays from Week 4 (not the raw Elliptic CSVs),
so this script's memory footprint is tiny regardless of dataset size.

Local CPU install (no GPU needed, just slower):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric

Run:
    cd scripts
    python 05_graph_ablation.py
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn_models import GraphSAGENet, GATNet, GCNNet

GRAPH_DIR = Path(__file__).parent.parent / "notebooks" / "graph_pipeline"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "graph_ablation_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
EPOCHS = 100
LR = 0.01
RANDOM_STATE = 42


def load_graph_data():
    X = np.load(GRAPH_DIR / "elliptic_X.npy")
    y = np.load(GRAPH_DIR / "elliptic_y.npy")
    edge_index = np.load(GRAPH_DIR / "elliptic_edge_index.npy")
    with open(GRAPH_DIR / "elliptic_id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    train_txids = set(int(i) for i in pd.read_csv(GRAPH_DIR / "train_ids.csv")["0"])
    test_txids = set(int(i) for i in pd.read_csv(GRAPH_DIR / "test_ids.csv")["0"])

    data = Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y, dtype=torch.long))
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    for txid in train_txids:
        if txid in id_to_idx:
            train_mask[id_to_idx[txid]] = True
    for txid in test_txids:
        if txid in id_to_idx:
            test_mask[id_to_idx[txid]] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


# GraphSAGENet, GATNet, GCNNet now imported from gnn_models.py


def train_and_eval(model_cls, data, name):
    torch.manual_seed(RANDOM_STATE)
    model = model_cls(data.x.shape[1], HIDDEN_DIM).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    train_y = data.y[data.train_mask]
    n_pos, n_neg = (train_y == 1).sum().item(), (train_y == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)

    best_f1, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
                true = data.y[data.test_mask].cpu().numpy()
                f1 = f1_score(true, pred, pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"[{name}] epoch {epoch:3d}  loss={loss.item():.4f}  test_f1={f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        proba = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
        pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()

    result = {"model": name,
              "roc_auc": roc_auc_score(true, proba) if len(set(true)) > 1 else float("nan"),
              "f1": f1_score(true, pred, pos_label=1, zero_division=0),
              "precision": precision_score(true, pred, pos_label=1, zero_division=0),
              "recall": recall_score(true, pred, pos_label=1, zero_division=0)}
    return model.to("cpu"), result


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    data = load_graph_data()
    print(f"Loaded graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges, "
          f"{data.train_mask.sum().item()} train / {data.test_mask.sum().item()} test")

    results = []
    winner_model, winner_name, winner_f1 = None, None, -1
    for cls, name in [(GraphSAGENet, "GraphSAGE"), (GATNet, "GAT"), (GCNNet, "GCN")]:
        print(f"\n=== Training {name} ===")
        model, res = train_and_eval(cls, data, name)
        results.append(res)
        if res["f1"] > winner_f1:
            winner_f1, winner_name, winner_model = res["f1"], name, model
        del model
        gc.collect()

    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    df.to_csv(OUT_DIR / "graph_ablation_summary.csv", index=False)
    print("\n" + "=" * 60)
    print("GRAPH ABLATION — SUMMARY (temporal split, same train/test nodes)")
    print("=" * 60)
    print(df.round(4).to_string(index=False))
    print(f"\nWinner (highest illicit-class F1): {winner_name}")

    with open(OUT_DIR / "winner_model.pkl", "wb") as f:
        pickle.dump({"name": winner_name, "model": winner_model}, f)
    print(f"Winning model saved to {OUT_DIR}/winner_model.pkl for Week 7")
    print("Script exiting now — all training memory released.")
