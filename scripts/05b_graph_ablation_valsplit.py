"""
Week 5b - Graph ablation with a proper VALIDATION split (new file).
=====================================================================
Why: 05_graph_ablation.py picks the best epoch by looking at TEST-set F1, then
reports that same test F1. That is test-set leakage - the reported numbers are
optimistic. Here:
  train      = labelled nodes in time steps 1-29
  validation = labelled nodes in time steps 30-34   (epoch selection + threshold)
  test       = labelled nodes in time steps 35-49   (touched once, at the end)
Same models (gnn_models.py), same data arrays from Week 4.

It also saves every node's predicted probability for the winner to
notebooks/graph_valsplit_report/gnn_node_proba.npy, which Week 8 (fusion) needs.

Run (needs torch + torch_geometric, CPU is fine):
    cd scripts
    python 05b_graph_ablation_valsplit.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from common_eval import best_f1_threshold, evaluate
from gnn_models import GATNet, GCNNet, GraphSAGENet

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "notebooks" / "graph_pipeline"
OUT_DIR = ROOT / "notebooks" / "graph_valsplit_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EPOCHS, LR, SEED = 64, 200, 0.01, 42
TRAIN_STEPS, VAL_STEPS, TEST_STEPS = range(1, 30), range(30, 35), range(35, 50)


def load():
    X = np.load(GRAPH_DIR / "elliptic_X.npy")
    y = np.load(GRAPH_DIR / "elliptic_y.npy")
    ei = np.load(GRAPH_DIR / "elliptic_edge_index.npy")
    ts = np.load(GRAPH_DIR / "elliptic_timeSteps.npy")
    # standardise features using train-period statistics only
    tr_rows = np.isin(ts, list(TRAIN_STEPS))
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    X = (X - mu) / sd
    labelled = y >= 0
    masks = {k: torch.tensor(labelled & np.isin(ts, list(r))) for k, r in
             [("train", TRAIN_STEPS), ("val", VAL_STEPS), ("test", TEST_STEPS)]}
    # message passing in both directions: Elliptic edges are directed money flows
    ei_t = torch.tensor(ei, dtype=torch.long)
    ei_t = torch.cat([ei_t, ei_t.flip(0)], dim=1)
    d = Data(x=torch.tensor(X, dtype=torch.float), edge_index=ei_t, y=torch.tensor(y))
    return d, masks, ts


def run(cls, name, d, m):
    torch.manual_seed(SEED)
    model = cls(d.x.shape[1], HIDDEN).to(DEVICE)
    d = d.to(DEVICE)
    m = {k: v.to(DEVICE) for k, v in m.items()}
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    ytr = d.y[m["train"]]
    w = torch.tensor([1.0, (ytr == 0).sum().item() / max((ytr == 1).sum().item(), 1)], device=DEVICE)
    best, best_state, patience = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(d.x, d.edge_index)[m["train"]], d.y[m["train"]], weight=w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(d.x, d.edge_index), 1)[:, 1]
        yv, pv = d.y[m["val"]].cpu().numpy(), p[m["val"]].cpu().numpy()
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yv, pv)
        if ap > best:
            best, best_state, patience = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 20 == 0:
            print(f"[{name}] ep {ep:3d} loss {loss.item():.4f} val PR-AUC {ap:.4f}")
        if patience >= 30:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(d.x, d.edge_index), 1)[:, 1].cpu().numpy()
    yv, yt = d.y[m["val"]].cpu().numpy(), d.y[m["test"]].cpu().numpy()
    thr = best_f1_threshold(yv, p[m["val"].cpu().numpy()])
    res = evaluate(yt, p[m["test"].cpu().numpy()], thr)
    res["model"] = name
    return model.cpu(), res, p


if __name__ == "__main__":
    d, masks, ts = load()
    print({k: int(v.sum()) for k, v in masks.items()})
    rows, best = [], (None, -1, None, None)
    for cls, name in [(GraphSAGENet, "GraphSAGE"), (GATNet, "GAT"), (GCNNet, "GCN")]:
        model, res, p = run(cls, name, d, masks)
        rows.append(res)
        print(f"{name}: test PR-AUC {res['pr_auc']:.4f} F1 {res['f1']:.4f}")
        if res["pr_auc"] > best[1]:
            best = (name, res["pr_auc"], model, p)
    df = pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
    df.to_csv(OUT_DIR / "graph_valsplit_summary.csv")
    print(df.round(4).to_string())
    np.save(OUT_DIR / "gnn_node_proba.npy", best[3])
    with open(OUT_DIR / "winner_model.pkl", "wb") as f:
        pickle.dump({"name": best[0], "model": best[2], "hidden": HIDDEN}, f)
    json.dump({"winner": best[0]}, open(OUT_DIR / "winner.json", "w"))
    print(f"Winner {best[0]}; node probabilities saved for Week 8 fusion.")
