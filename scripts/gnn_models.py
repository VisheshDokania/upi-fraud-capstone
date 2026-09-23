"""
Shared GNN model classes for Week 5 and Week 7.

WHY THIS FILE EXISTS: pickle stores a class reference as (module_name,
class_name). If GraphSAGENet/GATNet/GCNNet were defined directly inside
05_graph_ablation.py, running that script tags them as "__main__.GraphSAGENet"
(since Python names the script you run directly __main__). When
07_explainability.py later tries to unpickle that model, IT is __main__ now,
and it never defined that class -- hence:

    AttributeError: Can't get attribute 'GraphSAGENet' on <module '__main__'...>

Defining the classes here instead means they're tagged "gnn_models.GraphSAGENet"
regardless of which script imports them, so pickle can always find them by
importing this file.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class GATNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)
