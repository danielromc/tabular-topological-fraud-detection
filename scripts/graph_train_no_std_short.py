"""
Quick GraphSAGE training without standardization for comparison.
Saves metrics in metrics_graphsage_val_no_std_short.json
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)

# ===== Config =====
SEED = 2025
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 200
PATIENCE = 30
ART_GRAPH = Path("./artifacts_graph")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ===== Carga =====
data = torch.load(ART_GRAPH / "graph_data.pt", weights_only=False)
print(data)

print('Using raw features (no standardization)')

class HeteroSAGE(nn.Module):
    def __init__(self, in_dims, hidden_dim, edge_types, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(d, hidden_dim) for ntype, d in in_dims.items()
        })
        self.conv1 = HeteroConv({
            et: SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")
            for et in edge_types
        }, aggr="sum")
        self.conv2 = HeteroConv({
            et: SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")
            for et in edge_types
        }, aggr="sum")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout
    def forward(self, x_dict, edge_index_dict):
        h = {ntype: self.proj[ntype](x) for ntype, x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        logits = self.classifier(h["claim"]).squeeze(-1)
        return logits

in_dims = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
edge_types = list(data.edge_types)
model = HeteroSAGE(in_dims, HIDDEN_DIM, edge_types, DROPOUT)
with torch.no_grad():
    _ = model(data.x_dict, data.edge_index_dict)

# pos_weight
y = data["claim"].y.float()
train_mask = data["claim"].train_mask
val_mask = data["claim"].val_mask
n_pos = y[train_mask].sum().item()
n_neg = (1 - y[train_mask]).sum().item()
pos_weight_val = n_neg / n_pos
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.x_dict, data.edge_index_dict)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = y[mask].cpu().numpy().astype(int)
    return {"auc_pr": average_precision_score(y_true, probs), "auc_roc": roc_auc_score(y_true, probs), "probs": probs, "y_true": y_true}

history = []
best_auc_pr = -1.0
best_epoch = -1
best_state = None
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x_dict, data.edge_index_dict)
    loss = loss_fn(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    val_metrics = evaluate(val_mask)
    history.append({"epoch": epoch, "loss": float(loss.item()), "auc_pr_val": float(val_metrics["auc_pr"]), "auc_roc_val": float(val_metrics["auc_roc"])})
    improved = val_metrics["auc_pr"] > best_auc_pr
    if improved:
        best_auc_pr = val_metrics["auc_pr"]
        best_epoch = epoch
        best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= PATIENCE:
        break

model.load_state_dict(best_state)
val_eval = evaluate(val_mask)
metrics = {"model": "graphsage_no_std_short", "best_epoch": best_epoch, "training_time_sec": 0, "val": {"auc_pr": float(val_eval["auc_pr"]), "auc_roc": float(val_eval["auc_roc"])}}
with open(ART_GRAPH / "metrics_graphsage_val_no_std_short.json", "w") as f:
    json.dump(metrics, f, indent=2)
print('Saved metrics_graphsage_val_no_std_short.json')
