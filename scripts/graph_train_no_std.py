"""
GraphSAGE training without standardizing claim features (for comparison).
Saves model_graphsage_no_std.pt, metrics_graphsage_val_no_std.json, training_log_no_std.csv
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
    precision_score, recall_score, f1_score, confusion_matrix,
)

# ===== Config =====
SEED = 2025
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
PATIENCE = 40
ART_GRAPH = Path("./artifacts_graph")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ===== Carga =====
data = torch.load(ART_GRAPH / "graph_data.pt", weights_only=False)
print(data)

# NOTE: No standardization here — keep raw features
print('\nUsing raw claim features (no standardization)')

# ===== Arquitectura =====
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

# ===== Preparación =====
in_dims = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
edge_types = list(data.edge_types)

model = HeteroSAGE(in_dims, HIDDEN_DIM, edge_types, DROPOUT)
print(f"\nModelo: {model}")

with torch.no_grad():
    _ = model(data.x_dict, data.edge_index_dict)

# pos_weight
y = data["claim"].y.float()
train_mask = data["claim"].train_mask
val_mask   = data["claim"].val_mask
test_mask  = data["claim"].test_mask

n_pos = y[train_mask].sum().item()
n_neg = (1 - y[train_mask]).sum().item()
pos_weight_val = n_neg / n_pos
print(f"\npos_weight = {n_neg:.0f}/{n_pos:.0f} = {pos_weight_val:.3f}")

pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.x_dict, data.edge_index_dict)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = y[mask].cpu().numpy().astype(int)
    return {
        "auc_pr":  average_precision_score(y_true, probs),
        "auc_roc": roc_auc_score(y_true, probs),
        "probs":   probs,
        "y_true":  y_true,
    }

history = []
best_auc_pr = -1.0
best_epoch = -1
best_state = None
epochs_no_improve = 0

print(f"\nEntrenamiento (max {MAX_EPOCHS} epochs, early stopping paciencia {PATIENCE})...")
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x_dict, data.edge_index_dict)
    loss = loss_fn(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    val_metrics = evaluate(val_mask)
    history.append({
        "epoch": epoch,
        "loss":  float(loss.item()),
        "auc_pr_val":  float(val_metrics["auc_pr"]),
        "auc_roc_val": float(val_metrics["auc_roc"]),
    })

    improved = val_metrics["auc_pr"] > best_auc_pr
    if improved:
        best_auc_pr = val_metrics["auc_pr"]
        best_epoch = epoch
        best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or improved:
        marker = " *" if improved else ""
        print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  "
              f"AUC-PR val={val_metrics['auc_pr']:.4f}  "
              f"AUC-ROC val={val_metrics['auc_roc']:.4f}{marker}")

    if epochs_no_improve >= PATIENCE:
        print(f"  Early stopping en epoch {epoch} (sin mejora en {PATIENCE} epochs)")
        break

elapsed = time.time() - t0
print(f"\nTiempo de entrenamiento: {elapsed:.1f} s")
print(f"Mejor epoch: {best_epoch}  |  AUC-PR val: {best_auc_pr:.4f}")

model.load_state_dict(best_state)

val_eval = evaluate(val_mask)
print(f"\n--- VAL (mejor modelo) ---")
print(f"AUC-PR:  {val_eval['auc_pr']:.4f}")
print(f"AUC-ROC: {val_eval['auc_roc']:.4f}")

probs_v = val_eval["probs"]
y_v = val_eval["y_true"]
print(f"\n--- Sweep de thresholds (val) ---")
sweep_rows = []
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
    pred = (probs_v >= t).astype(int)
    if pred.sum() == 0: continue
    sweep_rows.append({
        "thr": t,
        "n_pos": int(pred.sum()),
        "P": round(precision_score(y_v, pred, zero_division=0), 4),
        "R": round(recall_score(y_v, pred, zero_division=0), 4),
        "F1": round(f1_score(y_v, pred, zero_division=0), 4),
    })
print(pd.DataFrame(sweep_rows).to_string(index=False))

# Guarda con sufijos "no_std"
torch.save({
    "state_dict": model.state_dict(),
    "in_dims": in_dims,
    "edge_types": edge_types,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
}, ART_GRAPH / "model_graphsage_no_std.pt")

pd.DataFrame(history).to_csv(ART_GRAPH / "training_log_no_std.csv", index=False)

metrics = {
    "model": "graphsage_no_std",
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "lr": LR,
    "seed": SEED,
    "best_epoch": best_epoch,
    "training_time_sec": elapsed,
    "val": {
        "auc_pr":  float(val_eval["auc_pr"]),
        "auc_roc": float(val_eval["auc_roc"]),
        "sweep": sweep_rows,
    },
    "pos_weight": pos_weight_val,
    "n_params": sum(p.numel() for p in model.parameters()),
}
with open(ART_GRAPH / "metrics_graphsage_val_no_std.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✔ Guardado: model_graphsage_no_std.pt, metrics_graphsage_val_no_std.json, training_log_no_std.csv")
