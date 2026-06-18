"""
Rápida comparación: features escaladas vs sin escalar
Entrena 2 modelos (50 épocas cada uno) y compara AUC-PR val
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

ART_GRAPH = Path("./artifacts_graph")
torch.manual_seed(2025)
np.random.seed(2025)

# Load
data = torch.load(ART_GRAPH / "graph_data.pt", weights_only=False)
y = data["claim"].y.float()
train_mask = data["claim"].train_mask
val_mask = data["claim"].val_mask

class HeteroSAGE(nn.Module):
    def __init__(self, in_dims, hidden_dim, edge_types, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleDict({ntype: nn.Linear(d, hidden_dim) for ntype, d in in_dims.items()})
        self.conv1 = HeteroConv({et: SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean") for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean") for et in edge_types}, aggr="sum")
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.dropout = dropout
    def forward(self, x_dict, edge_index_dict):
        h = {ntype: self.proj[ntype](x) for ntype, x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        return self.classifier(h["claim"]).squeeze(-1)

def train_model(data, label, standardize=False, epochs=50):
    if standardize:
        x_claim = data["claim"].x
        mean = x_claim[train_mask].mean(dim=0, keepdim=True)
        std = x_claim[train_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
        data["claim"].x = (x_claim - mean) / std
        print(f"  Standardized (range: [{data['claim'].x.min():.2f}, {data['claim'].x.max():.2f}])")
    
    in_dims = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    edge_types = list(data.edge_types)
    model = HeteroSAGE(in_dims, 64, edge_types, 0.3)
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    
    n_pos = y[train_mask].sum().item()
    n_neg = (1 - y[train_mask]).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    best_auc_pr = -1.0
    best_state = None
    no_improve = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        optim.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optim.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data.x_dict, data.edge_index_dict)
            probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
            y_val = y[val_mask].cpu().numpy().astype(int)
            auc_pr = average_precision_score(y_val, probs)
            auc_roc = roc_auc_score(y_val, probs)
        
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}")
        
        if no_improve >= 20:
            print(f"    Early stop at epoch {epoch}")
            break
    
    return {"auc_pr": best_auc_pr, "auc_roc": auc_roc}

print("\n=== Variante 1: SIN estandarización ===")
data_no_std = torch.load(ART_GRAPH / "graph_data.pt", weights_only=False)
res_no_std = train_model(data_no_std, "no_std", standardize=False, epochs=50)
print(f"  ✓ Final: AUC-PR={res_no_std['auc_pr']:.4f}  AUC-ROC={res_no_std['auc_roc']:.4f}")

print("\n=== Variante 2: CON estandarización ===")
data_std = torch.load(ART_GRAPH / "graph_data.pt", weights_only=False)
res_std = train_model(data_std, "std", standardize=True, epochs=50)
print(f"  ✓ Final: AUC-PR={res_std['auc_pr']:.4f}  AUC-ROC={res_std['auc_roc']:.4f}")

print("\n" + "="*60)
print("COMPARACIÓN (50 épocas)")
print("="*60)
print(f"{'Variante':<25} {'AUC-PR':<12} {'AUC-ROC':<12}")
print("-"*60)
print(f"{'Sin estandarización':<25} {res_no_std['auc_pr']:<12.4f} {res_no_std['auc_roc']:<12.4f}")
print(f"{'Con estandarización':<25} {res_std['auc_pr']:<12.4f} {res_std['auc_roc']:<12.4f}")
print("-"*60)
delta_pr = res_std['auc_pr'] - res_no_std['auc_pr']
delta_roc = res_std['auc_roc'] - res_no_std['auc_roc']
print(f"{'Diferencia (std - no_std)':<25} {delta_pr:<12.4f} {delta_roc:<12.4f}")
print("="*60)

# Save results
results = {
    "epochs": 50,
    "no_std": res_no_std,
    "std": res_std,
    "delta_pr": float(delta_pr),
    "delta_roc": float(delta_roc),
}
with open(ART_GRAPH / "scaling_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved: {ART_GRAPH / 'scaling_comparison.json'}")
