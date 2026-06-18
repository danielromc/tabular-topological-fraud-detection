"""pipeline/graph_train.py — Fase 2.3: entrenamiento GraphSAGE heterogéneo."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


class HeteroSAGE(nn.Module):
    def __init__(self, in_dims, hidden_dim, edge_types, dropout=0.3, aggr="mean"):
        super().__init__()
        self.proj = nn.ModuleDict({ntype: nn.Linear(d, hidden_dim) for ntype,d in in_dims.items()})
        self.conv1 = HeteroConv({et: SAGEConv((hidden_dim,hidden_dim), hidden_dim, aggr=aggr)
                                  for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden_dim,hidden_dim), hidden_dim, aggr=aggr)
                                  for et in edge_types}, aggr="sum")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        h = {n: self.proj[n](x) for n,x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k,v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k,v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k,v in h.items()}
        return self.classifier(h["claim"]).squeeze(-1)


def _build_train_only_edge_index(data, train_mask):
    """
    Devuelve un edge_index_dict que solo contiene aristas con extremo claim
    en train_mask (auditoría L4). Esto evita que el message-passing durante
    el entrenamiento traiga información de val/test al embedding de los
    nodos train. La evaluación sigue usando el grafo completo (data.edge_index_dict).
    """
    train_only = {}
    n_total = 0
    n_kept = 0
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        n_total += ei.shape[1]
        if src_t == "claim" and dst_t == "claim":
            keep = train_mask[ei[0]] & train_mask[ei[1]]
        elif src_t == "claim":
            keep = train_mask[ei[0]]
        elif dst_t == "claim":
            keep = train_mask[ei[1]]
        else:
            keep = torch.ones(ei.shape[1], dtype=torch.bool)
        train_only[et] = ei[:, keep]
        n_kept += int(keep.sum())
    return train_only, n_total, n_kept


def run(cfg):
    seed = cfg["seed"]
    gcfg = cfg["graph"]
    art_graph = Path(cfg["paths"]["artifacts_graph"])
    torch.manual_seed(seed); np.random.seed(seed)

    data = torch.load(art_graph / "graph_data.pt", weights_only=False)
    print(data)

# Las features ya vienen pre-procesadas desde graph_build (log + scale + clip).
# Solo verificamos que el rango sea sano.
    print(f"\nFeatures pre-procesadas (desde graph_build). "
      f"Rango: [{data['claim'].x.min():.2f}, {data['claim'].x.max():.2f}]")

    in_dims = {n: data[n].x.shape[1] for n in data.node_types}
    edge_types = list(data.edge_types)
    model = HeteroSAGE(in_dims, gcfg["hidden_dim"], edge_types,
                       gcfg["dropout"], gcfg["aggregation"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros: {n_params:,}")

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    y = data["claim"].y.float()
    tm, vm = data["claim"].train_mask, data["claim"].val_mask

    # Edge_index aislado para entrenamiento (auditoría L4)
    edge_index_train, n_total_edges, n_train_edges = _build_train_only_edge_index(data, tm)
    print(f"\nAristas totales: {n_total_edges}  "
          f"|  En training (sin val/test): {n_train_edges} "
          f"({n_train_edges/n_total_edges*100:.1f}%)")

    n_pos = y[tm].sum().item(); n_neg = (1-y[tm]).sum().item()
    pw = n_neg / n_pos
    print(f"pos_weight = {pw:.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=gcfg["learning_rate"],
                                  weight_decay=gcfg["weight_decay"])

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        logits = model(data.x_dict, data.edge_index_dict)
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        yt = y[mask].cpu().numpy().astype(int)
        return {"auc_pr":average_precision_score(yt,probs),
                "auc_roc":roc_auc_score(yt,probs),
                "probs":probs,"y_true":yt}

    history=[]; best_auc_pr=-1.0; best_epoch=-1; best_state=None; noimp=0
    max_ep = gcfg["max_epochs"]; pat = gcfg["patience"]
    print(f"\nEntrenamiento (max {max_ep} epochs, paciencia {pat})...")
    t0 = time.time()
    for epoch in range(1, max_ep+1):
        model.train(); optimizer.zero_grad()
        # Forward de entrenamiento usa edge_index_train (sin aristas a val/test).
        # La pérdida se calcula sobre nodos tm con esos embeddings honestos.
        logits = model(data.x_dict, edge_index_train)
        loss = loss_fn(logits[tm], y[tm])
        loss.backward(); optimizer.step()

        ev = evaluate(vm)
        history.append({"epoch":epoch,"loss":float(loss.item()),
                        "auc_pr_val":float(ev["auc_pr"]),"auc_roc_val":float(ev["auc_roc"])})
        improved = ev["auc_pr"] > best_auc_pr
        if improved:
            best_auc_pr=ev["auc_pr"]; best_epoch=epoch; noimp=0
            best_state = {k:v.clone().detach() for k,v in model.state_dict().items()}
        else:
            noimp += 1
        if epoch%10==0 or improved:
            mk = " *" if improved else ""
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  AUC-PR={ev['auc_pr']:.4f}  AUC-ROC={ev['auc_roc']:.4f}{mk}")
        if noimp >= pat:
            print(f"  Early stopping en epoch {epoch}")
            break

    elapsed = time.time()-t0
    print(f"\nTiempo: {elapsed:.1f}s  |  Mejor epoch {best_epoch}  AUC-PR val {best_auc_pr:.4f}")
    model.load_state_dict(best_state)

    ev = evaluate(vm)
    print(f"\n--- VAL ---\nAUC-PR: {ev['auc_pr']:.4f}  AUC-ROC: {ev['auc_roc']:.4f}")
    rows=[]
    for t in [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80]:
        pred = (ev["probs"]>=t).astype(int)
        if pred.sum()==0: continue
        rows.append({"thr":t,"n_pos":int(pred.sum()),
                     "P":round(precision_score(ev["y_true"],pred,zero_division=0),4),
                     "R":round(recall_score(ev["y_true"],pred,zero_division=0),4),
                     "F1":round(f1_score(ev["y_true"],pred,zero_division=0),4)})
    print(f"\n--- Sweep val ---\n{pd.DataFrame(rows).to_string(index=False)}")

    torch.save({
        "state_dict":model.state_dict(),"in_dims":in_dims,"edge_types":edge_types,
        "hidden_dim":gcfg["hidden_dim"],"dropout":gcfg["dropout"],"aggregation":gcfg["aggregation"],
    }, art_graph/"model_graphsage_baseline.pt")
    pd.DataFrame(history).to_csv(art_graph/"training_log.csv", index=False)

    with open(art_graph/"metrics_graphsage_val.json","w") as f:
        json.dump({"model":"graphsage_baseline_hetero",
                   "hidden_dim":gcfg["hidden_dim"],"dropout":gcfg["dropout"],
                   "lr":gcfg["learning_rate"],"weight_decay":gcfg["weight_decay"],
                   "seed":seed,"best_epoch":best_epoch,"training_time_sec":elapsed,
                   "val":{"auc_pr":float(ev["auc_pr"]),"auc_roc":float(ev["auc_roc"]),"sweep":rows},
                   "pos_weight":pw,"n_params":n_params}, f, indent=2)
    print(f"\n[OK] Guardado: model_graphsage_baseline.pt, metrics_graphsage_val.json, training_log.csv")
