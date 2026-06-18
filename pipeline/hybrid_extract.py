"""pipeline/hybrid_extract.py — Fase 3.1: extracción de SCORES (score-stacking).

CAMBIO: en lugar de extraer 64 embeddings y concatenarlos con 44 tabulares,
ahora producimos los scores predichos por XGBoost (sobre tabulares) y
GraphSAGE (sobre grafo) para cada claim. El meta-modelo de fase 3.2/3.3
aprende a combinar esos 2 scores via LogisticRegression.

Razón: el stacking de features anterior diluía la señal del XGB porque las
embeddings GNN dominaban la importancia (96.5% gain). El stacking de scores
preserva las dos rutas de información de forma explícita.

Salida en artifacts_hybrid/:
  - X_hybrid.pkl           (DataFrame 2 cols: score_xgb, score_gnn)
  - claim_embeddings.npy   (64 embeddings GNN — conservados para explainability)
  - feature_names_hybrid.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv


class HeteroSAGEEmb(nn.Module):
    """Modelo del fase 2.3 con método para devolver el embedding post-conv2."""
    def __init__(self, in_dims, hidden_dim, edge_types, dropout=0.3, aggr="mean"):
        super().__init__()
        self.proj = nn.ModuleDict({n: nn.Linear(d, hidden_dim) for n,d in in_dims.items()})
        self.conv1 = HeteroConv({et: SAGEConv((hidden_dim,hidden_dim), hidden_dim, aggr=aggr)
                                  for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden_dim,hidden_dim), hidden_dim, aggr=aggr)
                                  for et in edge_types}, aggr="sum")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def forward_embedding(self, x_dict, edge_index_dict):
        h = {n: self.proj[n](x) for n,x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k,v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k,v in h.items()}
        return h["claim"]

    def forward(self, x_dict, edge_index_dict):
        emb = self.forward_embedding(x_dict, edge_index_dict)
        return self.classifier(emb).squeeze(-1)


def run(cfg):
    seed = cfg["seed"]
    art        = Path(cfg["paths"]["artifacts"])
    art_graph  = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_hybrid.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(seed)

    # ===== 1. Score XGBoost tabular sobre todos los claims =====
    X_tab = pd.read_pickle(art/"X_all.pkl")
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(art/"model_tuned.json")
    score_xgb = m_xgb.predict_proba(X_tab)[:, 1].astype(np.float32)
    print(f"score_xgb: shape={score_xgb.shape}  "
          f"range=[{score_xgb.min():.4f}, {score_xgb.max():.4f}]")

    # ===== 2. Score GraphSAGE sobre todos los claims =====
    # weights_only=False solo para HeteroData; checkpoint con True (S1)
    data = torch.load(art_graph/"graph_data.pt", weights_only=False)
    ckpt = torch.load(art_graph/"model_graphsage_baseline.pt", weights_only=True)
    model = HeteroSAGEEmb(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                           ckpt["dropout"], ckpt.get("aggregation","mean"))
    with torch.no_grad(): _ = model(data.x_dict, data.edge_index_dict)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        score_gnn = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        embeddings = model.forward_embedding(data.x_dict, data.edge_index_dict)
    emb_np = embeddings.cpu().numpy().astype(np.float32)
    print(f"score_gnn: shape={score_gnn.shape}  "
          f"range=[{score_gnn.min():.4f}, {score_gnn.max():.4f}]")
    print(f"embeddings (para explainability): shape={emb_np.shape}")

    # ===== 3. Construir X_hybrid (2 features) =====
    X_hybrid = pd.DataFrame({
        "score_xgb": score_xgb,
        "score_gnn": score_gnn,
    }, index=X_tab.index)

    # Persistir embeddings (compatibilidad con explainability)
    np.save(art_hybrid/"claim_embeddings.npy", emb_np)
    X_hybrid.to_pickle(art_hybrid/"X_hybrid.pkl")

    feature_names = {
        "all": ["score_xgb", "score_gnn"],
        "n_total": 2,
        "stacking_type": "score_level",
        "n_embeddings_kept_for_xai": int(emb_np.shape[1]),
    }
    with open(art_hybrid/"feature_names_hybrid.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # ===== 4. Sanity checks =====
    y = np.load(art/"y_all.npy")
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(score_xgb, score_gnn)[0]
        print(f"\nCorrelación Spearman entre scores: rho = {rho:.4f}")
    except ImportError:
        rho = float(np.corrcoef(score_xgb, score_gnn)[0, 1])
        print(f"\nCorrelación Pearson entre scores: r = {rho:.4f}")

    print(f"Score medio fraudes:    XGB={score_xgb[y==1].mean():.4f}  "
          f"GNN={score_gnn[y==1].mean():.4f}")
    print(f"Score medio legítimas:  XGB={score_xgb[y==0].mean():.4f}  "
          f"GNN={score_gnn[y==0].mean():.4f}")
    print(f"\n[OK] Guardado: X_hybrid.pkl (2 cols), claim_embeddings.npy, feature_names_hybrid.json")
