"""pipeline/graph_build.py — Fase 2.2: construcción del HeteroData.

Aplica log-transform + StandardScaler (fit en train) a las features tabulares
de los nodos claim antes de guardarlas en el grafo. Esto evita que features
de cola larga (Cost_claims_year, LossRatio, etc.) dominen los gradientes
de GraphSAGE.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler


# Features con distribución de cola larga que necesitan log-transform.
# Identificadas por análisis empírico: Cost_claims_year max=260853 (std=3419),
# LossRatio max=1071, Premium max=2116, etc. Sin log, los z-scores
# resultantes de StandardScaler producen valores extremos (±90) que
# descalibran los pesos de GraphSAGE.
HEAVY_TAIL_COLS = [
    "Cost_claims_year",
    "LossRatio",
    "Cost_per_value",
    "Premium",
    "Value_vehicle",
    "Cylinder_capacity",
    "Weight",
    "Power",
]

def preprocess_claim_features(X_claim, train_mask_np):
     """
     Pipeline:
       0) Sanitización: NaN -> 0, +Inf -> max float32, -Inf -> 0
          y aviso si se detectan negativos en HEAVY_TAIL_COLS
       1) log1p sobre HEAVY_TAIL_COLS (las que existan en X_claim)
       2) StandardScaler con fit solo en train (evita leakage)
       3) Clip final a [-5, 5] como red de seguridad
     Devuelve un np.ndarray float32 listo para tensorizar.
     """
     fnames = list(X_claim.columns)
     x = X_claim.to_numpy().astype(np.float32).copy()

     # Paso 0: sanitización defensiva. Sin esto, log1p(NaN)=NaN y StandardScaler
     # produce coeficientes NaN silenciosamente, descalibrando todo el grafo.
     n_nan = int(np.isnan(x).sum())
     n_inf = int(np.isinf(x).sum())
     if n_nan or n_inf:
         print(f"  [WARN] Detectados {n_nan} NaN y {n_inf} Inf en features de claim. "
               f"Imputando NaN→0, +Inf→max(float32), -Inf→0.")
         x = np.nan_to_num(x, nan=0.0,
                           posinf=float(np.finfo(np.float32).max),
                           neginf=0.0)

     # Paso 1: log-transform a features de cola larga
     applied_log = []
     for col in HEAVY_TAIL_COLS:
         if col in fnames:
             i = fnames.index(col)
             n_neg = int((x[:, i] < 0).sum())
             if n_neg:
                 print(f"  [WARN] {col}: {n_neg} valores negativos forzados a 0 antes de log1p.")
             # np.maximum es explícito y trazable; np.clip(x,0,None) hacía lo mismo silenciosamente.
             x[:, i] = np.log1p(np.maximum(x[:, i], 0.0))
             applied_log.append(col)
     if applied_log:
         print(f"  log1p aplicado a: {applied_log}")
     # Paso 2: StandardScaler con fit solo en train
     scaler = StandardScaler()
     scaler.fit(x[train_mask_np])
     x_scaled = scaler.transform(x).astype(np.float32)
     # Paso 3: clip de seguridad
     x_scaled = np.clip(x_scaled, -5.0, 5.0)

     print(f"  Rango final: [{x_scaled.min():.2f}, {x_scaled.max():.2f}]  "
           f"std medio: {x_scaled.std():.2f}")

     return x_scaled, scaler


# def preprocess_claim_features(X_claim, train_mask_np):
#      """
#      Pipeline:
#        1) log1p sobre HEAVY_TAIL_COLS (las que existan en X_claim)
#        2) StandardScaler con fit solo en train (evita leakage)
#        3) Clip final a [-5, 5] como red de seguridad
#      Devuelve un np.ndarray float32 listo para tensorizar.
#      """
#      fnames = list(X_claim.columns)
#      x = X_claim.to_numpy().astype(np.float32).copy()
#      # Paso 1: log-transform a features de cola larga
#      applied_log = []
#      for col in HEAVY_TAIL_COLS:
#          if col in fnames:
#              i = fnames.index(col)
#              x[:, i] = np.log1p(np.clip(x[:, i], 0, None))
#              applied_log.append(col)
#      if applied_log:
#          print(f"  log1p aplicado a: {applied_log}")
#      # Paso 2: StandardScaler con fit solo en train
#      scaler = StandardScaler()
#      scaler.fit(x[train_mask_np])
#      x_scaled = scaler.transform(x).astype(np.float32)
#      # Paso 3: clip de seguridad
#      x_scaled = np.clip(x_scaled, -5.0, 5.0)

#      print(f"  Rango final: [{x_scaled.min():.2f}, {x_scaled.max():.2f}]  "
#            f"std medio: {x_scaled.std():.2f}")

#      return x_scaled, scaler


def run(cfg):
    art       = Path(cfg["paths"]["artifacts"])
    art_graph = Path(cfg["paths"]["artifacts_graph"])
    art_graph.mkdir(exist_ok=True, parents=True)

    claims    = pd.read_csv(cfg["paths"]["claims_csv"])
    providers = pd.read_csv(cfg["paths"]["providers_csv"])
    edges     = pd.read_csv(cfg["paths"]["edges_csv"])

    X_claim   = pd.read_pickle(art / "X_all.pkl")
    y_all     = np.load(art / "y_all.npy")
    idx_tr    = np.load(art / "idx_train.npy")
    idx_v     = np.load(art / "idx_val.npy")
    idx_te    = np.load(art / "idx_test.npy")
    linked    = np.load(art / "linked_mask.npy")

    assert len(X_claim) == len(claims)
    print(f"Carga OK: {len(claims)} claims, {len(providers)} providers, {len(edges)} edges")

    claim_id_to_ix = {cid: i for i, cid in enumerate(claims["claim_id"].tolist())}
    providers = providers.reset_index(drop=True)
    prov_id_to_ix = {pid: i for i, pid in enumerate(providers["Provider_ID"].tolist())}
    n_prov = len(providers)

    # ===== Pre-procesado de features =====
    print("\nPre-procesando features de claim...")
    train_mask_np = np.zeros(len(claims), dtype=bool)
    train_mask_np[idx_tr] = True

    x_claim_np, scaler = preprocess_claim_features(X_claim, train_mask_np)
    x_claim = torch.tensor(x_claim_np, dtype=torch.float32)

    # Provider features (one-hot, no necesitan escalado)
    type_to_ix = {"WSH": 0, "CLN": 1, "LAW": 2}
    x_prov = torch.zeros((n_prov, 3), dtype=torch.float32)
    for i, t in enumerate(providers["Provider_type"]):
        x_prov[i, type_to_ix[t]] = 1.0

    print(f"\nx_claim={tuple(x_claim.shape)}  x_provider={tuple(x_prov.shape)}")

    def build_edge_index(sub, src_map, dst_map):
        src = sub["source"].map(src_map).to_numpy()
        dst = sub["target"].map(dst_map).to_numpy()
        return torch.tensor(np.vstack([src.astype(np.int64), dst.astype(np.int64)]),
                            dtype=torch.long)

    ei_wsh = build_edge_index(edges[edges.edge_type=="claim_workshop"], claim_id_to_ix, prov_id_to_ix)
    ei_cln = build_edge_index(edges[edges.edge_type=="claim_clinic"],   claim_id_to_ix, prov_id_to_ix)
    ei_law = build_edge_index(edges[edges.edge_type=="claim_lawyer"],   claim_id_to_ix, prov_id_to_ix)
    ei_cc  = build_edge_index(edges[edges.edge_type=="claim_claim_event"], claim_id_to_ix, claim_id_to_ix)

    print(f"Aristas: wsh={ei_wsh.shape[1]}, cln={ei_cln.shape[1]}, law={ei_law.shape[1]}, cc={ei_cc.shape[1]}")

    data = HeteroData()
    data["claim"].x = x_claim
    data["provider"].x = x_prov

    y = torch.tensor(y_all.astype(np.int64), dtype=torch.long)
    data["claim"].y = y
    n_claims = len(claims)
    train_mask = torch.zeros(n_claims, dtype=torch.bool); train_mask[idx_tr] = True
    val_mask   = torch.zeros(n_claims, dtype=torch.bool); val_mask[idx_v]   = True
    test_mask  = torch.zeros(n_claims, dtype=torch.bool); test_mask[idx_te] = True
    data["claim"].train_mask  = train_mask
    data["claim"].val_mask    = val_mask
    data["claim"].test_mask   = test_mask
    data["claim"].linked_mask = torch.tensor(linked, dtype=torch.bool)

    data["claim","at_workshop","provider"].edge_index = ei_wsh
    data["claim","at_clinic","provider"].edge_index   = ei_cln
    data["claim","at_lawyer","provider"].edge_index   = ei_law
    data["claim","co_event","claim"].edge_index       = ei_cc
    data["provider","rev_at_workshop","claim"].edge_index = ei_wsh.flip(0)
    data["provider","rev_at_clinic","claim"].edge_index   = ei_cln.flip(0)
    data["provider","rev_at_lawyer","claim"].edge_index   = ei_law.flip(0)
    data["claim","co_event_rev","claim"].edge_index       = ei_cc.flip(0)

    print("\n" + "="*70)
    print("VALIDACIÓN")
    print("="*70)
    print(data)

    torch.save(data, art_graph / "graph_data.pt")

    # Guardar scaler por si más adelante se necesita en inferencia
    import pickle
    with open(art_graph / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "node_types":{"claim":{"n":int(x_claim.shape[0]),"feat_dim":int(x_claim.shape[1])},
                      "provider":{"n":int(x_prov.shape[0]),"feat_dim":int(x_prov.shape[1])}},
        "edge_types":{et[1]:int(data[et].edge_index.shape[1]) for et in data.edge_types},
        "splits":{"train":int(train_mask.sum()),"val":int(val_mask.sum()),
                  "test":int(test_mask.sum()),"linked":int(data['claim'].linked_mask.sum())},
        "target":{"n_fraud":int(y.sum()),"rate":float(y.float().mean())},
        "preprocessing":{
            "log_transformed_features": [c for c in HEAVY_TAIL_COLS if c in X_claim.columns],
            "scaler_fit_on": "train_only",
            "clip_range": [-5.0, 5.0],
        },
    }
    with open(art_graph / "graph_meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"\n[OK] Guardado: {art_graph}/graph_data.pt y graph_meta.json")