"""
explainability/witness_t3a.py — XAI Caso de estudio:
visualiza el subgrafo de un testigo T3A (víctima legítima de multichoque).

Para cada uno de los 5 Linked_to_fraud_event, dibuja el ecosistema:
  - El testigo (claim legítimo)
  - El claim fraudulento con el que comparte Event_ID (perpetrador)
  - Los proveedores de cada uno
  - Las probabilidades asignadas por XGBoost / GraphSAGE / Híbrido

Demuestra visualmente por qué el grafo se equivoca en algunos casos y
cómo el híbrido los corrige usando las features tabulares.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import torch
import xgboost as xgb
import joblib


def run(cfg):
    art        = Path(cfg["paths"]["artifacts"])
    art_graph  = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_cmp    = Path(cfg["paths"]["artifacts_cmp"])
    out_dir    = art_cmp / "explainability"
    out_dir.mkdir(exist_ok=True, parents=True)

    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    edges  = pd.read_csv(cfg["paths"]["edges_csv"])
    linked_mask = np.load(art / "linked_mask.npy")
    y_all = np.load(art / "y_all.npy")

    # Cargar probabilidades de los 3 modelos
    X_tab = pd.read_pickle(art / "X_all.pkl")
    m_xgb = xgb.XGBClassifier(); m_xgb.load_model(art / "model_tuned.json")
    proba_xgb = m_xgb.predict_proba(X_tab)[:, 1]

    # GNN
    from pipeline.graph_train import HeteroSAGE
    data = torch.load(art_graph / "graph_data.pt", weights_only=False)
    x_claim = data["claim"].x; tm = data["claim"].train_mask
    mean = x_claim[tm].mean(dim=0, keepdim=True)
    std  = x_claim[tm].std(dim=0, keepdim=True).clamp(min=1e-6)
    data["claim"].x = (x_claim - mean) / std
    ckpt = torch.load(art_graph / "model_graphsage_baseline.pt", weights_only=False)
    m_gnn = HeteroSAGE(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                       ckpt["dropout"], ckpt.get("aggregation","mean"))
    with torch.no_grad(): _ = m_gnn(data.x_dict, data.edge_index_dict)
    m_gnn.load_state_dict(ckpt["state_dict"]); m_gnn.eval()
    with torch.no_grad():
        logits = m_gnn(data.x_dict, data.edge_index_dict)
        proba_gnn = torch.sigmoid(logits).cpu().numpy()

    # Híbrido
    X_hyb = pd.read_pickle(art_hybrid / "X_hybrid.pkl")
    m_hyb = joblib.load(art_hybrid / "model_hybrid_tuned.pkl")
    proba_hyb = m_hyb.predict_proba(X_hyb)[:, 1]

    # Identificar testigos (5)
    linked_idx = np.where(linked_mask)[0]
    print(f"Encontrados {len(linked_idx)} testigos T3A")

    # Para cada testigo, encontrar su pareja (Event_ID)
    cases_summary = []
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    for ax_i, gi in enumerate(linked_idx):
        witness = claims.iloc[gi]
        event_id = witness["Event_ID"]
        # Pareja: el otro claim con mismo Event_ID
        partners = claims[(claims["Event_ID"] == event_id) &
                           (claims["claim_id"] != witness["claim_id"])]
        if len(partners) == 0:
            print(f"  ⚠ {witness['claim_id']}: sin pareja, saltando")
            continue
        perp = partners.iloc[0]
        gi_perp = claims.index.get_loc(perp.name)

        # Construir subgrafo
        G = nx.Graph()
        # Nodos claim
        G.add_node(witness["claim_id"], node_type="claim_legit",
                    label="Testigo legítimo")
        G.add_node(perp["claim_id"], node_type="claim_fraud",
                    label="Perpetrador")
        # Arista co_event
        G.add_edge(witness["claim_id"], perp["claim_id"], edge_type="co_event")

        # Proveedor del testigo (taller/clínica/abogado)
        for col, prefix in [("Provider_workshop_ID", "WSH"),
                             ("Provider_clinic_ID", "CLN"),
                             ("Provider_lawyer_ID", "LAW")]:
            pw = witness.get(col)
            if pd.notna(pw):
                G.add_node(pw, node_type="provider_legit", label=f"Proveedor legítimo ({prefix})")
                G.add_edge(witness["claim_id"], pw, edge_type=col)
            pp = perp.get(col)
            if pd.notna(pp):
                G.add_node(pp, node_type="provider_fraud", label=f"Proveedor cómplice ({prefix})")
                G.add_edge(perp["claim_id"], pp, edge_type=col)

        # Layout manual: testigo izquierda, perpetrador derecha
        pos = {}
        pos[witness["claim_id"]] = (-1.5, 0)
        pos[perp["claim_id"]]    = (1.5, 0)
        # Proveedores arriba y abajo según pertenencia
        wp_provs = [n for n in G.neighbors(witness["claim_id"])
                    if G.nodes[n].get("node_type") in ("provider_legit","provider_fraud")]
        pp_provs = [n for n in G.neighbors(perp["claim_id"])
                    if G.nodes[n].get("node_type") in ("provider_legit","provider_fraud")]
        for j, p in enumerate(wp_provs):
            pos[p] = (-2.0 - 0.2*j, 1.5 - 1.2*j)
        for j, p in enumerate(pp_provs):
            pos[p] = (2.0 + 0.2*j, 1.5 - 1.2*j)

        ax = axes[ax_i]
        # Aristas
        for u, v, d in G.edges(data=True):
            et = d.get("edge_type", "")
            if et == "co_event":
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                        width=2.5, edge_color="purple",
                                        style="dashed")
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                        width=1.0, edge_color="grey", alpha=0.6)

        # Nodos por tipo
        for ntype, color, size in [
            ("claim_legit",     "#1f77b4", 1100),
            ("claim_fraud",     "#d62728", 1100),
            ("provider_legit",  "#9edae5", 700),
            ("provider_fraud",  "#ff9896", 700),
        ]:
            nl = [n for n in G.nodes if G.nodes[n].get("node_type") == ntype]
            if nl:
                nx.draw_networkx_nodes(G, pos, nodelist=nl, ax=ax,
                                        node_color=color, node_size=size,
                                        edgecolors="black", linewidths=0.8)

        # Etiquetas
        labels = {}
        for n in G.nodes:
            if G.nodes[n].get("node_type", "").startswith("claim"):
                labels[n] = n.replace("CLM_", "")
            else:
                labels[n] = n
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

        # Probabilidades de cada modelo, abajo
        p_xgb = proba_xgb[gi]
        p_gnn = proba_gnn[gi]
        p_hyb = proba_hyb[gi]
        verdict = lambda p: "✓" if p < 0.5 else "✗"
        title = (
            f"{witness['claim_id']} (testigo)\n"
            f"Event {event_id}\n"
            f"\n"
            f"XGBoost: {p_xgb:.3f} {verdict(p_xgb)}\n"
            f"GNN:     {p_gnn:.3f} {verdict(p_gnn)}\n"
            f"Híbrido: {p_hyb:.3f} {verdict(p_hyb)}"
        )
        ax.set_title(title, fontsize=9, family="monospace")
        ax.axis("off")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2, 2.5)

        cases_summary.append({
            "claim_id_testigo": witness["claim_id"],
            "claim_id_perpetrador": perp["claim_id"],
            "Event_ID": event_id,
            "Claims_type": witness["Claims_type"],
            "wsh_testigo": witness.get("Provider_workshop_ID"),
            "wsh_perpetrador": perp.get("Provider_workshop_ID"),
            "is_fraud_testigo": bool(y_all[gi]),
            "is_fraud_perpetrador": bool(y_all[gi_perp]),
            "proba_xgb_testigo": float(p_xgb),
            "proba_gnn_testigo": float(p_gnn),
            "proba_hyb_testigo": float(p_hyb),
            "proba_xgb_perpetrador": float(proba_xgb[gi_perp]),
            "proba_gnn_perpetrador": float(proba_gnn[gi_perp]),
            "proba_hyb_perpetrador": float(proba_hyb[gi_perp]),
        })

    # Leyenda compartida
    legend_handles = [
        Patch(color="#1f77b4", label="Claim testigo (legítimo)"),
        Patch(color="#d62728", label="Claim perpetrador (fraude)"),
        Patch(color="#9edae5", label="Proveedor legítimo"),
        Patch(color="#ff9896", label="Proveedor cómplice"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, 0.04), fontsize=10)
    fig.suptitle("Casos de estudio: 5 testigos T3A — víctimas legítimas de multichoques coordinados",
                  fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_witness_t3a_subgraphs.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ fig_witness_t3a_subgraphs.png")

    pd.DataFrame(cases_summary).to_csv(out_dir / "case_study_witness_t3a.csv",
                                         index=False)
    print(f"✔ case_study_witness_t3a.csv")
    print(f"\n--- Resumen de los 5 testigos ---")
    summary_df = pd.DataFrame(cases_summary)[
        ["claim_id_testigo","Event_ID","proba_xgb_testigo",
         "proba_gnn_testigo","proba_hyb_testigo"]]
    print(summary_df.to_string(index=False))
