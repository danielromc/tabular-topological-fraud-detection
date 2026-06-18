"""
explainability/ego_ring_provider.py — XAI Caso de estudio:
visualiza la ego-red de un proveedor fraudulento (anillo).

Selecciona automáticamente un taller de anillo con buen volumen de claims
y dibuja:
  - Nodo central: el proveedor
  - Nodos periféricos: los claims que ese proveedor atiende
  - Color: rojo = fraude real, azul = legítimo

Sirve para mostrar visualmente la "constelación de fraude" en torno a un
proveedor cómplice, demostrando que el grafo tiene señal explotable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx


def run(cfg, target_provider=None):
    """
    target_provider: ID del proveedor a visualizar (e.g. 'WSH_001').
                     Si None, elige automáticamente uno de anillo A con
                     buen volumen.
                     También se puede pasar 'auto_legit' para elegir
                     automáticamente un proveedor legítimo con volumen
                     similar a los anillos (para comparación visual).
    """
    art_cmp = Path(cfg["paths"]["artifacts_cmp"])
    out_dir = art_cmp / "explainability"
    out_dir.mkdir(exist_ok=True, parents=True)

    claims    = pd.read_csv(cfg["paths"]["claims_csv"])
    providers = pd.read_csv(cfg["paths"]["providers_csv"])
    edges     = pd.read_csv(cfg["paths"]["edges_csv"])

    # Selección automática
    if target_provider is None:
        # Buscar talleres anillo A con volumen alto
        ring_a_wsh = providers[(providers.Provider_type == "WSH") &
                                (providers.Ring_ID == "A")]["Provider_ID"].tolist()
        wsh_volume = claims["Provider_workshop_ID"].value_counts()
        candidates = [(p, wsh_volume.get(p, 0)) for p in ring_a_wsh]
        candidates.sort(key=lambda x: -x[1])
        target_provider = candidates[0][0] if candidates else "WSH_001"
    elif target_provider == "auto_legit":
        # Elegir un taller legítimo con volumen comparable
        # (mediana del volumen de anillo)
        ring_provs = providers[providers.Is_fraudulent &
                                (providers.Provider_type == "WSH")]["Provider_ID"].tolist()
        wsh_volume = claims["Provider_workshop_ID"].value_counts()
        ring_vols = [wsh_volume.get(p, 0) for p in ring_provs]
        target_vol = int(np.median(ring_vols))
        # Buscar legítimo con volumen cercano
        legit_provs = providers[(~providers.Is_fraudulent) &
                                 (providers.Provider_type == "WSH")]["Provider_ID"].tolist()
        legit_with_vol = [(p, wsh_volume.get(p, 0)) for p in legit_provs]
        # Filtrar los que tengan volumen entre 0.7x y 1.3x del target
        legit_filtered = [(p, v) for p, v in legit_with_vol
                          if 0.7*target_vol <= v <= 1.3*target_vol]
        if legit_filtered:
            legit_filtered.sort(key=lambda x: abs(x[1] - target_vol))
            target_provider = legit_filtered[0][0]
        else:
            # Fallback: el legítimo de mayor volumen
            legit_with_vol.sort(key=lambda x: -x[1])
            target_provider = legit_with_vol[0][0]

    # Info del proveedor
    prov_info = providers[providers.Provider_ID == target_provider].iloc[0]
    prov_type = prov_info["Provider_type"]
    ring_id = prov_info["Ring_ID"]
    fraud_ratio = prov_info["Fraud_ratio"]
    is_fraud_prov = bool(prov_info["Is_fraudulent"])
    fr_str = f"{fraud_ratio:.2f}" if pd.notna(fraud_ratio) else "N/A (legítimo)"
    print(f"Proveedor objetivo: {target_provider}  tipo={prov_type}  "
          f"ring={ring_id}  Fraud_ratio_target={fr_str}")

    # Determinar columna de proveedor según tipo
    col_map = {"WSH": "Provider_workshop_ID",
               "CLN": "Provider_clinic_ID",
               "LAW": "Provider_lawyer_ID"}
    col = col_map[prov_type]

    # Claims conectados al proveedor
    connected_claims = claims[claims[col] == target_provider].copy()
    n_total = len(connected_claims)
    n_fraud = int(connected_claims["is_fraud"].sum())
    fraud_ratio_real = n_fraud / n_total if n_total > 0 else 0
    print(f"Claims conectados: {n_total}  fraudes: {n_fraud}  "
          f"ratio real: {fraud_ratio_real:.2%}")

    # Construir grafo bipartito
    G = nx.Graph()
    G.add_node(target_provider, node_type="provider")
    for _, row in connected_claims.iterrows():
        cid = row["claim_id"]
        G.add_node(cid, node_type="claim", is_fraud=bool(row["is_fraud"]),
                   fraud_type=row.get("Fraud_type", "legitimate"),
                   ring=row.get("Ring_ID", "none"),
                   claims_type=row["Claims_type"],
                   cost=row["Cost_claims_year"])
        G.add_edge(target_provider, cid)

    # Layout: proveedor en el centro, claims alrededor
    pos = {}
    pos[target_provider] = (0, 0)
    n_claims = n_total
    for i, cid in enumerate(connected_claims["claim_id"]):
        angle = 2 * np.pi * i / n_claims
        # Radio variable según fraude/legit para separar visualmente
        is_fraud = bool(connected_claims.iloc[i]["is_fraud"])
        radius = 1.0 if is_fraud else 1.3
        pos[cid] = (radius * np.cos(angle), radius * np.sin(angle))

    # Dibujar
    fig, ax = plt.subplots(figsize=(10, 10))

    # Aristas
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.7, alpha=0.5,
                            edge_color="grey")

    # Nodos claim
    fraud_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "claim"
                   and G.nodes[n].get("is_fraud")]
    legit_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "claim"
                   and not G.nodes[n].get("is_fraud")]

    nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, ax=ax,
                            node_color="#d62728", node_size=160,
                            edgecolors="black", linewidths=0.6,
                            label=f"Fraude (n={len(fraud_nodes)})")
    nx.draw_networkx_nodes(G, pos, nodelist=legit_nodes, ax=ax,
                            node_color="#1f77b4", node_size=160,
                            edgecolors="black", linewidths=0.6,
                            label=f"Legítimo (n={len(legit_nodes)})")

    # Nodo proveedor (central, grande). Color: oro si fraudulento, plata si legítimo
    prov_color = "#ffd700" if is_fraud_prov else "#c0c0c0"
    nx.draw_networkx_nodes(G, pos, nodelist=[target_provider], ax=ax,
                            node_color=prov_color, node_size=2200,
                            edgecolors="black", linewidths=2.0)
    nx.draw_networkx_labels(G, pos, labels={target_provider: target_provider},
                             font_size=11, font_weight="bold")

    title_extra = (f"Anillo {ring_id}, Fraud_ratio diseñado: {fr_str}"
                   if is_fraud_prov else f"Proveedor LEGÍTIMO (sin pertenencia a anillo)")
    ax.set_title(
        f"Ego-red del proveedor {target_provider}\n"
        f"{title_extra}\n"
        f"Total claims: {n_total}  |  Fraude: {n_fraud} ({fraud_ratio_real:.1%})",
        fontsize=13)
    ax.axis("off")
    ax.legend(handles=[
        Patch(color=prov_color, label=f"Proveedor {target_provider}"),
        Patch(color="#d62728", label=f"Claim fraudulento ({len(fraud_nodes)})"),
        Patch(color="#1f77b4", label=f"Claim legítimo ({len(legit_nodes)})"),
    ], loc="lower right", fontsize=10, frameon=True)

    fig.tight_layout()
    fname = out_dir / f"fig_ego_ring_{target_provider}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ {fname.name}")

    # Tabla de los claims para anexo
    rows = []
    for _, row in connected_claims.iterrows():
        rows.append({
            "claim_id":   row["claim_id"],
            "Claims_type":row["Claims_type"],
            "Cost":       round(float(row["Cost_claims_year"]), 2),
            "is_fraud":   bool(row["is_fraud"]),
            "Fraud_type": row.get("Fraud_type"),
            "Ring_ID":    row.get("Ring_ID"),
        })
    detail = pd.DataFrame(rows).sort_values(["is_fraud","Cost"], ascending=[False, False])
    csv_path = out_dir / f"ego_ring_{target_provider}_detail.csv"
    detail.to_csv(csv_path, index=False)
    print(f"✔ {csv_path.name}")
