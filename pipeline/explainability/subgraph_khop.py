"""
explainability/subgraph_khop.py — Visualización de un vecindario K-hop
del grafo bipartito claims ↔ proveedores.

Dado un claim semilla, expande K saltos (claim→proveedor→claim→proveedor→...)
y dibuja el subgrafo resultante mostrando:
  - Claims: círculos (rojo=fraude, azul=legítimo)
  - Proveedores: cuadrados (rosa=cómplice, gris=legítimo)
  - Aristas por tipo (workshop, clinic, lawyer, co_event)
  - El claim semilla destacado con borde grueso

Esto permite ver cómo dos claims que fueron al mismo taller están a 2 saltos,
o cómo un claim conectado a una clínica comparte esa clínica con otro claim
que a su vez fue a un taller del anillo → cadena de 4 saltos.
"""
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def run(cfg, seed_claim=None, k_hops=2, max_nodes=120):
    """
    seed_claim: claim_id desde el cual expandir. Si None, elige automáticamente
                un fraude organizado con conexiones ricas.
    k_hops: número de saltos a expandir (2 = claim→prov→claim).
    max_nodes: límite de nodos para que el gráfico sea legible.
    """
    art_cmp = Path(cfg["paths"]["artifacts_cmp"])
    out_dir = art_cmp / "explainability"
    out_dir.mkdir(exist_ok=True, parents=True)

    claims    = pd.read_csv(cfg["paths"]["claims_csv"])
    providers = pd.read_csv(cfg["paths"]["providers_csv"])
    edges     = pd.read_csv(cfg["paths"]["edges_csv"])

    # Construir el grafo completo bipartito
    G_full = nx.Graph()

    # Nodos claim
    for _, row in claims.iterrows():
        G_full.add_node(row["claim_id"], node_type="claim",
                        is_fraud=bool(row["is_fraud"]),
                        fraud_type=row.get("Fraud_type", "legitimate"),
                        ring=row.get("Ring_ID", "none"),
                        claims_type=row["Claims_type"],
                        cost=row["Cost_claims_year"])

    # Nodos proveedor
    prov_fraud_set = set(providers[providers.Is_fraudulent]["Provider_ID"])
    prov_ring_map = dict(zip(providers.Provider_ID, providers.Ring_ID))
    prov_type_map = dict(zip(providers.Provider_ID, providers.Provider_type))
    for _, row in providers.iterrows():
        G_full.add_node(row["Provider_ID"], node_type="provider",
                        is_fraudulent=bool(row["Is_fraudulent"]),
                        provider_type=row["Provider_type"],
                        ring=row["Ring_ID"])

    # Aristas
    edge_colors_map = {
        "claim_workshop":    "#2ca02c",   # verde
        "claim_clinic":      "#ff7f0e",   # naranja
        "claim_lawyer":      "#9467bd",   # morado
        "claim_claim_event": "#e377c2",   # rosa
    }
    for _, row in edges.iterrows():
        src, tgt, etype = row["source"], row["target"], row["edge_type"]
        if src in G_full and tgt in G_full:
            G_full.add_edge(src, tgt, edge_type=etype)

    print(f"Grafo completo: {G_full.number_of_nodes()} nodos, "
          f"{G_full.number_of_edges()} aristas")

    # Selección automática de semilla
    if seed_claim is None:
        # Buscar un fraude organizado de un anillo con conexión a clínica Y abogado
        # (para tener caminos multi-hop interesantes)
        candidates = claims[
            (claims.Fraud_type == "organized") &
            (claims.Provider_clinic_ID.notna()) &
            (claims.Provider_lawyer_ID.notna()) &
            (claims.Ring_ID.isin(["A", "B"]))
        ]
        if len(candidates) > 0:
            # Elegir el de mayor grado en el grafo
            best = None; best_deg = 0
            for _, row in candidates.iterrows():
                cid = row["claim_id"]
                deg = G_full.degree(cid)
                if deg > best_deg:
                    best_deg = deg; best = cid
            seed_claim = best
        else:
            seed_claim = claims[claims.is_fraud].iloc[0]["claim_id"]

    print(f"Claim semilla: {seed_claim}")
    seed_info = claims[claims.claim_id == seed_claim].iloc[0]
    print(f"  Claims_type: {seed_info['Claims_type']}")
    print(f"  is_fraud: {seed_info['is_fraud']}")
    print(f"  Fraud_type: {seed_info.get('Fraud_type')}")
    print(f"  Ring_ID: {seed_info.get('Ring_ID')}")
    print(f"  Workshop: {seed_info.get('Provider_workshop_ID')}")
    print(f"  Clinic: {seed_info.get('Provider_clinic_ID')}")
    print(f"  Lawyer: {seed_info.get('Provider_lawyer_ID')}")
    print(f"  Grado en grafo: {G_full.degree(seed_claim)}")

    # BFS de K saltos
    visited = set()
    queue = deque([(seed_claim, 0)])
    visited.add(seed_claim)
    nodes_by_hop = {0: [seed_claim]}

    while queue:
        node, depth = queue.popleft()
        if depth >= k_hops:
            continue
        for neighbor in G_full.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                hop = depth + 1
                if hop not in nodes_by_hop:
                    nodes_by_hop[hop] = []
                nodes_by_hop[hop].append(neighbor)

    # Limitar nodos si son demasiados
    total = len(visited)
    print(f"\nVecindario de {k_hops} saltos: {total} nodos")
    for h, nodes in sorted(nodes_by_hop.items()):
        n_claims = sum(1 for n in nodes if G_full.nodes[n].get("node_type") == "claim")
        n_provs = sum(1 for n in nodes if G_full.nodes[n].get("node_type") == "provider")
        print(f"  Hop {h}: {len(nodes)} nodos ({n_claims} claims, {n_provs} proveedores)")

    if total > max_nodes:
        print(f"  ⚠ Demasiados nodos ({total}), truncando a {max_nodes}")
        # Priorizar: semilla + hop 1 completo + hop 2 parcial
        keep = set(nodes_by_hop.get(0, []))
        keep.update(nodes_by_hop.get(1, []))
        remaining = max_nodes - len(keep)
        if remaining > 0 and 2 in nodes_by_hop:
            keep.update(nodes_by_hop[2][:remaining])
        visited = keep

    # Extraer subgrafo
    sub = G_full.subgraph(visited).copy()
    print(f"\nSubgrafo final: {sub.number_of_nodes()} nodos, "
          f"{sub.number_of_edges()} aristas")

    # Conteos
    n_claims_fraud = sum(1 for n in sub.nodes
                         if sub.nodes[n].get("node_type") == "claim"
                         and sub.nodes[n].get("is_fraud"))
    n_claims_legit = sum(1 for n in sub.nodes
                         if sub.nodes[n].get("node_type") == "claim"
                         and not sub.nodes[n].get("is_fraud"))
    n_prov_fraud = sum(1 for n in sub.nodes
                       if sub.nodes[n].get("node_type") == "provider"
                       and sub.nodes[n].get("is_fraudulent"))
    n_prov_legit = sum(1 for n in sub.nodes
                       if sub.nodes[n].get("node_type") == "provider"
                       and not sub.nodes[n].get("is_fraudulent", False))
    print(f"  Claims: {n_claims_fraud} fraude + {n_claims_legit} legítimos")
    print(f"  Proveedores: {n_prov_fraud} cómplices + {n_prov_legit} legítimos")

    # Layout
    # Usar spring_layout con semilla fija para reproducibilidad
    pos = nx.spring_layout(sub, seed=42, k=2.0/np.sqrt(sub.number_of_nodes()),
                            iterations=80)

    # Dibujar
    fig, ax = plt.subplots(figsize=(16, 12))

    # Aristas por tipo
    for etype, color in edge_colors_map.items():
        edge_list = [(u, v) for u, v, d in sub.edges(data=True)
                     if d.get("edge_type") == etype]
        if edge_list:
            style = "dashed" if etype == "claim_claim_event" else "solid"
            nx.draw_networkx_edges(sub, pos, edgelist=edge_list, ax=ax,
                                    edge_color=color, width=1.0, alpha=0.5,
                                    style=style)

    # Claims fraudulentos (excluido la semilla)
    fraud_claims = [n for n in sub.nodes
                    if sub.nodes[n].get("node_type") == "claim"
                    and sub.nodes[n].get("is_fraud")
                    and n != seed_claim]
    nx.draw_networkx_nodes(sub, pos, nodelist=fraud_claims, ax=ax,
                            node_color="#d62728", node_size=120,
                            edgecolors="black", linewidths=0.5,
                            node_shape="o")

    # Claims legítimos
    legit_claims = [n for n in sub.nodes
                    if sub.nodes[n].get("node_type") == "claim"
                    and not sub.nodes[n].get("is_fraud")
                    and n != seed_claim]
    nx.draw_networkx_nodes(sub, pos, nodelist=legit_claims, ax=ax,
                            node_color="#1f77b4", node_size=120,
                            edgecolors="black", linewidths=0.5,
                            node_shape="o")

    # Proveedores cómplices (cuadrados)
    fraud_provs = [n for n in sub.nodes
                   if sub.nodes[n].get("node_type") == "provider"
                   and sub.nodes[n].get("is_fraudulent")]
    nx.draw_networkx_nodes(sub, pos, nodelist=fraud_provs, ax=ax,
                            node_color="#ff9896", node_size=400,
                            edgecolors="#d62728", linewidths=1.5,
                            node_shape="s")

    # Proveedores legítimos (cuadrados)
    legit_provs = [n for n in sub.nodes
                   if sub.nodes[n].get("node_type") == "provider"
                   and not sub.nodes[n].get("is_fraudulent", False)]
    nx.draw_networkx_nodes(sub, pos, nodelist=legit_provs, ax=ax,
                            node_color="#c0c0c0", node_size=400,
                            edgecolors="black", linewidths=1.0,
                            node_shape="s")

    # Semilla: claim destacado (grande, borde grueso, estrella)
    seed_color = "#d62728" if seed_info["is_fraud"] else "#1f77b4"
    nx.draw_networkx_nodes(sub, pos, nodelist=[seed_claim], ax=ax,
                            node_color=seed_color, node_size=500,
                            edgecolors="gold", linewidths=3.0,
                            node_shape="o")

    # Etiquetas solo para proveedores y la semilla
    labels = {}
    labels[seed_claim] = seed_claim.replace("CLM_", "")
    for n in sub.nodes:
        if sub.nodes[n].get("node_type") == "provider":
            labels[n] = n
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax,
                             font_size=7, font_weight="bold")

    # Título
    ax.set_title(
        f"Vecindario de {k_hops} saltos desde {seed_claim}\n"
        f"({seed_info['Claims_type']}, {seed_info.get('Fraud_type','legitimate')}, "
        f"Ring {seed_info.get('Ring_ID','none')})\n"
        f"{sub.number_of_nodes()} nodos  |  {sub.number_of_edges()} aristas  |  "
        f"Claims: {n_claims_fraud} fraude + {n_claims_legit} legítimos  |  "
        f"Proveedores: {n_prov_fraud} cómplices + {n_prov_legit} legítimos",
        fontsize=12)
    ax.axis("off")

    # Leyenda
    legend_handles = [
        mpatches.Patch(color=seed_color, label=f"★ Claim semilla ({seed_claim})"),
        mpatches.Patch(color="#d62728", label=f"Claim fraudulento ({n_claims_fraud})"),
        mpatches.Patch(color="#1f77b4", label=f"Claim legítimo ({n_claims_legit})"),
        mpatches.Patch(color="#ff9896", label=f"Proveedor cómplice ({n_prov_fraud})"),
        mpatches.Patch(color="#c0c0c0", label=f"Proveedor legítimo ({n_prov_legit})"),
        plt.Line2D([],[],color="#2ca02c",lw=2, label="Arista workshop"),
        plt.Line2D([],[],color="#ff7f0e",lw=2, label="Arista clinic"),
        plt.Line2D([],[],color="#9467bd",lw=2, label="Arista lawyer"),
        plt.Line2D([],[],color="#e377c2",lw=2, ls="--", label="Arista co_event"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
               frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fname = out_dir / f"fig_subgraph_{k_hops}hop_{seed_claim}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ {fname.name}")
