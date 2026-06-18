"""pipeline/graph_inspect.py — Fase 2.1: inspección topológica del grafo."""
from pathlib import Path
import pandas as pd
import numpy as np


def run(cfg):
    claims    = pd.read_csv(cfg["paths"]["claims_csv"])
    providers = pd.read_csv(cfg["paths"]["providers_csv"])
    edges     = pd.read_csv(cfg["paths"]["edges_csv"])

    print("="*70)
    print("RESUMEN GENERAL")
    print("="*70)
    print(f"claims.csv   : {len(claims):>6d}")
    print(f"providers.csv: {len(providers):>6d}")
    print(f"edges.csv    : {len(edges):>6d}")

    print("\n--- Proveedores por tipo ---")
    print(providers["Provider_type"].value_counts().to_string())
    print("\n--- Fraudulentos por tipo ---")
    print(providers.groupby("Provider_type")["Is_fraudulent"].agg(["sum","count"]).to_string())
    print("\n--- Por Ring_ID ---")
    print(providers["Ring_ID"].value_counts().to_string())

    print("\n--- Aristas por tipo ---")
    print(edges["edge_type"].value_counts().to_string())

    # Coverage
    n = len(claims)
    cw = claims["Provider_workshop_ID"].notna().sum()
    cc = claims["Provider_clinic_ID"].notna().sum()
    cl = claims["Provider_lawyer_ID"].notna().sum()
    print(f"\n--- Coverage de claims ---")
    print(f"Con workshop: {cw} ({cw/n*100:.1f}%)")
    print(f"Con clinic:   {cc} ({cc/n*100:.1f}%)")
    print(f"Con lawyer:   {cl} ({cl/n*100:.1f}%)")

    has_any = (claims["Provider_workshop_ID"].notna() |
               claims["Provider_clinic_ID"].notna() |
               claims["Provider_lawyer_ID"].notna())
    n_iso = (~has_any).sum()
    print(f"\nClaims aislados: {n_iso} ({n_iso/n*100:.1f}%)")
    if n_iso > 0:
        print(f"  de los cuales fraude: {claims.loc[~has_any,'is_fraud'].sum()}")

    # Concentración de fraude
    print("\n--- Concentración de fraude por proveedor (top por tasa) ---")
    for col,label in [("Provider_workshop_ID","Workshop"),
                       ("Provider_clinic_ID","Clinic"),
                       ("Provider_lawyer_ID","Lawyer")]:
        sub = claims[[col,"is_fraud"]].dropna(subset=[col])
        agg = sub.groupby(col)["is_fraud"].agg(["sum","count"])
        agg["rate"] = agg["sum"]/agg["count"]
        agg = agg.merge(providers[["Provider_ID","Is_fraudulent","Ring_ID"]],
                        left_index=True, right_on="Provider_ID")
        print(f"\n{label} (top 5 con ≥5 claims):")
        top = agg[agg["count"]>=5].sort_values("rate",ascending=False).head(5)
        for _,r in top.iterrows():
            mark = f"[RING {r.Ring_ID}]" if r.Is_fraudulent else "[legit]"
            print(f"  {r.Provider_ID}: {int(r['sum'])}/{int(r['count'])} = {r['rate']*100:.1f}%  {mark}")

    # Balance final
    fraud_claims = claims[claims["is_fraud"]]
    ring_provs = set(providers[providers["Is_fraudulent"]]["Provider_ID"])
    connected = (fraud_claims["Provider_workshop_ID"].isin(ring_provs) |
                 fraud_claims["Provider_clinic_ID"].isin(ring_provs) |
                 fraud_claims["Provider_lawyer_ID"].isin(ring_provs))
    leg_claims = claims[~claims["is_fraud"]]
    leg_connected = (leg_claims["Provider_workshop_ID"].isin(ring_provs) |
                      leg_claims["Provider_clinic_ID"].isin(ring_provs) |
                      leg_claims["Provider_lawyer_ID"].isin(ring_provs))
    print(f"\n--- Balance ---")
    print(f"Fraudes conectados a proveedor de anillo: {connected.sum()}/{len(fraud_claims)} ({connected.mean()*100:.1f}%)")
    print(f"Legítimas conectadas a proveedor de anillo: {leg_connected.sum()}/{len(leg_claims)} ({leg_connected.mean()*100:.1f}%)")
