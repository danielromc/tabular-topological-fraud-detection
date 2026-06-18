"""
explainability/shap_local.py — XAI Caso de estudio:
explicación SHAP local de un siniestro del top-30 marcado como fraude
por el modelo híbrido.

Cumple el requisito EIOPA de proporcionar "explicación individual" sobre
por qué se marcó un siniestro como sospechoso, con contribuciones por
feature visualizadas en waterfall plot.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

try:
    import shap
except ImportError:
    raise ImportError(
        "Falta el paquete 'shap'. Instálalo con:\n"
        "  pip install shap"
    )


def run(cfg, target_claim=None, n_top_features=15, subtype=None):
    """
    target_claim: claim_id concreto a explicar. Si None y subtype=None,
                  elige el #1 del top.
    n_top_features: cuántas features mostrar en el waterfall.
    subtype: si se especifica, ignora target_claim y elige automáticamente
             un caso del subtipo dado. Valores válidos:
               'organized'  → un fraude de anillo (señal vendrá del grafo)
               'opp_fp'     → oportunista con proveedor fraudulento (mezcla)
               'opp_lp'     → oportunista con proveedor legítimo
                              (señal SOLO de features tabulares)
    """
    art        = Path(cfg["paths"]["artifacts"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_cmp    = Path(cfg["paths"]["artifacts_cmp"])
    out_dir    = art_cmp / "explainability"
    out_dir.mkdir(exist_ok=True, parents=True)

    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    X_hyb  = pd.read_pickle(art_hybrid / "X_hybrid.pkl")
    y_all  = np.load(art / "y_all.npy")
    idx_te = np.load(art / "idx_test.npy")

    m_hyb = joblib.load(art_hybrid / "model_hybrid_tuned.pkl")

    proba_test = m_hyb.predict_proba(X_hyb.iloc[idx_te])[:, 1]
    test_df = claims.iloc[idx_te].reset_index(drop=True).copy()
    test_df["proba"] = proba_test
    test_df["y"] = y_all[idx_te]

    # Selección por subtipo si se pidió
    if subtype is not None:
        if subtype == "organized":
            cand = test_df[(test_df.y == 1) & (test_df.Fraud_type == "organized")]
        elif subtype == "opp_fp":
            cand = test_df[(test_df.y == 1) &
                            test_df.Event_ID.astype(str).str.startswith("siniestro_opp_fp")]
        elif subtype == "opp_lp":
            cand = test_df[(test_df.y == 1) &
                            test_df.Event_ID.astype(str).str.startswith("siniestro_opp_lp")]
        else:
            raise ValueError(f"subtype no válido: {subtype}")
        # Elegir el de mayor probabilidad detectada (caso "exitoso" del modelo)
        cand = cand.sort_values("proba", ascending=False)
        if len(cand) == 0:
            raise ValueError(f"No hay casos del subtipo {subtype} en test")
        target_claim = cand.iloc[0]["claim_id"]
        print(f"Subtipo solicitado: {subtype}  →  claim elegido: {target_claim}  "
              f"(proba={cand.iloc[0]['proba']:.4f})")
    elif target_claim is None:
        # Top fraude real con probabilidad alta
        candidates = test_df[test_df.y == 1].sort_values("proba", ascending=False)
        target_claim = candidates.iloc[0]["claim_id"]

    target_row = test_df[test_df["claim_id"] == target_claim]
    if len(target_row) == 0:
        # Buscar en el dataset completo (puede no estar en test)
        target_row = claims[claims["claim_id"] == target_claim]
        if len(target_row) == 0:
            raise ValueError(f"claim_id {target_claim} no encontrado")

    target_global_idx = claims.index[claims["claim_id"] == target_claim][0]
    print(f"Claim a explicar: {target_claim}")
    print(f"  Cost_claims_year: {target_row.iloc[0]['Cost_claims_year']:.2f}")
    print(f"  Claims_type: {target_row.iloc[0]['Claims_type']}")
    print(f"  Provider_workshop_ID: {target_row.iloc[0].get('Provider_workshop_ID', 'N/A')}")
    print(f"  is_fraud REAL: {bool(y_all[target_global_idx])}")
    print(f"  Probabilidad híbrido: {m_hyb.predict_proba(X_hyb.iloc[[target_global_idx]])[:,1][0]:.4f}")

    # Construir explainer
    print("\nCalculando SHAP values...")
    explainer = shap.TreeExplainer(m_hyb)
    shap_values_target = explainer(X_hyb.iloc[[target_global_idx]])

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values_target[0], max_display=n_top_features,
                          show=False)
    plt.title(f"Explicación SHAP — Claim {target_claim}\n"
              f"(Probabilidad híbrido: {m_hyb.predict_proba(X_hyb.iloc[[target_global_idx]])[:,1][0]:.3f})",
              fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"fig_shap_local_{target_claim}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✔ {out_path.name}")

    # Tabla con las contribuciones
    contributions = pd.DataFrame({
        "feature": X_hyb.columns,
        "valor": X_hyb.iloc[target_global_idx].values,
        "shap_value": shap_values_target.values[0],
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions["tipo"] = contributions["feature"].apply(
        lambda c: "gnn_embedding" if c.startswith("gnn_emb_") else "tabular")
    contributions = contributions.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    csv_path = out_dir / f"shap_local_{target_claim}_contributions.csv"
    contributions.to_csv(csv_path, index=False)
    print(f"✔ {csv_path.name}")
    print(f"\n--- Top 15 contribuciones ---")
    print(contributions.head(15)[["feature","valor","shap_value","tipo"]].to_string(index=False))
