"""
pipeline/analysis_by_claims_type.py — Análisis de rendimiento por tipo de siniestro.

Genera tabla y gráfico de recall desglosado por Claims_type para los 3 modelos.
No modifica nada del pipeline existente, solo consume los modelos ya entrenados.

Salida en artifacts_comparison/:
  - claims_type_recall_table.csv
  - claims_type_fraud_distribution.csv
  - fig_recall_by_claims_type.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import xgboost as xgb
from sklearn.metrics import precision_recall_curve


def run(cfg):
    art        = Path(cfg["paths"]["artifacts"])
    art_graph  = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_cmp    = Path(cfg["paths"]["artifacts_cmp"])
    art_cmp.mkdir(exist_ok=True, parents=True)
    seed = cfg["seed"]

    # --- Carga de datos ---
    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    y_all  = np.load(art / "y_all.npy")
    idx_v  = np.load(art / "idx_val.npy")
    idx_te = np.load(art / "idx_test.npy")
    y_val  = y_all[idx_v]
    y_test = y_all[idx_te]

    # --- Probabilidades de los 3 modelos ---
    # XGBoost
    X_tab = pd.read_pickle(art / "X_all.pkl")
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(art / "model_tuned.json")
    proba_xgb_val  = m_xgb.predict_proba(X_tab.iloc[idx_v])[:, 1]
    proba_xgb_test = m_xgb.predict_proba(X_tab.iloc[idx_te])[:, 1]

    # GraphSAGE
    from .graph_train import HeteroSAGE
    torch.manual_seed(seed)
    data = torch.load(art_graph / "graph_data.pt", weights_only=False)
    x_claim = data["claim"].x; tm = data["claim"].train_mask
    mean = x_claim[tm].mean(dim=0, keepdim=True)
    std  = x_claim[tm].std(dim=0, keepdim=True).clamp(min=1e-6)
    data["claim"].x = (x_claim - mean) / std
    ckpt = torch.load(art_graph / "model_graphsage_baseline.pt", weights_only=False)
    m_gnn = HeteroSAGE(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                       ckpt["dropout"], ckpt.get("aggregation", "mean"))
    with torch.no_grad(): _ = m_gnn(data.x_dict, data.edge_index_dict)
    m_gnn.load_state_dict(ckpt["state_dict"]); m_gnn.eval()
    with torch.no_grad():
        logits = m_gnn(data.x_dict, data.edge_index_dict)
        proba_gnn_all = torch.sigmoid(logits).cpu().numpy()
    proba_gnn_val  = proba_gnn_all[idx_v]
    proba_gnn_test = proba_gnn_all[idx_te]

    # Híbrido
    X_hyb = pd.read_pickle(art_hybrid / "X_hybrid.pkl")
    m_hyb = joblib.load(art_hybrid / "model_hybrid_tuned.pkl")
    proba_hyb_val  = m_hyb.predict_proba(X_hyb.iloc[idx_v])[:, 1]
    proba_hyb_test = m_hyb.predict_proba(X_hyb.iloc[idx_te])[:, 1]

    print("✔ Modelos cargados")

    # --- Umbrales best-F1 sobre VAL ---
    def best_f1_thr(yv, pv):
        prec, rec, thr = precision_recall_curve(yv, pv)
        f1 = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
        return float(thr[int(np.argmax(f1))])

    thr_xgb = best_f1_thr(y_val, proba_xgb_val)
    thr_gnn = best_f1_thr(y_val, proba_gnn_val)
    thr_hyb = best_f1_thr(y_val, proba_hyb_val)
    print(f"Umbrales best-F1: XGB={thr_xgb:.4f}  GNN={thr_gnn:.4f}  HYB={thr_hyb:.4f}")

    # --- Predicciones binarias ---
    pred_xgb = (proba_xgb_test >= thr_xgb).astype(int)
    pred_gnn = (proba_gnn_test >= thr_gnn).astype(int)
    pred_hyb = (proba_hyb_test >= thr_hyb).astype(int)

    # --- Info de test ---
    test_claims = claims.iloc[idx_te].reset_index(drop=True).copy()
    test_claims["y_true"] = y_test
    test_claims["pred_xgb"] = pred_xgb
    test_claims["pred_gnn"] = pred_gnn
    test_claims["pred_hyb"] = pred_hyb
    test_claims["proba_xgb"] = proba_xgb_test
    test_claims["proba_gnn"] = proba_gnn_test
    test_claims["proba_hyb"] = proba_hyb_test

    # --- Distribución de fraude por Claims_type en test ---
    dist_rows = []
    all_types = sorted(test_claims["Claims_type"].unique())
    for ct in all_types:
        mask = test_claims["Claims_type"] == ct
        n_total = int(mask.sum())
        n_fraud = int(test_claims.loc[mask, "y_true"].sum())
        n_legit = n_total - n_fraud
        dist_rows.append({
            "Claims_type": ct,
            "n_total_test": n_total,
            "n_fraude_test": n_fraud,
            "n_legitimo_test": n_legit,
            "tasa_fraude": round(n_fraud / n_total * 100, 2) if n_total > 0 else 0,
        })
    dist_df = pd.DataFrame(dist_rows).sort_values("n_fraude_test", ascending=False)
    dist_df.to_csv(art_cmp / "claims_type_fraud_distribution.csv", index=False)
    print(f"\n--- Distribución de fraude por tipo (TEST) ---")
    print(dist_df.to_string(index=False))

    # --- Recall por Claims_type × modelo ---
    rows = []
    for ct in all_types:
        mask_fraud = (test_claims["Claims_type"] == ct) & (test_claims["y_true"] == 1)
        n_fraud = int(mask_fraud.sum())

        if n_fraud == 0:
            rows.append({
                "Claims_type": ct, "n_fraude_test": 0,
                "recall_XGBoost": None, "cazados_XGBoost": 0,
                "recall_GraphSAGE": None, "cazados_GraphSAGE": 0,
                "recall_Hibrido": None, "cazados_Hibrido": 0,
                "nota": "Sin fraudes en test",
            })
            continue

        cazados_xgb = int(test_claims.loc[mask_fraud, "pred_xgb"].sum())
        cazados_gnn = int(test_claims.loc[mask_fraud, "pred_gnn"].sum())
        cazados_hyb = int(test_claims.loc[mask_fraud, "pred_hyb"].sum())

        nota = ""
        if n_fraud < 5:
            nota = f"Muestra pequeña (n={n_fraud})"

        rows.append({
            "Claims_type": ct,
            "n_fraude_test": n_fraud,
            "recall_XGBoost": round(cazados_xgb / n_fraud, 4),
            "cazados_XGBoost": cazados_xgb,
            "recall_GraphSAGE": round(cazados_gnn / n_fraud, 4),
            "cazados_GraphSAGE": cazados_gnn,
            "recall_Hibrido": round(cazados_hyb / n_fraud, 4),
            "cazados_Hibrido": cazados_hyb,
            "nota": nota,
        })

    recall_df = pd.DataFrame(rows).sort_values("n_fraude_test", ascending=False)
    recall_df.to_csv(art_cmp / "claims_type_recall_table.csv", index=False)
    print(f"\n--- Recall por Claims_type (TEST, umbral best-F1) ---")
    print(recall_df.to_string(index=False))

    # --- Gráfico ---
    # Filtrar tipos con al menos 1 fraude para el gráfico
    plot_df = recall_df[recall_df["n_fraude_test"] > 0].copy()
    plot_df = plot_df.sort_values("n_fraude_test", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    w = 0.25
    colors = {"XGBoost": "#1f77b4", "GraphSAGE": "#0C855C", "Híbrido": "#f54927"}

    bars_xgb = ax.bar(x - w, plot_df["recall_XGBoost"].fillna(0), w,
                       label="XGBoost", color=colors["XGBoost"], alpha=0.8)
    bars_gnn = ax.bar(x,     plot_df["recall_GraphSAGE"].fillna(0), w,
                       label="GraphSAGE", color=colors["GraphSAGE"], alpha=0.8)
    bars_hyb = ax.bar(x + w, plot_df["recall_Hibrido"].fillna(0), w,
                       label="XGB+GNN", color=colors["Híbrido"], alpha=0.8)

    # Etiquetas con cazados/total
    for bars, col_cazados in [(bars_xgb, "cazados_XGBoost"),
                               (bars_gnn, "cazados_GraphSAGE"),
                               (bars_hyb, "cazados_Hibrido")]:
        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            h = bar.get_height()
            if row["n_fraude_test"] > 0:
                label = f"{int(row[col_cazados])}/{int(row['n_fraude_test'])}"
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        label, ha="center", va="bottom", fontsize=7, rotation=90)

    # Marcar tipos con muestra pequeña
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if 0 < row["n_fraude_test"] < 5:
            ax.text(x[i], -0.08, "⚠", ha="center", fontsize=10, color="orange")

    labels = [f"{row['Claims_type']}\n(n={int(row['n_fraude_test'])})"
              for _, row in plot_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Recall por tipo de siniestro",
                 fontsize=13)
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(art_cmp / "fig_recall_by_claims_type.pdf", dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ fig_recall_by_claims_type.pdf")
    print(f"✔ claims_type_recall_table.csv")
    print(f"✔ claims_type_fraud_distribution.csv")
