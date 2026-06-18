"""
explainability/shap_analysis.py — Explicabilidad SHAP para XGBoost tabular e hibrido.

Bloques:
A) SHAP XGBoost tabular
B) SHAP XGBoost hibrido (tabular + embeddings GNN)
C) SHAP individual para fraudes capturados solo por el hibrido vs GNN

Notas:
- NO aplica SHAP directo a GraphSAGE; se usa SHAP del hibrido como proxy.
- No reentrena ningun modelo.
- Las salidas se guardan bajo artifacts_comparison/explainability/shap_analysis.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

try:
    import shap
except ImportError as exc:
    print("ERROR: faltan dependencias para SHAP.")
    print("Ejecuta: pip install shap")
    raise exc

THR_GNN = 0.9108480215072632
THR_HYB = 0.7544639110565186


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe {label}: {path}")


def to_shap_matrix(shap_values):
    """Normaliza salida de SHAP a matriz [n_samples, n_features]."""
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError("SHAP devolvio una lista vacia")
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    return arr


def scalar_expected_value(expected_value):
    ev = np.asarray(expected_value)
    if ev.ndim == 0:
        return float(ev)
    if ev.size == 1:
        return float(ev.reshape(-1)[0])
    return float(ev.reshape(-1)[-1])


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", str(value))
    return safe[:120]


def topk_shap_features(values_row: np.ndarray, feature_names: list[str], k: int = 5):
    idx = np.argsort(np.abs(values_row))[::-1][:k]
    out = []
    for i in idx:
        out.append((feature_names[i], float(values_row[i]), int(i)))
    return out


def dominant_block(top_features: list[tuple[str, float, int]]) -> str:
    tab = 0
    emb = 0
    for name, _, _ in top_features:
        if str(name).startswith("gnn_emb_"):
            emb += 1
        else:
            tab += 1
    if tab > emb:
        return "tabular"
    if emb > tab:
        return "embedding"
    return "mixto"


def run(cfg):
    art = Path(cfg["paths"]["artifacts"])
    art_graph = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_cmp = Path(cfg["paths"]["artifacts_cmp"])
    out_dir = art_cmp / "explainability" / "shap_analysis"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 72)
    print("  SHAP ANALYSIS")
    print("=" * 72)

    # ===== Carga base =====
    ensure_exists(art / "idx_test.npy", "idx_test")
    ensure_exists(art / "y_all.npy", "y_all")
    ensure_exists(art / "model_tuned.json", "model_tuned.json")
    ensure_exists(art / "X_all.pkl", "X_all.pkl")
    ensure_exists(art_hybrid / "model_hybrid_tuned.pkl", "model_hybrid_tuned.pkl")
    ensure_exists(art_hybrid / "X_hybrid.pkl", "X_hybrid.pkl")
    ensure_exists(art_graph / "probas_test_gnn.npy", "probas_test_gnn.npy")
    ensure_exists(Path(cfg["paths"]["claims_csv"]), "claims.csv")

    idx_test = np.load(art / "idx_test.npy")
    y_all = np.load(art / "y_all.npy")
    y_test = y_all[idx_test]

    # =========================
    # Bloque A — XGBoost tabular
    # =========================
    print("\n" + "=" * 72)
    print("  BLOQUE A — SHAP XGBOOST TABULAR")
    print("=" * 72)

    X_tab = pd.read_pickle(art / "X_all.pkl")
    X_tab_test = X_tab.iloc[idx_test].copy()

    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(art / "model_tuned.json")

    explainer_xgb = shap.TreeExplainer(model_xgb)
    shap_values_xgb = to_shap_matrix(explainer_xgb.shap_values(X_tab_test))

    np.save(art / "shap_values_xgb_test.npy", shap_values_xgb)
    print("Guardado:", art / "shap_values_xgb_test.npy")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_tab_test, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_xgb_summary_bar.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Guardado:", out_dir / "fig_shap_xgb_summary_bar.pdf")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_tab_test, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_xgb_beeswarm.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Guardado:", out_dir / "fig_shap_xgb_beeswarm.pdf")

    # =========================
    # Bloque B — XGBoost hibrido
    # =========================
    print("\n" + "=" * 72)
    print("  BLOQUE B — SHAP XGBOOST HIBRIDO")
    print("=" * 72)

    X_hyb = pd.read_pickle(art_hybrid / "X_hybrid.pkl")
    X_hyb_test = X_hyb.iloc[idx_test].copy()

    model_hyb = joblib.load(art_hybrid / "model_hybrid_tuned.pkl")

    explainer_hyb = shap.TreeExplainer(model_hyb)
    shap_values_hyb = to_shap_matrix(explainer_hyb.shap_values(X_hyb_test))

    np.save(art_hybrid / "shap_values_hybrid_test.npy", shap_values_hyb)
    print("Guardado:", art_hybrid / "shap_values_hybrid_test.npy")

    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values_hyb, X_hyb_test, plot_type="bar", max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_hybrid_summary_bar.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Guardado:", out_dir / "fig_shap_hybrid_summary_bar.pdf")

    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values_hyb, X_hyb_test, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_hybrid_beeswarm.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Guardado:", out_dir / "fig_shap_hybrid_beeswarm.pdf")

    abs_mean_hyb = np.mean(np.abs(shap_values_hyb), axis=0)
    feature_names_hyb = X_hyb_test.columns.tolist()

    if len(feature_names_hyb) != 109:
        print(f"WARNING: se esperaban 109 features en hibrido y hay {len(feature_names_hyb)}")

    tab_idx = np.arange(0, min(45, len(feature_names_hyb)))
    emb_idx = np.arange(45, min(109, len(feature_names_hyb)))

    sum_tab = float(abs_mean_hyb[tab_idx].sum()) if len(tab_idx) else 0.0
    sum_emb = float(abs_mean_hyb[emb_idx].sum()) if len(emb_idx) else 0.0
    total = sum_tab + sum_emb if (sum_tab + sum_emb) > 0 else 1e-12
    pct_tab = 100.0 * sum_tab / total
    pct_emb = 100.0 * sum_emb / total

    print("\nContribucion agregada por SHAP (|SHAP| medio):")
    print(f"  Tabular (0-44):   {sum_tab:.6f} ({pct_tab:.2f}%)")
    print(f"  Embeddings (45-108): {sum_emb:.6f} ({pct_emb:.2f}%)")

    GAIN_EMB = 97.1
    GAIN_TAB = 2.9
    print("\nComparacion con attribution por gain (reportado):")
    print(f"  Gain -> embeddings: {GAIN_EMB:.1f}% | tabular: {GAIN_TAB:.1f}%")
    print(f"  SHAP -> embeddings: {pct_emb:.2f}% | tabular: {pct_tab:.2f}%")
    if pct_tab > GAIN_TAB:
        print("  Comentario: SHAP asigna relativamente mas peso a tabulares que gain,")
        print("  lo cual es coherente cuando hay features correlacionadas y gain es greedy.")
    else:
        print("  Comentario: SHAP no incrementa el peso tabular frente a gain en esta corrida.")

    # ================================================
    # Bloque C — Casos de fraude capturados solo hibrido
    # ================================================
    print("\n" + "=" * 72)
    print("  BLOQUE C — SHAP INDIVIDUAL CASOS SOLO HIBRIDO")
    print("=" * 72)

    proba_gnn = np.load(art_graph / "probas_test_gnn.npy")
    proba_hyb = (
        np.load(art_hybrid / "probas_test_hybrid_raw.npy")
        if (art_hybrid / "probas_test_hybrid_raw.npy").exists()
        else model_hyb.predict_proba(X_hyb_test)[:, 1]
    )

    proba_xgb = model_xgb.predict_proba(X_tab_test)[:, 1]

    pred_gnn = (proba_gnn >= THR_GNN).astype(int)
    pred_hyb = (proba_hyb >= THR_HYB).astype(int)
    solo_hibrido_idx = np.where((y_test == 1) & (pred_gnn == 0) & (pred_hyb == 1))[0]

    print(f"Fraudes solo-hibrido detectados: {len(solo_hibrido_idx)}")

    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    claims_test = claims.iloc[idx_test].reset_index(drop=True)

    base_value_hyb = scalar_expected_value(explainer_hyb.expected_value)

    rows = []
    for local_idx in solo_hibrido_idx:
        row = claims_test.iloc[int(local_idx)]
        claim_id = row.get("claim_id", f"idx_{int(local_idx)}")

        top5 = topk_shap_features(shap_values_hyb[int(local_idx)], feature_names_hyb, k=5)
        block = dominant_block(top5)

        explanation = shap.Explanation(
            values=shap_values_hyb[int(local_idx)],
            base_values=base_value_hyb,
            data=X_hyb_test.iloc[int(local_idx)].values,
            feature_names=feature_names_hyb,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        out_name = f"fig_shap_force_{sanitize_filename(claim_id)}.pdf"
        plt.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close()

        print("\nCaso solo-hibrido:")
        print(
            f"  claim_id={claim_id} | Claims_type={row.get('Claims_type', np.nan)} | "
            f"Ring_ID={row.get('Ring_ID', np.nan)} | Fraud_type={row.get('Fraud_type', np.nan)}"
        )
        print(
            f"  Cost_claims_year={row.get('Cost_claims_year', np.nan)} | "
            f"LossRatio={row.get('LossRatio', np.nan)} | z_score_type={row.get('z_score_type', np.nan)}"
        )
        print(
            f"  Provider_workshop_ID={row.get('Provider_workshop_ID', np.nan)} | "
            f"Provider_clinic_ID={row.get('Provider_clinic_ID', np.nan)} | "
            f"Provider_lawyer_ID={row.get('Provider_lawyer_ID', np.nan)}"
        )
        print(
            f"  proba_xgb={proba_xgb[int(local_idx)]:.4f} | "
            f"proba_gnn={proba_gnn[int(local_idx)]:.4f} | "
            f"proba_hyb={proba_hyb[int(local_idx)]:.4f}"
        )

        rec = {
            "claim_id": claim_id,
            "Claims_type": row.get("Claims_type", np.nan),
            "Ring_ID": row.get("Ring_ID", np.nan),
            "Fraud_type": row.get("Fraud_type", np.nan),
            "Cost_claims_year": row.get("Cost_claims_year", np.nan),
            "LossRatio": row.get("LossRatio", np.nan),
            "z_score_type": row.get("z_score_type", np.nan),
            "proba_xgb": float(proba_xgb[int(local_idx)]),
            "proba_gnn": float(proba_gnn[int(local_idx)]),
            "proba_hyb": float(proba_hyb[int(local_idx)]),
            "top1_feature": top5[0][0] if len(top5) > 0 else None,
            "top1_shap": top5[0][1] if len(top5) > 0 else None,
            "top2_feature": top5[1][0] if len(top5) > 1 else None,
            "top2_shap": top5[1][1] if len(top5) > 1 else None,
            "top3_feature": top5[2][0] if len(top5) > 2 else None,
            "top3_shap": top5[2][1] if len(top5) > 2 else None,
            "top4_feature": top5[3][0] if len(top5) > 3 else None,
            "top4_shap": top5[3][1] if len(top5) > 3 else None,
            "top5_feature": top5[4][0] if len(top5) > 4 else None,
            "top5_shap": top5[4][1] if len(top5) > 4 else None,
            "bloque_dominante": block,
        }
        rows.append(rec)

    casos_df = pd.DataFrame(rows)
    casos_out = out_dir / "casos_solo_hibrido.csv"
    casos_df.to_csv(casos_out, index=False)
    print("\nGuardado:", casos_out)

    print("\nResumen de casos solo-hibrido")
    if len(casos_df) == 0:
        print("  No hay casos solo-hibrido con los thresholds actuales.")
    else:
        print("  Distribucion bloque_dominante:")
        print(casos_df["bloque_dominante"].value_counts(dropna=False).to_string())

        print("\n  Distribucion por Claims_type:")
        print(casos_df["Claims_type"].value_counts(dropna=False).to_string())

        print("\n  Distribucion por Ring_ID:")
        print(casos_df["Ring_ID"].value_counts(dropna=False).to_string())

        mean_loss = float(pd.to_numeric(casos_df["LossRatio"], errors="coerce").mean())
        frac_opp = float((casos_df["Fraud_type"].astype(str) == "opportunistic").mean())
        frac_none_ring = float((casos_df["Ring_ID"].astype(str).str.lower() == "none").mean())
        dom = casos_df["bloque_dominante"].value_counts().idxmax()

        print("\nInterpretacion automatica:")
        print(
            f"  Los casos solo-hibrido muestran bloque dominante '{dom}', "
            f"con {frac_opp*100:.1f}% oportunistas y {frac_none_ring*100:.1f}% sin anillo; "
            f"LossRatio medio={mean_loss:.3f}."
        )

    print("\n" + "=" * 72)
    print("  FIN SHAP ANALYSIS")
    print("=" * 72)
    print(f"\n✔ Salidas SHAP guardadas en: {out_dir.resolve()}")


def main():
    import yaml

    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"No encuentro {config_path} en el directorio actual.")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
