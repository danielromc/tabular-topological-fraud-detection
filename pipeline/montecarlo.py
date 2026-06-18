"""
pipeline/montecarlo.py — Montecarlo con N semillas configurable.

Para cada semilla ejecuta el pipeline completo (prep → tune XGB → graph_train →
hybrid_extract → hybrid_tune) y captura métricas en TEST de los 3 modelos.

Las carpetas artifacts/artifacts_graph/artifacts_hybrid se sobrescriben en cada
iteración porque son de trabajo. Solo guardamos los resultados finales en:

  artifacts_comparison/montecarlo/
    ├── montecarlo_raw.csv          (1 fila por modelo x semilla = 3*N filas)
    ├── montecarlo_summary.csv      (media ± std por modelo)
    ├── fig_montecarlo_boxplots.pdf (3 paneles: AUC-PR, AUC-ROC, Precision@30)
    ├── fig_montecarlo_cm_means.pdf (matrices de confusión promedio)
    └── results_seed_<s>.json       (detalle por semilla)

Duración aproximada: ~4-6 min por semilla en CPU x N semillas.
"""
import json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
import joblib
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from ._thresholds import best_fbeta_thr

DEFAULT_SEEDS = [2025, 42, 7, 123, 2024, 99, 777, 1, 500, 314]


def _resolve_seeds(cfg):
    """Resuelve lista de semillas desde config con fallback retrocompatible."""
    mc_cfg = cfg.get("montecarlo", {})

    # Opción 1: lista explícita de semillas
    seeds_cfg = mc_cfg.get("seeds")
    if isinstance(seeds_cfg, list) and len(seeds_cfg) > 0:
        return [int(s) for s in seeds_cfg]

    # Opción 2: generar N semillas reproducibles
    n_seeds = int(mc_cfg.get("n_seeds", len(DEFAULT_SEEDS)))
    seed_base = int(mc_cfg.get("seed_base", cfg.get("seed", 2025)))
    rng = np.random.default_rng(seed_base)
    # Muestra sin reemplazo en rango amplio para evitar duplicados
    seeds = rng.choice(np.arange(1, 1_000_000, dtype=np.int64), size=n_seeds, replace=False)
    return [int(s) for s in seeds.tolist()]


def run(cfg):
    seeds = _resolve_seeds(cfg)
    art_cmp = Path(cfg["paths"]["artifacts_cmp"])
    mc_dir = art_cmp / "montecarlo"
    mc_dir.mkdir(exist_ok=True, parents=True)

    print(f"Montecarlo con {len(seeds)} semillas")
    print(f"Primeras semillas: {seeds[:10]}{' ...' if len(seeds) > 10 else ''}")
    print(f"Salida: {mc_dir.resolve()}")
    t_global = time.time()

    all_rows = []

    for i, seed in enumerate(seeds, start=1):
        t_seed = time.time()
        print(f"\n{'#'*72}")
        print(f"# ITERACIÓN {i}/{len(seeds)} — SEED={seed}")
        print('#'*72)

        # Clonar cfg y sobrescribir seed
        cfg_iter = {k: (v.copy() if isinstance(v, dict) else v)
                    for k, v in cfg.items()}
        cfg_iter["seed"] = seed

        # -------- Pipeline completo --------
        from pipeline import (prep, xgb_tune, graph_build, graph_train,
                              hybrid_extract, hybrid_tune)

        # Limpiar artefactos previos (solo archivos, no carpetas)
        for d in [cfg["paths"]["artifacts"], cfg["paths"]["artifacts_graph"],
                  cfg["paths"]["artifacts_hybrid"]]:
            dp = Path(d)
            if dp.exists():
                for f in dp.iterdir():
                    if f.is_file():
                        f.unlink()

        print(f"\n[{seed}] Paso 1/5: prep")
        prep.run(cfg_iter)
        print(f"\n[{seed}] Paso 2/5: tune XGBoost ({cfg_iter['xgb_tune']['n_trials']} trials)")
        xgb_tune.run(cfg_iter)
        print(f"\n[{seed}] Paso 3/5: graph_build + graph_train")
        graph_build.run(cfg_iter)
        graph_train.run(cfg_iter)
        print(f"\n[{seed}] Paso 4/5: hybrid_extract")
        hybrid_extract.run(cfg_iter)
        print(f"\n[{seed}] Paso 5/5: hybrid_tune")
        hybrid_tune.run(cfg_iter)

        # -------- Evaluación en test de los 3 modelos --------
        seed_results = evaluate_all_models(cfg_iter, seed)

        for row in seed_results:
            all_rows.append(row)

        # Guardar detalle por semilla
        with open(mc_dir / f"results_seed_{seed}.json", "w") as f:
            json.dump(seed_results, f, indent=2)

        elapsed = time.time() - t_seed
        print(f"\n[{seed}] ✔ Iteración completada en {elapsed/60:.1f} min")

    # -------- Agregación final --------
    df = pd.DataFrame(all_rows)
    df.to_csv(mc_dir / "montecarlo_raw.csv", index=False)

    # Resumen media ± std por modelo
    agg_cols = ["auc_pr", "auc_roc", "precision", "recall", "f1",
                "TN", "FP", "FN", "TP", "precision_at_30"]
    summary = df.groupby("modelo")[agg_cols].agg(["mean", "std", "min", "max"])
    summary.to_csv(mc_dir / "montecarlo_summary.csv")
    print("\n" + "="*72)
    print("RESUMEN MEDIA ± DESVIACIÓN TÍPICA")
    print("="*72)
    print(summary.round(4).to_string())

    # -------- Figura 1: Boxplots --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"XGBoost": "#1f77b4",
              "GraphSAGE":  "#0C855C",
              "XGB+GNN":  "#f54927"}
    order = ["XGBoost", "GraphSAGE", "XGB+GNN"]
    metrics_plot = [("auc_pr", "AUC-PR"),
                    ("auc_roc", "AUC-ROC"),
                    ("f1", "F1-score")]

    for ax, (metric, title) in zip(axes, metrics_plot):
        data = [df[df.modelo == m][metric].values for m in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.6,
                         medianprops=dict(color="black", linewidth=2),
                         flierprops=dict(marker="o", markerfacecolor="black", markersize=4))
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.7)
        # Scatter de cada punto
        for j, m in enumerate(order):
            vals = df[df.modelo == m][metric].values
            x_jit = np.random.normal(j+1, 0.06, size=len(vals))
            ax.scatter(x_jit, vals, alpha=0.6, color="black", s=20, zorder=3)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=10)
    fig.suptitle("Distribución de métricas por modelo",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(mc_dir / "fig_montecarlo_boxplots.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ fig_montecarlo_boxplots.pdf")

    # -------- Figura 2: Matrices de confusión promedio --------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, name in zip(axes, order):
        sub = df[df.modelo == name]
        cm_mean = np.array([[sub.TN.mean(), sub.FP.mean()],
                             [sub.FN.mean(), sub.TP.mean()]])
        cm_std = np.array([[sub.TN.std(), sub.FP.std()],
                            [sub.FN.std(), sub.TP.std()]])
        ax.imshow(cm_mean, cmap="Blues", vmin=0, vmax=cm_mean.max())
        for i in range(2):
            for j in range(2):
                txt = f"{cm_mean[i,j]:.1f}\n± {cm_std[i,j]:.1f}"
                color = "white" if cm_mean[i,j] > cm_mean.max()/2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)
        p_mean, p_std = sub.precision.mean(), sub.precision.std()
        r_mean, r_std = sub.recall.mean(), sub.recall.std()
        f1_mean, f1_std = sub.f1.mean(), sub.f1.std()
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["No fraude", "Fraude"])
        ax.set_yticklabels(["No fraude", "Fraude"])
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Realidad", fontsize=11)
        ax.set_title(f"{name}\nP={p_mean:.2f}±{p_std:.2f}  "
                     f"R={r_mean:.2f}±{r_std:.2f}  F1={f1_mean:.2f}±{f1_std:.2f}",
                     fontsize=10)
    fig.suptitle(f"Matriz de confusión promedio en el conjunto de prueba",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(mc_dir / "fig_montecarlo_cm_means.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_montecarlo_cm_means.pdf")

    t_total = time.time() - t_global
    print(f"\n{'='*72}")
    print(f"✔ MONTECARLO COMPLETADO en {t_total/60:.1f} minutos")
    print(f"✔ Resultados en {mc_dir.resolve()}")
    print('='*72)


def evaluate_all_models(cfg, seed):
    """Evalúa los 3 modelos entrenados con esta semilla y devuelve 3 filas."""
    import torch
    from pipeline.graph_train import HeteroSAGE

    art = Path(cfg["paths"]["artifacts"])
    art_graph = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])

    y_all = np.load(art / "y_all.npy")
    idx_v = np.load(art / "idx_val.npy")
    idx_te = np.load(art / "idx_test.npy")
    y_val = y_all[idx_v]
    y_test = y_all[idx_te]

    # XGBoost
    X_tab = pd.read_pickle(art / "X_all.pkl")
    m_xgb = xgb.XGBClassifier(); m_xgb.load_model(art / "model_tuned.json")
    proba_xgb_val = m_xgb.predict_proba(X_tab.iloc[idx_v])[:,1]
    proba_xgb_test = m_xgb.predict_proba(X_tab.iloc[idx_te])[:,1]

    # GraphSAGE
    torch.manual_seed(seed)
    # weights_only=False solo para HeteroData; checkpoint del modelo seguro con True (S1)
    data = torch.load(art_graph / "graph_data.pt", weights_only=False)
    x_claim = data["claim"].x
    tm = data["claim"].train_mask
    mean = x_claim[tm].mean(dim=0, keepdim=True)
    std = x_claim[tm].std(dim=0, keepdim=True).clamp(min=1e-6)
    data["claim"].x = (x_claim - mean) / std

    ckpt = torch.load(art_graph / "model_graphsage_baseline.pt", weights_only=True)
    m_gnn = HeteroSAGE(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                       ckpt["dropout"], ckpt.get("aggregation", "mean"))
    with torch.no_grad(): _ = m_gnn(data.x_dict, data.edge_index_dict)
    m_gnn.load_state_dict(ckpt["state_dict"])
    m_gnn.eval()
    with torch.no_grad():
        logits = m_gnn(data.x_dict, data.edge_index_dict)
        proba_gnn_all = torch.sigmoid(logits).cpu().numpy()
    proba_gnn_val = proba_gnn_all[idx_v]
    proba_gnn_test = proba_gnn_all[idx_te]

    # Híbrido — meta LogReg sobre [score_xgb, score_gnn] (score-stacking)
    X_hyb = pd.read_pickle(art_hybrid / "X_hybrid.pkl")
    m_hyb = joblib.load(art_hybrid / "model_hybrid_tuned.pkl")
    proba_hyb_val = m_hyb.predict_proba(X_hyb.iloc[idx_v])[:,1]
    proba_hyb_test = m_hyb.predict_proba(X_hyb.iloc[idx_te])[:,1]

    beta = float(cfg["eval"].get("fbeta", 1.0))
    thr_xgb = best_fbeta_thr(y_val, proba_xgb_val, beta=beta)
    thr_gnn = best_fbeta_thr(y_val, proba_gnn_val, beta=beta)
    thr_hyb = best_fbeta_thr(y_val, proba_hyb_val, beta=beta)

    rows = []
    for name, proba, thr in [("XGBoost", proba_xgb_test, thr_xgb),
                               ("GraphSAGE", proba_gnn_test, thr_gnn),
                               ("XGB+GNN", proba_hyb_test, thr_hyb)]:
        pred = (proba >= thr).astype(int)
        cm = confusion_matrix(y_test, pred)
        top_ix = np.argsort(proba)[::-1][:30]
        p30 = int(y_test[top_ix].sum()) / 30
        rows.append({
            "seed": seed,
            "modelo": name,
            "threshold": round(thr, 4),
            "auc_pr": round(average_precision_score(y_test, proba), 4),
            "auc_roc": round(roc_auc_score(y_test, proba), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
            "precision_at_30": round(p30, 4),
        })
    return rows
