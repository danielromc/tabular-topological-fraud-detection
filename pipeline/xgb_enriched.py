"""
pipeline/xgb_enriched.py — XGBoost enriquecido con features relacionales honestas.

Modelo intermedio entre el baseline tabular (42 features) y el GNN (topología).
Añade 6 features descriptivas por proveedor, calculadas EXCLUSIVAMENTE sobre
el conjunto de entrenamiento para evitar data leakage.

Features añadidas (6):
  Volumen (nº de claims atendidos por el proveedor en train):
    wsh_volume_train, cln_volume_train, law_volume_train     (3 numéricas)
  Coste medio (coste promedio de los claims del proveedor en train):
    wsh_avg_cost_train, cln_avg_cost_train, law_avg_cost_train (3 numéricas)

EXCLUIDAS POR ARTEFACTO DE INYECCIÓN:
  - has_workshop, has_clinic, has_lawyer, n_providers
    (todo fraude tiene proveedor por diseño del script R)

EXCLUIDAS POR TARGET ENCODING:
  - wsh/cln/law_fraud_rate_smooth
    (codifican la variable objetivo como feature, dando al XGBoost
     acceso directo a la señal que el GNN debe aprender por sí solo)

Total: 42 + 6 = 48 features.

Ejecución:
  python main.py xgb_enriched
"""
import json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def compute_enriched_features(claims, X_tab, idx_train, y_all):
    """
    Calcula las 6 features relacionales usando SOLO datos de train.
    Devuelve X_enriched (DataFrame de n x 48) para TODO el dataset.
    
    FLUJO ANTI-LEAKAGE:
    1. Se identifican los claims de train.
    2. Se calculan volumen y coste medio por proveedor SOLO sobre train.
    3. Se mapean esas estadísticas a TODOS los claims (train+val+test)
       via join por Provider_ID.
    4. Claims sin proveedor reciben valores por defecto (0 para volumen,
       media global para coste).
    
    NO se incluyen:
    - has_workshop/clinic/lawyer, n_providers (artefactos de inyección)
    - fraud_rate por proveedor (target encoding: convierte y en x)
    """
    n = len(claims)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[idx_train] = True

    # Coste medio global en train (para default de avg_cost)
    global_avg_cost = claims.loc[train_mask, "Cost_claims_year"].mean()
    print(f"  Coste medio global train: {global_avg_cost:.2f}")

    # --- Features por tipo de proveedor: volume y avg_cost ---
    provider_cols = [
        ("Provider_workshop_ID", "wsh"),
        ("Provider_clinic_ID",   "cln"),
        ("Provider_lawyer_ID",   "law"),
    ]

    enriched = pd.DataFrame(index=claims.index)

    for col, prefix in provider_cols:
        # Subset de train con proveedor no-nulo
        train_with_prov = claims[train_mask & claims[col].notna()].copy()

        if len(train_with_prov) == 0:
            enriched[f"{prefix}_volume_train"] = 0
            enriched[f"{prefix}_avg_cost_train"] = global_avg_cost
            continue

        # Agregar por proveedor (SOLO train)
        agg = train_with_prov.groupby(col).agg(
            volume=(col, "count"),
            avg_cost=("Cost_claims_year", "mean"),
        ).reset_index()
        agg.columns = [col, "volume", "avg_cost"]

        # Mapear a todo el dataset via join
        mapping_vol = dict(zip(agg[col], agg["volume"]))
        mapping_cost = dict(zip(agg[col], agg["avg_cost"]))

        enriched[f"{prefix}_volume_train"] = (
            claims[col].map(mapping_vol).fillna(0)
        )
        enriched[f"{prefix}_avg_cost_train"] = (
            claims[col].map(mapping_cost).fillna(global_avg_cost)
        )

        # Diagnóstico
        n_provs_mapped = claims[col].isin(agg[col].values).sum()
        n_provs_unseen = claims[col].notna().sum() - n_provs_mapped
        print(f"  {prefix}: {len(agg)} proveedores en train, "
              f"{n_provs_unseen} claims con proveedor no visto en train → default")

    # Concatenar con las 42 features tabulares originales
    X_enriched = pd.concat([X_tab.reset_index(drop=True),
                             enriched.reset_index(drop=True)], axis=1)

    print(f"\n  X_enriched: {X_enriched.shape} "
          f"({X_tab.shape[1]} tab + {enriched.shape[1]} relacional)")

    return X_enriched


def run(cfg):
    seed = cfg["seed"]
    art = Path(cfg["paths"]["artifacts"])
    art_cmp = Path(cfg["paths"]["artifacts_cmp"])
    out_dir = art_cmp / "xgb_enriched"
    out_dir.mkdir(exist_ok=True, parents=True)

    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    X_tab = pd.read_pickle(art / "X_all.pkl")
    y_all = np.load(art / "y_all.npy")
    idx_tr = np.load(art / "idx_train.npy")
    idx_v  = np.load(art / "idx_val.npy")
    idx_te = np.load(art / "idx_test.npy")

    print("="*70)
    print("XGBOOST ENRIQUECIDO — 42 tab + 6 relacionales = 48 features")
    print("="*70)

    # =========================================================
    # 1. Construir features enriquecidas
    # =========================================================
    print("\n[1/4] Calculando features relacionales (solo sobre train)...")
    X_enriched = compute_enriched_features(claims, X_tab, idx_tr, y_all)
    X_enriched.to_pickle(out_dir / "X_enriched.pkl")

    Xtr = X_enriched.iloc[idx_tr]; ytr = y_all[idx_tr]
    Xv  = X_enriched.iloc[idx_v];  yv  = y_all[idx_v]
    Xte = X_enriched.iloc[idx_te]; yte = y_all[idx_te]

    n_pos = int(ytr.sum()); n_neg = int(len(ytr) - n_pos)
    spw_base = n_neg / n_pos
    print(f"\n  Train: {Xtr.shape}  fraudes={n_pos}")
    print(f"  Val:   {Xv.shape}   fraudes={yv.sum()}")
    print(f"  SPW base: {spw_base:.3f}")

    # =========================================================
    # 2. Optuna (50 ensayos)
    # =========================================================
    print(f"\n[2/4] Optuna — 50 ensayos sobre {X_enriched.shape[1]} features...")
    n_trials = cfg["xgb_tune"]["n_trials"]

    def objective(trial):
        params = {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "tree_method": "hist", "random_state": seed, "n_jobs": -1,
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw_base * trial.suggest_float("spw_mult", 0.3, 2.0),
        }
        m = xgb.XGBClassifier(**params, n_estimators=2000, early_stopping_rounds=50)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        proba = m.predict_proba(Xv)[:, 1]
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        return average_precision_score(yv, proba)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                 study_name="xgb_enriched")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Mejor trial #{study.best_trial.number}  AUC-PR val: {study.best_value:.4f}")

    # Reentrenar con mejores hiperparámetros
    best = dict(study.best_params)
    best_n_est = study.best_trial.user_attrs["best_iteration"] + 1
    spw_final = spw_base * best.pop("spw_mult")
    final_params = {"objective": "binary:logistic", "eval_metric": "aucpr",
                    "tree_method": "hist", "random_state": seed, "n_jobs": -1,
                    "scale_pos_weight": spw_final, **best}
    model = xgb.XGBClassifier(**final_params, n_estimators=best_n_est)
    model.fit(Xtr, ytr, verbose=False)

    # Guardar
    model.save_model(out_dir / "model_enriched.json")
    with open(out_dir / "optuna_enriched_best_params.json", "w") as f:
        json.dump({**final_params, "n_estimators": best_n_est}, f, indent=2)

    # =========================================================
    # 3. Evaluación en test
    # =========================================================
    print(f"\n[3/4] Evaluación en test...")

    proba_val = model.predict_proba(Xv)[:, 1]
    proba_test = model.predict_proba(Xte)[:, 1]

    # Threshold best-F1 sobre val
    prec_v, rec_v, thr_v = precision_recall_curve(yv, proba_val)
    f1_v = 2*prec_v[:-1]*rec_v[:-1]/(prec_v[:-1]+rec_v[:-1]+1e-12)
    thr_bestF1 = float(thr_v[int(np.argmax(f1_v))])

    auc_pr = average_precision_score(yte, proba_test)
    auc_roc = roc_auc_score(yte, proba_test)
    pred = (proba_test >= thr_bestF1).astype(int)
    cm = confusion_matrix(yte, pred)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)

    # Top-30
    top_ix = np.argsort(proba_test)[::-1][:30]
    p30 = int(yte[top_ix].sum()) / 30

    print(f"\n  === TEST ===")
    print(f"  AUC-PR:  {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Threshold best-F1 val: {thr_bestF1:.4f}")
    print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print(f"  Precision@30: {p30:.4f} ({int(yte[top_ix].sum())}/30)")

    # Feature importance
    fi = pd.DataFrame({"feature": X_enriched.columns,
                        "gain": model.feature_importances_})
    fi["tipo"] = fi["feature"].apply(
        lambda c: "relacional" if c in [
            "wsh_volume_train", "cln_volume_train", "law_volume_train",
            "wsh_avg_cost_train", "cln_avg_cost_train", "law_avg_cost_train",
        ] else "tabular"
    )
    fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)
    fi.to_csv(out_dir / "feature_importance_enriched.csv", index=False)
    total_rel = fi.loc[fi.tipo == "relacional", "gain"].sum()
    total_tab = fi.loc[fi.tipo == "tabular", "gain"].sum()
    print(f"\n  --- Gain total ---")
    print(f"  Relacional: {total_rel:.4f} ({total_rel/(total_rel+total_tab)*100:.1f}%)")
    print(f"  Tabular:    {total_tab:.4f} ({total_tab/(total_rel+total_tab)*100:.1f}%)")
    print(f"\n  --- Top 15 features ---")
    print(fi.head(15)[["feature", "gain", "tipo"]].to_string(index=False))

    # Guardar métricas
    metrics = {
        "model": "xgboost_enriched_55feats",
        "n_features": int(X_enriched.shape[1]),
        "n_features_tabular": 42,
        "n_features_relacional": 6,
        "test": {
            "auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "threshold_bestF1": float(thr_bestF1),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "precision_at_30": float(p30),
        },
        "feature_importance": {
            "gain_relacional": float(total_rel),
            "gain_tabular": float(total_tab),
            "pct_relacional": float(total_rel/(total_rel+total_tab)),
        },
    }
    with open(out_dir / "metrics_enriched.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # =========================================================
    # 4. Comparativa con los otros modelos
    # =========================================================
    print(f"\n[4/4] Comparativa rápida...")

    # Cargar métricas del baseline
    baseline_metrics_path = art / "metrics_tuned_test.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            bl = json.load(f)
        bl_auc_pr = bl["test"]["auc_pr"]
        bl_auc_roc = bl["test"]["auc_roc"]
    else:
        bl_auc_pr = bl_auc_roc = None

    # Cargar métricas del GNN
    gnn_metrics_path = Path(cfg["paths"]["artifacts_graph"]) / "metrics_graphsage_test.json"
    if gnn_metrics_path.exists():
        with open(gnn_metrics_path) as f:
            gn = json.load(f)
        gn_auc_pr = gn["test"]["auc_pr"]
        gn_auc_roc = gn["test"]["auc_roc"]
    else:
        gn_auc_pr = gn_auc_roc = None

    # Cargar métricas del híbrido
    hyb_metrics_path = Path(cfg["paths"]["artifacts_hybrid"]) / "metrics_hybrid_test.json"
    if hyb_metrics_path.exists():
        with open(hyb_metrics_path) as f:
            hy = json.load(f)
        hy_auc_pr = hy["test"]["auc_pr"]
        hy_auc_roc = hy["test"]["auc_roc"]
    else:
        hy_auc_pr = hy_auc_roc = None

    print(f"\n  {'Modelo':<30s} {'AUC-PR':>8s} {'AUC-ROC':>8s}")
    print(f"  {'-'*48}")
    if bl_auc_pr is not None:
        print(f"  {'XGBoost baseline (42 feat)':<30s} {bl_auc_pr:>8.4f} {bl_auc_roc:>8.4f}")
    print(f"  {'XGBoost enriquecido (48 feat)':<30s} {auc_pr:>8.4f} {auc_roc:>8.4f}")
    if gn_auc_pr is not None:
        print(f"  {'GraphSAGE':<30s} {gn_auc_pr:>8.4f} {gn_auc_roc:>8.4f}")
    if hy_auc_pr is not None:
        print(f"  {'Híbrido XGB+GNN':<30s} {hy_auc_pr:>8.4f} {hy_auc_roc:>8.4f}")

    print(f"\n✔ Todo guardado en {out_dir.resolve()}")
