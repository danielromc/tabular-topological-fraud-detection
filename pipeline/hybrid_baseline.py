"""pipeline/hybrid_baseline.py — Fase 3.2: meta = shallow XGB sobre 2 scores.

CAMBIO frente a LogReg: XGBoost shallow (max_depth=3) puede capturar
interacciones no lineales como "si score_gnn > 0.9, ignora score_xgb",
que la regresión logística no podía expresar (combinaba linealmente).
La profundidad limitada y pocos estimadores controlan el overfitting
sobre las 2 features.

Entrenado sobre VAL (scores honestos: las redes XGB y GNN no vieron las
labels de val durante su training).

Salida en artifacts_hybrid/:
  - model_hybrid_baseline.pkl     (XGBClassifier shallow, persistido vía joblib)
  - metrics_hybrid_baseline_val.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def run(cfg):
    seed = cfg["seed"]
    art        = Path(cfg["paths"]["artifacts"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])

    X = pd.read_pickle(art_hybrid/"X_hybrid.pkl")
    y = np.load(art/"y_all.npy")
    idx_val = np.load(art/"idx_val.npy")

    X_val_meta = X.iloc[idx_val].values
    y_val = y[idx_val]

    n_pos = int(y_val.sum()); n_neg = len(y_val) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"Val: n={len(y_val)}  fraudes={n_pos} ({n_pos/len(y_val)*100:.2f}%)")
    print(f"Features meta: {list(X.columns)}")
    print(f"scale_pos_weight = {spw:.3f}")

    # Shallow XGB: capacidad muy limitada porque solo hay 2 features.
    # max_depth=3 permite hasta 8 hojas → suficiente para reglas tipo
    # "si gnn>X y xgb<Y entonces fraude"; pocos estimadores y reg fuerte.
    meta = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        min_child_weight=5.0,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=1.0,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
    )
    meta.fit(X_val_meta, y_val)
    print(f"\nMeta entrenado. Importancias: "
          f"{dict(zip(X.columns, [float(v) for v in meta.feature_importances_]))}")

    # Métricas in-sample sobre val (informativo)
    proba_val = meta.predict_proba(X_val_meta)[:, 1]
    auc_pr = average_precision_score(y_val, proba_val)
    auc_roc = roc_auc_score(y_val, proba_val)
    print(f"\nAUC-PR val (in-sample): {auc_pr:.4f}  AUC-ROC val: {auc_roc:.4f}")

    pred_05 = (proba_val >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_val, pred_05)
    print(f"thr=0.5: P={precision_score(y_val,pred_05,zero_division=0):.4f}  "
          f"R={recall_score(y_val,pred_05,zero_division=0):.4f}  "
          f"F1={f1_score(y_val,pred_05,zero_division=0):.4f}")
    print(f"CM:\n{cm_05}")

    # Persistir vía joblib (compatible con comparison.py y montecarlo.py)
    joblib.dump(meta, art_hybrid/"model_hybrid_baseline.pkl")

    metrics = {
        "model": "xgboost_shallow_meta",
        "stacking_type": "score_level",
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "feature_importances": {k: float(v)
                                 for k, v in zip(X.columns, meta.feature_importances_)},
        "trained_on": "val_split",
        "params": {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1,
                    "scale_pos_weight": spw},
        "val_in_sample": {
            "auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "threshold_0.5": {
                "precision": float(precision_score(y_val, pred_05, zero_division=0)),
                "recall": float(recall_score(y_val, pred_05, zero_division=0)),
                "f1": float(f1_score(y_val, pred_05, zero_division=0)),
                "confusion_matrix": cm_05.tolist(),
            }
        },
    }
    with open(art_hybrid/"metrics_hybrid_baseline_val.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Guardado: model_hybrid_baseline.pkl, metrics_hybrid_baseline_val.json")
