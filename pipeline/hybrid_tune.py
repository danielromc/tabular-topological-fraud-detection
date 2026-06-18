"""pipeline/hybrid_tune.py — Fase 3.3: tuning del shallow XGB meta vía Optuna.

Sustituye GridSearchCV por Optuna (TPE sampler) para mantener coherencia
metodológica con xgb_tune.py de fase 1: mismo número de trials, misma
semilla, mismo sampler. Cross-validation 3-fold dentro de cada trial sobre
val mantiene la honestidad (no entrenar y evaluar en exactamente el mismo
fold).

Salida en artifacts_hybrid/:
  - model_hybrid_tuned.pkl
  - optuna_hybrid_study.pkl
  - optuna_hybrid_best_params.json
  - optuna_hybrid_trials.csv
  - metrics_hybrid_tuned_val.json
"""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def run(cfg):
    seed = cfg["seed"]
    n_trials = cfg["hybrid"]["xgb_tune_trials"]
    art        = Path(cfg["paths"]["artifacts"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])

    X = pd.read_pickle(art_hybrid/"X_hybrid.pkl")
    y = np.load(art/"y_all.npy")
    idx_val = np.load(art/"idx_val.npy")

    X_val_meta = X.iloc[idx_val].values
    y_val = y[idx_val]

    n_pos = int(y_val.sum()); n_neg = len(y_val) - n_pos
    spw_base = n_neg / max(n_pos, 1)
    print(f"Val: n={len(y_val)}  fraudes={n_pos}")
    print(f"Features meta: {list(X.columns)}")
    print(f"spw_base = {spw_base:.3f}")

    # CV interna 3-fold (consistente con honestidad de GridSearchCV anterior)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_splits = list(cv.split(X_val_meta, y_val))

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
            "max_depth":         trial.suggest_int("max_depth", 2, 5),
            "n_estimators":      trial.suggest_int("n_estimators", 20, 200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight":  trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "scale_pos_weight":  spw_base * trial.suggest_float("spw_mult", 0.3, 2.0),
        }
        # 3-fold CV sobre val
        aucprs = []
        for fold_tr, fold_va in cv_splits:
            m = xgb.XGBClassifier(**params)
            m.fit(X_val_meta[fold_tr], y_val[fold_tr])
            proba = m.predict_proba(X_val_meta[fold_va])[:, 1]
            aucprs.append(average_precision_score(y_val[fold_va], proba))
        return float(np.mean(aucprs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                 study_name="hybrid_meta_xgb")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nLanzando {n_trials} trials Optuna (TPE) con 3-fold CV interna...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[OK] Mejor trial #{study.best_trial.number}")
    print(f"  AUC-PR cv-mean: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Reentrenar el modelo final sobre TODO val con los mejores params
    best = dict(study.best_params)
    spw_final = spw_base * best.pop("spw_mult")
    final_params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "random_state": seed, "n_jobs": -1,
        "scale_pos_weight": spw_final, **best,
    }
    best_meta = xgb.XGBClassifier(**final_params)
    best_meta.fit(X_val_meta, y_val)
    fi = dict(zip(X.columns, [float(v) for v in best_meta.feature_importances_]))
    print(f"\n  Feature importances finales: {fi}")

    # Métricas in-sample sobre val
    proba = best_meta.predict_proba(X_val_meta)[:, 1]
    auc_pr = average_precision_score(y_val, proba)
    auc_roc = roc_auc_score(y_val, proba)
    pred_05 = (proba >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_val, pred_05)
    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_ix = int(np.argmax(f1_arr))

    print(f"\n--- Tuned val (in-sample tras refit) ---")
    print(f"AUC-PR: {auc_pr:.4f}  AUC-ROC: {auc_roc:.4f}")
    print(f"thr=0.5: P={precision_score(y_val,pred_05,zero_division=0):.4f}  "
          f"R={recall_score(y_val,pred_05,zero_division=0):.4f}  "
          f"F1={f1_score(y_val,pred_05,zero_division=0):.4f}")
    print(f"bestF1 ({thr[best_ix]:.4f}): P={prec[best_ix]:.4f}  R={rec[best_ix]:.4f}  "
          f"F1={f1_arr[best_ix]:.4f}")

    # Persistir
    joblib.dump(best_meta, art_hybrid/"model_hybrid_tuned.pkl")
    with open(art_hybrid/"optuna_hybrid_study.pkl", "wb") as f:
        pickle.dump(study, f)
    with open(art_hybrid/"optuna_hybrid_best_params.json", "w") as f:
        json.dump(final_params, f, indent=2)
    study.trials_dataframe().to_csv(art_hybrid/"optuna_hybrid_trials.csv", index=False)

    metrics = {
        "model": "xgboost_shallow_meta_optuna",
        "stacking_type": "score_level",
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "best_params": final_params,
        "best_cv_score": float(study.best_value),
        "feature_importances": fi,
        "val_in_sample": {
            "auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "threshold_0.5": {
                "precision": float(precision_score(y_val, pred_05, zero_division=0)),
                "recall": float(recall_score(y_val, pred_05, zero_division=0)),
                "f1": float(f1_score(y_val, pred_05, zero_division=0)),
                "confusion_matrix": cm_05.tolist(),
            },
            "threshold_bestF1": {
                "threshold": float(thr[best_ix]),
                "precision": float(prec[best_ix]),
                "recall": float(rec[best_ix]),
                "f1": float(f1_arr[best_ix]),
            },
        },
    }
    with open(art_hybrid/"metrics_hybrid_tuned_val.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Guardado: model_hybrid_tuned.pkl, optuna_hybrid_study.pkl, "
          f"optuna_hybrid_best_params.json, optuna_hybrid_trials.csv, "
          f"metrics_hybrid_tuned_val.json")
