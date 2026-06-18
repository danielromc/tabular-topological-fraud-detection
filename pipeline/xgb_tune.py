"""pipeline/xgb_tune.py — Fase 1.3: Optuna sobre XGBoost."""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def run(cfg):
    random_state = cfg["seed"]
    n_trials = cfg["xgb_tune"]["n_trials"]
    art = Path(cfg["paths"]["artifacts"])

    X = pd.read_pickle(art / "X_all.pkl")
    y = np.load(art / "y_all.npy")
    idx_train = np.load(art / "idx_train.npy")
    idx_val   = np.load(art / "idx_val.npy")
    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_val,   y_val   = X.iloc[idx_val],   y[idx_val]

    n_pos = int(y_train.sum()); n_neg = int(len(y_train)-n_pos)
    spw_base = n_neg / n_pos
    print(f"spw base = {spw_base:.3f}")

    def objective(trial):
        params = {
            "objective":"binary:logistic","eval_metric":"aucpr","tree_method":"hist",
            "random_state":random_state,"n_jobs":-1,
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
        m = xgb.XGBClassifier(**params,
                              n_estimators=cfg["xgb_tune"]["n_estimators_max"],
                              early_stopping_rounds=cfg["xgb_tune"]["early_stopping_rounds"])
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = m.predict_proba(X_val)[:,1]
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        return average_precision_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_fraud")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nLanzando {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✔ Mejor trial #{study.best_trial.number}")
    print(f"  AUC-PR val: {study.best_value:.4f}")
    print(f"  best_iteration: {study.best_trial.user_attrs['best_iteration']}")
    for k,v in study.best_params.items(): print(f"    {k}: {v}")

    # Reentrenar modelo final
    best = dict(study.best_params)
    best_n_est = study.best_trial.user_attrs["best_iteration"]+1
    spw_final = spw_base * best.pop("spw_mult")
    final_params = {"objective":"binary:logistic","eval_metric":"aucpr","tree_method":"hist",
                    "random_state":random_state,"n_jobs":-1,
                    "scale_pos_weight":spw_final, **best}
    model = xgb.XGBClassifier(**final_params, n_estimators=best_n_est)
    model.fit(X_train, y_train, verbose=False)

    proba_val = model.predict_proba(X_val)[:,1]
    auc_pr, auc_roc = average_precision_score(y_val,proba_val), roc_auc_score(y_val,proba_val)
    pred_05 = (proba_val>=0.5).astype(int)
    cm_05 = confusion_matrix(y_val, pred_05)
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1_arr = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    best_ix = int(np.argmax(f1_arr))

    print(f"\n--- Tuned val ---")
    print(f"AUC-PR: {auc_pr:.4f}   AUC-ROC: {auc_roc:.4f}")
    print(f"thr=0.5:  CM:\n{cm_05}")
    print(f"  P={precision_score(y_val,pred_05):.4f} R={recall_score(y_val,pred_05):.4f} F1={f1_score(y_val,pred_05):.4f}")
    print(f"bestF1 ({thr[best_ix]:.4f}): P={prec[best_ix]:.4f} R={rec[best_ix]:.4f} F1={f1_arr[best_ix]:.4f}")

    with open(art/"optuna_study.pkl","wb") as f: pickle.dump(study, f)
    with open(art/"optuna_best_params.json","w") as f:
        json.dump({**final_params, "n_estimators":best_n_est}, f, indent=2)
    study.trials_dataframe().to_csv(art/"optuna_trials.csv", index=False)
    model.save_model(art/"model_tuned.json")

    with open(art/"metrics_tuned_val.json","w") as f:
        json.dump({
            "model":"xgboost_tuned_optuna","n_trials":n_trials,
            "best_trial_number":study.best_trial.number,
            "final_params":{**final_params,"n_estimators":best_n_est},
            "val":{
                "auc_pr":float(auc_pr),"auc_roc":float(auc_roc),
                "threshold_0.5":{"precision":float(precision_score(y_val,pred_05,zero_division=0)),
                                 "recall":float(recall_score(y_val,pred_05,zero_division=0)),
                                 "f1":float(f1_score(y_val,pred_05,zero_division=0)),
                                 "confusion_matrix":cm_05.tolist()},
                "threshold_bestF1":{"threshold":float(thr[best_ix]),
                                    "precision":float(prec[best_ix]),
                                    "recall":float(rec[best_ix]),
                                    "f1":float(f1_arr[best_ix])}
            }
        }, f, indent=2)

    print(f"\n✔ Guardado: model_tuned.json, optuna_study.pkl, optuna_best_params.json, optuna_trials.csv, metrics_tuned_val.json")
