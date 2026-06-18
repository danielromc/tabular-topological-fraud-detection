"""pipeline/xgb_baseline.py — Fase 1.2: XGBoost con defaults + scale_pos_weight."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def run(cfg):
    random_state = cfg["seed"]
    art = Path(cfg["paths"]["artifacts"])

    X = pd.read_pickle(art / "X_all.pkl")
    y = np.load(art / "y_all.npy")
    idx_train = np.load(art / "idx_train.npy")
    idx_val   = np.load(art / "idx_val.npy")

    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_val,   y_val   = X.iloc[idx_val],   y[idx_val]
    print(f"Train: {X_train.shape}, fraudes={y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"Val:   {X_val.shape}, fraudes={y_val.sum()} ({y_val.mean()*100:.3f}%)")

    n_pos = int(y_train.sum()); n_neg = int(len(y_train) - n_pos)
    spw = n_neg / n_pos
    print(f"\nscale_pos_weight = {n_neg}/{n_pos} = {spw:.3f}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        n_estimators=cfg["xgb_baseline"]["n_estimators"],
        early_stopping_rounds=cfg["xgb_baseline"]["early_stopping_rounds"],
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"\nMejor iteración: {model.best_iteration}  |  best aucpr val: {model.best_score:.4f}")

    proba_val = model.predict_proba(X_val)[:, 1]
    auc_pr  = average_precision_score(y_val, proba_val)
    auc_roc = roc_auc_score(y_val, proba_val)
    print(f"\nAUC-PR val: {auc_pr:.4f}  |  AUC-ROC val: {auc_roc:.4f}")

    pred_05 = (proba_val >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_val, pred_05)
    print(f"\n--- Threshold 0.5 ---\nCM:\n{cm_05}")
    print(f"P={precision_score(y_val, pred_05):.4f}  R={recall_score(y_val, pred_05):.4f}  F1={f1_score(y_val, pred_05):.4f}")

    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1_arr = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1]+1e-12)
    best_ix = int(np.argmax(f1_arr))
    thr_bestF1 = float(thr[best_ix])

    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba_val >= t).astype(int)
        if pred.sum() == 0: continue
        rows.append({"threshold": round(float(t),3), "n_pos_pred": int(pred.sum()),
                     "precision": round(precision_score(y_val,pred,zero_division=0),4),
                     "recall":    round(recall_score(y_val,pred,zero_division=0),4),
                     "f1":        round(f1_score(y_val,pred,zero_division=0),4)})
    sweep = pd.DataFrame(rows)
    print(f"\n--- Sweep thresholds val ---\n{sweep.to_string(index=False)}")
    sweep.to_csv(art / "threshold_sweep_baseline_val.csv", index=False)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rec, prec, linewidth=2, label=f"Baseline (AUC-PR={auc_pr:.3f})")
    ax.axhline(y_val.mean(), color="grey", ls="--", lw=1, label=f"No-skill ({y_val.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curva PR — Baseline XGBoost (val)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig(art / "pr_curve_baseline_val.png", dpi=120)
    plt.close(fig)

    model.save_model(art / "model_baseline.json")

    metrics = {
        "model": "xgboost_baseline",
        "n_estimators_used": int(model.best_iteration)+1,
        "scale_pos_weight": spw,
        "val": {
            "auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "threshold_0.5": {
                "precision": float(precision_score(y_val, pred_05, zero_division=0)),
                "recall":    float(recall_score(y_val, pred_05, zero_division=0)),
                "f1":        float(f1_score(y_val, pred_05, zero_division=0)),
                "confusion_matrix": cm_05.tolist(),
            },
            "threshold_bestF1": {
                "threshold": thr_bestF1,
                "precision": float(prec[best_ix]), "recall": float(rec[best_ix]),
                "f1": float(f1_arr[best_ix]),
            },
        },
    }
    with open(art / "metrics_baseline_val.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✔ Guardado: model_baseline.json, metrics_baseline_val.json, pr_curve_baseline_val.png")
