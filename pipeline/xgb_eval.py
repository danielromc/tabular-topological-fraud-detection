"""pipeline/xgb_eval.py — Fase 1.4: evaluación XGBoost en TEST."""
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
from pipeline._thresholds import best_fbeta_thr


def run(cfg):
    art = Path(cfg["paths"]["artifacts"])
    op_thr = cfg["eval"]["operational_threshold"]
    beta = float(cfg["eval"].get("fbeta", 1.0))

    X = pd.read_pickle(art / "X_all.pkl")
    y = np.load(art / "y_all.npy")
    idx_train = np.load(art/"idx_train.npy")
    idx_val   = np.load(art/"idx_val.npy")
    idx_test  = np.load(art/"idx_test.npy")
    linked_mask = np.load(art/"linked_mask.npy")

    model = xgb.XGBClassifier()
    model.load_model(art/"model_tuned.json")

    X_val, y_val   = X.iloc[idx_val],   y[idx_val]
    X_test, y_test = X.iloc[idx_test],  y[idx_test]

    proba_val = model.predict_proba(X_val)[:,1]
    thr_bestF = best_fbeta_thr(y_val, proba_val, beta=beta)
    print(f"Threshold best-F{beta:g} sobre VAL: {thr_bestF:.4f}")

    proba_test = model.predict_proba(X_test)[:,1]
    auc_pr_test  = average_precision_score(y_test, proba_test)
    auc_roc_test = roc_auc_score(y_test, proba_test)
    print(f"\n=== TEST: AUC-PR={auc_pr_test:.4f}  AUC-ROC={auc_roc_test:.4f} ===")
    print(f"(no-skill = {y_test.mean():.4f})")

    results = {}
    for label, thr in [(f"thr_{op_thr}_alta_captura", op_thr),
                        (f"thr_bestF{beta:g}_val", thr_bestF)]:
        pred = (proba_test >= thr).astype(int)
        cm = confusion_matrix(y_test, pred)
        results[label] = {"threshold":float(thr),
                          "precision":float(precision_score(y_test,pred,zero_division=0)),
                          "recall":float(recall_score(y_test,pred,zero_division=0)),
                          "f1":float(f1_score(y_test,pred,zero_division=0)),
                          "confusion_matrix":cm.tolist(),"n_pos_pred":int(pred.sum())}
        print(f"\n--- {label} (thr={thr:.4f}) ---\nCM:\n{cm}")
        print(f"P={results[label]['precision']:.4f} R={results[label]['recall']:.4f} F1={results[label]['f1']:.4f}")

    # Subtipos
    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    test_info = claims.iloc[idx_test][["Fraud_type","Ring_ID","Claims_type","Event_ID","is_fraud"]].reset_index(drop=True)
    subtype_rows = []
    for label, thr_val in [("thr_op", op_thr), (f"thr_bestF{beta:g}", thr_bestF)]:
        pred = (proba_test >= thr_val).astype(int)
        for ft in ["organized","opportunistic"]:
            m = (test_info.Fraud_type==ft).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":f"fraud_type={ft}","n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
        for ring in ["A","B","C","lobo_1","lobo_2","lobo_3"]:
            m = (test_info.Ring_ID==ring).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":f"ring={ring}","n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
        for prefix,lab in [("siniestro_opp_inj","opp_inj"),
                            ("siniestro_opp_fp","opp_fp (provider fraud)"),
                            ("siniestro_opp_lp","opp_lp (provider legit)")]:
            m = test_info.Event_ID.astype(str).str.startswith(prefix).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":lab,"n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
    pd.DataFrame(subtype_rows).to_csv(art/"subtype_recall_test.csv", index=False)
    print(f"\n--- Recall por subtipo ---\n{pd.DataFrame(subtype_rows).to_string(index=False)}")

    # Linked — batch predict + sets para lookup O(1)
    linked_idx_global = np.where(linked_mask)[0]
    set_train = set(idx_train.tolist())
    set_val   = set(idx_val.tolist())
    set_test  = set(idx_test.tolist())
    proba_linked = model.predict_proba(X.iloc[linked_idx_global])[:, 1]
    linked_rows = []
    for k, gi in enumerate(linked_idx_global):
        gi_int = int(gi)
        if gi_int in set_train: sp = "train"
        elif gi_int in set_val: sp = "val"
        elif gi_int in set_test: sp = "test"
        else: sp = "?"
        linked_rows.append({"global_idx":gi_int,"split":sp,"is_fraud":bool(y[gi]),
                            "Event_ID":claims.iloc[gi]["Event_ID"],
                            "Claims_type":claims.iloc[gi]["Claims_type"],
                            "Cost_claims_year":float(claims.iloc[gi]["Cost_claims_year"]),
                            "proba_fraud":float(proba_linked[k])})
    linked_df = pd.DataFrame(linked_rows)
    linked_df.to_csv(art/"linked_eval.csv", index=False)
    print(f"\n--- Linked ---\n{linked_df.to_string(index=False)}")

    # Feature importance
    fi = pd.DataFrame({"feature":X.columns,"gain":model.feature_importances_}).sort_values("gain",ascending=False).reset_index(drop=True)
    fi.to_csv(art/"feature_importance.csv", index=False)
    print(f"\n--- Top 15 features ---\n{fi.head(15).to_string(index=False)}")

    # Plots
    fig, ax = plt.subplots(figsize=(6,5))
    prec_t, rec_t, _ = precision_recall_curve(y_test, proba_test)
    ax.plot(rec_t, prec_t, linewidth=2, label=f"Test (AUC-PR={auc_pr_test:.3f})")
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"No-skill ({y_test.mean():.3f})")
    for label, r in results.items():
        ax.plot(r["recall"], r["precision"], "o", markersize=10,
                label=f"{label}: P={r['precision']:.2f} R={r['recall']:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curva PR — Modelo tuneado (TEST)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1); fig.tight_layout()
    fig.savefig(art/"pr_curve_tuned_test.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,(label,r) in zip(axes, results.items()):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=14,
                        color="white" if cm[i,j]>cm.max()/2 else "black")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(f"{label}\nthr={r['threshold']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f}")
    fig.tight_layout(); fig.savefig(art/"confusion_tuned_test.png", dpi=120); plt.close(fig)

    # Final metrics
    op_alta = {k:v for k,v in results.items() if "alta_captura" in k}
    op_bestF = {k:v for k,v in results.items() if "bestF" in k}
    metrics = {
        "model":"xgboost_tuned_clean_42feats",
        "notes":"Baseline tabular honesto. Features relacionales descartadas por contaminacion.",
        "fbeta_used": beta,
        "test":{
            "n":int(len(y_test)),"n_fraud":int(y_test.sum()),
            "auc_pr":float(auc_pr_test),"auc_roc":float(auc_roc_test),
            "no_skill_aucpr":float(y_test.mean()),
            "op_alta_captura":list(op_alta.values())[0],
            "op_best_fbeta":list(op_bestF.values())[0],
        },
        "val_threshold_bestF_used":float(thr_bestF),
    }
    with open(art/"metrics_tuned_test.json","w") as f: json.dump(metrics, f, indent=2)
    print(f"\n✔ Guardado en {art}")
