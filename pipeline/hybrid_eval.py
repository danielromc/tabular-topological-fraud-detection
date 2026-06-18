"""pipeline/hybrid_eval.py — Fase 3.4: evaluación del híbrido (LogReg meta) en TEST."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from ._thresholds import best_fbeta_thr


def run(cfg):
    top_n = cfg["eval"]["top_n_investigation"]
    op_thr = cfg["eval"]["operational_threshold"]
    beta = float(cfg["eval"].get("fbeta", 1.0))
    art = Path(cfg["paths"]["artifacts"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])

    X = pd.read_pickle(art_hybrid/"X_hybrid.pkl")
    y = np.load(art/"y_all.npy")
    idx_tr = np.load(art/"idx_train.npy"); idx_v = np.load(art/"idx_val.npy"); idx_te = np.load(art/"idx_test.npy")
    linked_mask = np.load(art/"linked_mask.npy")

    # Score-stacking: meta-modelo es shallow XGB sobre [score_xgb, score_gnn]
    model = joblib.load(art_hybrid/"model_hybrid_tuned.pkl")
    fi_meta = dict(zip(X.columns, [float(v) for v in model.feature_importances_]))
    print(f"Meta cargado. Features: {list(X.columns)}")
    print(f"  Feature importances: {fi_meta}")

    Xv, yv   = X.iloc[idx_v], y[idx_v]
    Xte, yte = X.iloc[idx_te], y[idx_te]

    proba_val = model.predict_proba(Xv)[:,1]
    thr_bestF = best_fbeta_thr(yv, proba_val, beta=beta)
    print(f"Threshold best-F{beta:g} val: {thr_bestF:.4f}")

    proba_test = model.predict_proba(Xte)[:,1]
    auc_pr = average_precision_score(yte, proba_test)
    auc_roc = roc_auc_score(yte, proba_test)
    print(f"\n=== TEST: AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f} ===")

    results = {}
    for label, thr in [(f"thr_{op_thr}",op_thr),(f"thr_bestF{beta:g}_val",thr_bestF)]:
        pred = (proba_test>=thr).astype(int); cm = confusion_matrix(yte,pred)
        results[label] = {"threshold":float(thr),
                          "precision":float(precision_score(yte,pred,zero_division=0)),
                          "recall":float(recall_score(yte,pred,zero_division=0)),
                          "f1":float(f1_score(yte,pred,zero_division=0)),
                          "confusion_matrix":cm.tolist(),"n_pos_pred":int(pred.sum())}
        print(f"\n--- {label} (thr={thr:.4f}) ---")
        print(f"CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        print(f"P={results[label]['precision']:.4f} R={results[label]['recall']:.4f} F1={results[label]['f1']:.4f}")

    # Subtipos
    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    test_info = claims.iloc[idx_te][["Fraud_type","Ring_ID","Event_ID","is_fraud"]].reset_index(drop=True)
    subtype_rows = []
    for label, thr_val in [("thr_op",op_thr),(f"thr_bestF{beta:g}",thr_bestF)]:
        pred = (proba_test>=thr_val).astype(int)
        for ft in ["organized","opportunistic"]:
            m = (test_info.Fraud_type==ft).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":f"fraud_type={ft}","n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
        for ring in ["A","B","C","lobo_1","lobo_2","lobo_3"]:
            m = (test_info.Ring_ID==ring).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":f"ring={ring}","n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
        for prefix,lab in [("siniestro_opp_inj","opp_inj"),
                            ("siniestro_opp_fp","opp_fp"),
                            ("siniestro_opp_lp","opp_lp")]:
            m = test_info.Event_ID.astype(str).str.startswith(prefix).to_numpy(); n = int(m.sum())
            if n: subtype_rows.append({"threshold":label,"grupo":lab,"n":n,
                                        "cazados":int(pred[m].sum()),"recall":round(pred[m].sum()/n,4)})
    pd.DataFrame(subtype_rows).to_csv(art_hybrid/"subtype_recall_hybrid_test.csv", index=False)
    print(f"\n--- Recall por subtipo ---\n{pd.DataFrame(subtype_rows).to_string(index=False)}")

    # Linked — batch predict + sets
    linked_idx = np.where(linked_mask)[0]
    set_train = set(idx_tr.tolist())
    set_val   = set(idx_v.tolist())
    set_test  = set(idx_te.tolist())
    proba_linked = model.predict_proba(X.iloc[linked_idx])[:, 1]
    linked_rows = []
    for k, gi in enumerate(linked_idx):
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
    pd.DataFrame(linked_rows).to_csv(art_hybrid/"linked_eval_hybrid.csv", index=False)
    print(f"\n--- Linked ---\n{pd.DataFrame(linked_rows).to_string(index=False)}")

    # Top-N
    test_df = claims.iloc[idx_te].reset_index(drop=True).copy()
    test_df["proba_fraude"] = proba_test; test_df["is_fraud_real"] = yte.astype(bool)
    cols = ["proba_fraude","is_fraud_real","claim_id","ID","Current_Year","Claims_type",
            "Cost_claims_year","Premium","N_claims_history","Driver_Age","Vehicle_Age",
            "Type_risk","Area","Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID",
            "Event_ID","Fraud_type","Ring_ID"]
    top_test = test_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols].reset_index(drop=True)
    top_test.insert(0,"rank",range(1,len(top_test)+1))
    top_test.to_csv(art_hybrid/f"top{top_n}_hybrid_test.csv", index=False)
    n_hits = int(top_test["is_fraud_real"].sum())
    print(f"\n=== TOP-{top_n} TEST: {n_hits}/{top_n}  Precision@{top_n}={n_hits/top_n:.3f} ===")

    # Top-N operativo
    op_df = claims.copy(); op_df["proba_fraude"] = model.predict_proba(X)[:,1]
    split = np.full(len(op_df),"test",dtype=object)
    split[idx_tr]="train"; split[idx_v]="val"
    op_df["_split"]=split
    cols_op = ["proba_fraude","claim_id","ID","Current_Year","Claims_type","Cost_claims_year",
               "Premium","N_claims_history","Driver_Age","Vehicle_Age","Type_risk","Area",
               "Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID","_split"]
    top_op = op_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols_op].reset_index(drop=True)
    top_op.insert(0,"rank",range(1,len(top_op)+1))
    top_op.to_csv(art_hybrid/f"top{top_n}_hybrid_operativo.csv", index=False)

    # Importancias del meta (shallow XGB)
    total = sum(fi_meta.values()) + 1e-12
    fi = pd.DataFrame({
        "feature": list(fi_meta.keys()),
        "importancia": list(fi_meta.values()),
        "pct": [v / total for v in fi_meta.values()],
    }).sort_values("importancia", ascending=False).reset_index(drop=True)
    fi.to_csv(art_hybrid/"feature_importance_hybrid.csv", index=False)
    print(f"\n--- Importancias del meta (shallow XGB) ---")
    for _, r in fi.iterrows():
        print(f"  {r['feature']}: gain={r['importancia']:.4f}  "
              f"({r['pct']*100:.1f}% del peso)")

    # Plots
    prec_t, rec_t, _ = precision_recall_curve(yte, proba_test)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rec_t, prec_t, lw=2, label=f"Híbrido (AUC-PR={auc_pr:.3f})")
    ax.axhline(yte.mean(), color="grey", ls="--", lw=1, label=f"No-skill ({yte.mean():.3f})")
    for label, r in results.items():
        m = "o" if "bestF1" not in label else "s"
        ax.plot(r["recall"], r["precision"], m, ms=10,
                label=f"{label}: P={r['precision']:.2f} R={r['recall']:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Curva PR — Híbrido (TEST)")
    ax.legend(loc="lower left",fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(art_hybrid/"pr_curve_hybrid_test.png", dpi=120); plt.close(fig)

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
        ax.set_title(f"{label} thr={r['threshold']:.3f}\nP={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    fig.tight_layout(); fig.savefig(art_hybrid/"confusion_hybrid_test.png", dpi=120); plt.close(fig)

    metrics = {"model":"xgboost_shallow_meta_score_stacking",
               "stacking_type":"score_level",
               "n_features":int(X.shape[1]),
               "fbeta_used": beta,
               "test":{"n":int(len(yte)),"n_fraud":int(yte.sum()),
                       "auc_pr":float(auc_pr),"auc_roc":float(auc_roc),
                       "no_skill_aucpr":float(yte.mean()),
                       **results,
                       "precision_at_N":n_hits/top_n,"n_hits_topN":int(n_hits)},
               "val_threshold_bestF_used":float(thr_bestF),
               "meta_feature_importances": fi_meta}
    with open(art_hybrid/"metrics_hybrid_test.json","w") as f: json.dump(metrics,f,indent=2)
    print(f"\n[OK] Guardado en {art_hybrid}")
