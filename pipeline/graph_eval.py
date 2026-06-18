"""pipeline/graph_eval.py — Fase 2.4: evaluación GraphSAGE en TEST."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from .graph_train import HeteroSAGE
from ._thresholds import best_fbeta_thr


def run(cfg):
    seed = cfg["seed"]
    top_n = cfg["eval"]["top_n_investigation"]
    op_thr = cfg["eval"]["operational_threshold"]
    beta = float(cfg["eval"].get("fbeta", 1.0))
    art = Path(cfg["paths"]["artifacts"])
    art_graph = Path(cfg["paths"]["artifacts_graph"])
    torch.manual_seed(seed)

    # weights_only=False necesario para HeteroData (objeto pyg, no dict puro).
    # El checkpoint del modelo sí se carga con weights_only=True (S1).
    data = torch.load(art_graph / "graph_data.pt", weights_only=False)
    # Features already scaled in graph_build.py (no rescaling needed)

    ckpt = torch.load(art_graph/"model_graphsage_baseline.pt", weights_only=True)
    model = HeteroSAGE(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                       ckpt["dropout"], ckpt.get("aggregation","mean"))
    with torch.no_grad(): _ = model(data.x_dict, data.edge_index_dict)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        proba_all = torch.sigmoid(logits).cpu().numpy()

    y_all = data["claim"].y.cpu().numpy()
    test_mask = data["claim"].test_mask.cpu().numpy()
    val_mask  = data["claim"].val_mask.cpu().numpy()
    linked_mask = data["claim"].linked_mask.cpu().numpy()
    idx_test = np.where(test_mask)[0]
    idx_val  = np.where(val_mask)[0]
    idx_train = np.load(art/"idx_train.npy")

    y_test = y_all[idx_test]; proba_test = proba_all[idx_test]
    auc_pr = average_precision_score(y_test, proba_test)
    auc_roc = roc_auc_score(y_test, proba_test)
    print(f"=== TEST: AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f} ===")

    np.save(art_graph / "probas_test_gnn.npy", proba_test)
    print(f"\n[OK] Guardado: {art_graph / 'probas_test_gnn.npy'}")

    y_val = y_all[idx_val]; proba_val = proba_all[idx_val]
    thr_bestF = best_fbeta_thr(y_val, proba_val, beta=beta)
    print(f"Threshold best-F{beta:g} val: {thr_bestF:.4f}")

    def metrics_at(thr):
        pred = (proba_test>=thr).astype(int); cm = confusion_matrix(y_test,pred)
        return {"threshold":float(thr),
                "precision":float(precision_score(y_test,pred,zero_division=0)),
                "recall":float(recall_score(y_test,pred,zero_division=0)),
                "f1":float(f1_score(y_test,pred,zero_division=0)),
                "n_pos_pred":int(pred.sum()),"confusion_matrix":cm.tolist()}

    op_alta = metrics_at(op_thr); op_bestF = metrics_at(thr_bestF)
    for label,r in [(f"Alta captura thr={op_thr}",op_alta),
                    (f"Best-F{beta:g} val thr={thr_bestF:.4f}",op_bestF)]:
        cm = r["confusion_matrix"]
        print(f"\n--- {label} ---")
        print(f"CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        print(f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    # Subtipos
    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    test_info = claims.iloc[idx_test][["Fraud_type","Ring_ID","Event_ID","is_fraud"]].reset_index(drop=True)
    subtype_rows=[]
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
    pd.DataFrame(subtype_rows).to_csv(art_graph/"subtype_recall_graph_test.csv", index=False)
    print(f"\n--- Recall por subtipo ---\n{pd.DataFrame(subtype_rows).to_string(index=False)}")

    # Linked — sets para lookup O(1)
    linked_idx = np.where(linked_mask)[0]
    set_train = set(idx_train.tolist())
    set_val   = set(idx_val.tolist())
    set_test  = set(idx_test.tolist())
    linked_rows = []
    for gi in linked_idx:
        gi_int = int(gi)
        if gi_int in set_train: sp = "train"
        elif gi_int in set_val: sp = "val"
        elif gi_int in set_test: sp = "test"
        else: sp = "?"
        linked_rows.append({"global_idx":gi_int,"split":sp,"is_fraud":bool(y_all[gi]),
                            "Event_ID":claims.iloc[gi]["Event_ID"],
                            "Claims_type":claims.iloc[gi]["Claims_type"],
                            "Cost_claims_year":float(claims.iloc[gi]["Cost_claims_year"]),
                            "proba_fraud":float(proba_all[gi])})
    linked_df = pd.DataFrame(linked_rows)
    linked_df.to_csv(art_graph/"linked_eval_graph.csv", index=False)
    print(f"\n--- Linked ---\n{linked_df.to_string(index=False)}")

    # Top-N
    test_df = claims.iloc[idx_test].reset_index(drop=True).copy()
    test_df["proba_fraude"] = proba_test
    test_df["is_fraud_real"] = y_test.astype(bool)
    cols = ["proba_fraude","is_fraud_real","claim_id","ID","Current_Year","Claims_type",
            "Cost_claims_year","Premium","N_claims_history","Driver_Age","Vehicle_Age",
            "Type_risk","Area","Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID",
            "Event_ID","Fraud_type","Ring_ID"]
    top_test = test_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols].reset_index(drop=True)
    top_test.insert(0,"rank",range(1,len(top_test)+1))
    top_test.to_csv(art_graph/f"top{top_n}_graph_test.csv", index=False)
    n_hits = int(top_test["is_fraud_real"].sum())
    print(f"\n=== TOP-{top_n} TEST: {n_hits}/{top_n}  Precision@{top_n}={n_hits/top_n:.3f} ===")

    # Top-N operativo
    op_df = claims.copy(); op_df["proba_fraude"] = proba_all
    split = np.full(len(op_df),"test",dtype=object)
    split[idx_train]="train"; split[idx_val]="val"
    op_df["_split"]=split
    cols_op = ["proba_fraude","claim_id","ID","Current_Year","Claims_type","Cost_claims_year",
               "Premium","N_claims_history","Driver_Age","Vehicle_Age","Type_risk","Area",
               "Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID","_split"]
    top_op = op_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols_op].reset_index(drop=True)
    top_op.insert(0,"rank",range(1,len(top_op)+1))
    top_op.to_csv(art_graph/f"top{top_n}_graph_operativo.csv", index=False)

    # Plots
    prec_t, rec_t, _ = precision_recall_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rec_t, prec_t, linewidth=2, label=f"GraphSAGE (AUC-PR={auc_pr:.3f})")
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"No-skill ({y_test.mean():.3f})")
    ax.plot(op_alta["recall"], op_alta["precision"], "o", ms=10,
            label=f"thr={op_thr}: P={op_alta['precision']:.2f} R={op_alta['recall']:.2f}")
    ax.plot(op_bestF["recall"], op_bestF["precision"], "s", ms=10,
            label=f"best-F{beta:g}: P={op_bestF['precision']:.2f} R={op_bestF['recall']:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curva PR — GraphSAGE (TEST)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1); fig.tight_layout()
    fig.savefig(art_graph/"pr_curve_graphsage_test.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,(label,r) in zip(axes,[("thr_op",op_alta),(f"best-F{beta:g}",op_bestF)]):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=14,
                        color="white" if cm[i,j]>cm.max()/2 else "black")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(f"{label} thr={r['threshold']:.3f}\nP={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    fig.tight_layout(); fig.savefig(art_graph/"confusion_graphsage_test.png", dpi=120); plt.close(fig)

    metrics = {"model":"graphsage_baseline_hetero",
               "fbeta_used": beta,
               "test":{"n":int(len(y_test)),"n_fraud":int(y_test.sum()),
                       "auc_pr":float(auc_pr),"auc_roc":float(auc_roc),
                       "no_skill_aucpr":float(y_test.mean()),
                       "op_alta_captura":op_alta,"op_best_fbeta_val":op_bestF,
                       "precision_at_N":n_hits/top_n,"n_hits_topN":int(n_hits)},
               "threshold_bestF_from_val":float(thr_bestF)}
    with open(art_graph/"metrics_graphsage_test.json","w") as f: json.dump(metrics,f,indent=2)
    print(f"\n[OK] Guardado en {art_graph}")
