"""pipeline/comparison.py — Comparación final XGB vs GNN vs Híbrido (+Ensemble)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
import joblib
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from matplotlib.patches import Patch
from .graph_train import HeteroSAGE
from ._thresholds import best_fbeta_thr


def run(cfg):
    seed = cfg["seed"]
    top_n = cfg["eval"]["top_n_investigation"]
    beta = float(cfg["eval"].get("fbeta", 1.0))
    art        = Path(cfg["paths"]["artifacts"])
    art_graph  = Path(cfg["paths"]["artifacts_graph"])
    art_hybrid = Path(cfg["paths"]["artifacts_hybrid"])
    art_cmp    = Path(cfg["paths"]["artifacts_cmp"])
    art_cmp.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(seed)

    claims = pd.read_csv(cfg["paths"]["claims_csv"])
    y_all = np.load(art/"y_all.npy")
    idx_tr = np.load(art/"idx_train.npy"); idx_v = np.load(art/"idx_val.npy"); idx_te = np.load(art/"idx_test.npy")
    linked_mask = np.load(art/"linked_mask.npy")
    y_test = y_all[idx_te]; y_val = y_all[idx_v]

    # XGBoost — predict una sola vez sobre todo el dataset y luego indexa (P4)
    X_tab = pd.read_pickle(art/"X_all.pkl")
    m_xgb = xgb.XGBClassifier(); m_xgb.load_model(art/"model_tuned.json")
    proba_xgb_all  = m_xgb.predict_proba(X_tab)[:,1]
    proba_xgb_test = proba_xgb_all[idx_te]
    proba_xgb_val  = proba_xgb_all[idx_v]
    print("✔ XGBoost cargado")

    # GraphSAGE — weights_only=True solo en checkpoint (HeteroData necesita False)
    data = torch.load(art_graph/"graph_data.pt", weights_only=False)
    # Features already scaled in graph_build.py (no rescaling needed)

    ckpt = torch.load(art_graph/"model_graphsage_baseline.pt", weights_only=True)
    m_gnn = HeteroSAGE(ckpt["in_dims"], ckpt["hidden_dim"], ckpt["edge_types"],
                       ckpt["dropout"], ckpt.get("aggregation","mean"))
    with torch.no_grad():
        _ = m_gnn(data.x_dict, data.edge_index_dict)
    m_gnn.load_state_dict(ckpt["state_dict"])
    m_gnn.eval()
    with torch.no_grad():
        logits = m_gnn(data.x_dict, data.edge_index_dict)
        proba_gnn_all = torch.sigmoid(logits).cpu().numpy()
    proba_gnn_test = proba_gnn_all[idx_te]
    proba_gnn_val = proba_gnn_all[idx_v]
    print("✔ GraphSAGE cargado")

    # Híbrido — meta LogReg sobre [score_xgb, score_gnn] (score-stacking)
    X_hyb = pd.read_pickle(art_hybrid/"X_hybrid.pkl")
    m_hyb = joblib.load(art_hybrid/"model_hybrid_tuned.pkl")
    proba_hyb_all  = m_hyb.predict_proba(X_hyb)[:,1]
    proba_hyb_val  = proba_hyb_all[idx_v]
    proba_hyb_test = proba_hyb_all[idx_te]
    print(f"✔ Híbrido cargado (shallow XGB meta sobre {list(X_hyb.columns)})")

    thr_xgb = best_fbeta_thr(y_val, proba_xgb_val, beta=beta)
    thr_gnn = best_fbeta_thr(y_val, proba_gnn_val, beta=beta)
    thr_hyb = best_fbeta_thr(y_val, proba_hyb_val, beta=beta)
    print(f"\nUmbrales best-F{beta:g} val: XGBoost={thr_xgb:.4f}  GNN={thr_gnn:.4f}  Híbrido={thr_hyb:.4f}")

    models = [
        ("XGBoost", proba_xgb_test, thr_xgb),
        ("GraphSAGE", proba_gnn_test, thr_gnn),
        ("XGB+GNN", proba_hyb_test, thr_hyb),
    ]

    def recall_at_k(proba, k):
        top_ix = np.argsort(proba)[::-1][:k]
        hits = int(y_test[top_ix].sum())
        total = int(y_test.sum())
        return round(hits / total, 4), hits, total

    # Tabla maestra
    rows = []
    for name, proba, thr in models:
        pred = (proba>=thr).astype(int); cm = confusion_matrix(y_test,pred)
        top_ix = np.argsort(proba)[::-1][:top_n]
        hits = int(y_test[top_ix].sum())
        r50, h50, total_fraud = recall_at_k(proba, 50)
        r100, h100, _ = recall_at_k(proba, 100)
        rows.append({"Modelo":name,"AUC-PR":round(average_precision_score(y_test,proba),4),
                     "AUC-ROC":round(roc_auc_score(y_test,proba),4),
                     "Threshold":round(thr,4),
                     "Precision":round(precision_score(y_test,pred,zero_division=0),4),
                     "Recall":round(recall_score(y_test,pred,zero_division=0),4),
                     "F1":round(f1_score(y_test,pred,zero_division=0),4),
                     "TN":int(cm[0][0]),"FP":int(cm[0][1]),
                     "FN":int(cm[1][0]),"TP":int(cm[1][1]),
                     f"P@{top_n}":round(hits/top_n,4),
                     "Hits@30":f"{hits}/{top_n}",
                     "Recall@50":r50,
                     "Hits@50":f"{h50}/{total_fraud}",
                     "Recall@100":r100,
                     "Hits@100":f"{h100}/{total_fraud}"})
    master = pd.DataFrame(rows)
    master.to_csv(art_cmp/"master_metrics_table.csv", index=False)
    try:
        md = master.to_markdown(index=False)
    except ImportError:
        md = master.to_string(index=False)
    with open(art_cmp/"master_metrics_table.md","w",encoding="utf-8") as f:
        f.write("# Tabla maestra de métricas en TEST\n\n" + md)
    print("\n=== TABLA MAESTRA (TEST) ===")
    print(master.to_string(index=False))

    # Curvas PR
    plt.rcParams.update({"font.size":11})
    colors = {"XGBoost":"#1f77b4","GraphSAGE":"#0C855C","XGB+GNN":"#f54927"}
    fig, ax = plt.subplots(figsize=(7,5.5))
    for name, proba, thr in models:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc = average_precision_score(y_test, proba)
        ax.plot(rec, prec, lw=2.2, label=f"{name} (AUC-PR={auc:.3f})", color=colors[name])
        pred = (proba>=thr).astype(int)
        p = precision_score(y_test,pred,zero_division=0); r = recall_score(y_test,pred,zero_division=0)
        ax.plot(r, p, "o", ms=9, color=colors[name], markeredgecolor="black", markeredgewidth=0.8)
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"Clasificador aleatorio ({y_test.mean():.3f})")
    ax.set_xlabel("Recall",fontsize=12); ax.set_ylabel("Precision",fontsize=12)
    ax.set_title("Curvas Precision-Recall",fontsize=12)
    ax.legend(loc="upper right",fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); fig.tight_layout()
    fig.savefig(art_cmp/"fig_pr_curves_unified.pdf", dpi=600, bbox_inches="tight"); plt.close(fig)
    print("✔ fig_pr_curves_unified.pdf")

    # Matrices de confusión (un panel por modelo)
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models)+2, 4.5))
    for ax,(name,proba,thr) in zip(axes,models):
        pred = (proba>=thr).astype(int); cm = confusion_matrix(y_test,pred)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=15,fontweight="bold",
                        color="white" if cm[i,j]>cm.max()/2 else "black")
        p = precision_score(y_test,pred,zero_division=0); r = recall_score(y_test,pred,zero_division=0)
        f1 = f1_score(y_test,pred,zero_division=0)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["No fraude","Fraude"]); ax.set_yticklabels(["No fraude","Fraude"])
        ax.set_xlabel("Predicción",fontsize=11); ax.set_ylabel("Realidad",fontsize=11)
        ax.set_title(f"{name}\nthr={thr:.3f}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}",fontsize=11)
    fig.suptitle("Matrices de confusión en el conjunto de prueba",fontsize=13,y=1.02)
    fig.tight_layout(); fig.savefig(art_cmp/"fig_confusion_matrices_unified.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_confusion_matrices_unified.pdf")

    # ===== FEATURE IMPORTANCE =====
    # (a) XGBoost baseline: gain normalizado
    fi_xgb = pd.DataFrame({"feature": X_tab.columns,
                            "importancia": m_xgb.feature_importances_}) \
                .sort_values("importancia", ascending=False).head(15)

    # (b) Híbrido: feature_importances_ del meta (shallow XGB sobre 2 scores)
    fi_hyb_full = pd.DataFrame({
        "feature": list(X_hyb.columns),
        "importancia": [float(v) for v in m_hyb.feature_importances_],
    })
    fi_hyb_full["tipo"] = fi_hyb_full["feature"].apply(
        lambda c: "score_gnn" if c == "score_gnn" else "score_xgb")
    fi_hyb = fi_hyb_full.sort_values("importancia", ascending=False).head(15)

    # (c) GNN: gradient saliency sobre features del nodo-claim
    m_gnn.eval()
    x_claim_req = data["claim"].x.clone().detach().requires_grad_(True)
    x_dict_sal = {"claim": x_claim_req, "provider": data["provider"].x}
    logits_s = m_gnn(x_dict_sal, data.edge_index_dict)
    proba_s  = torch.sigmoid(logits_s)
    y_torch = data["claim"].y
    mask_test_fraud = torch.zeros(len(y_torch), dtype=torch.bool)
    mask_test_fraud[idx_te] = True
    mask_test_fraud = mask_test_fraud & (y_torch == 1)
    loss_saliency = proba_s[mask_test_fraud].sum()
    loss_saliency.backward()
    grads = x_claim_req.grad.detach().abs()
    saliency_per_feat = grads[mask_test_fraud].mean(dim=0).cpu().numpy()
    fi_gnn = pd.DataFrame({
        "feature": X_tab.columns,
        "importancia": saliency_per_feat,
    }).sort_values("importancia", ascending=False).head(15)

    # Guardamos las 3 tablas
    fi_xgb.to_csv(art_cmp / "feature_importance_xgb.csv", index=False)
    fi_hyb_full.sort_values("importancia", ascending=False).to_csv(
        art_cmp / "feature_importance_hybrid_full.csv", index=False)
    fi_gnn.to_csv(art_cmp / "feature_importance_gnn_saliency.csv", index=False)

    # Figura feature importance unificada
    COLORS = {"XGBoost": "#1f77b4",
              "GraphSAGE":        "#0C855C",
              "XGB+GNN":  "#f54927"}
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    panels = [
        (axes[0], fi_xgb, "XGBoost\n(gain relativo)",
         COLORS["XGBoost"], None),
        (axes[1], fi_gnn, "GraphSAGE\n(gradient saliency)",
         COLORS["GraphSAGE"], None),
        (axes[2], fi_hyb, "XGB+GNN\n(gain relativo)",
         COLORS["XGB+GNN"], fi_hyb["tipo"] if "tipo" in fi_hyb.columns else None),
    ]
    for ax, dfp, title, color, tipo_col in panels:
        dfp_sorted = dfp.iloc[::-1].reset_index(drop=True)
        if tipo_col is not None:
            tipos = dfp_sorted["tipo"].to_numpy()
            bars = ax.barh(range(len(dfp_sorted)), dfp_sorted["importancia"],
                           color="#f54927", edgecolor="black", linewidth=0.8)
        else:
            ax.barh(range(len(dfp_sorted)), dfp_sorted["importancia"],
                    color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(dfp_sorted)))
        ax.set_yticklabels(dfp_sorted["feature"], fontsize=9)
        ax.set_xlabel("Importancia relativa", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    
    fig.suptitle("Importancia de variables — Top 15 por modelo",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(art_cmp / "fig_feature_importance_unified.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_feature_importance_unified.pdf")

    # ===== CURVAS ROC UNIFICADAS =====
    fig, ax = plt.subplots(figsize=(7, 5.5))
    models_plot = [
        ("XGBoost", proba_xgb_test, COLORS["XGBoost"]),
        ("GraphSAGE", proba_gnn_test, COLORS["GraphSAGE"]),
        ("XGB+GNN", proba_hyb_test, COLORS["XGB+GNN"]),
    ]
    for name, proba, color in models_plot:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio", alpha=0.5)
    ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=12)
    ax.set_title("Curvas ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(art_cmp / "fig_roc_curves_unified.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_roc_curves_unified.pdf")

    # Subtype recall
    test_info = claims.iloc[idx_te][["Fraud_type","Ring_ID","Event_ID","is_fraud"]].reset_index(drop=True)
    grupos = [
        ("Anillo A",(test_info.Ring_ID=="A").to_numpy()),
        ("Anillo B",(test_info.Ring_ID=="B").to_numpy()),
        ("Anillo C",(test_info.Ring_ID=="C").to_numpy()),
        ("Aislados",test_info.Ring_ID.isin(["lobo_1","lobo_2","lobo_3"]).to_numpy()),
        ("Opp_pf",test_info.Event_ID.astype(str).str.startswith("siniestro_opp_fp").to_numpy()),
        ("Opp_pl",test_info.Event_ID.astype(str).str.startswith("siniestro_opp_lp").to_numpy()),
    ]
    subtype_data = []
    for name, proba, thr in models:
        pred = (proba>=thr).astype(int)
        for label, mask in grupos:
            n = mask.sum()
            r = 0.0 if n==0 else pred[mask].sum()/n
            subtype_data.append({"modelo":name,"grupo":label,"n":int(n),"recall":round(r,4)})
    pd.DataFrame(subtype_data).to_csv(art_cmp/"subtype_recall_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(13,5))
    labels = [g[0] for g in grupos]
    n_per = [int(g[1].sum()) for g in grupos]
    x = np.arange(len(labels))
    n_models = len(models)
    w = 0.8 / n_models   # ancho dinámico según nº de modelos
    st = pd.DataFrame(subtype_data)
    for i,(name,_,_) in enumerate(models):
        recalls = [st[(st.modelo==name)&(st.grupo==l)]["recall"].iloc[0] for l in labels]
        offset = (i - (n_models - 1) / 2) * w
        bars = ax.bar(x + offset, recalls, w, label=name, color=colors[name])
        for b,v in zip(bars,recalls):
            ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l,n in zip(labels,n_per)], fontsize=10)
    ax.set_ylabel("Recall",fontsize=12)
    ax.set_title("Comparación de Recall por subtipo de fraude",fontsize=12)
    ax.set_ylim(0,1.15); ax.legend(loc="upper right",fontsize=10); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(art_cmp/"fig_subtype_recall_barplot.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_subtype_recall_barplot.pdf")

    # Análisis de errores
    pred_xgb = (proba_xgb_test>=thr_xgb).astype(int)
    pred_gnn = (proba_gnn_test>=thr_gnn).astype(int)
    pred_hyb = (proba_hyb_test>=thr_hyb).astype(int)
    test_df = claims.iloc[idx_te].reset_index(drop=True).copy()
    test_df["y_true"]=y_test; test_df["pred_xgb"]=pred_xgb
    test_df["pred_gnn"]=pred_gnn; test_df["pred_hyb"]=pred_hyb
    test_df["proba_xgb"]=proba_xgb_test; test_df["proba_gnn"]=proba_gnn_test
    test_df["proba_hyb"]=proba_hyb_test

    fraud_df = test_df[test_df.y_true==1].copy()
    only_hyb = fraud_df[(fraud_df.pred_hyb==1)&(fraud_df.pred_xgb==0)&(fraud_df.pred_gnn==0)]
    no_one = fraud_df[(fraud_df.pred_hyb==0)&(fraud_df.pred_xgb==0)&(fraud_df.pred_gnn==0)]
    hyb_vs_xgb = fraud_df[(fraud_df.pred_hyb==1)&(fraud_df.pred_xgb==0)]
    hyb_vs_gnn = fraud_df[(fraud_df.pred_hyb==1)&(fraud_df.pred_gnn==0)]
    reg_vs_xgb = fraud_df[(fraud_df.pred_xgb==1)&(fraud_df.pred_hyb==0)]
    reg_vs_gnn = fraud_df[(fraud_df.pred_gnn==1)&(fraud_df.pred_hyb==0)]

    print(f"\nTotal fraudes test: {len(fraud_df)}")
    print(f"  Detectados por los 3: {len(fraud_df[(fraud_df.pred_xgb==1)&(fraud_df.pred_gnn==1)&(fraud_df.pred_hyb==1)])}")
    print(f"  Solo híbrido:         {len(only_hyb)}")
    print(f"  Nadie:                {len(no_one)}")
    print(f"  Híbrido > XGB:        {len(hyb_vs_xgb)}  |  Regresiones: {len(reg_vs_xgb)}")
    print(f"  Híbrido > GNN:        {len(hyb_vs_gnn)}  |  Regresiones: {len(reg_vs_gnn)}")

    cols_err = ["claim_id","Claims_type","Cost_claims_year","Fraud_type","Ring_ID",
                "Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID",
                "Event_ID","y_true","proba_xgb","proba_gnn","proba_hyb",
                "pred_xgb","pred_gnn","pred_hyb"]
    only_hyb[cols_err].to_csv(art_cmp/"errors_only_hybrid_catches.csv", index=False)
    no_one[cols_err].to_csv(art_cmp/"errors_no_one_catches.csv", index=False)

    # Caso de estudio testigos — sets para lookup O(1)
    linked_idx = np.where(linked_mask)[0]
    set_train = set(idx_tr.tolist())
    set_val   = set(idx_v.tolist())
    set_test  = set(idx_te.tolist())
    rows = []
    for gi in linked_idx:
        gi_int = int(gi)
        if gi_int in set_train: sp = "train"
        elif gi_int in set_val: sp = "val"
        elif gi_int in set_test: sp = "test"
        else: sp = "?"
        rows.append({"global_idx":gi_int,"split":sp,
                     "claim_id":claims.iloc[gi]["claim_id"],
                     "Event_ID":claims.iloc[gi]["Event_ID"],
                     "Claims_type":claims.iloc[gi]["Claims_type"],
                     "Cost_claims_year":float(claims.iloc[gi]["Cost_claims_year"]),
                     "proba_xgb":float(proba_xgb_all[gi]),
                     "proba_gnn":float(proba_gnn_all[gi]),
                     "proba_hyb":float(proba_hyb_all[gi])})
    linked_df = pd.DataFrame(rows)
    linked_df.to_csv(art_cmp/"case_study_linked_testigos.csv", index=False)
    print(f"\n--- Testigos T3A ---\n{linked_df.to_string(index=False)}")

    # Top hits bar chart
    fig, ax = plt.subplots(figsize=(7,4))
    names = [n for n,_,_ in models]
    hits_vals = [master[master.Modelo==n]["Hits@30"].iloc[0] for n in names]
    hits_num = [int(h.split("/")[0]) for h in hits_vals]
    bars = ax.bar(names, hits_num, color=[colors[n] for n in names])
    ax.axhline(top_n, color="grey", ls="--", label=f"Máximo ({top_n}/{top_n})")
    for b,h in zip(bars,hits_num):
        ax.text(b.get_x()+b.get_width()/2, h+0.5, f"{h}/{top_n}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Aciertos en top-{top_n}", fontsize=12)
    ax.set_title(f"Precision en el Top {top_n}", fontsize=11)
    ax.set_ylim(0, top_n+3); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(art_cmp/"fig_top30_hits_comparison.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("✔ fig_top30_hits_comparison.pdf")

    # Resumen errores CSV
    pd.DataFrame([
        {"categoria":"Total fraudes en test","n":len(fraud_df)},
        {"categoria":"Detectados por los 3 modelos",
         "n":len(fraud_df[(fraud_df.pred_xgb==1)&(fraud_df.pred_gnn==1)&(fraud_df.pred_hyb==1)])},
        {"categoria":"Solo híbrido","n":len(only_hyb)},
        {"categoria":"Nadie detecta","n":len(no_one)},
        {"categoria":"Híbrido > XGBoost","n":len(hyb_vs_xgb)},
        {"categoria":"Híbrido > GraphSAGE","n":len(hyb_vs_gnn)},
        {"categoria":"Regresiones híbrido vs XGB","n":len(reg_vs_xgb)},
        {"categoria":"Regresiones híbrido vs GNN","n":len(reg_vs_gnn)},
    ]).to_csv(art_cmp/"error_analysis.csv", index=False)

    print(f"\n✔ TODO GUARDADO EN: {art_cmp.resolve()}")
