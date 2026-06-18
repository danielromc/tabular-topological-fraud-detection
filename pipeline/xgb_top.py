"""pipeline/xgb_top.py — Fase 1.5: Top-N para investigación (XGBoost)."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def run(cfg):
    top_n = cfg["eval"]["top_n_investigation"]
    art = Path(cfg["paths"]["artifacts"])

    X = pd.read_pickle(art/"X_all.pkl")
    y = np.load(art/"y_all.npy")
    idx_train = np.load(art/"idx_train.npy")
    idx_val   = np.load(art/"idx_val.npy")
    idx_test  = np.load(art/"idx_test.npy")
    claims = pd.read_csv(cfg["paths"]["claims_csv"])

    model = xgb.XGBClassifier()
    model.load_model(art/"model_tuned.json")

    # Top-N evaluable en test
    X_test = X.iloc[idx_test]
    proba_test = model.predict_proba(X_test)[:,1]
    y_test = y[idx_test]

    test_df = claims.iloc[idx_test].reset_index(drop=True).copy()
    test_df["proba_fraude"] = proba_test
    test_df["is_fraud_real"] = y_test.astype(bool)

    cols_test = ["proba_fraude","is_fraud_real","claim_id","ID","Current_Year",
                 "Claims_type","Cost_claims_year","Premium","N_claims_history",
                 "Driver_Age","Vehicle_Age","Type_risk","Area",
                 "Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID",
                 "Event_ID","Fraud_type","Ring_ID"]
    top_test = test_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols_test].reset_index(drop=True)
    top_test.insert(0,"rank",range(1,len(top_test)+1))
    top_test.to_csv(art/f"top{top_n}_test.csv", index=False)

    n_hits = int(top_test["is_fraud_real"].sum())
    total = int(y_test.sum())
    print(f"=== TOP-{top_n} TEST (evaluable) ===")
    print(f"Aciertos: {n_hits}/{top_n}  |  Precision@{top_n}={n_hits/top_n:.3f}")
    print(f"Recall@{top_n}: {n_hits/total:.3f} ({n_hits} de {total} fraudes reales)")
    hits = top_test[top_test.is_fraud_real]
    if len(hits):
        print(f"\nPor Fraud_type:\n{hits['Fraud_type'].value_counts().to_string()}")
        print(f"\nPor Ring_ID:\n{hits['Ring_ID'].value_counts().to_string()}")

    # Top-N operativo (dataset completo)
    proba_all = model.predict_proba(X)[:,1]
    op_df = claims.copy()
    op_df["proba_fraude"] = proba_all
    split = np.full(len(op_df),"test",dtype=object)
    split[idx_train] = "train"; split[idx_val] = "val"
    op_df["_split"] = split
    cols_op = ["proba_fraude","claim_id","ID","Current_Year","Claims_type",
               "Cost_claims_year","Premium","N_claims_history","Driver_Age","Vehicle_Age",
               "Type_risk","Area","Provider_workshop_ID","Provider_clinic_ID","Provider_lawyer_ID","_split"]
    top_op = op_df.sort_values("proba_fraude",ascending=False).head(top_n)[cols_op].reset_index(drop=True)
    top_op.insert(0,"rank",range(1,len(top_op)+1))
    top_op.to_csv(art/f"top{top_n}_operativo.csv", index=False)
    print(f"\n✔ top{top_n}_test.csv, top{top_n}_operativo.csv")
