"""
pipeline/prep.py — Fase 1.1: preparación de features y split por grupo.

CAMBIO IMPORTANTE (auditoría L3):
El split aleatorio estratificado anterior repartía las múltiples filas
de un mismo cliente (ID, años 2015-2018) entre train/val/test, permitiendo
memorización de identidad. Ahora se usa GroupShuffleSplit por `ID`:
todas las filas de un cliente caen en el mismo split.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def run(cfg):
    random_state = cfg["seed"]
    csv_path = cfg["paths"]["claims_csv"]
    out_dir = Path(cfg["paths"]["artifacts"])
    out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)
    print(f"Cargado: {df.shape}")

    LEAK_COLS = cfg["prep"]["leak_cols"]
    ID_COLS = ["claim_id", "ID", "Event_ID"]
    PROVIDER_ID_COLS = ["Provider_workshop_ID", "Provider_clinic_ID", "Provider_lawyer_ID"]
    CONTAMINATED_RELATIONAL = cfg["prep"]["contaminated_relational"]
    DATE_COLS = ["Date_start_contract", "Date_last_renewal", "Date_next_renewal",
                 "Date_birth", "Date_driving_licence"]
    RESERVED = ["Linked_to_fraud_event"]
    TARGET = "is_fraud"

    drop_cols = (LEAK_COLS + ID_COLS + PROVIDER_ID_COLS + CONTAMINATED_RELATIONAL
                 + DATE_COLS + RESERVED + [TARGET])

    # Filtrar drop_cols a las que efectivamente existen en df (z_score_type puede
    # no estar si el inyector R cambia)
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df[TARGET].astype(int).to_numpy()
    linked_mask = df["Linked_to_fraud_event"].astype(bool).to_numpy()
    print(f"Tasa fraude global: {y.mean()*100:.3f}%")
    print(f"Filas Linked_to_fraud_event: {linked_mask.sum()}")

    X = df.drop(columns=drop_cols).copy()
    X["Current_Year"] = X["Current_Year"].astype(str)

    CAT_COLS = ["Payment", "Type_risk", "Area", "Second_driver",
                "Type_fuel", "Claims_type", "Current_Year"]
    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=False, dtype=np.uint8)

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.uint8)

    # ============================================================
    # Split por grupo (cliente ID) — NO aleatorio puro
    # ============================================================
    n = len(X)
    idx_all = np.arange(n)
    test_size = cfg["prep"]["test_size"]              # 0.30 → 70/30 primer corte
    val_frac = cfg["prep"]["val_fraction_of_test"]    # 0.50 → 15/15 segundo corte
    groups = df["ID"].to_numpy()

    print(f"\nClientes únicos: {len(np.unique(groups))}  "
          f"(media de filas por cliente: {n/len(np.unique(groups)):.2f})")

    # Primer corte: train (70%) vs temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_temp = next(gss1.split(idx_all, y, groups=groups))

    # Segundo corte sobre temp: val (50%) vs test (50%) → 15/15
    groups_temp = groups[idx_temp]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    sub_val, sub_test = next(gss2.split(idx_temp, y[idx_temp], groups=groups_temp))
    idx_val = idx_temp[sub_val]
    idx_test = idx_temp[sub_test]

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    print("\n--- Splits (por cliente) ---")
    for name, idx in splits.items():
        print(f"  {name:5s}: n={len(idx):5d}  fraudes={y[idx].sum()}  "
              f"tasa={y[idx].mean()*100:.3f}%  "
              f"clientes={len(np.unique(groups[idx]))}")

    # Verificación: ningún cliente comparte filas entre splits
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_val])), \
        "Cliente compartido entre train y val"
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_test])), \
        "Cliente compartido entre train y test"
    assert set(groups[idx_val]).isdisjoint(set(groups[idx_test])), \
        "Cliente compartido entre val y test"
    print("[OK] Verificado: ningún cliente está en más de un split.")

    # Aviso si la estratificación se desvía mucho (GroupShuffleSplit no estratifica)
    rate_global = y.mean()
    for name, idx in splits.items():
        rate = y[idx].mean()
        if rate_global > 0 and abs(rate - rate_global) / rate_global > 0.30:
            print(f"  [WARN] tasa fraude en {name} ({rate*100:.2f}%) desviada >30% "
                  f"respecto a la global ({rate_global*100:.2f}%)")

    # Guardar
    X.to_pickle(out_dir / "X_all.pkl")
    np.save(out_dir / "y_all.npy", y)
    np.save(out_dir / "idx_train.npy", idx_train)
    np.save(out_dir / "idx_val.npy", idx_val)
    np.save(out_dir / "idx_test.npy", idx_test)
    np.save(out_dir / "linked_mask.npy", linked_mask)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    summary = {
        "n_rows": int(n), "n_features": int(X.shape[1]),
        "fraud_rate_global": float(y.mean()),
        "random_state": random_state,
        "split_strategy": "GroupShuffleSplit_by_ID",
        "splits": {k: {"n": int(len(v)), "fraudes": int(y[v].sum()),
                       "fraud_rate": float(y[v].mean()),
                       "n_clientes": int(len(np.unique(groups[v])))}
                   for k, v in splits.items()},
        "dropped_columns": drop_cols,
    }
    with open(out_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Guardado en {out_dir.resolve()}")
