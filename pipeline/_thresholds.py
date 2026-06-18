"""
pipeline/_thresholds.py — Selección de threshold operacional.

Centraliza la lógica de elegir el threshold que maximiza F-beta sobre VAL.
Por defecto beta=0.5 (prioriza precisión, reduce falsos positivos), configurable
desde cfg["eval"]["fbeta"].
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def best_fbeta_thr(y_true, proba, beta: float = 1.0) -> float:
    """
    Devuelve el threshold sobre `proba` que maximiza F_beta(y_true, pred).

    beta = 1.0 → F1 (precisión y recall equiponderados).
    beta < 1.0 → prioriza precisión (recomendado para fraude por coste de FP).
    beta > 1.0 → prioriza recall.

    Si la curva PR no permite calcular ningún F_beta válido, devuelve 0.5
    como fallback seguro.
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)
    p, r = prec[:-1], rec[:-1]
    denom = (beta ** 2) * p + r
    fbeta = np.where(denom > 0, (1 + beta ** 2) * p * r / (denom + 1e-12), 0.0)
    if fbeta.size == 0 or not np.isfinite(fbeta).any():
        return 0.5
    return float(thr[int(np.argmax(fbeta))])
