"""
explainability/xai_suite.py — Genera la suite XAI ampliada para el TFM.

Llama a los módulos individuales con casos contrastantes:

  EGO-REDES (2 figuras):
    - Anillo: ego-red de un proveedor de anillo A (auto)
    - Comparación: ego-red de un proveedor legítimo de volumen similar

  TESTIGOS (1 figura): los 5 testigos T3A

  SHAP (3 figuras): casos contrastantes
    - Organized: señal vendrá del grafo
    - Opp_fp: oportunista con proveedor cómplice (mezcla)
    - Opp_lp: oportunista con proveedor legítimo (señal solo tabular)
"""
from . import ego_ring_provider, witness_t3a, shap_local


def run(cfg):
    print("\n" + "="*70)
    print("XAI SUITE — Set ampliado para el TFM")
    print("="*70)

    print("\n[1/6] Ego-red proveedor de anillo (anillo A, alto volumen)")
    ego_ring_provider.run(cfg, target_provider=None)  # auto: ring A

    print("\n[2/6] Ego-red proveedor LEGÍTIMO de comparación")
    ego_ring_provider.run(cfg, target_provider="auto_legit")

    print("\n[3/6] Subgrafos de los 5 testigos T3A")
    witness_t3a.run(cfg)

    print("\n[4/6] SHAP local — caso ORGANIZED (señal del grafo)")
    shap_local.run(cfg, subtype="organized")

    print("\n[5/6] SHAP local — caso OPP_FP (señal mixta)")
    shap_local.run(cfg, subtype="opp_fp")

    print("\n[6/6] SHAP local — caso OPP_LP (señal solo tabular)")
    shap_local.run(cfg, subtype="opp_lp")

    print("\n" + "="*70)
    print("✔ XAI SUITE COMPLETADA")
    print("="*70)
