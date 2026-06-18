"""
main.py — Orquestador único del pipeline de detección de fraude.

USO DESDE TERMINAL (abre CMD o Anaconda Prompt en la carpeta del proyecto):

  python main.py <fase>

Fases disponibles:
  prep              Fase 1.1 — preparación de features y splits
  baseline          Fase 1.2 — XGBoost con defaults
  tune              Fase 1.3 — Optuna sobre XGBoost
  eval_test         Fase 1.4 — evaluación en test (XGBoost)
  top_invest        Fase 1.5 — top-30 para investigación
  fase1             TODO lo anterior en orden

  graph_inspect     Fase 2.1 — inspección topológica del grafo
  graph_build       Fase 2.2 — construcción del HeteroData
  graph_train       Fase 2.3 — entrenamiento GraphSAGE
  graph_eval        Fase 2.4 — evaluación en test (GraphSAGE)
  fase2             TODO lo anterior en orden

  hybrid_extract    Fase 3.1 — extracción de embeddings
  hybrid_baseline   Fase 3.2 — baseline híbrido
  hybrid_tune       Fase 3.3 — Optuna híbrido
  hybrid_eval       Fase 3.4 — evaluación final híbrido
  fase3             TODO lo anterior en orden

  comparison        Comparación final XGB vs GNN vs Híbrido
  all               TODAS las fases desde cero (reproduce el proyecto completo)

EJEMPLOS:
  python main.py prep
  python main.py fase1
  python main.py all
  python main.py graph_train

Antes de lanzar, verifica config.yaml: ahí están todos los parámetros
(semilla, nº de trials Optuna, hiperparámetros del GNN, etc.).
"""
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: falta PyYAML. Instálalo con: pip install pyyaml")
    sys.exit(1)

CONFIG_PATH = Path("config.yaml")

def load_config():
    if not CONFIG_PATH.exists():
        print(f"ERROR: no encuentro {CONFIG_PATH} en el directorio actual.")
        print("Asegúrate de ejecutar desde la carpeta donde está config.yaml.")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def banner(text):
    print("\n" + "="*72)
    print(f"  {text}")
    print("="*72)

# -----------------------------------------------------------------
# Importar módulos de fase solo cuando se usen (import tardío)
# Esto evita cargar PyTorch si solo quieres hacer prep, por ejemplo.
# -----------------------------------------------------------------
def run_prep(cfg):
    banner("FASE 1.1 — Preparación")
    from pipeline import prep as mod
    mod.run(cfg)

def run_baseline(cfg):
    banner("FASE 1.2 — Baseline XGBoost")
    from pipeline import xgb_baseline as mod
    mod.run(cfg)

def run_tune(cfg):
    banner("FASE 1.3 — Optuna XGBoost")
    from pipeline import xgb_tune as mod
    mod.run(cfg)

def run_eval_test(cfg):
    banner("FASE 1.4 — Evaluación XGBoost en test")
    from pipeline import xgb_eval as mod
    mod.run(cfg)

def run_top_invest(cfg):
    banner("FASE 1.5 — Top-30 para investigación")
    from pipeline import xgb_top as mod
    mod.run(cfg)

def run_graph_inspect(cfg):
    banner("FASE 2.1 — Inspección del grafo")
    from pipeline import graph_inspect as mod
    mod.run(cfg)

def run_graph_build(cfg):
    banner("FASE 2.2 — Construcción HeteroData")
    from pipeline import graph_build as mod
    mod.run(cfg)

def run_graph_train(cfg):
    banner("FASE 2.3 — Entrenamiento GraphSAGE")
    from pipeline import graph_train as mod
    mod.run(cfg)

def run_graph_eval(cfg):
    banner("FASE 2.4 — Evaluación GraphSAGE en test")
    from pipeline import graph_eval as mod
    mod.run(cfg)

def run_hybrid_extract(cfg):
    banner("FASE 3.1 — Extracción de embeddings")
    from pipeline import hybrid_extract as mod
    mod.run(cfg)

def run_hybrid_baseline(cfg):
    banner("FASE 3.2 — Baseline híbrido")
    from pipeline import hybrid_baseline as mod
    mod.run(cfg)
 
def run_hybrid_tune(cfg):
    banner("FASE 3.3 — Optuna híbrido")
    from pipeline import hybrid_tune as mod
    mod.run(cfg)

def run_hybrid_eval(cfg):
    banner("FASE 3.4 — Evaluación híbrido en test")
    from pipeline import hybrid_eval as mod
    mod.run(cfg)

def run_comparison(cfg):
    banner("COMPARACIÓN FINAL — XGB vs GNN vs Híbrido")
    from pipeline import comparison as mod
    mod.run(cfg)

def run_montecarlo(cfg):
    banner("MINI-MONTECARLO — N semillas sobre pipeline completo")
    from pipeline import montecarlo as mod
    mod.run(cfg)

def run_xai_ego(cfg):
    banner("XAI — Ego-red de proveedor de anillo")
    from pipeline.explainability import ego_ring_provider as mod
    mod.run(cfg)

def run_xai_witness(cfg):
    banner("XAI — Subgrafos de testigos T3A")
    from pipeline.explainability import witness_t3a as mod
    mod.run(cfg)

def run_xai_shap(cfg):
    banner("XAI — Explicación SHAP local sobre el híbrido")
    from pipeline.explainability import shap_local as mod
    mod.run(cfg)

def run_xai_shap_analysis(cfg):
    banner("XAI — SHAP completo de comparación y casos individuales")
    from pipeline.explainability import shap_analysis as mod
    mod.run(cfg)

def run_xai_subgraph(cfg):
    banner("XAI — Subgrafo K-hop del grafo bipartito")
    from pipeline.explainability import subgraph_khop as mod
    mod.run(cfg)

def run_analysis_claims_type(cfg):
    banner("ANÁLISIS — Recall por tipo de siniestro")
    from pipeline import analysis_by_claims_type as mod
    mod.run(cfg)

def run_xai_all(cfg):
    banner("XAI — Todos los casos de estudio")
    run_xai_ego(cfg)
    run_xai_witness(cfg)
    run_xai_shap(cfg)

def run_xai_suite(cfg):
    banner("XAI SUITE — Set ampliado para TFM (6 figuras)")
    from pipeline.explainability import xai_suite as mod
    mod.run(cfg)

# -----------------------------------------------------------------
# Grupos de fases
# -----------------------------------------------------------------
def run_fase1(cfg):
    banner("FASE 1 COMPLETA — XGBoost tabular")
    run_prep(cfg)
    run_baseline(cfg)
    run_tune(cfg)
    run_eval_test(cfg)
    run_top_invest(cfg)

def run_fase2(cfg):
    banner("FASE 2 COMPLETA — GraphSAGE")
    run_graph_inspect(cfg)
    run_graph_build(cfg)
    run_graph_train(cfg)
    run_graph_eval(cfg)

def run_fase3(cfg):
    banner("FASE 3 COMPLETA — Híbrido")
    run_hybrid_extract(cfg)
    run_hybrid_baseline(cfg)
    run_hybrid_tune(cfg)
    run_hybrid_eval(cfg)

def run_all(cfg):
    banner("PIPELINE COMPLETO DESDE CERO")
    t0 = time.time()
    run_fase1(cfg)
    run_fase2(cfg)
    run_fase3(cfg)
    run_comparison(cfg)
    elapsed = time.time() - t0
    banner(f"✔ PIPELINE COMPLETO. Tiempo total: {elapsed/60:.1f} minutos")

# -----------------------------------------------------------------
# Mapa de comandos
# -----------------------------------------------------------------
COMMANDS = {
    "prep":            run_prep,
    "baseline":        run_baseline,
    "tune":            run_tune,
    "eval_test":       run_eval_test,
    "top_invest":      run_top_invest,
    "fase1":           run_fase1,
    "graph_inspect":   run_graph_inspect,
    "graph_build":     run_graph_build,
    "graph_train":     run_graph_train,
    "graph_eval":      run_graph_eval,
    "fase2":           run_fase2,
    "hybrid_extract":  run_hybrid_extract,
    "hybrid_baseline": run_hybrid_baseline,
    "hybrid_tune":     run_hybrid_tune,
    "hybrid_eval":     run_hybrid_eval,
    "fase3":           run_fase3,
    "comparison":      run_comparison,
    "montecarlo":      run_montecarlo,
    "xai_ego":         run_xai_ego,
    "xai_witness":     run_xai_witness,
    "xai_shap":        run_xai_shap,
    "xai_shap_analysis": run_xai_shap_analysis,
    "xai_subgraph":    run_xai_subgraph,
    "analysis_claims_type": run_analysis_claims_type,
    "xai_all":         run_xai_all,
    "xai_suite":       run_xai_suite,
    "all":             run_all,
}

def print_usage():
    print(__doc__)
    print("\nComandos disponibles:")
    for cmd in COMMANDS:
        print(f"  python main.py {cmd}")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd in ("-h", "--help", "help"):
        print_usage()
        sys.exit(0)

    if cmd not in COMMANDS:
        print(f"ERROR: comando '{cmd}' no reconocido.\n")
        print_usage()
        sys.exit(1)

    cfg = load_config()
    print(f"Config cargado de: {CONFIG_PATH.resolve()}")
    print(f"Semilla global: {cfg['seed']}")

    COMMANDS[cmd](cfg)

if __name__ == "__main__":
    main()
