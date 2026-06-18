# Pipeline de detección de fraude

Repositorio con un flujo híbrido para detección de fraude que combina:

- modelo tabular con XGBoost,
- modelo topológico con GraphSAGE,
- y una capa híbrida tipo stacking.

El proyecto está pensado para ejecutarse de principio a fin desde `main.py` y se
configura desde `config.yaml`.

## Estructura del proyecto

```
tabular-topological-fraud-detection/
├── README.md
├── LICENSE
├── config.yaml              ← parámetros del pipeline
├── main.py                  ← orquestador único
├── claims.csv               ← datos de entrada tabulares
├── providers.csv            ← datos de entrada de proveedores
├── edges.csv                ← relaciones para el grafo
├── pipeline/                ← módulos por fase
│   ├── prep.py
│   ├── xgb_baseline.py
│   ├── xgb_tune.py
│   ├── xgb_eval.py
│   ├── xgb_top.py
│   ├── graph_inspect.py
│   ├── graph_build.py
│   ├── graph_train.py
│   ├── graph_eval.py
│   ├── hybrid_extract.py
│   ├── hybrid_baseline.py
│   ├── hybrid_tune.py
│   ├── hybrid_eval.py
│   └── comparison.py
├── scripts/                 ← scripts auxiliares/experimentales
├── lib/                     ← librerías estáticas de visualización
├── artifacts/               ← salidas fase tabular
├── artifacts_graph/         ← salidas fase topológica
├── artifacts_hybrid/        ← salidas fase híbrida
└── artifacts_comparison/    ← métricas y figuras finales
```

## ¿Qué contiene cada dato?

- `claims.csv`: base principal de reclamaciones para el entrenamiento tabular.
- `providers.csv`: información auxiliar de proveedores.
- `edges.csv`: relaciones (aristas) necesarias para construir el grafo.

Estos archivos se generan en el proyecto en R y se incluyen aquí para que el
pipeline de Python sea reproducible sin pasos intermedios.

## Requisitos

Se recomienda usar la versión más actualziada de Python.

Paquetes principales usados por el proyecto:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `xgboost`
- `optuna`
- `pyyaml`
- `torch`
- `torch-geometric`
- `networkx`
- `shap`
- `scipy`


## ¿Cómo ejecutar?

El punto de entrada es `main.py`.

```bash
python main.py all
```

Comandos disponibles:

| comando | qué hace |
|---|---|
| `python main.py prep` | Preparación de features y splits |
| `python main.py fase1` | XGBoost completo (prep → baseline → tune → eval → top) |
| `python main.py fase2` | GraphSAGE completo |
| `python main.py fase3` | Flujo híbrido completo |
| `python main.py comparison` | Comparación final entre modelos |
| `python main.py all` | Pipeline completo desde cero |
| `python main.py --help` | Ver todas las opciones |

## Flujo de trabajo

### Fase 1: tabular

- limpieza y preparación de datos,
- entrenamiento baseline con XGBoost,
- ajuste de hiperparámetros,
- evaluación en test,
- ranking top-30 para investigación.

### Fase 2: topológica

- inspección del grafo,
- construcción de `HeteroData`,
- entrenamiento de GraphSAGE,
- evaluación en test.

### Fase 3: híbrida

- extracción de embeddings,
- baseline híbrido,
- tuning híbrido,
- evaluación híbrida final.

### Comparación final

- tabla maestra de métricas,
- curvas PR y ROC,
- matrices de confusión,
- importancias de variables,
- análisis por subtipo.

## Configuración

Toda la configuración vive en `config.yaml`.

Ejemplo: cambiar la semilla.

```yaml
seed: 42
```

Ejemplo: cambiar la dimensión oculta de GraphSAGE.

```yaml
graph:
  hidden_dim: 128
```

Luego ejecutar solo la parte afectada o el pipeline completo.

## Salidas generadas

- `artifacts/`: features, índices de split, métricas tabulares y resultados de XGBoost.
- `artifacts_graph/`: grafo, checkpoint y métricas de GraphSAGE.
- `artifacts_hybrid/`: embeddings, modelos e indicadores híbridos.
- `artifacts_comparison/`: tablas y figuras comparativas finales.

