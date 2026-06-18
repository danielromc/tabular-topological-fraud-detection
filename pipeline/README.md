# Refactor del pipeline — cómo usarlo

## Estructura tras el refactor

```
Test 2/
├── config.yaml          ← TODOS los parámetros (cámbialos aquí)
├── main.py              ← orquestador único
├── pipeline/            ← módulos de cada fase
│   ├── __init__.py
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
│
├── claims.csv, providers.csv, edges.csv
├── artifacts/           ← fase 1
├── artifacts_graph/     ← fase 2
├── artifacts_hybrid/    ← fase 3
└── artifacts_comparison/ ← comparación final
```

## Cómo ejecutar

**Abres CMD o Anaconda Prompt** (NO Spyder) en la carpeta del proyecto y escribes:

```cmd
cd "C:\Users\danie\OneDrive\Documentos\Proyectos de Python\Test 2"
python main.py fase1
```

**Si prefieres ejecutar desde Spyder:**
Abre `main.py`, y al final del archivo (antes de `if __name__ == "__main__"`) añade temporalmente:
```python
sys.argv = ["main.py", "fase1"]   # o la fase que quieras
```
Luego ejecútalo normalmente con Play.

## Comandos disponibles

| comando | qué hace |
|---|---|
| `python main.py prep` | Solo preparación |
| `python main.py fase1` | Todo XGBoost (prep → baseline → tune → eval → top) |
| `python main.py fase2` | Todo GraphSAGE |
| `python main.py fase3` | Todo híbrido |
| `python main.py comparison` | Comparación final |
| `python main.py all` | Pipeline completo desde cero (~15 min CPU) |
| `python main.py --help` | Ver lista completa |

## Cómo cambiar un parámetro

**Ejemplo 1: probar con semilla 42 en vez de 2025.**
Abres `config.yaml`, cambias:
```yaml
seed: 42
```
Lanzas: `python main.py all`. Fin.

**Ejemplo 2: probar GraphSAGE con 128 dims en vez de 64.**
En `config.yaml`:
```yaml
graph:
  hidden_dim: 128
```
Lanzas solo lo afectado:
```cmd
python main.py graph_train
python main.py graph_eval
python main.py hybrid_extract
python main.py hybrid_tune
python main.py hybrid_eval
python main.py comparison
```

**Ejemplo 3: duplicar los trials de Optuna.**
```yaml
xgb_tune:
  n_trials: 100
hybrid:
  xgb_tune_trials: 100
```

## Ventaja para la memoria

En el apéndice puedes pegar el `config.yaml` completo y decir:

> *"La configuración experimental completa está documentada en el fichero
> `config.yaml`. La ejecución del pipeline de principio a fin se realiza con
> el comando `python main.py all`, garantizando reproducibilidad total."*

Esto satisface directamente el principio de trazabilidad que exige EIOPA
para modelos de alto impacto en el sector asegurador.
