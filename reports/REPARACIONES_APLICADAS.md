# Reparaciones aplicadas a `TFM_xG_MachineLearning_v6`

Este documento resume los cambios realizados para dejar el proyecto más limpio, reproducible y defendible como TFM.

## 1. Reparaciones críticas

### 1.1 Se evita el fallback silencioso a datos sintéticos

Antes, si el usuario ejecutaba:

```bash
python run_pipeline.py --source statsbomb
```

y `statsbombpy` no estaba instalado o fallaba la descarga, el proyecto podía caer a modo sintético. Esto era peligroso porque podía generar resultados sintéticos creyendo que eran reales.

Ahora:

- `--source statsbomb` exige StatsBomb real o una caché local legible.
- Si no existe caché y no se puede usar `statsbombpy`, el pipeline falla explícitamente.
- Solo `--source auto` puede caer a sintético, y lo deja indicado en logs.

Archivo modificado:

```text
src/xg/data/loader.py
```

### 1.2 Lectura de caché más robusta

La lectura de caché ahora intenta formatos portables:

1. parquet si hay motor compatible;
2. CSV;
3. pickle como compatibilidad antigua.

Además, la escritura guarda CSV de respaldo. Esto permite ejecutar el proyecto aunque falte `pyarrow`.

Archivo modificado:

```text
src/xg/utils/io.py
```

### 1.3 Caché sintética parametrizada

Antes la caché sintética se llamaba siempre `synthetic_shots_raw`, aunque se generase con distinto `n_matches` o `seed`. Eso podía reutilizar un dataset incorrecto.

Ahora el nombre incluye parámetros:

```text
synthetic_shots_raw_n{n_matches}_seed{seed}
```

Archivo modificado:

```text
src/xg/data/loader.py
```

## 2. Reparaciones de resultados

### 2.1 Rankings recalculados desde una única fuente

Se recalcularon los resultados de análisis aplicado desde:

```text
data/processed/shots_scored.csv
```

para que fueran coherentes entre sí:

- `team_xg_ranking.csv`
- `player_xg_ranking.csv`
- `top_finishers_over_xg.csv`
- `finishers_significance.csv`
- `finishers_bayesian_hierarchical.csv`
- `finishers_freq_vs_bayes.csv`
- `bayesian_em_trace.csv`
- `benchmark_vs_statsbomb.csv`

Esto elimina diferencias de xG por jugador provocadas por ejecuciones distintas.

### 2.2 Corrección por comparaciones múltiples

El contraste frecuentista de finalizadores ahora incluye corrección Benjamini-Hochberg:

- `p_value_fdr_bh`
- `significant_fdr`

Esto evita presentar como hallazgos fuertes resultados que solo son significativos sin corregir por el número de jugadores analizados.

Archivo modificado:

```text
src/xg/analysis/aggregation.py
```

## 3. Reparaciones de documentación

### 3.1 README actualizado

Se sustituyeron cifras sintéticas/desactualizadas por los resultados reales del ZIP:

- test externo UEFA Euro 2024;
- HistGradientBoosting calibrado como modelo principal;
- calibración descrita como mejora ligera, no drástica;
- benchmark frente a StatsBomb explicado con cautela.

Archivo modificado:

```text
README.md
```

### 3.2 Memoria actualizada

La sección de interpretación de resultados se reescribió con cifras coherentes con los CSV finales:

- 5.829 disparos brutos;
- 5.606 disparos no penales;
- 507 goles no penales;
- Euro 2024 como test externo;
- Brier, Log Loss, ECE y bootstrap;
- LOCO;
- benchmark frente a StatsBomb;
- finalización con FDR y modelo bayesiano.

Archivo modificado:

```text
reports/MEMORIA_CONTENIDO.md
```

## 4. Reparaciones de notebooks

Los notebooks se limpiaron para que:

- no tengan outputs antiguos embebidos;
- usen holdout por competición cuando esté disponible;
- no sobrescriban resultados oficiales del pipeline;
- no afirmen que la calibración mejora de forma drástica.

Archivos modificados:

```text
notebooks/*.ipynb
build_notebooks.py
```

## 5. Limpieza del proyecto

Se eliminan del ZIP final:

- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- `src/xg.egg-info/`
- cachés sintéticas antiguas no parametrizadas.

## 6. Estado de pruebas

Se ejecutó la suite de tests tras los cambios:

```text
23 passed
```

## 7. Advertencia metodológica que se mantiene

El proyecto es defendible, pero los resultados de rankings de jugadores deben presentarse con cautela. La lectura recomendada es:

- los rankings `goles - xG` son exploratorios;
- el contraste frecuentista ayuda, pero puede inflar hallazgos;
- la conclusión principal debe apoyarse en el modelo bayesiano jerárquico y en la corrección FDR.
