# Expected Goals (xG) con Machine Learning

> Trabajo de Fin de Máster · Máster en Ingeniería Matemática y Computacional

Modelo de *Expected Goals* (xG) para fútbol que estima la probabilidad de gol de
cada disparo a partir de su contexto geométrico, táctico y espacial. El proyecto
trata el xG como lo que es —un problema de **estimación de probabilidad
calibrada**, no de clasificación— y construye un pipeline reproducible de
extremo a extremo: adquisición de datos, ingeniería de variables, comparación de
modelos, calibración, validación estadística e interpretación aplicada por
jugador y equipo, incluyendo un modelo jerárquico bayesiano de finalización.

## Tesis del trabajo

Un buen modelo de xG no es necesariamente el que mejor *ordena* los disparos
(ROC-AUC), sino el que mejor *estima sus probabilidades* (Brier, Log Loss) y
mantiene una calibración razonable (ECE). El resultado principal del proyecto,
ejecutado con datos reales de StatsBomb y UEFA Euro 2024 como competición de
test no vista, es que el **HistGradientBoosting calibrado** ofrece el mejor
comportamiento probabilístico.

| Modelo | ROC-AUC | PR-AUC | Brier ↓ | ECE ↓ | Log Loss ↓ | Media predicha | Tasa real |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting calibrado | 0.744 | 0.203 | 0.0659 | 0.0176 | 0.2406 | 0.0886 | 0.0752 |
| HistGradientBoosting sin calibrar | 0.741 | 0.206 | 0.0665 | 0.0206 | 0.2424 | 0.0872 | 0.0752 |
| Regresión logística + B-splines | 0.747 | 0.232 | 0.1732 | 0.2899 | 0.5249 | 0.3651 | 0.0752 |
| Regresión logística | 0.761 | 0.252 | 0.1745 | 0.2989 | 0.5293 | 0.3741 | 0.0752 |

La regresión logística presenta un ROC-AUC superior, pero sus probabilidades no
son válidas como xG: la media predicha queda muy alejada de la tasa real de gol.
Por ello se selecciona el modelo calibrado de boosting, que no maximiza solo la
capacidad de ranking, sino la calidad de las probabilidades. La calibración
isotónica mejora ligeramente el Brier, el Log Loss y el ECE respecto a la versión
sin calibrar; la mejora se presenta como moderada, no como una mejora drástica.

### Resultado externo frente a StatsBomb xG

Sobre 5.606 disparos no penales, el modelo propio alcanza una correlación de
Pearson de 0.827 y una correlación de Spearman de 0.804 frente al xG oficial de
StatsBomb. La suma total de xG propio es 503.73 frente a 507 goles reales y
519.44 de StatsBomb. StatsBomb obtiene mejor Brier (0.0679 frente a 0.0719), por
lo que este benchmark se interpreta como una validación de coherencia externa,
no como una superación del modelo profesional de referencia.

### Nota sobre esta versión reparada

Esta versión corrige los principales problemas detectados en la revisión: caché
sintética parametrizada, lectura portable de cachés CSV/parquet, prohibición de
fallback silencioso a sintético cuando se pide StatsBomb explícitamente, outputs
de finalización recalculados desde un único `shots_scored.csv`, corrección FDR
Benjamini-Hochberg en el contraste de finalizadores y limpieza de artefactos
temporales del ZIP.

## Instalacion

Requiere Python >= 3.10. Para máxima reproducibilidad se recomienda Python 3.11 o 3.12. Con Python 3.14 se puede ejecutar el núcleo del proyecto, pero algunas dependencias opcionales pueden requerir versiones recientes. Se recomienda un entorno virtual.

```bash
git clone <url-del-repositorio>
cd TFM_xG_MachineLearning

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Instalacion como paquete editable (recomendado)
pip install -e .                         # núcleo
pip install -e ".[data,notebooks,dev]"    # recomendado para ejecutar pipeline, notebooks y tests
# Evitar ".[all]" al principio si se usa Python 3.14; instala dependencias pesadas opcionales.
```

La instalacion editable expone el paquete `xg` y el comando `xg-pipeline`.

## Uso rapido

```bash
# Pipeline completo con datos reales de StatsBomb (requiere ".[data]")
xg-pipeline --source statsbomb --holdout-competition "UEFA Euro 2024"

# Pipeline completo en modo sintetico (sin conexion, totalmente reproducible)
xg-pipeline --source synthetic
# equivalente: python run_pipeline.py --source synthetic

# Atajos de Makefile
python run_pipeline.py --source synthetic
pytest -q               # suite de pruebas (23 tests)
python execute_notebooks.py --source statsbomb
```

El modo `synthetic` genera disparos a partir de un modelo logistico generador
con **xG verdadero conocido** (un "oraculo"), lo que permite validar la
calibracion y el scoring contra la verdad de base sin necesidad de red.

## Estructura del proyecto

```
TFM_xG_MachineLearning/
├── pyproject.toml              Paquete instalable (PEP 621) + extras
├── requirements.txt            Dependencias con rangos cerrados (reproducibilidad)
├── Makefile                    Targets de reproducibilidad
├── run_pipeline.py             Punto de entrada (delega en xg.cli)
├── src/xg/                     Paquete principal
│   ├── config.py               Fuente unica de verdad (rutas, semilla, esquema)
│   ├── utils/      io.py        E/S y cache (parquet con respaldo a pickle)
│   ├── data/       loader.py    Adquisicion StatsBomb + generador sintetico
│   ├── features/   engineering.py  Geometria, contexto tactico, splines
│   ├── models/     catalog.py    Catalogo de modelos (pipelines sklearn)
│   │               evaluation.py  Holdout por competicion, LOCO, OOF, metricas
│   │               statistics.py  Bootstrap, comparacion pareada, Wilson
│   │               bayesian.py    Modelo jerarquico de finalizacion (EM+Laplace)
│   ├── bayes/      finishing.py  Alternativa Beta-Binomial conjugada
│   ├── analysis/   aggregation.py  Rankings, Poisson-binomial, benchmark
│   ├── visualization/ plots.py   18 figuras reproducibles (campo propio)
│   ├── pipeline.py             Orquestador end-to-end (9 pasos)
│   └── cli.py                  Interfaz de linea de comandos
├── notebooks/                  7 notebooks Jupyter (analisis paso a paso)
├── tests/                      Suite de pruebas (pytest)
├── data/   raw/ processed/ results/   Datos en cada etapa
├── artifacts/models/           Modelos entrenados (.joblib)
├── figures/                    Figuras generadas (.png)
└── reports/                    Guia de ejecucion en Windows
```

## El pipeline en 9 pasos

1. **Adquisicion** — disparos de StatsBomb Open Data (o sinteticos), una fila por
   disparo, con deteccion de defensores en el cono de tiro via *freeze frames*.
2. **Ingenieria de variables** — distancia y angulo a porteria (geometria), sus
   transformaciones no lineales, contexto tactico y `game_state` (marcador).
3. **Particiones** — *holdout* por competicion (generalizacion a un torneo no
   visto), evitando fuga de informacion temporal.
4. **Comparacion de modelos** — 6 modelos por validacion cruzada, metrica
   principal Brier; mas validacion *Leave-One-Competition-Out* (LOCO).
5. **Entrenamiento + calibracion** — mejor modelo + calibracion isotonica;
   baseline logistico y logistica con B-splines como rival lineal serio.
6. **Evaluacion en test** — metricas con IC *bootstrap* (2000 replicas) y
   comparacion pareada del mejor modelo frente al baseline.
7. **Figuras de evaluacion** — ROC, PR, curvas de calibracion, fiabilidad.
8. **Interpretabilidad** — importancia por permutacion (y SHAP si esta disponible).
9. **Analisis aplicado** — scoring honesto *out-of-fold* de todos los disparos,
   rankings por jugador/equipo, contraste de finalizadores (Poisson-binomial) y
   **modelo jerarquico bayesiano** con encogimiento; benchmark vs. xG oficial.

## Dos contribuciones metodologicas destacadas

**Scoring honesto del analisis aplicado.** Los rankings de finalizacion se
calculan con xG estimado por validacion cruzada *out-of-fold*: cada disparo se
puntua con un modelo que no lo vio en entrenamiento. Esto evita el optimismo de
re-sustitucion que comprimiria artificialmente la dispersion de (goles - xG).

**Finalizacion: frecuentista vs. bayesiano.** El contraste Poisson-binomial
evalua cada jugador por separado y, con muchos jugadores y pocos disparos,
produce falsos positivos. El modelo jerarquico bayesiano (efecto aleatorio de
jugador en log-odds con prior N(0, tau^2), ajustado por Bayes empirico EM+Laplace)
aplica *partial pooling*: encoge las estimaciones ruidosas hacia la media
poblacional en proporcion a la incertidumbre. En datos sin habilidad de
finalizacion real, el frecuentista marca falsos finalizadores excepcionales
mientras que el bayesiano, correctamente, no marca ninguno.

## Reproducibilidad

- Semilla global fija (`config.RANDOM_STATE = 42`) en todos los componentes.
- Toda la aleatoriedad pasa por generadores sembrados.
- Dependencias con rangos de version cerrados en `requirements.txt`.
- Pipelines de scikit-learn encapsulan preprocesamiento + modelo (cero fuga en CV).
- 23 pruebas automatizadas cubren geometria, particiones, metricas, OOF y el
  modelo bayesiano. Ejecutar con `make test`.

## Notebooks

Los notebooks en `notebooks/` reproducen el analisis paso a paso con explicacion
matematica. Se generan con `python build_notebooks.py` y se ejecutan con
`make notebooks` (requiere `pip install -e ".[notebooks]"`).

1. `01_data_acquisition` — adquisicion y estructura de datos.
2. `02_exploratory_data_analysis` — exploracion y mapas de disparo.
3. `03_feature_engineering` — geometria y variables derivadas.
4. `04_modeling_and_comparison` — comparacion de modelos y LOCO.
5. `05_calibration_and_interpretability` — calibracion e importancia.
6. `06_player_team_analysis` — rankings y contraste de finalizadores.
7. `07_bayesian_hierarchical_finishing` — modelo jerarquico y shrinkage.

## Licencia

MIT. Datos de StatsBomb sujetos a su licencia de uso de datos abiertos.
