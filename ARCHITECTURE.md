# Arquitectura del proyecto

Este documento describe la organización del paquete `xg` y el flujo de datos del
pipeline. Para la guia de uso, ver `README.md`; para el contenido academico, la
memoria del TFM es la referencia autoritativa.

## Organizacion del paquete (src layout)

El proyecto es un paquete Python instalable (`pip install -e .`) bajo `src/xg/`,
con separacion de responsabilidades por subpaquetes:

```
src/xg/
├── config.py            Fuente unica de verdad: rutas, semilla, geometria,
│                        esquema de variables, catalogo de modelos.
├── utils/
│   └── io.py            E/S y cache (parquet con respaldo automatico a pickle).
├── data/
│   └── loader.py        Adquisicion: StatsBomb (con freeze frames) y generador
│                        sintetico con xG-oraculo conocido.
├── features/
│   └── engineering.py   Geometria (distancia, angulo), transformaciones no
│                        lineales, contexto tactico, game_state; preprocesadores
│                        estandar y de B-splines.
├── models/
│   ├── catalog.py       Catalogo de modelos como pipelines de scikit-learn.
│   ├── evaluation.py    Particiones (holdout por competicion, LOCO), scoring
│                        out-of-fold, metricas (Brier, ECE, Log Loss).
│   ├── statistics.py    Bootstrap, comparacion pareada, intervalos de Wilson.
│   └── bayesian.py      Modelo jerarquico de finalizacion (EM+Laplace; MCMC opc.)
├── bayes/
│   └── finishing.py     Implementacion alternativa Beta-Binomial conjugada.
├── analysis/
│   └── aggregation.py   Rankings jugador/equipo, contraste Poisson-binomial,
│                        benchmark vs xG oficial, importancia de variables.
├── visualization/
│   └── plots.py         18 figuras reproducibles (dibujante de campo propio).
├── pipeline.py          Orquestador end-to-end de 9 pasos.
└── cli.py               Interfaz de linea de comandos (xg-pipeline).
```

## Flujo de datos (de extremo a extremo)

```
        FUENTES                StatsBomb Open Data  |  Generador sintetico
           |                          (data/loader.py)
           v
   data/loader.py        ->  shots_raw  (una fila por disparo + competition)
           |
           v
 features/engineering.py ->  shots_features (geometria + contexto + game_state)
           |
           v
      Particiones         ->  holdout por competicion  (models/evaluation.py)
           |
           +--> Comparacion CV (6 modelos, Brier)  +  LOCO (media +/- DT)
           |
           v
  Mejor modelo + calibracion isotonica  (HistGradientBoosting)
           |
           +--> Evaluacion en test  (bootstrap 2000 + Wilson)
           +--> Interpretabilidad   (permutacion + SHAP)
           |
           v
    Scoring out-of-fold de TODOS los disparos  (honesto, sin re-sustitucion)
           |
           +--> Rankings jugador/equipo
           +--> Finalizadores: Poisson-binomial (frecuentista)
           +--> Finalizadores: jerarquico bayesiano (shrinkage)  [models/bayesian.py]
           +--> Benchmark vs xG oficial de StatsBomb
           |
           v
       ENTREGABLES   artifacts/models/ (.joblib)  ·  data/results/ (.csv)
                     figures/ (.png)              ·  data/processed/ (datos)
```

## Principios de diseno

- **Fuente unica de verdad.** Toda la configuracion (rutas, semilla, esquema,
  catalogo de modelos) vive en `config.py`. Ningun otro modulo fija constantes.
- **Cero fuga de informacion.** Todo el preprocesamiento se encapsula en
  pipelines de scikit-learn, de modo que se ajusta solo con los datos de
  entrenamiento dentro de cada pliegue de validacion cruzada.
- **Reproducibilidad.** Semilla global, generadores sembrados, dependencias con
  rangos cerrados y 23 pruebas automatizadas.
- **Portabilidad.** El dibujante de campo es propio (sin mplsoccer); las
  dependencias pesadas (xgboost, shap, pymc) son opcionales y el codigo degrada
  con elegancia a alternativas nativas si no estan presentes.
- **Honestidad metodologica.** El scoring del analisis aplicado es out-of-fold;
  el game_state respeta el orden cronologico real; el rival lineal incluye
  B-splines para que la comparacion con el boosting sea justa.
