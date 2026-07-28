# Registro de cambios

## v1.2.0 — Rigor metodológico avanzado (revisión crítica de tribunal)

Cambios orientados a resolver las objeciones de una evaluación exigente de
Ingeniería Matemática.

### Nuevo
- **Regresión logística con B-splines** (`logistic_splines`): rival lineal serio
  del *boosting*, captura la no linealidad geometría→probabilidad. Nuevo
  `features.build_spline_preprocessor`.
- **Validación Leave-One-Competition-Out** (`evaluation.leave_one_competition_out`):
  rota las cuatro competiciones como test y reporta media ± desviación típica
  entre torneos. Salida: `data/outputs/loco_validation.csv` y figuras
  `loco_roc_auc.png`, `loco_brier.png`.
- **Significancia en rankings de finalizadores**
  (`analysis.overperformers_with_significance`): contraste Poisson-binomial con
  p-valor e IC; distingue habilidad de ruido. Salida: `finishers_significance.csv`.
- **Benchmark vs. xG oficial de StatsBomb** (`analysis.benchmark_vs_statsbomb`):
  correlaciones, MAE y Brier comparado. Salida: `benchmark_vs_statsbomb.csv`.
- **Diagrama de fiabilidad con IC de Wilson** (`visualization.plot_reliability_with_ci`):
  muestra dónde la calibración es fiable y dónde es ruidosa. Figura
  `reliability_wilson.png`.
- **`ARCHITECTURE.md`**: mapa visual del flujo de datos y de módulos.
- Nuevas pruebas unitarias (suite de 19 en total).

### Mejorado
- Exclusión de penaltis ahora **justificada con el dato empírico** (se registra
  la conversión real en el log), no por convención.
- README: sección de decisiones metodológicas ampliada con todas las técnicas
  anteriores.

## v1.1.0 — Validación estadística, tests y resultados reales

Mejoras orientadas a rigor académico (revisión para TFM):

### Nuevo
- **`src/statistics.py`**: validación estadística rigurosa.
  - Intervalos de confianza por *bootstrap* (no paramétrico) para cualquier
    métrica (ROC-AUC, PR-AUC, Brier, Log Loss).
  - Comparación pareada por *bootstrap* entre dos modelos sobre el mismo test,
    con intervalo de confianza y p-valor empírico bilateral de la diferencia.
  - Tabla de fiabilidad (calibración por tramos) con intervalos de Wilson.
- **`tests/`**: suite de pruebas unitarias (`pytest`) sobre geometría,
  ingeniería de variables, métricas, partición por competición y estadística.
- **Paso 6b del pipeline**: la validación estadística se ejecuta y se exporta
  automáticamente (`test_metrics_bootstrap_ci.csv`,
  `model_comparison_paired_bootstrap.csv`, `reliability_table.csv`).
- **Sección 4.5** en el notebook 04 con la validación estadística e interpretación.

### Mejorado
- **Importancia por permutación** ahora usa ROC-AUC por defecto (interpretable)
  y se añade `permutation_importance_report` con AUC y Brier y su desviación
  típica entre repeticiones (estabilidad). Antes los valores en escala Brier
  resultaban diminutos y poco legibles.
- **README**: la sección de resultados refleja ahora los **resultados reales de
  StatsBomb** (test = Eurocopa 2024), claramente separados del modo sintético,
  con lectura honesta de por qué el boosting gana en calibración aunque la
  logística tenga mayor AUC.

### Notas
- Los CSV de la validación estadística y de importancia se **regeneran** al
  ejecutar `python run_pipeline.py --source statsbomb`. Tras esta actualización,
  basta una ejecución para producir todos los resultados de forma coherente.

## v1.0.0 — Versión inicial
- Pipeline xG end-to-end, ingeniería de variables, comparación de modelos,
  calibración, interpretabilidad, análisis jugador/equipo, notebooks y figuras.
- Partición de test por competición (holdout Eurocopa 2024) para medir
  generalización entre torneos.
