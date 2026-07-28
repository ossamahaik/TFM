"""
config.py
=========
Configuración central del proyecto de Expected Goals (xG).

Este módulo concentra TODAS las constantes, rutas, semillas y parámetros del
proyecto. El resto del código importa desde aquí, de modo que no exista ningún
valor "hardcodeado" disperso. Centralizar la configuración es una buena práctica
de ingeniería de software (single source of truth) y facilita la reproducibilidad
y la trazabilidad de los experimentos.

Decisiones de diseño
--------------------
* Todas las rutas se construyen de forma RELATIVA a la raíz del proyecto, que se
  resuelve dinámicamente a partir de la ubicación de este fichero. Esto permite
  ejecutar el proyecto desde cualquier directorio de trabajo (notebooks, scripts,
  CI) sin romper las rutas.
* Una única semilla global (`RANDOM_STATE`) gobierna toda la aleatoriedad del
  pipeline (splits, modelos, remuestreos). La reproducibilidad es un criterio
  académico explícito.
* Las dimensiones del campo siguen la convención de StatsBomb (120 x 80), lo que
  permite calcular variables geométricas (distancia y ángulo) de forma exacta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Rutas del proyecto (relativas, reproducibles)
# --------------------------------------------------------------------------- #
# Este fichero vive en src/xg/config.py -> la raíz del proyecto es parents[2]
# (src/xg/config.py -> src/xg -> src -> raíz). La resolución dinámica permite
# ejecutar desde notebooks, scripts o CI sin romper rutas.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
# Resultados tabulares (métricas, rankings, validación estadística).
OUTPUTS_DIR: Path = DATA_DIR / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
# Artefactos de modelos entrenados (.joblib).
MODELS_DIR: Path = PROJECT_ROOT / "artifacts" / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_dirs() -> None:
    """Crea (idempotentemente) todos los directorios de salida del proyecto."""
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Geometría del campo (convención StatsBomb)
# --------------------------------------------------------------------------- #
PITCH_LENGTH: float = 120.0   # eje x
PITCH_WIDTH: float = 80.0     # eje y
GOAL_WIDTH: float = 7.32      # metros reales de la portería (~8 yardas)
# La portería se sitúa en x = 120; sus postes en y = 36 y y = 44 (centro = 40)
GOAL_CENTER: Tuple[float, float] = (120.0, 40.0)
GOAL_POST_LEFT: Tuple[float, float] = (120.0, 36.0)
GOAL_POST_RIGHT: Tuple[float, float] = (120.0, 44.0)

# --------------------------------------------------------------------------- #
# Fuente de datos (StatsBomb Open Data)
# --------------------------------------------------------------------------- #
# Competiciones por defecto para el pipeline reproducible. Se eligen torneos
# completos y bien documentados de la StatsBomb Open Data. El usuario puede
# sobreescribir esta lista. (competition_id, season_id, etiqueta legible)
DEFAULT_COMPETITIONS: List[Dict] = [
    {"competition_id": 43, "season_id": 3, "label": "FIFA World Cup 2018"},
    {"competition_id": 55, "season_id": 43, "label": "UEFA Euro 2020"},
    {"competition_id": 43, "season_id": 106, "label": "FIFA World Cup 2022"},
    {"competition_id": 55, "season_id": 282, "label": "UEFA Euro 2024"},
]

# Competición reservada como conjunto de test "fuera de muestra" (holdout).
# El modelo se entrena y valida con el resto de torneos y se evalúa sobre este,
# que es posterior en el tiempo. Esto evalúa la capacidad de GENERALIZACIÓN del
# modelo a una competición nueva (validación temporal entre torneos), un
# criterio metodológicamente más exigente que una partición aleatoria.
# Poner a None para volver a la partición aleatoria estratificada clásica.
HOLDOUT_COMPETITION: str = "UEFA Euro 2024"

# Si no hay conexión / statsbombpy no está disponible, el data_loader genera un
# dataset sintético físicamente realista para que el pipeline sea ejecutable.
SYNTHETIC_N_MATCHES: int = 220          # nº de partidos simulados
SYNTHETIC_SHOTS_PER_MATCH: Tuple[int, int] = (18, 34)  # rango uniforme

# --------------------------------------------------------------------------- #
# Definición del problema de aprendizaje
# --------------------------------------------------------------------------- #
TARGET: str = "is_goal"

# Reparto train / validation / test. Se usa un test estrictamente reservado
# (hold-out) y validación cruzada estratificada sobre el train+val.
TEST_SIZE: float = 0.20
VALIDATION_SIZE: float = 0.20   # proporción del *resto* tras separar el test
CV_FOLDS: int = 5

# --------------------------------------------------------------------------- #
# Esquema de variables (feature schema)
# --------------------------------------------------------------------------- #
# Variables numéricas y categóricas que alimentarán el ColumnTransformer.
# Mantener el esquema aquí evita inconsistencias entre entrenamiento y scoring.
NUMERIC_FEATURES: List[str] = [
    "distance_to_goal",
    "shot_angle",
    "distance_x",
    "distance_y",
    "log_distance",
    "inverse_distance",
    "angle_distance_interaction",
    "minute",
    "n_defenders_in_cone",
    "n_defenders_within_3m",
    "distance_to_nearest_defender",
    "gk_distance_to_goal",
    "gk_distance_to_shot",
    "goal_diff",
    "shot_in_box",
]

CATEGORICAL_FEATURES: List[str] = [
    "body_part",
    "shot_type",
    "play_pattern",
    "under_pressure",
    "is_first_time",
    "assisted",
    "is_open_play",
    "game_state",
]

@dataclass(frozen=True)
class ModelSpec:
    """Especificación declarativa de un modelo a entrenar."""
    name: str
    needs_scaling: bool           # si requiere escalado de numéricas (modelos lineales)
    supports_native_nan: bool = False  # si tolera NaN sin imputación previa


# Catálogo de modelos a comparar. Los basados en árboles no necesitan escalado;
# la regresión logística sí. Esto se respeta en el ColumnTransformer por modelo.
MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec("logistic_regression", needs_scaling=True),
    ModelSpec("random_forest", needs_scaling=False),
    ModelSpec("gradient_boosting", needs_scaling=False),
    ModelSpec("hist_gradient_boosting", needs_scaling=False, supports_native_nan=True),
    # xgboost / lightgbm se añaden dinámicamente en models.py si están instalados.
]

# Métrica principal de selección de modelo. Se prioriza la calidad probabilística
# (Brier / LogLoss) porque xG es un problema de estimación de probabilidad, no de
# clasificación dura. ROC-AUC se reporta como medida de ranking.
PRIMARY_METRIC: str = "brier_score"   # menor es mejor

# --------------------------------------------------------------------------- #
# Estilo de figuras
# --------------------------------------------------------------------------- #
FIG_DPI: int = 200
FIG_FORMAT: str = "png"
SAVE_FIGURES: bool = True
