"""
models.py
=========
Definición, entrenamiento, calibración y persistencia de modelos xG.

Estrategia de modelado
----------------------
El xG es un problema de **estimación de probabilidad** sobre un evento raro
(la tasa base de gol ronda el 10-12 %). Por ello:

* Se comparan varias familias (lineal, bagging, boosting) bajo un protocolo
  idéntico, encapsulando preprocesamiento + estimador en un único `Pipeline`.
  Esto evita data leakage y hace que la validación cruzada sea honesta.
* La métrica de selección prioriza la calidad PROBABILÍSTICA (Brier/LogLoss),
  no la clasificación dura, porque interesa que p̂ ≈ P(gol) y no un 0/1.
* Se aplica **calibración** (Platt / isotónica) sobre el mejor modelo, ya que
  muchos clasificadores (RF, boosting) producen probabilidades sesgadas.
* `class_weight="balanced"` (donde el estimador lo soporta) atiende el
  desbalanceo sin recurrir a sobremuestreo agresivo, que tiende a empeorar la
  calibración.

El módulo es extensible: si `xgboost` o `lightgbm` están instalados, se añaden
automáticamente al catálogo; si no, se usa `HistGradientBoostingClassifier` de
scikit-learn (boosting por histogramas equivalente en espíritu a LightGBM).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xg import config, features

logger = logging.getLogger(__name__)

# Imports opcionales de boosting de terceros.
try:  # pragma: no cover
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except Exception:  # noqa: BLE001
    _HAS_XGB = False

try:  # pragma: no cover
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except Exception:  # noqa: BLE001
    _HAS_LGBM = False


# --------------------------------------------------------------------------- #
# Catálogo de estimadores
# --------------------------------------------------------------------------- #
def _base_estimators() -> Dict[str, object]:
    """Diccionario nombre -> estimador base (sin preprocesar)."""
    rs = config.RANDOM_STATE
    estimators: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=rs
        ),
        "logistic_splines": LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0, random_state=rs
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=rs,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=rs
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, max_depth=None, learning_rate=0.05,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=rs
        ),
    }
    if _HAS_XGB:
        estimators["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=rs, n_jobs=-1, tree_method="hist",
        )
    if _HAS_LGBM:
        estimators["lightgbm"] = LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=rs, n_jobs=-1, verbose=-1,
        )
    return estimators


def available_model_names() -> List[str]:
    return list(_base_estimators().keys())


def _needs_scaling(name: str) -> bool:
    """Solo los modelos lineales requieren escalado de las variables numéricas."""
    return name == "logistic_regression"


def build_model_pipeline(name: str) -> Pipeline:
    """
    Construye un Pipeline completo (preprocesador + estimador) para un modelo.

    Encapsular el preprocesamiento dentro del Pipeline es la práctica correcta:
    en validación cruzada, imputadores/escaladores/OHE se reajustan en cada fold
    usando únicamente los datos de entrenamiento de ese fold.

    El modelo ``logistic_splines`` recibe un preprocesador con expansión en
    B-splines de las variables numéricas (no linealidad explícita); el resto usan
    el preprocesador estándar (con escalado solo para la logística lineal).
    """
    estimators = _base_estimators()
    if name not in estimators:
        raise KeyError(f"Modelo desconocido: {name}. Disponibles: {list(estimators)}")
    if name == "logistic_splines":
        pre = features.build_spline_preprocessor()
    else:
        pre = features.build_preprocessor(scale_numeric=_needs_scaling(name))
    return Pipeline([("preprocess", pre), ("model", estimators[name])])


# --------------------------------------------------------------------------- #
# Calibración
# --------------------------------------------------------------------------- #
def calibrate_pipeline(
    fitted_or_unfitted: Pipeline, method: str = "isotonic", cv: int = 5
) -> CalibratedClassifierCV:
    """
    Envuelve un Pipeline en un calibrador.

    * "isotonic": no paramétrico, flexible; recomendado con suficientes datos.
    * "sigmoid" (Platt): paramétrico, robusto con pocos datos.

    `CalibratedClassifierCV` con `cv` realiza la calibración por validación
    cruzada interna, evitando reutilizar los mismos datos para ajustar y calibrar.
    """
    return CalibratedClassifierCV(estimator=fitted_or_unfitted, method=method, cv=cv)


# --------------------------------------------------------------------------- #
# Optimización de hiperparámetros (opcional, vía Optuna si está disponible)
# --------------------------------------------------------------------------- #
def tune_hist_gradient_boosting(X, y, n_trials: int = 40, cv: int = 3):
    """
    Optimiza HistGradientBoosting con Optuna si está instalado; si no, devuelve
    el Pipeline por defecto. La función está aislada para no acoplar el resto del
    pipeline a una dependencia opcional.
    """
    try:  # pragma: no cover
        import optuna
        from sklearn.model_selection import cross_val_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:  # noqa: BLE001
        logger.info("Optuna no disponible; se usa configuración por defecto.")
        return build_model_pipeline("hist_gradient_boosting")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 700),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
        }
        pre = features.build_preprocessor(scale_numeric=False)
        est = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, **params)
        pipe = Pipeline([("preprocess", pre), ("model", est)])
        return cross_val_score(pipe, X, y, cv=cv, scoring="neg_brier_score").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    pre = features.build_preprocessor(scale_numeric=False)
    best = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, **study.best_params)
    logger.info("Mejores hiperparámetros: %s", study.best_params)
    return Pipeline([("preprocess", pre), ("model", best)])


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def save_model(model, name: str) -> Path:
    """Serializa un modelo entrenado en models/ con joblib."""
    config.ensure_dirs()
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Modelo guardado en %s", path)
    return path


def load_model(name: str):
    """Carga un modelo serializado desde models/."""
    path = config.MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
