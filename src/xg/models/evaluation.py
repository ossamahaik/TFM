"""
evaluation.py
=============
Evaluación rigurosa y comparación de modelos xG.

Filosofía de evaluación
-----------------------
Un buen modelo xG no es el que mejor "clasifica" goles, sino el que mejor
**estima la probabilidad** de gol. Por eso el conjunto de métricas combina:

* **Discriminación / ranking**: ROC-AUC y PR-AUC (Average Precision). Miden si
  el modelo ordena correctamente disparos peligrosos frente a inocuos. PR-AUC es
  especialmente informativa con clases desbalanceadas como aquí.

* **Calidad probabilística (proper scoring rules)**: Log Loss y Brier Score.
  Son *proper scoring rules*: se minimizan cuando p̂ coincide con la
  probabilidad real, penalizando tanto el sesgo como la sobreconfianza.

* **Calibración**: curva de fiabilidad y **ECE** (Expected Calibration Error).
  Un xG bien calibrado cumple que, entre los disparos con xG≈0.3, alrededor del
  30 % acaban en gol. La calibración es CRÍTICA para que el xG agregado por
  jugador/equipo sea interpretable.

Toda la evaluación se realiza sobre un test hold-out reservado y, en paralelo,
mediante validación cruzada estratificada para estimar la varianza.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from xg import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Particiones reproducibles
# --------------------------------------------------------------------------- #
def make_splits(X: pd.DataFrame, y: pd.Series):
    """
    Genera particiones train / validation / test estratificadas y reproducibles.

    Primero se reserva el test (hold-out). Del resto se separa la validación.
    La estratificación mantiene la tasa de gol en las tres particiones, lo que
    es esencial dado el desbalanceo.
    """
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=config.VALIDATION_SIZE,
        stratify=y_tr_val, random_state=config.RANDOM_STATE,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_tr_val, y_tr_val)


def make_competition_holdout_splits(
    X: pd.DataFrame,
    y: pd.Series,
    competition: pd.Series,
    holdout: str,
):
    """
    Particiona reservando como test TODOS los disparos de una competición.

    A diferencia de :func:`make_splits` (partición aleatoria), aquí el conjunto
    de test es una competición completa que el modelo no ve durante el
    entrenamiento. Esto evalúa la **generalización entre torneos** (validación
    temporal/externa), un criterio metodológicamente más exigente y realista:
    reproduce el uso en producción, donde se entrena con el pasado y se predice
    sobre una competición futura.

    El resto de competiciones se reparte en train/validation de forma
    estratificada para preservar la tasa de gol.

    Parameters
    ----------
    X, y : matriz de diseño y target alineados por índice.
    competition : Serie con la etiqueta de competición de cada disparo (mismo
        índice que ``X``).
    holdout : etiqueta de la competición reservada como test.

    Returns
    -------
    Igual estructura que :func:`make_splits`:
    ``(X_train, y_train), (X_val, y_val), (X_test, y_test), (X_tr_val, y_tr_val)``
    """
    competition = competition.reindex(X.index)
    is_holdout = competition == holdout
    if is_holdout.sum() == 0:
        raise ValueError(
            f"No se han encontrado disparos de la competición de test '{holdout}'. "
            f"Competiciones disponibles: {sorted(competition.dropna().unique())}"
        )
    if (~is_holdout).sum() == 0:
        raise ValueError(
            f"Todos los disparos pertenecen a '{holdout}'; no quedan datos para "
            "entrenar. Revisa DEFAULT_COMPETITIONS y HOLDOUT_COMPETITION."
        )

    X_test, y_test = X[is_holdout], y[is_holdout]
    X_tr_val, y_tr_val = X[~is_holdout], y[~is_holdout]

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=config.VALIDATION_SIZE,
        stratify=y_tr_val, random_state=config.RANDOM_STATE,
    )
    logger.info(
        "Holdout por competición '%s': test=%d (tasa gol %.4f) | "
        "train+val=%d (tasa gol %.4f)",
        holdout, len(y_test), float(y_test.mean()),
        len(y_tr_val), float(y_tr_val.mean()),
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_tr_val, y_tr_val)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    ECE: media ponderada (por nº de muestras en cada bin) de |confianza - acierto|.
    Cuantifica en una sola cifra el desajuste de calibración. Menor es mejor.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = np.mean(y_prob[mask])
        acc = np.mean(y_true[mask])
        ece += (np.sum(mask) / n) * abs(conf - acc)
    return float(ece)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula el panel completo de métricas para un conjunto de predicciones."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-15
    y_prob_clip = np.clip(y_prob, eps, 1 - eps)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "base_rate": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_prob)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom_p = (tp + fp) or 1
    denom_r = (tp + fn) or 1
    metrics["precision"] = tp / denom_p
    metrics["recall"] = tp / denom_r
    metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"]
        / ((metrics["precision"] + metrics["recall"]) or 1)
    )
    return metrics


# --------------------------------------------------------------------------- #
# Validación cruzada
# --------------------------------------------------------------------------- #
def cross_validated_probabilities(pipeline: Pipeline, X, y, n_splits: int = config.CV_FOLDS):
    """
    Probabilidades fuera de muestra (out-of-fold) por validación cruzada
    estratificada. Permite estimar métricas sin optimismo de re-sustitución.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba", n_jobs=-1)
    return oof[:, 1]


def evaluate_models_cv(
    pipelines: Dict[str, Pipeline], X, y, return_probabilities: bool = False,
):
    """
    Evalúa cada modelo por CV out-of-fold y devuelve una tabla comparativa
    ordenada por la métrica principal (Brier, menor es mejor).

    Si ``return_probabilities`` es True, también devuelve un diccionario
    ``{modelo: (y_true, y_prob_oof)}`` con las probabilidades agregadas de
    cada modelo, reutilizables para análisis posteriores (p. ej. IC bootstrap)
    sin repetir la validación cruzada.
    """
    rows: List[Dict] = []
    oof_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    y_arr = np.asarray(y)
    for name, pipe in pipelines.items():
        logger.info("Evaluando por CV: %s", name)
        prob = cross_validated_probabilities(pipe, X, y)
        m = compute_metrics(y, prob)
        m["model"] = name
        rows.append(m)
        oof_results[name] = (y_arr, prob)
    df = pd.DataFrame(rows).set_index("model")
    ascending = True  # brier/log_loss: menor mejor
    df = df.sort_values(config.PRIMARY_METRIC, ascending=ascending)
    if return_probabilities:
        return df, oof_results
    return df


def evaluate_on_test(model, X_test, y_test, name: str) -> Tuple[Dict[str, float], np.ndarray]:
    """Evalúa un modelo ya entrenado sobre el test hold-out."""
    prob = model.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, prob)
    m["model"] = name
    return m, prob


def out_of_fold_scoring(
    build_pipeline_fn,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = config.CV_FOLDS,
    calibrate: bool = False,
    calibration_method: str = "isotonic",
) -> np.ndarray:
    """
    Asigna a CADA disparo una probabilidad xG estimada por un modelo que NO lo
    vio en entrenamiento (predicción *out-of-fold*).

    Motivación metodológica
    ------------------------
    El análisis aplicado (rankings de xG por jugador/equipo, sobre/infra-
    rendimiento en finalización) requiere puntuar TODOS los disparos del dataset,
    no solo el test. Si para ello se usara un único modelo entrenado y luego se
    re-puntearan sus propios datos de entrenamiento, el xG de esos disparos
    estaría sesgado por sobreajuste (optimismo de re-sustitución): la suma de xG
    se aproximaría artificialmente a los goles reales y los rankings de
    finalización quedarían comprimidos hacia cero. Eso contaminaría justamente la
    magnitud que se desea medir (goles − xG).

    La solución correcta es la validación cruzada *out-of-fold*: se reparte el
    dataset en ``n_splits`` particiones estratificadas; para cada disparo, su xG
    se predice con el modelo entrenado en los pliegues restantes. Así, todas las
    predicciones son fuera de muestra y el análisis agregado es honesto.

    Parameters
    ----------
    build_pipeline_fn : callable
        Función ``model_name -> Pipeline`` sin entrenar (``models.build_model_pipeline``).
    model_name : nombre del modelo a construir en cada pliegue.
    X, y : matriz de diseño y target alineados por índice.
    calibrate : si True, calibra el modelo dentro de cada pliegue de entrenamiento
        (la calibración usa su propia CV interna sobre el train del pliegue, sin
        ver el fold de test). Recomendado para que el xG agregado esté calibrado.

    Returns
    -------
    np.ndarray con la probabilidad xG out-of-fold de cada fila de ``X``, en el
    mismo orden que ``X``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    oof = np.full(len(X), np.nan, dtype=float)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_reset, y_reset)):
        pipe = build_pipeline_fn(model_name)
        if calibrate:
            from xg.models import catalog as _models  # import diferido (catalog: API de modelos)
            pipe = _models.calibrate_pipeline(pipe, method=calibration_method, cv=config.CV_FOLDS)
        pipe.fit(X_reset.iloc[tr_idx], y_reset.iloc[tr_idx])
        oof[te_idx] = pipe.predict_proba(X_reset.iloc[te_idx])[:, 1]
        logger.info("OOF scoring | pliegue %d/%d completado", fold + 1, n_splits)
    return oof


# --------------------------------------------------------------------------- #
# Validación Leave-One-Competition-Out (LOCO)
# --------------------------------------------------------------------------- #
def leave_one_competition_out(
    build_pipeline_fn,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    competition: pd.Series,
    calibrate: bool = False,
    calibration_method: str = "isotonic",
) -> pd.DataFrame:
    """
    Validación cruzada dejando fuera una competición completa en cada iteración.

    En cada pliegue, una competición actúa como test y las restantes como
    entrenamiento. A diferencia de un único *holdout*, este esquema estima la
    variabilidad del rendimiento ENTRE torneos (la fuente de incertidumbre
    dominante en xG), no solo el muestreo dentro de un torneo. Es el análogo a
    *leave-one-group-out* y proporciona una evaluación de generalización mucho
    más robusta.

    Parameters
    ----------
    build_pipeline_fn : callable
        Función que recibe ``model_name`` y devuelve un Pipeline sin entrenar
        (típicamente ``models.build_model_pipeline``).
    model_name : nombre del modelo a construir en cada pliegue.
    X, y, competition : datos y etiqueta de competición alineados por índice.
    calibrate : si True, calibra el modelo en cada pliegue (requiere
        ``models.calibrate_pipeline``; se importa de forma diferida).

    Returns
    -------
    DataFrame con una fila por competición de test y sus métricas, más una fila
    "MEDIA" y otra "DESV_TIP" que resumen el rendimiento medio y su dispersión
    entre torneos.
    """
    competition = competition.reindex(X.index)
    comps = sorted(competition.dropna().unique())
    rows = []
    for held in comps:
        test_mask = competition == held
        if test_mask.sum() == 0 or (~test_mask).sum() == 0:
            continue
        X_tr, y_tr = X[~test_mask], y[~test_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        if y_te.nunique() < 2 or y_tr.nunique() < 2:
            logger.warning("Pliegue '%s' omitido (una sola clase).", held)
            continue

        pipe = build_pipeline_fn(model_name)
        if calibrate:
            from xg.models import catalog as _models  # import diferido (catalog: API de modelos)
            pipe = _models.calibrate_pipeline(pipe, method=calibration_method, cv=config.CV_FOLDS)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, prob)
        m["test_competition"] = held
        m["n_test"] = int(test_mask.sum())
        rows.append(m)

    result = pd.DataFrame(rows).set_index("test_competition")
    numeric_cols = result.select_dtypes("number").columns
    summary_mean = result[numeric_cols].mean()
    summary_std = result[numeric_cols].std(ddof=1)
    result.loc["MEDIA"] = summary_mean
    result.loc["DESV_TIP"] = summary_std
    return result


# --------------------------------------------------------------------------- #
# Utilidades para curvas (consumidas por visualization.py)
# --------------------------------------------------------------------------- #
def roc_points(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def pr_points(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return recall, precision


def calibration_points(y_true, y_prob, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def save_metrics_table(df: pd.DataFrame, name: str):
    """Persiste una tabla de métricas en data/outputs como CSV."""
    config.ensure_dirs()
    path = config.OUTPUTS_DIR / f"{name}.csv"
    df.to_csv(path)
    logger.info("Métricas guardadas en %s", path)
    return path
