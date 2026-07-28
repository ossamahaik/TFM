"""
statistics.py
=============
Validación estadística y probabilística rigurosa de los modelos xG.

La comparación de modelos por un único valor puntual de una métrica es
insuficiente para un análisis serio: no informa sobre la **incertidumbre** de
esa estimación ni sobre si las diferencias entre modelos son **estadísticamente
distinguibles** del ruido. Este módulo añade:

* **Intervalos de confianza por bootstrap** para cualquier métrica, sin asumir
  una distribución paramétrica (apropiado para AUC, Brier, ECE, etc.).
* **Comparación pareada por bootstrap** entre dos modelos sobre el mismo
  conjunto de test, que estima la distribución de la *diferencia* de métrica y
  el correspondiente intervalo de confianza y p-valor empírico.
* **Tabla de fiabilidad (calibración por tramos)** con frecuencias observadas
  frente a probabilidad media predicha e intervalos de Wilson por bin.

Todos los procedimientos son reproducibles vía ``config.RANDOM_STATE``.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from xg import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Métricas individuales (envoltorios robustos para el bootstrap)
# --------------------------------------------------------------------------- #
def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    return float(log_loss(y_true, p, labels=[0, 1]))


METRIC_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": _safe_auc,
    "pr_auc": lambda yt, yp: float(average_precision_score(yt, yp)),
    "brier_score": lambda yt, yp: float(brier_score_loss(yt, yp)),
    "log_loss": _safe_logloss,
}


# --------------------------------------------------------------------------- #
# Intervalos de confianza por bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_metric_ci(
    y_true,
    y_prob,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Intervalo de confianza por bootstrap (remuestreo con reemplazo) para una
    métrica. Devuelve estimación puntual, límites del IC y error estándar.

    El bootstrap no paramétrico es apropiado aquí porque métricas como el AUC o
    el Brier no tienen una distribución muestral sencilla con datos desbalanceados.
    """
    if metric not in METRIC_FUNCS:
        raise ValueError(f"Métrica no soportada: {metric}. Opciones: {list(METRIC_FUNCS)}")
    func = METRIC_FUNCS[metric]
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(config.RANDOM_STATE)

    point = func(y_true, y_prob)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = func(y_true[idx], y_prob[idx])
    boots = boots[~np.isnan(boots)]

    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "estimate": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "std_error": float(np.std(boots, ddof=1)),
        "n_boot": int(len(boots)),
        "confidence": 1 - alpha,
    }


def bootstrap_metrics_table(
    y_true, y_prob, metrics=("roc_auc", "pr_auc", "brier_score", "log_loss"),
    n_boot: int = 2000, alpha: float = 0.05,
) -> pd.DataFrame:
    """Tabla con estimación puntual e IC bootstrap para varias métricas."""
    rows = [bootstrap_metric_ci(y_true, y_prob, m, n_boot, alpha) for m in metrics]
    return pd.DataFrame(rows).set_index("metric")


# --------------------------------------------------------------------------- #
# Comparación pareada de dos modelos
# --------------------------------------------------------------------------- #
def paired_bootstrap_comparison(
    y_true,
    y_prob_a,
    y_prob_b,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compara dos modelos (A y B) sobre el MISMO test mediante bootstrap pareado.

    En cada réplica se remuestrean los índices una sola vez y se evalúan ambos
    modelos sobre la misma muestra, de modo que la diferencia controla la
    variabilidad común del conjunto de test. Se reporta la diferencia media
    (A − B), su IC y un p-valor empírico bilateral para H0: diferencia = 0.

    Para métricas donde "mejor" es menor (brier_score, log_loss), una diferencia
    A − B negativa indica que A es mejor.
    """
    func = METRIC_FUNCS[metric]
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(y_prob_a, dtype=float)
    b = np.asarray(y_prob_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(config.RANDOM_STATE)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        diffs[i] = func(yt, a[idx]) - func(yt, b[idx])
    diffs = diffs[~np.isnan(diffs)]

    point = func(y_true, a) - func(y_true, b)
    lo = float(np.percentile(diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # p-valor empírico bilateral: proporción de réplicas que cruzan el cero
    # respecto al signo de la diferencia observada.
    p_two_sided = 2.0 * min(
        np.mean(diffs <= 0.0), np.mean(diffs >= 0.0)
    )
    p_two_sided = float(min(1.0, p_two_sided))
    return {
        "metric": metric,
        "diff_a_minus_b": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_two_sided,
        "significant_at_alpha": bool(lo > 0 or hi < 0),
        "confidence": 1 - alpha,
    }


# --------------------------------------------------------------------------- #
# Tabla de fiabilidad (calibración por tramos) con IC de Wilson
# --------------------------------------------------------------------------- #
def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de Wilson para una proporción (más fiable que el normal con n pequeño)."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """
    Tabla de fiabilidad: por cada tramo de probabilidad predicha, nº de disparos,
    probabilidad media predicha, frecuencia observada de gol e IC de Wilson.

    Es la base numérica de la curva de calibración: un modelo bien calibrado
    tiene la frecuencia observada dentro del IC alrededor de la diagonal.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(y_true[mask].sum())
        lo, hi = _wilson_interval(k, n)
        rows.append({
            "bin": f"[{edges[b]:.1f}, {edges[b+1]:.1f})",
            "n_shots": n,
            "mean_pred": float(y_prob[mask].mean()),
            "obs_freq": k / n,
            "obs_ci_low": lo,
            "obs_ci_high": hi,
        })
    return pd.DataFrame(rows)
