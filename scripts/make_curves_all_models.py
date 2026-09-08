"""
make_curves_all_models.py — Curvas ROC y Precision-Recall de LOS CINCO modelos.

Motivacion
----------
Las figuras de curvas ROC y PR del pipeline solo trazaban tres modelos
(HistGradientBoosting con y sin calibrar, logistica con splines y logistica),
de modo que el random forest y el gradient boosting quedaban fuera de la
evidencia grafica pese a estar presentados en la teoria y en las tablas. Este
script las regenera con las cinco configuraciones calibradas, en el mismo orden
expositivo del capitulo 2, para que texto, tablas y figuras cubran el mismo
conjunto de modelos.

Salidas (sobrescribe las figuras del proyecto):
  - figures/roc_curves.png
  - figures/pr_curves.png

Determinista (config.RANDOM_STATE = 42). Ejecutar desde la raiz del repo:
    .venv/Scripts/python.exe scripts/make_curves_all_models.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg import models  # noqa: E402
from xg import visualization  # noqa: E402
from xg.models import evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("curvas")

# Orden expositivo del capitulo 2 (seccion 2.5.1).
MODEL_ORDER = [
    ("logistic_regression", "Regresión logística"),
    ("logistic_splines", "Logística + B-splines"),
    ("random_forest", "Random forest"),
    ("gradient_boosting", "Gradient boosting"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
]


def main() -> None:
    np.random.seed(config.RANDOM_STATE)
    config.ensure_dirs()

    raw, used = data_loader.build_shots_dataframe(source="statsbomb")
    logger.info("Fuente: %s | disparos: %d", used, len(raw))
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(feat, drop_penalties=True, return_competition=True)
    (_, _), (_, _), (Xte, yte), (Xtrval, ytrval) = \
        evaluation.make_competition_holdout_splits(X, y, comp, config.HOLDOUT_COMPETITION)

    results = {}
    for key, label in MODEL_ORDER:
        cal = CalibratedClassifierCV(
            models.build_model_pipeline(key), method="isotonic", cv=config.CV_FOLDS
        )
        cal.fit(Xtrval, ytrval)
        prob = cal.predict_proba(Xte)[:, 1]
        results[f"{label} (calibrado)"] = (np.asarray(yte), prob)
        logger.info("  %-24s trazado", label)

    base_rate = float(np.asarray(yte).mean())
    visualization.plot_roc_curves(results, save_as="roc_curves")
    visualization.plot_pr_curves(results, base_rate, save_as="pr_curves")
    logger.info("Figuras regeneradas con los 5 modelos (tasa base %.4f)", base_rate)


if __name__ == "__main__":
    main()
