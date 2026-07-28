"""
run_all_models_test.py — Evaluación en test de LOS CINCO modelos comparados.

Motivación
----------
La comparación por validación cruzada cubre los cinco modelos del catálogo, pero
la evaluación sobre la competición no vista (UEFA Euro 2024) solo se reportaba
para cuatro configuraciones. Este script cierra esa laguna: entrena los cinco
modelos sobre train+val y los evalúa en el test hold-out, en dos variantes
(sin calibrar y calibrado por regresión isotónica), de modo que la memoria pueda
presentar la evidencia completa y ordenada de todos los modelos mencionados.

Salidas (CSV nuevos en ``data/results/``):
  - all_models_test.csv        : 5 modelos x {sin calibrar, calibrado} en test.
  - all_models_test_ci.csv     : IC bootstrap 95 % del Brier de cada configuración.

Determinista (config.RANDOM_STATE = 42). Ejecutar desde la raíz del repo:
    .venv/Scripts/python.exe scripts/run_all_models_test.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg import models  # noqa: E402
from xg.models import evaluation, statistics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("all_models_test")

# Orden de presentación: el mismo que sigue la exposición teórica del capítulo 2
# (de menor a mayor flexibilidad), para que la memoria pueda ordenar los
# resultados igual que la teoría.
MODEL_ORDER = [
    ("logistic_regression", "Regresión logística"),
    ("logistic_splines", "Logística + B-splines"),
    ("random_forest", "Random forest"),
    ("gradient_boosting", "Gradient boosting"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
]


def _metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "roc_auc": float(evaluation.roc_auc_score(y, p)),
        "pr_auc": float(evaluation.average_precision_score(y, p)),
        "brier": float(evaluation.brier_score_loss(y, p)),
        "ece": float(evaluation.expected_calibration_error(y, p, n_bins=10)),
        "log_loss": float(evaluation.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "mean_pred": float(p.mean()),
    }


def main() -> None:
    t0 = time.perf_counter()
    np.random.seed(config.RANDOM_STATE)
    config.ensure_dirs()

    raw, used = data_loader.build_shots_dataframe(source="statsbomb")
    logger.info("Fuente: %s | disparos brutos: %d", used, len(raw))
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(feat, drop_penalties=True, return_competition=True)
    (_, _), (_, _), (Xte, yte), (Xtrval, ytrval) = \
        evaluation.make_competition_holdout_splits(X, y, comp, config.HOLDOUT_COMPETITION)
    logger.info("train+val=%d | test=%d (tasa gol %.4f)", len(Xtrval), len(Xte), float(yte.mean()))

    rows, ci_rows = [], []
    for key, label in MODEL_ORDER:
        for calibrado in (False, True):
            pipe = models.build_model_pipeline(key)
            if calibrado:
                mdl = CalibratedClassifierCV(pipe, method="isotonic", cv=config.CV_FOLDS)
            else:
                mdl = pipe
            mdl.fit(Xtrval, ytrval)
            p = mdl.predict_proba(Xte)[:, 1]

            m = _metrics(yte, p)
            m["modelo"] = label
            m["calibrado"] = "sí" if calibrado else "no"
            m["clave"] = key
            rows.append(m)

            ci = statistics.bootstrap_metric_ci(
                np.asarray(yte).astype(int), p, metric="brier_score", n_boot=2000
            )
            ci_rows.append({
                "modelo": label, "calibrado": "sí" if calibrado else "no",
                "brier": m["brier"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            })
            logger.info(
                "  %-24s cal=%-2s | Brier=%.4f ECE=%.4f ROC=%.4f PR=%.4f media=%.4f",
                label, "sí" if calibrado else "no",
                m["brier"], m["ece"], m["roc_auc"], m["pr_auc"], m["mean_pred"],
            )

    cols = ["modelo", "calibrado", "clave", "roc_auc", "pr_auc", "brier", "ece",
            "log_loss", "mean_pred"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(config.OUTPUTS_DIR / "all_models_test.csv", index=False)
    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(config.OUTPUTS_DIR / "all_models_test_ci.csv", index=False)

    logger.info("== Hecho en %.1f s ==", time.perf_counter() - t0)
    print("\n=== Los cinco modelos en test (UEFA Euro 2024) ===")
    print(df.round(4).to_string(index=False))
    print("\n=== IC bootstrap 95 % del Brier ===")
    print(df_ci.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
