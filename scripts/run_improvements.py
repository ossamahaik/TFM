"""
run_improvements.py — Experimentos adicionales de refuerzo del TFM xG.

Genera CSV NUEVOS (no sobrescribe los canónicos) en ``data/results/``:

  1. ``model_comparison_calibration_ablation.csv`` — contrafáctico de calibración.
     Responde si la ventaja probabilística del boosting sobre la logística es un
     artefacto de ``class_weight`` o del modelo: compara la logística con y sin
     reponderado de clases, calibrada y sin calibrar, frente al HGB calibrado.

  2. ``feature_ablation_hgb.csv`` — ablation por familias de variables sobre el
     modelo seleccionado (HistGradientBoosting): geometría, +no lineales,
     +defensivas (freeze frames) y +contexto, de forma acumulativa. Cuantifica a
     nivel de MODELO (no solo por importancia de permutación) cuánto aporta cada
     familia, en particular la defensiva.

  3. ``ece_binning_comparison.csv`` — ECE del modelo calibrado en test con tramos
     uniformes frente a tramos por cuantiles, para no depender del binning.

  4. ``defensive_missingness.csv`` — cobertura real de valores ausentes de cada
     variable defensiva derivada de los freeze frames (antes de imputar).

Reutiliza EXACTAMENTE el mismo protocolo de datos y particiones que
``run_pipeline.py`` (fuente StatsBomb real, holdout ``UEFA Euro 2024``), de modo
que las cifras son comparables con las de la memoria. Determinista
(``config.RANDOM_STATE = 42``).

Ejecutar desde la raíz del repo:
    .venv/Scripts/python.exe scripts/run_improvements.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Permitir ejecución como script (añade src/ al path como hacen los tests).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg import models  # noqa: E402
from xg.models import evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("improvements")

CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "brier": float(evaluation.brier_score_loss(y, p)),
        "log_loss": float(evaluation.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "roc_auc": float(evaluation.roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": float(evaluation.average_precision_score(y, p)),
        "ece": float(evaluation.expected_calibration_error(y, p, n_bins=10)),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def ece_quantile(y, p, n_bins: int = 10) -> float:
    """ECE con tramos de igual frecuencia (cuantiles), robusto a distribuciones sesgadas."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    inner = edges[1:-1]
    idx = np.clip(np.digitize(p, inner), 0, n_bins - 1)
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _oof_proba(pipe, X, y) -> np.ndarray:
    """Probabilidades out-of-fold por CV estratificada (sin fuga: preprocesado en el pipeline)."""
    return cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]


def _std_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    return features.build_preprocessor(scale_numeric=scale_numeric)


def _logit(class_weight):
    return LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight,
                              random_state=config.RANDOM_STATE)


def _logit_pipeline(class_weight):
    return Pipeline([("preprocess", _std_preprocessor(scale_numeric=True)),
                     ("model", _logit(class_weight))])


def _hgb_pipeline():
    return models.build_model_pipeline("hist_gradient_boosting")


# --------------------------------------------------------------------------- #
# 1. Contrafáctico de calibración (C1)
# --------------------------------------------------------------------------- #
def calibration_ablation(Xtrval, ytrval, Xte, yte) -> pd.DataFrame:
    logger.info("== 1. Contrafáctico de calibración (logística ±class_weight ±calibración) ==")
    variants = {
        "logistica_balanced_sincal": lambda: _logit_pipeline("balanced"),
        "logistica_balanced_calibrada": lambda: CalibratedClassifierCV(
            _logit_pipeline("balanced"), method="isotonic", cv=config.CV_FOLDS),
        "logistica_sinpeso_sincal": lambda: _logit_pipeline(None),
        "logistica_sinpeso_calibrada": lambda: CalibratedClassifierCV(
            _logit_pipeline(None), method="isotonic", cv=config.CV_FOLDS),
        "hgb_calibrado": lambda: CalibratedClassifierCV(
            _hgb_pipeline(), method="isotonic", cv=config.CV_FOLDS),
    }
    rows = []
    for name, build in variants.items():
        # CV out-of-fold sobre train+val.
        p_cv = _oof_proba(build(), Xtrval, ytrval)
        m_cv = _metrics(ytrval, p_cv)
        m_cv.update({"variant": name, "split": "cv_oof"})
        rows.append(m_cv)
        # Test hold-out (UEFA Euro 2024): ajuste sobre train+val, evaluación en test.
        mdl = build()
        mdl.fit(Xtrval, ytrval)
        p_te = mdl.predict_proba(Xte)[:, 1]
        m_te = _metrics(yte, p_te)
        m_te.update({"variant": name, "split": "test"})
        rows.append(m_te)
        logger.info("  %-30s | test Brier=%.4f ECE=%.4f ROC=%.4f mean_pred=%.4f",
                    name, m_te["brier"], m_te["ece"], m_te["roc_auc"], m_te["mean_pred"])
    cols = ["variant", "split", "n", "base_rate", "mean_pred",
            "brier", "log_loss", "ece", "roc_auc", "pr_auc"]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# 2. Ablation por familias de variables (I2)
# --------------------------------------------------------------------------- #
FAMILIES = [
    ("geometria",
     ["distance_to_goal", "shot_angle", "distance_x", "distance_y"], []),
    ("+no_lineales",
     ["log_distance", "inverse_distance", "angle_distance_interaction", "shot_in_box"], []),
    ("+defensivas",
     ["n_defenders_in_cone", "n_defenders_within_3m", "distance_to_nearest_defender",
      "gk_distance_to_goal", "gk_distance_to_shot"], []),
    ("+contexto",
     ["minute", "goal_diff"],
     ["body_part", "shot_type", "play_pattern", "under_pressure",
      "is_first_time", "assisted", "is_open_play", "game_state"]),
]


def _subset_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def _hgb_estimator():
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=None, learning_rate=0.05, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=config.RANDOM_STATE)


def feature_ablation(Xtrval, ytrval) -> pd.DataFrame:
    logger.info("== 2. Ablation por familias de variables (HGB, CV out-of-fold) ==")
    num_cum, cat_cum, rows = [], [], []
    prev_brier = None
    for label, num_new, cat_new in FAMILIES:
        num_cum = num_cum + num_new
        cat_cum = cat_cum + cat_new
        pipe = Pipeline([("preprocess", _subset_preprocessor(num_cum, cat_cum)),
                         ("model", _hgb_estimator())])
        p = _oof_proba(pipe, Xtrval, ytrval)
        m = _metrics(ytrval, p)
        m["conjunto"] = label
        m["n_features"] = len(num_cum) + len(cat_cum)
        m["delta_brier"] = np.nan if prev_brier is None else m["brier"] - prev_brier
        prev_brier = m["brier"]
        rows.append(m)
        logger.info("  %-14s (%2d vars) | Brier=%.4f ROC=%.4f PR=%.4f ECE=%.4f | dBrier=%s",
                    label, m["n_features"], m["brier"], m["roc_auc"], m["pr_auc"], m["ece"],
                    "---" if np.isnan(m["delta_brier"]) else f"{m['delta_brier']:+.4f}")
    cols = ["conjunto", "n_features", "brier", "delta_brier", "log_loss",
            "roc_auc", "pr_auc", "ece", "mean_pred", "base_rate", "n"]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# 3. ECE uniforme vs cuantiles (M1)
# --------------------------------------------------------------------------- #
def ece_binning(Xtrval, ytrval, Xte, yte) -> pd.DataFrame:
    logger.info("== 3. ECE uniforme vs cuantiles (HGB calibrado, test) ==")
    cal = CalibratedClassifierCV(_hgb_pipeline(), method="isotonic", cv=config.CV_FOLDS)
    cal.fit(Xtrval, ytrval)
    p = cal.predict_proba(Xte)[:, 1]
    rows = []
    for nb in (10, 15):
        rows.append({"n_bins": nb, "modelo": "hgb_calibrado", "split": "test",
                     "ece_uniforme": evaluation.expected_calibration_error(yte, p, n_bins=nb),
                     "ece_cuantiles": ece_quantile(yte, p, n_bins=nb)})
        logger.info("  n_bins=%d | ECE uniforme=%.4f | ECE cuantiles=%.4f",
                    nb, rows[-1]["ece_uniforme"], rows[-1]["ece_cuantiles"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Cobertura de ausentes de variables defensivas
# --------------------------------------------------------------------------- #
def defensive_missingness(feat: pd.DataFrame) -> pd.DataFrame:
    logger.info("== 4. Cobertura de ausentes de variables defensivas ==")
    nonpen = feat[feat["is_penalty"] == 0] if "is_penalty" in feat.columns else feat
    defensive = ["n_defenders_in_cone", "n_defenders_within_3m",
                 "distance_to_nearest_defender", "gk_distance_to_goal", "gk_distance_to_shot"]
    rows = []
    n = len(nonpen)
    for col in defensive:
        n_missing = int(nonpen[col].isna().sum()) if col in nonpen.columns else n
        rows.append({"variable": col, "n": n, "n_ausentes": n_missing,
                     "pct_ausentes": round(100.0 * n_missing / n, 3)})
        logger.info("  %-30s | ausentes=%d (%.3f%%)", col, n_missing, 100.0 * n_missing / n)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.perf_counter()
    np.random.seed(config.RANDOM_STATE)
    config.ensure_dirs()

    logger.info("Cargando datos StatsBomb reales (caché local)...")
    raw, used = data_loader.build_shots_dataframe(source="statsbomb")
    logger.info("Fuente: %s | disparos brutos: %d", used, len(raw))
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(feat, drop_penalties=True, return_competition=True)
    (_, _), (_, _), (Xte, yte), (Xtrval, ytrval) = \
        evaluation.make_competition_holdout_splits(X, y, comp, config.HOLDOUT_COMPETITION)
    logger.info("train+val=%d | test(%s)=%d | tasa gol test=%.4f",
                len(Xtrval), config.HOLDOUT_COMPETITION, len(Xte), float(yte.mean()))

    out = config.OUTPUTS_DIR

    df1 = calibration_ablation(Xtrval, ytrval, Xte, yte)
    df1.to_csv(out / "model_comparison_calibration_ablation.csv", index=False)

    df2 = feature_ablation(Xtrval, ytrval)
    df2.to_csv(out / "feature_ablation_hgb.csv", index=False)

    df3 = ece_binning(Xtrval, ytrval, Xte, yte)
    df3.to_csv(out / "ece_binning_comparison.csv", index=False)

    df4 = defensive_missingness(feat)
    df4.to_csv(out / "defensive_missingness.csv", index=False)

    elapsed = time.perf_counter() - t0
    logger.info("== Hecho en %.1f s. CSV escritos en %s ==", elapsed, out)
    print("\n=== 1. Contrafáctico de calibración ===")
    print(df1.round(4).to_string(index=False))
    print("\n=== 2. Ablation por familias (HGB, CV) ===")
    print(df2.round(4).to_string(index=False))
    print("\n=== 3. ECE uniforme vs cuantiles ===")
    print(df3.round(4).to_string(index=False))
    print("\n=== 4. Ausentes defensivas ===")
    print(df4.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
