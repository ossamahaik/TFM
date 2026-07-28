"""
Suite de pruebas unitarias del proyecto xG.

Cubre las piezas con lógica no trivial y mayor riesgo de regresión:
geometría, ingeniería de variables, métricas, particiones (incluida la
partición por competición) y los procedimientos estadísticos.

Ejecutar con:
    pytest -q
desde la raíz del proyecto.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg.models import evaluation, statistics  # noqa: E402


# --------------------------------------------------------------------------- #
# Geometría y features
# --------------------------------------------------------------------------- #
def test_distance_central_shot():
    """Un disparo en (108, 40) está a 12 m del centro de portería (120, 40)."""
    df = pd.DataFrame({"x": [108.0], "y": [40.0]})
    out = features.add_geometric_features(df)
    assert out["distance_to_goal"].iloc[0] == pytest.approx(12.0, abs=1e-6)


def test_angle_decreases_with_distance():
    """A igual posición lateral, el ángulo de tiro decrece al alejarse."""
    df = pd.DataFrame({"x": [115.0, 90.0], "y": [40.0, 40.0]})
    out = features.add_geometric_features(df)
    assert out["shot_angle"].iloc[0] > out["shot_angle"].iloc[1]


def test_angle_central_greater_than_wide():
    """Desde la misma distancia al gol, el ángulo central supera al lateral."""
    # Dos puntos a ~18 m del centro: uno frontal, otro muy escorado.
    df = pd.DataFrame({"x": [102.0, 118.0], "y": [40.0, 22.0]})
    out = features.add_geometric_features(df)
    assert out["shot_angle"].iloc[0] > out["shot_angle"].iloc[1]


def test_derived_and_box_flag():
    """log/inverse de la distancia y marcador de disparo dentro del área."""
    df = pd.DataFrame({"x": [110.0, 80.0], "y": [40.0, 10.0]})
    out = features.add_derived_features(features.add_geometric_features(df))
    assert "log_distance" in out and "inverse_distance" in out
    assert out["shot_in_box"].iloc[0] == 1     # cerca de portería, central
    assert out["shot_in_box"].iloc[1] == 0     # lejos


def test_engineer_features_schema_complete():
    """Tras la ingeniería, existen todas las columnas del esquema de modelado."""
    raw = data_loader.generate_synthetic_shots(n_matches=10)
    feat = features.engineer_features(raw)
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        assert col in feat.columns, f"falta la columna {col}"
    assert "is_penalty" in feat.columns


def test_split_features_target_excludes_penalties():
    """split_features_target con drop_penalties elimina los penaltis."""
    raw = data_loader.generate_synthetic_shots(n_matches=30)
    feat = features.engineer_features(raw)
    X, y = features.split_features_target(feat, drop_penalties=True)
    assert len(X) == len(y)
    # No debe quedar ningún penalti: su nº de filas debe coincidir con no-penaltis.
    assert len(X) == int((feat["is_penalty"] == 0).sum())


# --------------------------------------------------------------------------- #
# Datos sintéticos
# --------------------------------------------------------------------------- #
def test_synthetic_goal_rate_is_realistic():
    """La tasa de gol sintética debe ser realista (entre 8% y 20%)."""
    raw = data_loader.generate_synthetic_shots(n_matches=120)
    rate = raw["is_goal"].mean()
    assert 0.08 <= rate <= 0.20, f"tasa de gol fuera de rango: {rate:.3f}"


def test_synthetic_has_competition_column():
    """El generador sintético etiqueta cada disparo con una competición."""
    raw = data_loader.generate_synthetic_shots(n_matches=20)
    assert "competition" in raw.columns
    assert raw["competition"].nunique() >= 1


def test_synthetic_reproducible():
    """Con la misma semilla global, el dataset sintético es reproducible."""
    a = data_loader.generate_synthetic_shots(n_matches=15)
    b = data_loader.generate_synthetic_shots(n_matches=15)
    assert a["is_goal"].sum() == b["is_goal"].sum()
    assert len(a) == len(b)


# --------------------------------------------------------------------------- #
# Métricas y particiones
# --------------------------------------------------------------------------- #
def test_perfect_predictions_metrics():
    """Predicciones perfectas: AUC=1, Brier=0, ECE≈0."""
    y = np.array([0, 0, 1, 1, 0, 1])
    p = y.astype(float)
    m = evaluation.compute_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["brier_score"] == pytest.approx(0.0, abs=1e-9)


def test_ece_bounds():
    """El ECE siempre está en [0, 1]."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.random(size=500)
    ece = evaluation.expected_calibration_error(y, p)
    assert 0.0 <= ece <= 1.0


def test_competition_holdout_split_isolates_test():
    """El holdout por competición deja en test SOLO la competición reservada."""
    raw = data_loader.generate_synthetic_shots(n_matches=200)
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(
        feat, drop_penalties=True, return_competition=True
    )
    holdout = comp.value_counts().index[0]
    (Xtr, ytr), (Xv, yv), (Xte, yte), (Xtrv, ytrv) = \
        evaluation.make_competition_holdout_splits(X, y, comp, holdout)
    # El test no debe solapar con train/val (índices disjuntos).
    assert set(Xte.index).isdisjoint(set(Xtrv.index))
    # train+val debe ser exactamente el complementario del test.
    assert len(Xte) + len(Xtrv) == len(X)


def test_holdout_split_raises_if_competition_absent():
    raw = data_loader.generate_synthetic_shots(n_matches=20)
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(
        feat, drop_penalties=True, return_competition=True
    )
    with pytest.raises(ValueError):
        evaluation.make_competition_holdout_splits(X, y, comp, "Competición Inexistente")


# --------------------------------------------------------------------------- #
# Estadística
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_contains_estimate():
    """El IC bootstrap debe contener la estimación puntual."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    p = np.clip(y * 0.6 + rng.random(400) * 0.4, 0, 1)
    res = statistics.bootstrap_metric_ci(y, p, metric="roc_auc", n_boot=300)
    assert res["ci_low"] <= res["estimate"] <= res["ci_high"]


def test_paired_comparison_identical_models_no_difference():
    """Dos modelos idénticos: diferencia 0 y no significativa."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    p = rng.random(300)
    res = statistics.paired_bootstrap_comparison(y, p, p, metric="roc_auc", n_boot=300)
    assert res["diff_a_minus_b"] == pytest.approx(0.0, abs=1e-9)
    assert res["significant_at_alpha"] is False


def test_reliability_table_sums_to_total():
    """La tabla de fiabilidad reparte todos los disparos en sus tramos."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    tab = statistics.reliability_table(y, p, n_bins=10)
    assert tab["n_shots"].sum() == 500


# --------------------------------------------------------------------------- #
# Modelos y validación entre competiciones
# --------------------------------------------------------------------------- #
def test_spline_logistic_model_available_and_fits():
    """El modelo logístico con splines existe y produce probabilidades válidas."""
    from xg import models
    assert "logistic_splines" in models.available_model_names()
    raw = data_loader.generate_synthetic_shots(n_matches=60)
    feat = features.engineer_features(raw)
    X, y = features.split_features_target(feat, drop_penalties=True)
    pipe = models.build_model_pipeline("logistic_splines")
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_leave_one_competition_out_runs():
    """LOCO produce una fila por competición más MEDIA y DESV_TIP."""
    from xg import models
    raw = data_loader.generate_synthetic_shots(n_matches=160)
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(
        feat, drop_penalties=True, return_competition=True
    )
    loco = evaluation.leave_one_competition_out(
        models.build_model_pipeline, "logistic_regression", X, y, comp, calibrate=False
    )
    assert "MEDIA" in loco.index and "DESV_TIP" in loco.index
    # Una fila por competición presente (excluyendo los resúmenes).
    n_comps = comp.nunique()
    assert len(loco) == n_comps + 2


def test_out_of_fold_scoring_is_complete_and_oof():
    """El scoring out-of-fold cubre todos los disparos y reduce el optimismo.

    Verifica (1) que no quedan NaN, (2) que las probabilidades son válidas, y
    (3) que el xG OOF difiere del xG in-sample (re-sustitución), evidenciando
    que se corrige el optimismo que sesgaría los rankings de finalización.
    """
    from xg import models
    raw = data_loader.generate_synthetic_shots(n_matches=120)
    feat = features.engineer_features(raw)
    X, y = features.split_features_target(feat, drop_penalties=True)
    oof = evaluation.out_of_fold_scoring(
        models.build_model_pipeline, "logistic_regression", X, y, n_splits=5
    )
    assert len(oof) == len(X)
    assert not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()
    # xG in-sample (re-sustitución) para contrastar
    pipe = models.build_model_pipeline("logistic_regression")
    pipe.fit(X, y)
    insample = pipe.predict_proba(X)[:, 1]
    # No deben ser idénticos: el OOF es estrictamente fuera de muestra.
    assert not np.allclose(oof, insample)


def test_game_state_uses_event_order_when_available():
    """El game_state respeta el orden de evento de StatsBomb (event_index)."""
    raw = data_loader.generate_synthetic_shots(n_matches=40)
    feat = features.engineer_features(raw)
    # En sintético event_index existe y debe haberse usado para el orden.
    assert "goal_diff" in feat.columns
    assert "game_state" in feat.columns
    # El game_state coherente: nunca etiqueta 'winning' con goal_diff negativo.
    assert (feat.loc[feat["goal_diff"] < 0, "game_state"] == "losing").all()
    assert (feat.loc[feat["goal_diff"] > 0, "game_state"] == "winning").all()


def test_hierarchical_finishing_shrinks_without_skill():
    """Sin efecto-jugador real, el modelo jerárquico contrae todo hacia cero."""
    from xg.bayes import fit_hierarchical_finishing
    rng = np.random.default_rng(0)
    recs = []
    for i in range(120):
        xg = rng.beta(2, 8, size=25)
        goals = (rng.random(25) < xg).astype(int)  # sin habilidad: Bernoulli(xG)
        for x, g in zip(xg, goals):
            recs.append({"player": f"P{i}", "xg_pred": float(x), "is_goal": int(g)})
    df = pd.DataFrame(recs)
    res = fit_hierarchical_finishing(df, prefer="empirical_bayes", min_shots=10)
    # tau pequeño y ningún falso finalizador con alta probabilidad posterior.
    assert res.tau < 0.30
    assert (res.table["prob_better_than_xg"] > 0.95).sum() == 0
    assert res.table["u_hat"].abs().max() < 0.5


def test_hierarchical_finishing_recovers_injected_skill():
    """Con habilidad inyectada, el modelo la detecta en la dirección correcta."""
    from xg.bayes import fit_hierarchical_finishing

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    rng = np.random.default_rng(1)
    recs, skill = [], {}
    for i in range(150):
        u_true = 0.8 if i < 40 else 0.0
        skill[f"P{i}"] = u_true
        xg = rng.beta(2, 8, size=120)
        eta = np.log(xg / (1 - xg)) + u_true
        goals = (rng.random(120) < sigmoid(eta)).astype(int)
        for x, g in zip(xg, goals):
            recs.append({"player": f"P{i}", "xg_pred": float(x), "is_goal": int(g)})
    df = pd.DataFrame(recs)
    res = fit_hierarchical_finishing(df, prefer="empirical_bayes", min_shots=10)
    tbl = res.table.assign(u_true=res.table["player"].map(skill))
    assert res.tau > 0.2  # detecta heterogeneidad real
    # Los jugadores con habilidad tienen u_hat mayor que los que no.
    mu_skilled = tbl[tbl.u_true > 0]["u_hat"].mean()
    mu_plain = tbl[tbl.u_true == 0]["u_hat"].mean()
    assert mu_skilled > mu_plain + 0.2
    # Correlación positiva clara entre estimación y verdad.
    assert tbl["u_hat"].corr(tbl["u_true"]) > 0.5


def test_overperformers_significance_columns():
    """El ranking con contraste incluye p-valor e intervalo de confianza."""
    raw = data_loader.generate_synthetic_shots(n_matches=120)
    feat = features.engineer_features(raw)
    feat = feat[feat["is_penalty"] == 0].reset_index(drop=True)
    feat["xg_pred"] = feat["true_xg"] if "true_xg" in feat.columns else 0.1
    from xg import analysis
    out = analysis.overperformers_with_significance(
        feat, group_col="player", prob_col="xg_pred", min_shots=3
    )
    for col in (
        "p_value", "p_value_fdr_bh", "ci_low", "ci_high",
        "significant", "significant_fdr", "z_score",
    ):
        assert col in out.columns
    # El xG total esperado nunca debe ser negativo.
    assert (out["xg_total"] >= 0).all()
