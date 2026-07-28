"""
run_advanced.py — Dos experimentos avanzados de refuerzo del TFM xG.

Genera CSV NUEVOS (no sobrescribe los canónicos) en ``data/results/``:

  A. Búsqueda de hiperparámetros del modelo seleccionado (HistGradientBoosting)
     mediante ``RandomizedSearchCV`` con validación cruzada interna de cinco
     pliegues sobre el conjunto de entrenamiento (métrica: Brier score), dejando
     el test hold-out (UEFA Euro 2024) como evaluación externa no tocada por la
     búsqueda —un diseño anidado honesto—. Responde a la carencia de optimización
     de hiperparámetros y cuantifica cuánto mejora la configuración por defecto.
       -> hyperparameter_search.csv        (mejores hiperparámetros)
       -> hyperparameter_comparison.csv    (por defecto vs. ajustado; CV y test)

  B. Validación de la aproximación de Laplace del modelo jerárquico bayesiano
     frente al posterior EXACTO calculado por cuadratura numérica (determinista,
     sin dependencias). Para cada jugador (>=12 disparos) compara la media, la
     desviación típica y la probabilidad posterior P(u>0) de Laplace con las
     exactas; y valida el hiperparámetro tau del EM frente al que maximiza la
     verosimilitud marginal exacta. Responde a la crítica de que la aproximación
     de Laplace solo es exacta asintóticamente aunque se aplique a jugadores con
     pocos disparos.
       -> bayes_laplace_validation.csv     (por jugador: Laplace vs. exacto)
       -> bayes_validation_summary.csv     (errores máximos/medios; tau)

Determinista (config.RANDOM_STATE = 42). Ejecutar desde la raíz del repo:
    .venv/Scripts/python.exe scripts/run_advanced.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg import models  # noqa: E402
from xg.models import bayesian, evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("advanced")

CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


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
    }


# --------------------------------------------------------------------------- #
# A. Búsqueda de hiperparámetros (nested: CV interna + holdout externo)
# --------------------------------------------------------------------------- #
def hyperparameter_search(Xtrval, ytrval, Xte, yte):
    logger.info("== A. Búsqueda de hiperparámetros (RandomizedSearchCV, CV interna 5x) ==")
    base = models.build_model_pipeline("hist_gradient_boosting")  # preprocesado + HGB
    space = {
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_iter": randint(200, 701),
        "model__max_leaf_nodes": randint(15, 64),
        "model__min_samples_leaf": randint(10, 81),
        "model__l2_regularization": loguniform(1e-3, 10.0),
        "model__max_depth": [None, 3, 5, 8],
    }
    search = RandomizedSearchCV(
        base, space, n_iter=40, scoring="neg_brier_score", cv=CV,
        random_state=config.RANDOM_STATE, n_jobs=3, refit=True, error_score="raise",
    )
    search.fit(Xtrval, ytrval)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    tuned_cv_brier = float(-search.best_score_)
    logger.info("  Mejores hiperparámetros: %s", best_params)
    logger.info("  Brier CV (ajustado): %.4f", tuned_cv_brier)

    # Brier CV de la configuración POR DEFECTO (misma CV interna, mismo protocolo).
    default_pipe = models.build_model_pipeline("hist_gradient_boosting")
    p_def_cv = cross_val_predict(default_pipe, Xtrval, ytrval, cv=CV, method="predict_proba")[:, 1]
    default_cv_brier = float(evaluation.brier_score_loss(np.asarray(ytrval).astype(int), p_def_cv))
    logger.info("  Brier CV (por defecto): %.4f", default_cv_brier)

    # Evaluación EXTERNA en el test hold-out: ambas configuraciones calibradas
    # (como el modelo final del pipeline), ajustadas sobre train+val.
    def _fit_cal(pipe):
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv=config.CV_FOLDS)
        cal.fit(Xtrval, ytrval)
        return cal.predict_proba(Xte)[:, 1]

    tuned_pipe = models.build_model_pipeline("hist_gradient_boosting")
    tuned_pipe.set_params(**search.best_params_)
    m_def = _metrics(yte, _fit_cal(models.build_model_pipeline("hist_gradient_boosting")))
    m_tun = _metrics(yte, _fit_cal(tuned_pipe))

    # CSV 1: mejores hiperparámetros.
    hp_rows = [{"hiperparametro": k, "valor": str(v)} for k, v in best_params.items()]
    hp_rows.append({"hiperparametro": "random_state", "valor": str(config.RANDOM_STATE)})
    df_hp = pd.DataFrame(hp_rows)

    # CSV 2: comparación por defecto vs ajustado (CV y test).
    rows = []
    rows.append({"config": "por_defecto", "split": "cv_oof", "brier": default_cv_brier})
    rows.append({"config": "ajustado", "split": "cv_oof", "brier": tuned_cv_brier})
    r_def = {"config": "por_defecto", "split": "test"}; r_def.update(m_def)
    r_tun = {"config": "ajustado", "split": "test"}; r_tun.update(m_tun)
    rows.append(r_def); rows.append(r_tun)
    df_cmp = pd.DataFrame(rows)
    logger.info("  Test por defecto: %s", {k: round(v, 4) for k, v in m_def.items()})
    logger.info("  Test ajustado   : %s", {k: round(v, 4) for k, v in m_tun.items()})
    return df_hp, df_cmp


# --------------------------------------------------------------------------- #
# B. Validación de la aproximación de Laplace por cuadratura numérica
# --------------------------------------------------------------------------- #
def _logpost_unnorm(u, eta, g, tau2):
    """log-posterior no normalizada de u (vector u; eta,g del jugador)."""
    # z_ij = eta_j + u ; log-verosimilitud Bernoulli + log-prior gaussiano
    #   Σ_j [ g_j (eta_j+u) - log(1+e^{eta_j+u}) ] - u^2/(2 tau2)
    z = eta[None, :] + u[:, None]                       # (n_u, n_shots)
    ll = (g[None, :] * z - np.logaddexp(0.0, z)).sum(axis=1)
    return ll - u * u / (2.0 * tau2)


def _exact_posterior(eta, g, tau2, u_center, u_scale, n=2001, half_width=12.0):
    """Media, desv. típica y P(u>0) exactas por cuadratura (trapezoidal)."""
    scale = max(u_scale, 0.05)
    lo, hi = u_center - half_width * scale, u_center + half_width * scale
    u = np.linspace(lo, hi, n)
    lp = _logpost_unnorm(u, eta, g, tau2)
    w = np.exp(lp - lp.max())
    Z = np.trapezoid(w, u)
    mean = np.trapezoid(u * w, u) / Z
    var = np.trapezoid((u - mean) ** 2 * w, u) / Z
    mask = u > 0
    p_pos = np.trapezoid(w[mask], u[mask]) / Z
    return float(mean), float(np.sqrt(max(var, 0.0))), float(p_pos)


def _marginal_loglik_tau(players, tau2, n=1601, lo=-4.0, hi=4.0):
    """log-verosimilitud marginal total sum_i log ∫ N(u;0,tau2) L_i(u) du (cuadratura)."""
    u = np.linspace(lo, hi, n)
    log_prior = -0.5 * np.log(2 * np.pi * tau2) - u * u / (2.0 * tau2)
    total = 0.0
    for eta, g in players:
        z = eta[None, :] + u[:, None]
        ll = (g[None, :] * z - np.logaddexp(0.0, z)).sum(axis=1)
        integrand = np.exp(log_prior + ll - (log_prior + ll).max())
        logZ = np.log(np.trapezoid(integrand, u)) + (log_prior + ll).max()
        total += logZ
    return float(total)


def bayes_validation():
    logger.info("== B. Validación de Laplace por cuadratura numérica ==")
    scored = pd.read_parquet(ROOT / "data" / "processed" / "shots_scored.parquet")
    scored = scored[scored.get("is_penalty", 0) == 0] if "is_penalty" in scored.columns else scored

    # Reproduce el ajuste EM+Laplace del pipeline (backend log-odds principal).
    res = bayesian.fit_hierarchical_finishing(
        scored, group_col="player", prob_col="xg_pred", min_shots=12)
    tau_hat = float(res.tau)
    tau2 = tau_hat ** 2
    logger.info("  EM+Laplace: tau_hat = %.4f | jugadores = %d", tau_hat, len(res.table))

    # Datos por jugador (eta, g) para la cuadratura.
    players = {}
    for player, grp in scored.groupby("player"):
        if len(grp) < 12:
            continue
        p = np.clip(grp["xg_pred"].to_numpy(float), 1e-6, 1 - 1e-6)
        players[player] = (np.log(p / (1 - p)), grp["is_goal"].to_numpy(float))

    rows = []
    for _, r in res.table.iterrows():
        player = r["player"]
        eta, g = players[player]
        u_lap, s_lap = float(r["u_hat"]), float(r["u_se"])
        p_lap = float(r["prob_better_than_xg"])
        u_ex, s_ex, p_ex = _exact_posterior(eta, g, tau2, u_lap, s_lap)
        rows.append({
            "player": player, "shots": int(r["shots"]),
            "u_hat_laplace": u_lap, "u_mean_exact": u_ex,
            "s_laplace": s_lap, "sd_exact": s_ex,
            "p_laplace": p_lap, "p_exact": p_ex,
            "abs_err_u": abs(u_lap - u_ex),
            "abs_err_s": abs(s_lap - s_ex),
            "abs_err_p": abs(p_lap - p_ex),
        })
    val = pd.DataFrame(rows)

    # tau que maximiza la verosimilitud marginal EXACTA (rejilla).
    plist = list(players.values())
    grid = np.round(np.arange(0.05, 0.701, 0.005), 4)
    lls = [_marginal_loglik_tau(plist, float(t) ** 2) for t in grid]
    tau_exact = float(grid[int(np.argmax(lls))])
    logger.info("  tau* (verosimilitud marginal exacta) = %.4f | tau_hat (EM) = %.4f",
                tau_exact, tau_hat)

    summary = pd.DataFrame([{
        "n_jugadores": len(val),
        "tau_em_laplace": round(tau_hat, 4),
        "tau_marginal_exacta": tau_exact,
        "abs_err_u_max": round(val["abs_err_u"].max(), 5),
        "abs_err_u_medio": round(val["abs_err_u"].mean(), 5),
        "abs_err_s_max": round(val["abs_err_s"].max(), 5),
        "abs_err_s_medio": round(val["abs_err_s"].mean(), 5),
        "abs_err_p_max": round(val["abs_err_p"].max(), 5),
        "abs_err_p_medio": round(val["abs_err_p"].mean(), 5),
        "corr_p_laplace_exacta": round(val["p_laplace"].corr(val["p_exact"]), 5),
    }])
    logger.info("  |Δu| máx=%.5f medio=%.5f | |ΔP(>xG)| máx=%.5f medio=%.5f | corr P=%.5f",
                summary["abs_err_u_max"][0], summary["abs_err_u_medio"][0],
                summary["abs_err_p_max"][0], summary["abs_err_p_medio"][0],
                summary["corr_p_laplace_exacta"][0])
    return val, summary


# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    np.random.seed(config.RANDOM_STATE)
    config.ensure_dirs()
    out = config.OUTPUTS_DIR

    raw, _ = data_loader.build_shots_dataframe(source="statsbomb")
    feat = features.engineer_features(raw)
    X, y, comp = features.split_features_target(feat, drop_penalties=True, return_competition=True)
    (_, _), (_, _), (Xte, yte), (Xtrval, ytrval) = \
        evaluation.make_competition_holdout_splits(X, y, comp, config.HOLDOUT_COMPETITION)

    df_hp, df_cmp = hyperparameter_search(Xtrval, ytrval, Xte, yte)
    df_hp.to_csv(out / "hyperparameter_search.csv", index=False)
    df_cmp.to_csv(out / "hyperparameter_comparison.csv", index=False)

    val, summary = bayes_validation()
    val.to_csv(out / "bayes_laplace_validation.csv", index=False)
    summary.to_csv(out / "bayes_validation_summary.csv", index=False)

    logger.info("== Hecho en %.1f s ==", time.perf_counter() - t0)
    print("\n=== A. Hiperparámetros (mejores) ===")
    print(df_hp.to_string(index=False))
    print("\n=== A. Comparación por defecto vs ajustado ===")
    print(df_cmp.round(4).to_string(index=False))
    print("\n=== B. Validación de Laplace (resumen) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
