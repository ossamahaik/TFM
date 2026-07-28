"""
run_pipeline.py
===============
Pipeline reproducible end-to-end del proyecto xG.

Ejecuta TODO el flujo de principio a fin y genera de forma automática todos los
entregables (modelos, métricas, figuras, CSVs, rankings). Es el punto de entrada
principal del proyecto y deja el repositorio en estado "listo para revisar".

Flujo:
    1. Adquisición de datos      (StatsBomb si hay red; sintético en su defecto)
    2. Ingeniería de variables
    3. Particiones reproducibles  (train / val / test estratificadas)
    4. Comparación de modelos por validación cruzada
    5. Entrenamiento del mejor modelo + calibración
    6. Evaluación final en test hold-out
    7. Interpretabilidad (importancia por permutación)
    8. Análisis por jugador y equipo
    9. Exportación de figuras, métricas y CSVs

Uso:
    python run_pipeline.py [--source auto|statsbomb|synthetic] [--max-matches N]
"""
from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd

from xg import config
from xg import analysis
from xg import data as data_loader
from xg import features
from xg import models
from xg import visualization
from xg.models import evaluation
from xg.models import statistics

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xg_pipeline")


def main(source: str = "auto", max_matches=None, holdout_competition=None) -> None:
    np.random.seed(config.RANDOM_STATE)
    config.ensure_dirs()
    if holdout_competition is None:
        holdout_competition = getattr(config, "HOLDOUT_COMPETITION", None)

    # --- 1. Datos ---------------------------------------------------------- #
    logger.info("== 1. Adquisición de datos ==")
    raw, used_source = data_loader.build_shots_dataframe(source=source, max_matches=max_matches)
    logger.info("Fuente de datos: %s | disparos: %d", used_source, len(raw))
    data_loader.save_processed(raw, "shots_raw")

    # --- 2. Feature engineering ------------------------------------------- #
    logger.info("== 2. Ingeniería de variables ==")
    feat = features.engineer_features(raw)
    data_loader.save_processed(feat, "shots_features")
    # Justificación empírica de excluir penaltis: su conversión es casi constante
    # y desligada de la geometría, por lo que distorsionaría el modelo.
    if "is_penalty" in feat.columns and feat["is_penalty"].sum() > 0:
        pens = feat[feat["is_penalty"] == 1]
        logger.info(
            "Penaltis: %d | conversión empírica = %.3f (se excluyen del modelo principal)",
            len(pens), float(pens[config.TARGET].mean()),
        )
    X, y, comp = features.split_features_target(
        feat, drop_penalties=True, return_competition=True
    )
    logger.info("Matriz de diseño: %s | tasa de gol: %.4f", X.shape, y.mean())

    # --- 3. Particiones ---------------------------------------------------- #
    logger.info("== 3. Particiones train/val/test ==")
    use_holdout = holdout_competition and (comp == holdout_competition).any()
    if holdout_competition and not use_holdout:
        logger.warning(
            "La competición de holdout '%s' no está presente en los datos; "
            "se usa partición aleatoria estratificada.", holdout_competition,
        )
    if use_holdout:
        logger.info(
            "Estrategia de test: holdout por competición = '%s' "
            "(generalización entre torneos)", holdout_competition,
        )
        (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_trval, y_trval) = \
            evaluation.make_competition_holdout_splits(X, y, comp, holdout_competition)
    else:
        logger.info("Estrategia de test: partición aleatoria estratificada")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_trval, y_trval) = \
            evaluation.make_splits(X, y)
    logger.info("train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # --- 4. Comparación de modelos por CV ---------------------------------- #
    logger.info("== 4. Comparación de modelos (CV) ==")
    names = models.available_model_names()
    pipelines = {n: models.build_model_pipeline(n) for n in names}
    cv_metrics, cv_oof = evaluation.evaluate_models_cv(
        pipelines, X_trval, y_trval, return_probabilities=True,
    )
    evaluation.save_metrics_table(cv_metrics, "model_comparison_cv")
    print("\n--- Comparación por validación cruzada (ordenado por Brier) ---")
    print(cv_metrics.round(4).to_string())

    # IC bootstrap al 95 % para todos los modelos en CV (no solo el punto medio).
    cv_ci_rows = []
    for name, (yt, prob) in cv_oof.items():
        t = statistics.bootstrap_metrics_table(yt, prob, n_boot=2000).reset_index()
        t.insert(0, "model", name)
        cv_ci_rows.append(t)
    cv_models_ci = pd.concat(cv_ci_rows, ignore_index=True)
    analysis.save_table(cv_models_ci, "model_comparison_cv_bootstrap_ci")

    best_name = cv_metrics.index[0]
    logger.info("Mejor modelo por %s: %s", config.PRIMARY_METRIC, best_name)

    # --- 4b. Validación Leave-One-Competition-Out (si hay >=2 torneos) ----- #
    if comp.nunique() > 1:
        logger.info("== 4b. Validación Leave-One-Competition-Out (LOCO) ==")
        loco = evaluation.leave_one_competition_out(
            models.build_model_pipeline, best_name, X, y, comp, calibrate=True
        )
        evaluation.save_metrics_table(loco, "loco_validation")
        print("\n--- LOCO: rendimiento dejando fuera cada competición ---")
        cols_show = ["n_test", "roc_auc", "pr_auc", "brier_score", "log_loss", "ece"]
        print(loco[cols_show].round(4).to_string())
        visualization.plot_loco_validation(loco, "roc_auc", save_as="loco_roc_auc")
        visualization.plot_loco_validation(loco, "brier_score", save_as="loco_brier")

    # --- 5. Entrenamiento final + calibración ------------------------------ #
    logger.info("== 5. Entrenamiento y calibración del mejor modelo ==")
    best_pipe = models.build_model_pipeline(best_name)
    best_pipe.fit(X_trval, y_trval)
    models.save_model(best_pipe, f"best_model_{best_name}_uncalibrated")

    calibrated = models.calibrate_pipeline(
        models.build_model_pipeline(best_name), method="isotonic", cv=config.CV_FOLDS
    )
    calibrated.fit(X_trval, y_trval)
    models.save_model(calibrated, f"best_model_{best_name}_calibrated")

    # También entrenamos y guardamos la regresión logística como baseline
    # interpretable de referencia.
    baseline = models.build_model_pipeline("logistic_regression")
    baseline.fit(X_trval, y_trval)
    models.save_model(baseline, "baseline_logistic_regression")

    # Rival lineal SERIO: regresión logística con bases de B-splines, que captura
    # la no linealidad geometría->probabilidad. Es la comparación justa frente al
    # boosting ("lineal con la no linealidad adecuada" vs. árboles), no un hombre
    # de paja. Se entrena y evalúa en el mismo test hold-out.
    spline_logit = models.build_model_pipeline("logistic_splines")
    spline_logit.fit(X_trval, y_trval)
    models.save_model(spline_logit, "logistic_splines")

    # --- 6. Evaluación en test hold-out ------------------------------------ #
    logger.info("== 6. Evaluación en test hold-out ==")
    test_rows = []
    roc_results, cal_results = {}, {}
    for label, mdl in [
        (f"{best_name} (sin calibrar)", best_pipe),
        (f"{best_name} (calibrado)", calibrated),
        ("logistic_splines", spline_logit),
        ("logistic_regression", baseline),
    ]:
        m, prob = evaluation.evaluate_on_test(mdl, X_test, y_test, label)
        test_rows.append(m)
        roc_results[label] = (y_test.values, prob)
        cal_results[label] = (y_test.values, prob)
    test_metrics = pd.DataFrame(test_rows).set_index("model")
    evaluation.save_metrics_table(test_metrics, "model_comparison_test")
    print("\n--- Métricas en test hold-out ---")
    print(test_metrics.round(4).to_string())

    # --- 6b. Validación estadística (bootstrap) ---------------------------- #
    logger.info("== 6b. Validación estadística (IC bootstrap + comparación) ==")
    cal_label = f"{best_name} (calibrado)"
    y_cal, p_cal = cal_results[cal_label]
    _, p_base = cal_results["logistic_regression"]

    # Intervalos de confianza al 95 % para el mejor modelo calibrado.
    ci_table = statistics.bootstrap_metrics_table(y_cal, p_cal, n_boot=2000)
    analysis.save_table(ci_table.reset_index(), "test_metrics_bootstrap_ci")
    print("\n--- IC bootstrap 95% (mejor modelo calibrado, test) ---")
    print(ci_table.round(4).to_string())

    # IC bootstrap al 95 % para TODOS los modelos en test, no solo el ganador:
    # evita presentar una única estimación puntual por modelo en la comparación.
    all_ci_rows = []
    for label, (yt, prob) in cal_results.items():
        t = statistics.bootstrap_metrics_table(yt, prob, n_boot=2000).reset_index()
        t.insert(0, "model", label)
        all_ci_rows.append(t)
    all_models_ci = pd.concat(all_ci_rows, ignore_index=True)
    analysis.save_table(all_models_ci, "model_comparison_test_bootstrap_ci")

    # Comparación pareada mejor-modelo vs. baseline logístico sobre el mismo test.
    cmp_rows = [
        statistics.paired_bootstrap_comparison(y_cal, p_cal, p_base, metric=m, n_boot=2000)
        for m in ("roc_auc", "pr_auc", "brier_score", "log_loss")
    ]
    cmp_table = pd.DataFrame(cmp_rows).set_index("metric")
    analysis.save_table(cmp_table.reset_index(), "model_comparison_paired_bootstrap")
    print("\n--- Comparación pareada (mejor calibrado − baseline) ---")
    print(cmp_table.round(4).to_string())

    # Tabla de fiabilidad (calibración por tramos) con IC de Wilson.
    rel_table = statistics.reliability_table(y_cal, p_cal, n_bins=10)
    analysis.save_table(rel_table, "reliability_table")
    visualization.plot_reliability_with_ci(rel_table, save_as="reliability_wilson")

    # --- 7. Figuras de evaluación ----------------------------------------- #
    logger.info("== 7. Figuras de evaluación ==")
    base_rate = float(y_test.mean())
    visualization.plot_roc_curves(roc_results, save_as="roc_curves")
    visualization.plot_pr_curves(roc_results, base_rate, save_as="pr_curves")
    visualization.plot_calibration_curves(cal_results, save_as="calibration_curves")
    best_prob = calibrated.predict_proba(X_test)[:, 1]
    visualization.plot_xg_distribution(best_prob, save_as="xg_distribution")
    visualization.plot_confusion(y_test.values, best_prob, save_as="confusion_matrix")
    visualization.plot_correlation_matrix(feat, config.NUMERIC_FEATURES, save_as="correlation_matrix")
    # Distribución marginal de cada variable (geométricas y defensivas): un panel
    # por variable, para mostrar su forma y no solo describirla en prosa.
    feat_nonpen = feat[feat["is_penalty"] == 0] if "is_penalty" in feat.columns else feat
    visualization.plot_feature_distributions(
        feat_nonpen,
        [("distance_to_goal", "Distancia a portería"),
         ("shot_angle", "Ángulo de visión (rad)"),
         ("distance_x", "Distancia en x"),
         ("distance_y", "Distancia en y"),
         ("log_distance", "Log-distancia"),
         ("inverse_distance", "Inversa de la distancia")],
        title="Distribución de las variables geométricas (disparos no penales)",
        save_as="dist_geometric",
    )
    visualization.plot_feature_distributions(
        feat_nonpen,
        [("n_defenders_in_cone", "Defensores en el cono"),
         ("n_defenders_within_3m", "Defensores a < 3 m"),
         ("distance_to_nearest_defender", "Distancia al defensor más cercano"),
         ("gk_distance_to_goal", "Distancia del portero a portería"),
         ("gk_distance_to_shot", "Distancia del portero al disparo")],
        title="Distribución de las variables defensivas (freeze frames)",
        save_as="dist_defensive",
    )
    visualization.plot_metric_comparison(cv_metrics, "brier_score", save_as="brier_comparison")
    visualization.plot_metric_comparison(cv_metrics, "roc_auc", save_as="roc_auc_comparison")

    # --- 8. Interpretabilidad ---------------------------------------------- #
    logger.info("== 8. Interpretabilidad (importancia por permutación) ==")
    importances = analysis.permutation_feature_importance(
        best_pipe, X_test, y_test, scoring="roc_auc"
    )
    importances.to_csv(config.OUTPUTS_DIR / "feature_importance_permutation.csv")
    visualization.plot_feature_importance(importances, save_as="feature_importance")
    # Informe ampliado: AUC y Brier, con desviación típica entre repeticiones.
    imp_report = analysis.permutation_importance_report(best_pipe, X_test, y_test)
    analysis.save_table(imp_report.reset_index().rename(columns={"index": "feature"}),
                        "feature_importance_report")

    # --- 9. Scoring completo + análisis jugador/equipo --------------------- #
    logger.info("== 9. Scoring y análisis agregado ==")
    scored = feat[feat["is_penalty"] == 0].copy()
    Xall, _ = features.split_features_target(feat, drop_penalties=True)
    scored = scored.reset_index(drop=True)
    # Scoring HONESTO out-of-fold: cada disparo recibe un xG predicho por un
    # modelo que NO lo vio en entrenamiento. Esto evita el optimismo de
    # re-sustitución que sesgaría los rankings de finalización (goles − xG).
    # Véase evaluation.out_of_fold_scoring para la justificación metodológica.
    logger.info(
        "Asignando xG out-of-fold (%s, calibrado) a los %d disparos del análisis...",
        best_name, len(Xall),
    )
    scored["xg_pred"] = evaluation.out_of_fold_scoring(
        models.build_model_pipeline, best_name, Xall, scored[config.TARGET].astype(int),
        n_splits=config.CV_FOLDS, calibrate=True,
    )
    if "true_xg" in scored.columns:
        # En modo sintético validamos contra el "oráculo" (xG generador).
        mae_oracle = float(np.mean(np.abs(scored["xg_pred"] - scored["true_xg"])))
        logger.info("MAE out-of-fold frente al xG generador (oráculo): %.4f", mae_oracle)
    data_loader.save_processed(scored, "shots_scored")

    team_agg = analysis.aggregate_by(scored, "team")
    analysis.save_table(team_agg, "team_xg_ranking")
    player_agg = analysis.aggregate_by(scored, "player")
    analysis.save_table(player_agg, "player_xg_ranking")
    finishers = analysis.overperformers(player_agg, min_shots=12, group_col="player")
    analysis.save_table(finishers, "top_finishers_over_xg")

    # Desempeño por subgrupos futbolísticos: resultado aplicado y, a la vez,
    # validación de calibración condicional (xG medio vs. tasa real por subgrupo).
    subgroup_perf = analysis.subgroup_performance(scored, prob_col="xg_pred", min_shots=30)
    if not subgroup_perf.empty:
        analysis.save_table(subgroup_perf, "subgroup_performance")
        visualization.plot_subgroup_performance(subgroup_perf, save_as="subgroup_performance")

    # Ranking de finalizadores CON contraste estadístico (Poisson-binomial):
    # distingue sobre-rendimiento real de simple ruido muestral.
    finishers_sig = analysis.overperformers_with_significance(
        scored, group_col="player", prob_col="xg_pred", min_shots=12
    )
    if not finishers_sig.empty:
        analysis.save_table(finishers_sig, "finishers_significance")
        n_sig = int(finishers_sig["significant"].sum())
        logger.info(
            "Finalizadores con sobre/infra-rendimiento significativo (alpha=0.05): "
            "%d de %d", n_sig, len(finishers_sig),
        )

    # --- 9b. Modelo jerárquico bayesiano de finalización ------------------- #
    # Complementa el contraste frecuentista con estimaciones regularizadas
    # (partial pooling / shrinkage), robustas a tamaños de muestra pequeños.
    # Inferencia por Bayes empírico (EM + Laplace); backend PyMC opcional.
    logger.info("== 9b. Modelo jerárquico bayesiano (shrinkage) ==")
    from xg.models import bayesian
    bayes_res = bayesian.fit_hierarchical_finishing(
        scored, group_col="player", prob_col="xg_pred", min_shots=12
    )
    logger.info("Bayes jerárquico — %s", bayes_res.shrinkage_summary())
    if not bayes_res.table.empty:
        analysis.save_table(bayes_res.table, "finishers_bayesian_hierarchical")
        # Traza de convergencia del EM (diagnóstico numérico).
        pd.DataFrame({"em_iter": range(len(bayes_res.tau_history)),
                      "tau": bayes_res.tau_history}).to_csv(
            config.OUTPUTS_DIR / "bayesian_em_trace.csv", index=False)
        visualization.plot_em_convergence(bayes_res.tau_history,
                                          save_as="bayesian_em_convergence")
        # Comparación directa frecuentista vs bayesiano (material de memoria).
        if not finishers_sig.empty:
            comp = bayesian.compare_frequentist_bayesian(finishers_sig, bayes_res)
            if not comp.empty:
                analysis.save_table(comp, "finishers_freq_vs_bayes")
                visualization.plot_bayesian_shrinkage(comp, save_as="bayesian_shrinkage")

        # Control de calidad: backend alternativo Beta-Binomial conjugado.
        # Ambas formulaciones deben converger a la misma conclusión cualitativa
        # (ausencia de evidencia fuerte de finalizadores por encima de su xG)
        # pese a estar parametrizadas de forma distinta (log-odds + Laplace vs.
        # probabilidad + prior conjugado).
        from xg.bayes import fit_hierarchical_finishing as fit_beta_binomial
        bb_res = fit_beta_binomial(
            scored, prefer="empirical_bayes", group_col="player",
            prob_col="xg_pred", min_shots=12,
        )
        if not bb_res.estimates.empty:
            analysis.save_table(bb_res.estimates, "finishers_beta_binomial")
            merged = bb_res.estimates.merge(
                bayes_res.table[["player", "shrunk_goals_minus_xg"]], on="player",
            )
            corr = merged["goals_minus_xg_shrunk"].corr(merged["shrunk_goals_minus_xg"])
            n_excludes_zero = int((bb_res.estimates["skill_hdi_low"] > 0).sum())
            logger.info(
                "Validación cruzada de backends bayesianos — correlación de "
                "estimaciones encogidas: %.4f | jugadores con IC > 0 (Beta-Binomial): "
                "%d de %d", corr, n_excludes_zero, len(bb_res.estimates),
            )

    # Benchmark frente al xG oficial de StatsBomb (validación externa).
    sb_bench = analysis.benchmark_vs_statsbomb(scored, pred_col="xg_pred", ref_col="statsbomb_xg")
    if sb_bench is not None:
        sb_bench.to_csv(config.OUTPUTS_DIR / "benchmark_vs_statsbomb.csv")
        logger.info("Benchmark vs StatsBomb xG guardado en data/outputs/benchmark_vs_statsbomb.csv")

    visualization.plot_shot_map(scored, "xg_pred", save_as="shot_map")
    visualization.plot_xg_heatmap(scored, "xg_pred", save_as="xg_heatmap")
    visualization.plot_ranking(team_agg, "xg_total", "team",
                               "Top equipos por xG total", save_as="ranking_teams_xg")
    visualization.plot_ranking(finishers, "goals_minus_xg", "player",
                               "Top finalizadores (goles − xG)", save_as="ranking_finishers")

    logger.info("== Pipeline finalizado. Resultados en data/results, figuras en figures, modelos en artifacts/models. ==")
