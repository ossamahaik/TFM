"""
run_critical_fixes.py — Cuatro experimentos que cierran las carencias criticas.

A. SELECCION CON CV CALIBRADA (critico).
   El pipeline seleccionaba el mejor modelo por Brier de validacion cruzada sobre
   modelos SIN calibrar y despues calibraba al ganador, de modo que el criterio
   premiaba a los modelos nativamente calibrados y penalizaba a los que solo
   necesitaban calibracion posterior, propiedad irrelevante una vez que se calibra
   a todos. Aqui se repite la comparacion con las CINCO configuraciones YA
   CALIBRADAS, que es la condicion en que realmente se despliegan.
     -> seleccion_cv_calibrada.csv

B. POTENCIA DEL CONTRASTE BAYESIANO (importante).
   El analisis de finalizacion arroja un resultado nulo (0 de 97 jugadores). Sin
   analisis de potencia no puede distinguirse "no hay habilidad" de "no hay datos
   para verla". Se simula, con los MISMOS jugadores y los MISMOS xG por disparo,
   una habilidad verdadera u_i ~ N(0, tau^2) para una rejilla de tau, y se mide
   que fraccion de jugadores con habilidad se detecta al umbral 0,95.
     -> potencia_bayesiana.csv

C. SENSIBILIDAD A min_shots Y AL UMBRAL (medio).
   Comprueba si el resultado nulo depende de dos cortes fijados por el analista.
     -> sensibilidad_bayesiana.csv

D. EVALUACION AGREGADA ENTRE COMPETICIONES (importante).
   El test externo tiene un solo torneo (1.304 disparos, 98 goles) y sus intervalos
   son anchos. Agregando las predicciones fuera de muestra de las cuatro rotaciones
   LOCO se obtiene una evaluacion sobre 5.606 disparos y 507 goles, todos predichos
   por un modelo que no vio su competicion.
     -> loco_pooled_metrics.csv

Determinista (config.RANDOM_STATE = 42). Ejecutar desde la raiz del repo:
    .venv/Scripts/python.exe scripts/run_critical_fixes.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg import config, features  # noqa: E402
from xg import data as data_loader  # noqa: E402
from xg import models  # noqa: E402
from xg.models import bayesian, evaluation, statistics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("criticos")

CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

MODEL_ORDER = [
    ("logistic_regression", "Regresión logística"),
    ("logistic_splines", "Logística + B-splines"),
    ("random_forest", "Random forest"),
    ("gradient_boosting", "Gradient boosting"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
]


def _m(y, p):
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


# --------------------------------------------------------------------------- #
# A. Seleccion con CV calibrada
# --------------------------------------------------------------------------- #
def seleccion_calibrada(Xtrval, ytrval):
    logger.info("== A. Seleccion por CV con los cinco modelos CALIBRADOS ==")
    rows = []
    for key, label in MODEL_ORDER:
        for cal in (False, True):
            pipe = models.build_model_pipeline(key)
            est = CalibratedClassifierCV(pipe, method="isotonic", cv=config.CV_FOLDS) if cal else pipe
            p = cross_val_predict(est, Xtrval, ytrval, cv=CV, method="predict_proba")[:, 1]
            r = _m(ytrval, p)
            r["modelo"], r["calibrado"], r["clave"] = label, "sí" if cal else "no", key
            rows.append(r)
            logger.info("  %-24s cal=%-2s | Brier=%.4f ECE=%.4f ROC=%.4f",
                        label, "sí" if cal else "no", r["brier"], r["ece"], r["roc_auc"])
    df = pd.DataFrame(rows)[["modelo", "calibrado", "clave", "roc_auc", "pr_auc",
                             "brier", "ece", "log_loss", "mean_pred"]]
    cal = df[df.calibrado == "sí"].sort_values("brier")
    logger.info("  >> Mejor por Brier CV CALIBRADA: %s (%.4f)",
                cal.iloc[0]["modelo"], cal.iloc[0]["brier"])
    return df


# --------------------------------------------------------------------------- #
# B/C. Utilidades bayesianas
# --------------------------------------------------------------------------- #
def _players(scored, min_shots):
    out = {}
    for name, g in scored.groupby("player"):
        if len(g) < min_shots:
            continue
        p = np.clip(g["xg_pred"].to_numpy(float), 1e-6, 1 - 1e-6)
        out[name] = (np.log(p / (1 - p)), g["is_goal"].to_numpy(float))
    return out


def potencia(scored, taus, n_rep=100, umbral=0.95, min_shots=12):
    logger.info("== B. Potencia del contraste bayesiano (%d reps por tau) ==", n_rep)
    base = _players(scored, min_shots)
    etas = {k: v[0] for k, v in base.items()}
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    for tau in taus:
        det_con, det_sin, taus_est = [], [], []
        for _ in range(n_rep):
            recs, verdad = [], {}
            for name, eta in etas.items():
                u = float(rng.normal(0.0, tau))
                verdad[name] = u
                p = 1.0 / (1.0 + np.exp(-(eta + u)))
                g = (rng.random(len(eta)) < p).astype(int)
                for e, gi in zip(eta, g):
                    recs.append({"player": name, "xg_pred": float(1 / (1 + np.exp(-e))),
                                 "is_goal": int(gi)})
            df = pd.DataFrame(recs)
            res = bayesian.fit_hierarchical_finishing(df, group_col="player",
                                                      prob_col="xg_pred", min_shots=min_shots)
            if res.table.empty:
                continue
            t = res.table.assign(u_true=res.table["player"].map(verdad))
            pos = t[t.u_true > 0.2]      # jugadores con habilidad apreciable
            neg = t[t.u_true.abs() < 0.05]
            if len(pos):
                det_con.append(float((pos["prob_better_than_xg"] > umbral).mean()))
            if len(neg):
                det_sin.append(float((neg["prob_better_than_xg"] > umbral).mean()))
            taus_est.append(float(res.tau))
        rows.append({
            "tau_verdadero": round(float(tau), 3),
            "n_jugadores": len(base),
            "potencia_u_mayor_0_2": round(float(np.mean(det_con)) if det_con else np.nan, 4),
            "falsos_positivos_u_cero": round(float(np.mean(det_sin)) if det_sin else np.nan, 4),
            "tau_estimado_medio": round(float(np.mean(taus_est)), 4),
            "umbral": umbral, "n_replicas": n_rep,
        })
        logger.info("  tau=%.2f -> potencia=%.3f | FP=%.3f | tau_est=%.3f",
                    tau, rows[-1]["potencia_u_mayor_0_2"],
                    rows[-1]["falsos_positivos_u_cero"], rows[-1]["tau_estimado_medio"])
    return pd.DataFrame(rows)


def sensibilidad(scored, min_shots_list, umbrales):
    logger.info("== C. Sensibilidad a min_shots y al umbral ==")
    rows = []
    for ms in min_shots_list:
        res = bayesian.fit_hierarchical_finishing(scored, group_col="player",
                                                  prob_col="xg_pred", min_shots=ms)
        if res.table.empty:
            continue
        for u in umbrales:
            n_sig = int((res.table["prob_better_than_xg"] > u).sum())
            rows.append({"min_shots": ms, "n_jugadores": len(res.table),
                         "tau_estimado": round(float(res.tau), 4), "umbral": u,
                         "n_significativos": n_sig,
                         "max_prob_posterior": round(float(res.table["prob_better_than_xg"].max()), 4)})
        logger.info("  min_shots=%-3d n=%-3d tau=%.4f | signif. por umbral %s",
                    ms, len(res.table), res.tau,
                    {u: int((res.table["prob_better_than_xg"] > u).sum()) for u in umbrales})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# D. Evaluacion agregada LOCO
# --------------------------------------------------------------------------- #
def loco_pooled(X, y, comp, model_key):
    logger.info("== D. Evaluacion agregada de las rotaciones LOCO ==")
    comp = comp.reindex(X.index)
    pred = pd.Series(np.nan, index=X.index, dtype=float)
    for held in sorted(comp.dropna().unique()):
        te = comp == held
        cal = CalibratedClassifierCV(models.build_model_pipeline(model_key),
                                     method="isotonic", cv=config.CV_FOLDS)
        cal.fit(X[~te], y[~te])
        pred[te] = cal.predict_proba(X[te])[:, 1]
        logger.info("  rotacion %-22s | %d disparos puntuados", held, int(te.sum()))
    yy = np.asarray(y).astype(int)
    pp = pred.to_numpy(float)
    m = _m(yy, pp)
    ci = statistics.bootstrap_metrics_table(yy, pp, n_boot=2000)
    rows = []
    for k, nombre in [("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"),
                      ("brier_score", "brier"), ("log_loss", "log_loss")]:
        if k in ci.index:
            rows.append({"metrica": k, "valor": round(m[nombre], 4),
                         "ci_low": round(float(ci.loc[k, "ci_low"]), 4),
                         "ci_high": round(float(ci.loc[k, "ci_high"]), 4)})
    # El ECE y la media predicha no llevan IC bootstrap, pero deben quedar en el CSV
    # para que las cifras citadas en la memoria sean trazables a su fuente.
    rows.append({"metrica": "ece", "valor": round(m["ece"], 4),
                 "ci_low": "", "ci_high": ""})
    rows.append({"metrica": "mean_pred", "valor": round(m["mean_pred"], 4),
                 "ci_low": "", "ci_high": ""})
    rows.append({"metrica": "base_rate", "valor": round(float(yy.mean()), 4),
                 "ci_low": "", "ci_high": ""})
    df = pd.DataFrame(rows)
    df.attrs["n"] = len(yy)
    logger.info("  agregado: n=%d goles=%d | Brier=%.4f ECE=%.4f ROC=%.4f",
                len(yy), int(yy.sum()), m["brier"], m["ece"], m["roc_auc"])
    return df, m, int(yy.sum()), len(yy)


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

    dfA = seleccion_calibrada(Xtrval, ytrval)
    dfA.to_csv(out / "seleccion_cv_calibrada.csv", index=False)

    scored = pd.read_parquet(ROOT / "data" / "processed" / "shots_scored.parquet")
    dfC = sensibilidad(scored, [8, 10, 12, 15, 20], [0.90, 0.95, 0.99])
    dfC.to_csv(out / "sensibilidad_bayesiana.csv", index=False)

    dfB = potencia(scored, [0.0, 0.2, 0.34, 0.5, 0.8], n_rep=100)
    dfB.to_csv(out / "potencia_bayesiana.csv", index=False)

    dfD, mD, gol, n = loco_pooled(X, y, comp, "hist_gradient_boosting")
    dfD.to_csv(out / "loco_pooled_metrics.csv", index=False)

    logger.info("== Hecho en %.1f s ==", time.perf_counter() - t0)
    print("\n=== A. Seleccion con CV calibrada ===")
    print(dfA.round(4).to_string(index=False))
    print("\n=== C. Sensibilidad ===")
    print(dfC.to_string(index=False))
    print("\n=== B. Potencia ===")
    print(dfB.to_string(index=False))
    print(f"\n=== D. LOCO agregado (n={n}, goles={gol}) ===")
    print(dfD.to_string(index=False))
    print("ECE agregado:", round(mD["ece"], 4), "| media pred:", round(mD["mean_pred"], 4))


if __name__ == "__main__":
    main()
