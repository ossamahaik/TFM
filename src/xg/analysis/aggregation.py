"""
analysis.py
===========
Análisis agregado por jugador y equipo, e interpretabilidad del modelo.

A partir del xG predicho a nivel de disparo, este módulo construye los análisis
que dan valor "de negocio" a un modelo xG:

* **Rendimiento ofensivo agregado**: suma de xG por jugador/equipo, goles reales,
  y la diferencia *goles − xG* (over/under-performance), que mide finalización
  por encima/por debajo de lo esperado.
* **Eficiencia**: goles por xG, xG por disparo.
* **Interpretabilidad**: importancia de variables por permutación (agnóstica al
  modelo) y, si `shap` está disponible, valores SHAP.

La importancia por permutación se prefiere como método base porque es válida
para cualquier estimador y mide el impacto real sobre la métrica de evaluación.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from xg import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agregaciones por jugador y equipo
# --------------------------------------------------------------------------- #
def aggregate_by(df: pd.DataFrame, group_col: str, prob_col: str = "xg_pred") -> pd.DataFrame:
    """
    Agrega xG y goles por una entidad (jugador o equipo).

    Devuelve, por entidad: nº de disparos, xG total, goles reales, diferencia
    goles−xG (finalización), xG por disparo y conversión.
    """
    g = df.groupby(group_col)
    out = pd.DataFrame(
        {
            "shots": g.size(),
            "xg_total": g[prob_col].sum(),
            "goals": g[config.TARGET].sum(),
        }
    )
    out["goals_minus_xg"] = out["goals"] - out["xg_total"]
    out["xg_per_shot"] = out["xg_total"] / out["shots"].clip(lower=1)
    out["conversion"] = out["goals"] / out["shots"].clip(lower=1)
    out = out.reset_index().sort_values("xg_total", ascending=False).reset_index(drop=True)
    return out


def overperformers(agg: pd.DataFrame, min_shots: int = 15, group_col: str = "player") -> pd.DataFrame:
    """Entidades que más superan su xG (mejores finalizadores), con filtro de volumen."""
    sub = agg[agg["shots"] >= min_shots].copy()
    return sub.sort_values("goals_minus_xg", ascending=False).reset_index(drop=True)


def overperformers_with_significance(
    df: pd.DataFrame,
    group_col: str = "player",
    prob_col: str = "xg_pred",
    min_shots: int = 15,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Sobre/infra-rendimiento en finalización CON contraste estadístico.

    Crítica habitual a los rankings de "goles − xG": con pocos disparos, la
    diferencia es esencialmente ruido. Aquí se formaliza el contraste.

    Bajo la hipótesis nula de que la habilidad finalizadora del jugador coincide
    con su xG, el número de goles G es suma de Bernoullis independientes con
    probabilidades p_i (los xG de cada disparo), es decir una distribución
    **Poisson-binomial** con media ``μ = Σ p_i`` y varianza ``σ² = Σ p_i(1−p_i)``.
    Se usa la aproximación normal (válida con suficientes disparos) para construir
    un intervalo de confianza de ``G − μ`` y un p-valor bilateral:

        z = (G − μ) / σ ,   p = 2·(1 − Φ(|z|))

    Una diferencia es estadísticamente significativa solo si su IC no contiene 0.
    Esto evita presentar como "hallazgo" lo que es variación muestral.
    """
    from scipy import stats as sps

    rows = []
    for entity, grp in df.groupby(group_col):
        p = grp[prob_col].to_numpy(dtype=float)
        n = len(p)
        if n < min_shots:
            continue
        goals = int(grp[config.TARGET].sum())
        mu = float(p.sum())                       # xG total esperado
        var = float(np.sum(p * (1.0 - p)))        # varianza Poisson-binomial
        sigma = float(np.sqrt(var)) if var > 0 else np.nan
        diff = goals - mu
        if sigma and sigma > 0:
            z = diff / sigma
            p_value = float(2 * (1 - sps.norm.cdf(abs(z))))
            zc = sps.norm.ppf(1 - alpha / 2)
            ci_low, ci_high = diff - zc * sigma, diff + zc * sigma
        else:
            z, p_value, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan
        rows.append({
            group_col: entity,
            "shots": n,
            "goals": goals,
            "xg_total": mu,
            "goals_minus_xg": diff,
            "std_error": sigma,
            "z_score": z,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": bool(p_value < alpha) if not np.isnan(p_value) else False,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Corrección por comparaciones múltiples. El ranking frecuentista evalúa a
    # muchos jugadores a la vez; sin ajustar, el número esperado de falsos
    # positivos crece con el número de contrastes. Se añade Benjamini-Hochberg
    # (FDR) para que la tabla sea defendible ante tribunal. La columna
    # ``significant`` mantiene el contraste individual clásico;
    # ``significant_fdr`` es la conclusión prudente recomendada.
    pvals = out["p_value"].to_numpy(dtype=float)
    valid = ~np.isnan(pvals)
    out["p_value_fdr_bh"] = np.nan
    out["significant_fdr"] = False
    if valid.any():
        idx = np.where(valid)[0]
        order = idx[np.argsort(pvals[valid])]
        m = len(order)
        adjusted = np.empty(m, dtype=float)
        prev = 1.0
        for rank in range(m, 0, -1):
            orig_idx = order[rank - 1]
            val = min(prev, pvals[orig_idx] * m / rank)
            adjusted[rank - 1] = val
            prev = val
        out.loc[order, "p_value_fdr_bh"] = np.clip(adjusted, 0, 1)
        out.loc[order, "significant_fdr"] = out.loc[order, "p_value_fdr_bh"] < alpha

    return out.sort_values("goals_minus_xg", ascending=False).reset_index(drop=True)


def subgroup_performance(
    df: pd.DataFrame, prob_col: str = "xg_pred", min_shots: int = 30,
) -> pd.DataFrame:
    """
    Desempeño del modelo por subgrupos futbolísticos con sentido táctico.

    Para cada subgrupo (banda de distancia, dentro/fuera del área, jugada
    abierta o a balón parado, parte del cuerpo) calcula el xG medio predicho
    frente a la tasa real de gol con su IC de Wilson, además del Brier score
    del subgrupo. Es a la vez un resultado aplicado (¿dónde se generan los
    goles?) y una validación de calibración condicional (¿el xG medio coincide
    con la tasa real dentro de cada subgrupo, no solo en global?).
    """
    from xg.models.statistics import _wilson_interval

    bands = pd.cut(
        df["distance_to_goal"],
        bins=[0, 6, 11, 16, 22, np.inf],
        labels=["0-6 m", "6-11 m", "11-16 m", "16-22 m", ">22 m"],
    )
    groups = {
        "distancia": bands,
        "zona": df["shot_in_box"].map({1: "Dentro del área", 0: "Fuera del área"}),
        "jugada": df["is_open_play"].map({1: "Jugada abierta", 0: "Balón parado"}),
        "parte_cuerpo": df["body_part"],
    }

    rows = []
    for dimension, group_values in groups.items():
        sub_df = df.assign(_group=group_values)
        for subgroup, grp in sub_df.groupby("_group", observed=True):
            n = len(grp)
            if n < min_shots:
                continue
            k = int(grp[config.TARGET].sum())
            p = grp[prob_col].to_numpy(dtype=float)
            y = grp[config.TARGET].to_numpy(dtype=int)
            ci_low, ci_high = _wilson_interval(k, n)
            rows.append({
                "dimension": dimension,
                "subgrupo": subgroup,
                "n_shots": n,
                "mean_xg_pred": float(p.mean()),
                "obs_goal_rate": k / n,
                "obs_ci_low": ci_low,
                "obs_ci_high": ci_high,
                "brier_score": float(np.mean((p - y) ** 2)),
            })
    out = pd.DataFrame(rows)
    return out


def benchmark_vs_statsbomb(
    df: pd.DataFrame, pred_col: str = "xg_pred", ref_col: str = "statsbomb_xg"
) -> Optional[pd.DataFrame]:
    """
    Compara el xG del modelo propio con el xG oficial de StatsBomb.

    StatsBomb publica su propio xG (un modelo profesional entrenado con millones
    de disparos y decenas de variables). Usarlo como referencia externa permite
    situar la calidad del modelo propio: si ambas estimaciones concuerdan a nivel
    agregado y de disparo, es una validación independiente potente.

    Devuelve métricas de acuerdo (correlación de Pearson y Spearman, MAE, sesgo
    medio) entre ambas estimaciones sobre los disparos con xG de StatsBomb
    disponible, o None si la columna de referencia no existe.
    """
    if ref_col not in df.columns:
        return None
    sub = df[[pred_col, ref_col, config.TARGET]].dropna(subset=[pred_col, ref_col])
    if sub.empty:
        return None
    from scipy import stats as sps

    a = sub[pred_col].to_numpy(dtype=float)
    b = sub[ref_col].to_numpy(dtype=float)
    y = sub[config.TARGET].to_numpy(dtype=int)
    # Brier de cada modelo frente al resultado real, para comparar quién predice mejor.
    brier_own = float(np.mean((a - y) ** 2))
    brier_sb = float(np.mean((b - y) ** 2))
    metrics = {
        "n_shots": int(len(sub)),
        "pearson_r": float(sps.pearsonr(a, b)[0]),
        "spearman_r": float(sps.spearmanr(a, b)[0]),
        "mae_vs_statsbomb": float(np.mean(np.abs(a - b))),
        "mean_bias_own_minus_sb": float(np.mean(a - b)),
        "brier_own_model": brier_own,
        "brier_statsbomb": brier_sb,
        "sum_xg_own": float(a.sum()),
        "sum_xg_statsbomb": float(b.sum()),
        "total_goals": int(y.sum()),
    }
    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


def save_table(df: pd.DataFrame, name: str):
    """Persiste una tabla agregada en data/outputs como CSV."""
    config.ensure_dirs()
    path = config.OUTPUTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("Tabla guardada en %s", path)
    return path


# --------------------------------------------------------------------------- #
# Interpretabilidad
# --------------------------------------------------------------------------- #
def permutation_feature_importance(
    model, X, y, n_repeats: int = 20, scoring: str = "roc_auc"
) -> pd.Series:
    """
    Importancia por permutación sobre el Pipeline completo (interpreta variables
    de entrada originales, no las post-OHE). Mide la caída de rendimiento al
    permutar cada variable: cuanto mayor la caída, más importante la variable.

    Por defecto usa ``roc_auc`` por ser más interpretable (la caída está en la
    misma escala que el AUC). Para la calidad probabilística puede usarse
    ``neg_brier_score`` o ``neg_log_loss``.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=config.RANDOM_STATE,
        scoring=scoring, n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def permutation_importance_report(
    model, X, y, n_repeats: int = 20,
    scorings=("roc_auc", "neg_brier_score"),
) -> pd.DataFrame:
    """
    Informe de importancia por permutación con varias métricas a la vez y su
    desviación típica entre repeticiones (mide la estabilidad de la importancia).

    Devuelve un DataFrame indexado por variable con columnas
    ``<scoring>_mean`` y ``<scoring>_std``, ordenado por la primera métrica.
    """
    cols = {}
    for sc in scorings:
        res = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=config.RANDOM_STATE,
            scoring=sc, n_jobs=-1,
        )
        cols[f"{sc}_mean"] = pd.Series(res.importances_mean, index=X.columns)
        cols[f"{sc}_std"] = pd.Series(res.importances_std, index=X.columns)
    report = pd.DataFrame(cols)
    return report.sort_values(f"{scorings[0]}_mean", ascending=False)


def shap_summary(model, X, max_display: int = 20) -> Optional[object]:
    """
    Calcula valores SHAP si la librería está instalada. Devuelve el objeto de
    explicación o None. Se aísla para no acoplar el proyecto a una dependencia
    opcional pesada.
    """
    try:  # pragma: no cover
        import shap

        # Se explica el estimador final sobre los datos ya preprocesados.
        pre = model.named_steps["preprocess"]
        est = model.named_steps["model"]
        X_trans = pre.transform(X)
        explainer = shap.Explainer(est, X_trans)
        return explainer(X_trans)
    except Exception as exc:  # noqa: BLE001
        logger.info("SHAP no disponible (%s).", exc)
        return None
