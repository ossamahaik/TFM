"""
bayesian.py — Modelo jerárquico bayesiano de habilidad finalizadora (EM + Laplace).

Motivación
----------
El contraste frecuentista de finalización (``analysis.overperformers_with_significance``)
formaliza si un jugador difiere significativamente de su xG mediante la
distribución Poisson-binomial. Ese enfoque tiene dos limitaciones bien conocidas:

1. **Sensibilidad a muestras pequeñas.** Un jugador con 4 goles en 8 disparos de
   bajo xG aparece como "sobre-rendidor extremo", pero la evidencia es débil.
2. **No comparte información entre jugadores.** Cada jugador se evalúa aislado,
   ignorando que la mayoría de futbolistas finalizan cerca de su xG.

El modelo jerárquico bayesiano resuelve ambas mediante *partial pooling*
(encogimiento / *shrinkage*): las estimaciones individuales se contraen hacia la
media poblacional en proporción a su incertidumbre, de modo que jugadores con
pocos disparos se regularizan fuertemente y solo los respaldados por evidencia
abundante se separan del promedio. Es el remedio canónico al problema de estimar
muchas medias relacionadas (Efron & Morris, 1975; James-Stein).

Especificación del modelo
-------------------------
Sea ``g_ij ∈ {0,1}`` el resultado del disparo ``j`` del jugador ``i`` y ``p_ij``
su xG estimado. Con log-odds base ``η_ij = logit(p_ij)`` se modela un efecto
aleatorio de jugador ``u_i`` que desplaza el log-odds:

    g_ij ~ Bernoulli( σ(η_ij + u_i) ),     σ(x) = 1/(1+e^{-x})
    u_i  ~ Normal(0, τ²)

``u_i`` es la habilidad de finalización (log-odds); ``exp(u_i)`` es el
multiplicador sobre las *odds* de marcar respecto del xG. ``τ²`` mide la
heterogeneidad real de finalización en la población.

Inferencia: Bayes empírico vía EM + Laplace
-------------------------------------------
Se estima ``τ²`` por máxima verosimilitud marginal con EM, aproximando la
marginal por jugador mediante Laplace:

- **Paso E.** Con ``τ²`` fijo se halla el MAP ``û_i`` (la log-posterior es
  cóncava) por Newton-Raphson, y su varianza posterior de Laplace
  ``s_i² = 1 / (−ℓ''(û_i))``, con

      ℓ(u)   = Σ_j [ g_ij(η_ij+u) − log(1+e^{η_ij+u}) ] − u²/(2τ²)
      ℓ'(u)  = Σ_j [ g_ij − σ(η_ij+u) ] − u/τ²
      ℓ''(u) = −Σ_j σ(η_ij+u)(1−σ(η_ij+u)) − 1/τ²

- **Paso M.** Se actualiza ``τ² ← (1/N) Σ_i (û_i² + s_i²)``.

Se itera hasta convergencia de ``τ``. Para muestreo completo del posterior
(R̂, ESS, HDI) está disponible ``fit_hierarchical_pymc`` si PyMC está instalado.

La fiabilidad de la aproximación de Laplace se valida en ``scripts/run_advanced.py``
frente a la posterior exacta calculada por cuadratura numérica (error despreciable en
la muestra del proyecto: |Δu| < 0.01 en log-odds, |ΔP(u>0)| < 0.012, correlación
0.9999; τ_EM = 0.3374 frente a τ exacto por verosimilitud marginal = 0.3150).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats as sps

from xg import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return special.expit(x)


def _logit(p: np.ndarray) -> np.ndarray:
    return special.logit(np.clip(p, 1e-9, 1 - 1e-9))


@dataclass
class HierarchicalFinishingResult:
    """Resultado del ajuste jerárquico de finalización (EM + Laplace)."""

    table: pd.DataFrame                          # una fila por jugador
    tau: float                                   # heterogeneidad poblacional
    tau_history: List[float] = field(default_factory=list)  # traza EM (diagnóstico)
    n_iter: int = 0
    converged: bool = False
    backend: str = "em_laplace"

    def shrinkage_summary(self) -> str:
        """Resumen legible del ajuste (logs y notebooks)."""
        n = len(self.table)
        over = int((self.table["prob_better_than_xg"] > 0.95).sum()) if n else 0
        under = int((self.table["prob_better_than_xg"] < 0.05).sum()) if n else 0
        if not self.converged and self.tau_history and \
                self.tau_history[-1] < self.tau_history[0] * 0.6:
            # τ desciende de forma sostenida hacia 0: diagnóstico (no fallo) de
            # ausencia de heterogeneidad real de finalización en la población.
            conv = " EM no estacionario: τ↓ hacia 0 (sin habilidad diferencial)."
        elif self.converged:
            conv = f" EM convergió en {self.n_iter} iter."
        else:
            conv = f" EM detenido en {self.n_iter} iter."
        return (
            f"τ̂ = {self.tau:.4f} (log-odds). Jugadores: {n}. "
            f"P>0.95 (sobre-xG): {over}; P<0.05 (bajo-xG): {under}.{conv}"
        )

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.table.sort_values("u_hat", ascending=False).head(n)


def _map_and_laplace(eta: np.ndarray, g: np.ndarray, tau2: float,
                     tol: float = 1e-9, max_iter: int = 100) -> Tuple[float, float]:
    """
    MAP del efecto de jugador por Newton-Raphson y su varianza de Laplace.

    Returns (u_hat, s2) con s2 = 1 / (−ℓ''(u_hat)).
    """
    u = 0.0
    for _ in range(max_iter):
        p = _sigmoid(eta + u)
        grad = float(np.sum(g - p) - u / tau2)
        hess = float(-np.sum(p * (1.0 - p)) - 1.0 / tau2)  # cóncava ⇒ hess < 0
        if hess == 0:
            break
        step = grad / hess
        u_new = u - step
        if abs(u_new - u) < tol:
            u = u_new
            break
        u = u_new
    p = _sigmoid(eta + u)
    info = float(np.sum(p * (1.0 - p)) + 1.0 / tau2)       # −ℓ''(u) > 0
    s2 = 1.0 / info if info > 0 else np.nan
    return u, s2


def fit_hierarchical_finishing(
    df: pd.DataFrame,
    group_col: str = "player",
    prob_col: str = "xg_pred",
    min_shots: int = 12,
    tau_init: float = 0.5,
    tol: float = 1e-6,
    atol: float = 1e-4,
    max_iter: int = 500,
) -> HierarchicalFinishingResult:
    """
    Ajusta el modelo jerárquico de finalización por Bayes empírico (EM+Laplace).

    Devuelve una tabla por jugador con ``u_hat`` (habilidad en log-odds),
    ``u_se`` (error estándar posterior), ``odds_multiplier`` = exp(u_hat),
    ``prob_better_than_xg`` = Φ(û/s), ``shrunk_goals_minus_xg`` (sobre-rendimiento
    en goles tras el encogimiento) e intervalos de credibilidad.
    """
    groups: Dict[str, Dict] = {}
    for entity, grp in df.groupby(group_col):
        if len(grp) < min_shots:
            continue
        p = np.clip(grp[prob_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        groups[entity] = {
            "eta": _logit(p),
            "g": grp[config.TARGET].to_numpy(dtype=float),
            "shots": int(len(grp)),
            "goals": float(grp[config.TARGET].sum()),
            "xg_total": float(p.sum()),
        }

    if not groups:
        return HierarchicalFinishingResult(table=pd.DataFrame(), tau=0.0)

    tau2 = float(tau_init ** 2)
    tau_history: List[float] = [float(np.sqrt(tau2))]
    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        sq_acc = 0.0
        for d in groups.values():
            u_hat, s2 = _map_and_laplace(d["eta"], d["g"], tau2)
            sq_acc += u_hat * u_hat + s2
        tau2_new = float(sq_acc / len(groups))
        tau_new = float(np.sqrt(tau2_new))
        tau_old = float(np.sqrt(tau2))
        tau_history.append(tau_new)
        # Criterio de convergencia DOBLE: relativo (robusto en régimen general)
        # y absoluto (necesario cuando τ→0, donde el cambio relativo decrece muy
        # despacio por el término de varianza de Laplace s² en el paso M). El EM
        # de Bayes empírico tiene convergencia sublineal cerca del óptimo; el
        # criterio absoluto evita iteraciones improductivas sin parar antes de
        # tiempo en presencia de heterogeneidad real.
        abs_change = abs(tau_new - tau_old)
        rel_change = abs_change / max(tau_old, 1e-6)
        if rel_change < tol or abs_change < atol or tau_new < 1e-3:
            tau2 = tau2_new
            converged = True
            break
        tau2 = tau2_new

    rows = []
    z975 = sps.norm.ppf(0.975)
    for entity, d in groups.items():
        u_hat, s2 = _map_and_laplace(d["eta"], d["g"], tau2)
        s = float(np.sqrt(s2)) if s2 and s2 > 0 else np.nan
        # goles esperados bajo la habilidad encogida (suma de σ(η+û))
        adj_goals = float(np.sum(_sigmoid(d["eta"] + u_hat)))
        prob_better = float(sps.norm.cdf(u_hat / s)) if s and s > 0 else np.nan
        rows.append({
            group_col: entity,
            "shots": d["shots"],
            "goals": int(d["goals"]),
            "xg_total": d["xg_total"],
            "goals_minus_xg": d["goals"] - d["xg_total"],
            "u_hat": u_hat,
            "u_se": s,
            "u_ci_low": u_hat - z975 * s if s else np.nan,
            "u_ci_high": u_hat + z975 * s if s else np.nan,
            "odds_multiplier": float(np.exp(u_hat)),
            "shrunk_goals_minus_xg": adj_goals - d["xg_total"],
            "prob_better_than_xg": prob_better,
        })
    table = pd.DataFrame(rows).sort_values("u_hat", ascending=False).reset_index(drop=True)

    return HierarchicalFinishingResult(
        table=table, tau=float(np.sqrt(tau2)), tau_history=tau_history,
        n_iter=n_iter, converged=converged, backend="em_laplace",
    )


def fit_hierarchical_pymc(
    df: pd.DataFrame,
    group_col: str = "player",
    prob_col: str = "xg_pred",
    min_shots: int = 12,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = config.RANDOM_STATE,
) -> HierarchicalFinishingResult:
    """
    Versión MCMC (NUTS) del modelo jerárquico, parametrización no centrada:

        μ ~ Normal(0,1); τ ~ HalfNormal(1); z_i ~ Normal(0,1); u_i = μ + τ·z_i
        g_ij ~ Bernoulli( σ(η_ij + u_i) )

    Requiere ``pip install -e ".[bayes]"``. Devuelve la misma estructura de
    resultado con la media posterior de ``u_i`` y diagnósticos R̂/ESS.
    """
    try:  # pragma: no cover
        import pymc as pm
        import arviz as az
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        raise ImportError(
            "fit_hierarchical_pymc requiere pymc y arviz. "
            "Instálalos con: pip install -e \".[bayes]\". "
            "Alternativa sin dependencias: fit_hierarchical_finishing (EM+Laplace)."
        ) from exc

    work = df[df.groupby(group_col)[group_col].transform("size") >= min_shots].copy()  # pragma: no cover
    cats = work[group_col].astype("category")
    idx = cats.cat.codes.to_numpy()
    labels = list(cats.cat.categories)
    eta = _logit(work[prob_col].to_numpy(dtype=float))
    y = work[config.TARGET].to_numpy(dtype=int)

    with pm.Model():  # pragma: no cover
        mu = pm.Normal("mu", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=len(labels))
        u = pm.Deterministic("u", mu + tau * z)
        pm.Bernoulli("obs", logit_p=eta + u[idx], observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          progressbar=False)

    post_u = idata.posterior["u"]  # pragma: no cover
    u_mean = post_u.mean(dim=["chain", "draw"]).to_numpy()
    u_sd = post_u.std(dim=["chain", "draw"]).to_numpy()
    hdi = az.hdi(idata, var_names=["u"])["u"].to_numpy()
    summ = az.summary(idata, var_names=["u"])

    agg = {e: (int((cats == e).sum()),
               float(work.loc[cats == e, config.TARGET].sum()),
               float(np.clip(work.loc[cats == e, prob_col], 1e-6, 1-1e-6).sum()))
           for e in labels}
    rows = []
    for i, e in enumerate(labels):
        shots, goals, xgt = agg[e]
        prob_better = float((post_u[:, :, i].values > 0).mean())
        rows.append({
            group_col: e, "shots": shots, "goals": int(goals), "xg_total": xgt,
            "goals_minus_xg": goals - xgt, "u_hat": float(u_mean[i]),
            "u_se": float(u_sd[i]), "u_ci_low": float(hdi[i, 0]),
            "u_ci_high": float(hdi[i, 1]), "odds_multiplier": float(np.exp(u_mean[i])),
            "shrunk_goals_minus_xg": np.nan, "prob_better_than_xg": prob_better,
        })
    table = pd.DataFrame(rows).sort_values("u_hat", ascending=False).reset_index(drop=True)
    return HierarchicalFinishingResult(
        table=table, tau=float(idata.posterior["tau"].mean()),
        tau_history=[], n_iter=draws, converged=bool(summ["r_hat"].max() < 1.01),
        backend="pymc",
    )


def compare_frequentist_bayesian(
    freq_table: pd.DataFrame,
    bayes_result: HierarchicalFinishingResult,
    group_col: str = "player",
) -> pd.DataFrame:
    """
    Fusiona el contraste frecuentista (Poisson-binomial) con la estimación
    bayesiana regularizada, jugador a jugador, para evidenciar el *shrinkage*:
    casos señalados como "extremos" por el método crudo se regularizan al
    incorporar la incertidumbre de muestra pequeña.
    """
    if freq_table.empty or bayes_result.table.empty:
        return pd.DataFrame()
    f = freq_table[[group_col, "shots", "goals_minus_xg", "p_value", "significant"]].copy()
    f = f.rename(columns={"goals_minus_xg": "freq_goals_minus_xg"})
    b = bayes_result.table[[group_col, "u_hat", "u_se", "odds_multiplier",
                            "shrunk_goals_minus_xg", "prob_better_than_xg"]].copy()
    merged = f.merge(b, on=group_col, how="inner")
    return merged.sort_values("u_hat", ascending=False).reset_index(drop=True)
