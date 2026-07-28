"""
finishing.py — Modelo jerárquico bayesiano de habilidad de finalización.

Motivación
==========
El contraste frecuentista de finalización (``analysis.overperformers_with_significance``)
evalúa a cada jugador de forma aislada mediante un test Poisson-binomial. Tiene
dos limitaciones bien conocidas:

1. **Estimación inestable con pocos disparos.** El estadístico ``goles − xG`` de
   un jugador con 12 disparos es dominado por el ruido binomial; un par de goles
   afortunados lo catapultan al "ranking de mejores finalizadores".
2. **Comparaciones múltiples.** Al testear cientos de jugadores a α = 0.05 se
   esperan decenas de falsos positivos; el control individual del error no
   controla el error familiar.

El enfoque jerárquico bayesiano resuelve ambas con *partial pooling*: en lugar de
estimar la habilidad de cada jugador por separado (*no pooling*) o asumir que
todos finalizan igual (*complete pooling*), se modela la habilidad individual
como una desviación respecto de una media poblacional, con una varianza que el
propio modelo estima a partir de los datos. El resultado es un **encogimiento
adaptativo** (*shrinkage*): los jugadores con pocos disparos se acercan a la
media (su señal es débil), y los jugadores con muchos disparos conservan su
estimación individual (su señal es fuerte).

Modelo
======
Sea el jugador ``j`` con disparos ``i = 1..n_j``, cada uno con xG ``p_ij``
estimado por el modelo de ML (calibrado, *out-of-fold*). Definimos el *offset*
en log-odds de cada disparo como ``η_ij = logit(p_ij)``. Modelamos el resultado:

    y_ij ~ Bernoulli( σ(η_ij + θ_j) )                 (verosimilitud)
    θ_j  ~ Normal(μ, τ²)                              (prior jerárquico)

donde ``θ_j`` es la **habilidad de finalización** del jugador en log-odds: un
``θ_j > 0`` indica que el jugador convierte por encima de lo que su xG predice
(buen finalizador), y ``θ_j < 0`` por debajo. El prior jerárquico
``Normal(μ, τ²)`` acopla a todos los jugadores: ``μ`` es la habilidad media de la
población (típicamente ≈ 0 si el xG está bien calibrado) y ``τ`` mide cuánta
heterogeneidad real de finalización existe.

Este módulo ofrece dos backends:

- ``fit_empirical_bayes`` (por defecto, sin dependencias externas): aproximación
  Beta-Binomial conjugada con estimación de hiperparámetros por máxima
  verosimilitud marginal (*empirical Bayes*). Determinista, en milisegundos.
- ``fit_pymc`` (opcional, requiere ``pymc``): el modelo logístico-jerárquico
  completo muestreado por MCMC (NUTS), con diagnósticos ArviZ.

Ambos devuelven estimaciones encogidas e intervalos de credibilidad por jugador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from xg import config


# --------------------------------------------------------------------------- #
# Estructura de resultados
# --------------------------------------------------------------------------- #
@dataclass
class HierarchicalFinishingResult:
    """Resultado del ajuste jerárquico de finalización."""

    estimates: pd.DataFrame          # una fila por jugador (estimaciones encogidas)
    hyperparams: Dict[str, float]    # mu, tau (y alpha,beta en Beta-Binomial)
    backend: str
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def top(self, n: int = 10, by: str = "skill_mean") -> pd.DataFrame:
        """Devuelve los ``n`` mejores finalizadores según la columna ``by``."""
        return self.estimates.sort_values(by, ascending=False).head(n)

    def credibly_nonzero(self) -> pd.DataFrame:
        """Jugadores cuyo IC de credibilidad al 95 % de la habilidad excluye 0."""
        e = self.estimates
        mask = (e["skill_hdi_low"] > 0) | (e["skill_hdi_high"] < 0)
        return e[mask].sort_values("skill_mean", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------------- #
    # API de conveniencia / compatibilidad
    # ----------------------------------------------------------------- #
    @property
    def tau(self) -> float:
        """Desviación típica poblacional de la habilidad (heterogeneidad)."""
        if "tau_mean" in self.hyperparams:
            return float(self.hyperparams["tau_mean"])
        # En Beta-Binomial, τ se deriva de la dispersión de las habilidades
        # estimadas en log-odds (proxy de la heterogeneidad poblacional).
        if self.estimates.empty:
            return 0.0
        return float(self.estimates["skill_mean"].std(ddof=0))

    @property
    def table(self) -> pd.DataFrame:
        """Tabla por jugador con columnas de conveniencia (u_hat, prob_better_than_xg)."""
        if self.estimates.empty:
            return self.estimates
        t = self.estimates.copy()
        # u_hat: habilidad estimada en log-odds (alias de skill_mean).
        t["u_hat"] = t["skill_mean"]
        # Probabilidad posterior de finalizar mejor que el xG, P(θ_j > 0).
        # Aproximación normal a partir del HDI simétrico en log-odds.
        half_width = (t["skill_hdi_high"] - t["skill_hdi_low"]) / 2.0
        sd = half_width / 1.959964  # HDI 95% -> sd
        sd = sd.replace(0.0, np.nan)
        from scipy import stats as _sps
        prob = _sps.norm.cdf((t["skill_mean"] / sd).to_numpy())
        t["prob_better_than_xg"] = pd.Series(prob, index=t.index).fillna(0.5)
        return t


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _prepare_groups(
    df: pd.DataFrame, group_col: str, prob_col: str, min_shots: int
) -> pd.DataFrame:
    """Agrega por jugador: nº de disparos, goles y xG total (esperado)."""
    rows = []
    for entity, grp in df.groupby(group_col):
        p = grp[prob_col].to_numpy(dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        n = len(p)
        if n < min_shots:
            continue
        rows.append({
            group_col: entity,
            "shots": n,
            "goals": int(grp[config.TARGET].sum()),
            "xg_total": float(p.sum()),
            "_probs": p,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Backend 1 — Empirical Bayes Beta-Binomial (sin dependencias externas)
# --------------------------------------------------------------------------- #
def fit_empirical_bayes(
    df: pd.DataFrame,
    group_col: str = "player",
    prob_col: str = "xg_pred",
    min_shots: int = 10,
    hdi_prob: float = 0.95,
) -> HierarchicalFinishingResult:
    r"""
    Ajuste jerárquico por *empirical Bayes* con conjugación Beta-Binomial.

    Parametrización
    ---------------
    Para cada jugador se define su **tasa de conversión esperada** bajo el xG,
    ``q_j = xG_total_j / n_j`` (la probabilidad media de gol que predice el
    modelo). Se modela la tasa de conversión *real* del jugador, ``r_j``, con un
    prior poblacional Beta cuyo modo se ancla en la tasa esperada del jugador:

        goles_j ~ Binomial(n_j, r_j)
        r_j     ~ Beta(α_j, β_j),   con  α_j = κ·q_j + 1,  β_j = κ·(1−q_j) + 1

    El hiperparámetro ``κ ≥ 0`` (concentración) es común a toda la población y se
    estima por **máxima verosimilitud marginal** integrando ``r_j`` (la marginal
    es Beta-Binomial). ``κ`` grande ⇒ priors concentrados ⇒ mucho encogimiento
    hacia el xG; ``κ`` pequeño ⇒ priors difusos ⇒ poco encogimiento. Así el
    *grado* de encogimiento lo deciden los datos, no el analista.

    Posterior
    ---------
    Por conjugación, el posterior de ``r_j`` es de nuevo Beta:

        r_j | datos ~ Beta(α_j + goles_j, β_j + n_j − goles_j)

    La **habilidad de finalización** se reporta en dos escalas:
    - ``conv_*``: tasa de conversión posterior (media e IC de credibilidad).
    - ``skill_*``: efecto en log-odds frente al xG, ``θ_j = logit(r_j) − logit(q_j)``,
      comparable directamente con el contraste frecuentista.

    Returns
    -------
    HierarchicalFinishingResult con un DataFrame (una fila por jugador) y los
    hiperparámetros estimados.
    """
    g = _prepare_groups(df, group_col, prob_col, min_shots)
    if g.empty:
        return HierarchicalFinishingResult(
            estimates=pd.DataFrame(), hyperparams={}, backend="empirical_bayes"
        )

    n = g["shots"].to_numpy(dtype=float)
    k = g["goals"].to_numpy(dtype=float)
    q = np.clip(g["xg_total"].to_numpy(dtype=float) / n, 1e-4, 1 - 1e-4)  # tasa esperada

    # --- Log-verosimilitud marginal Beta-Binomial como función de log κ ----- #
    # Marginal: P(k|n,α,β) = C(n,k) · B(k+α, n−k+β) / B(α,β)
    def neg_log_marginal(log_kappa: float) -> float:
        kappa = np.exp(log_kappa)
        a = kappa * q + 1.0
        b = kappa * (1.0 - q) + 1.0
        ll = (
            special.betaln(k + a, n - k + b)
            - special.betaln(a, b)
        )
        # término combinatorio constante en κ -> se omite para la optimización
        return -float(np.sum(ll))

    # Optimización 1-D robusta del hiperparámetro de concentración κ.
    res = optimize.minimize_scalar(
        neg_log_marginal, bounds=(np.log(1e-3), np.log(1e4)), method="bounded"
    )
    kappa_hat = float(np.exp(res.x))

    # --- Posterior Beta por jugador ---------------------------------------- #
    a = kappa_hat * q + 1.0
    b = kappa_hat * (1.0 - q) + 1.0
    a_post = a + k
    b_post = b + (n - k)

    conv_mean = a_post / (a_post + b_post)
    lo = (1 - hdi_prob) / 2
    hi = 1 - lo
    conv_lo = stats.beta.ppf(lo, a_post, b_post)
    conv_hi = stats.beta.ppf(hi, a_post, b_post)

    # Habilidad en log-odds frente al xG (offset = logit de la tasa esperada).
    logit = special.logit
    skill_mean = logit(conv_mean) - logit(q)
    skill_lo = logit(conv_lo) - logit(q)
    skill_hi = logit(conv_hi) - logit(q)

    # Encogimiento explícito: cuánto se ha movido la tasa cruda hacia el prior.
    raw_rate = k / n
    shrinkage = np.where(
        np.abs(raw_rate - q) > 1e-9,
        1.0 - np.abs(conv_mean - q) / np.abs(raw_rate - q),
        np.nan,
    )

    est = pd.DataFrame({
        group_col: g[group_col].to_numpy(),
        "shots": g["shots"].to_numpy(),
        "goals": g["goals"].to_numpy(),
        "xg_total": g["xg_total"].to_numpy(),
        "raw_conversion": raw_rate,
        "expected_conversion": q,
        "conv_mean": conv_mean,
        "conv_hdi_low": conv_lo,
        "conv_hdi_high": conv_hi,
        "skill_mean": skill_mean,
        "skill_hdi_low": skill_lo,
        "skill_hdi_high": skill_hi,
        "shrinkage": shrinkage,
        # goles esperados frente a posterior, para interpretabilidad directa
        "goals_minus_xg_raw": k - g["xg_total"].to_numpy(),
        "goals_minus_xg_shrunk": conv_mean * n - g["xg_total"].to_numpy(),
    }).sort_values("skill_mean", ascending=False).reset_index(drop=True)

    return HierarchicalFinishingResult(
        estimates=est,
        hyperparams={
            "kappa": kappa_hat,
            "population_mean_conversion": float(np.average(q, weights=n)),
            "neg_log_marginal": float(res.fun),
        },
        backend="empirical_bayes",
        diagnostics={"converged": bool(res.success)},
    )


# --------------------------------------------------------------------------- #
# Backend 2 — PyMC (modelo logístico jerárquico completo, opcional)
# --------------------------------------------------------------------------- #
def pymc_available() -> bool:
    """Indica si PyMC está disponible en el entorno."""
    try:  # pragma: no cover
        import pymc  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def fit_pymc(
    df: pd.DataFrame,
    group_col: str = "player",
    prob_col: str = "xg_pred",
    min_shots: int = 10,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    hdi_prob: float = 0.95,
    random_seed: int = config.RANDOM_STATE,
) -> HierarchicalFinishingResult:
    r"""
    Modelo logístico jerárquico completo muestreado con MCMC (NUTS) vía PyMC.

    Especificación (no centrada, para mejorar la geometría del muestreo):

        μ      ~ Normal(0, 1)                     # habilidad media poblacional
        τ      ~ HalfNormal(1)                    # heterogeneidad entre jugadores
        z_j    ~ Normal(0, 1)                     # efecto estandarizado por jugador
        θ_j    = μ + τ · z_j                       # habilidad del jugador (log-odds)
        y_ij   ~ Bernoulli( σ( logit(p_ij) + θ_{j(i)} ) )

    El *offset* ``logit(p_ij)`` ancla cada disparo en su xG calibrado, de modo que
    ``θ_j`` mide exclusivamente la desviación sistemática del jugador respecto del
    modelo. Devuelve la media posterior y el HDI de ``θ_j`` por jugador, además de
    los diagnósticos de convergencia (R-hat, ESS).

    Requiere ``pip install -e ".[bayes]"`` (PyMC + ArviZ).
    """
    if not pymc_available():  # pragma: no cover
        raise ImportError(
            "fit_pymc requiere PyMC. Instálalo con: pip install -e \".[bayes]\". "
            "Alternativamente usa fit_empirical_bayes (sin dependencias)."
        )
    import pymc as pm  # pragma: no cover
    import arviz as az  # pragma: no cover

    work = df[df.groupby(group_col)[group_col].transform("size") >= min_shots].copy()
    players = work[group_col].astype("category")
    idx = players.cat.codes.to_numpy()
    labels = list(players.cat.categories)
    eta = special.logit(np.clip(work[prob_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    y = work[config.TARGET].to_numpy(dtype=int)

    with pm.Model() as model:  # pragma: no cover
        mu = pm.Normal("mu", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=len(labels))
        theta = pm.Deterministic("theta", mu + tau * z)
        logit_p = eta + theta[idx]
        pm.Bernoulli("obs", logit_p=logit_p, observed=y)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            random_seed=random_seed, progressbar=False,
        )

    post = idata.posterior["theta"]  # pragma: no cover
    summary = az.summary(idata, var_names=["theta"], hdi_prob=hdi_prob)  # pragma: no cover
    skill_mean = post.mean(dim=["chain", "draw"]).to_numpy()
    hdi = az.hdi(idata, var_names=["theta"], hdi_prob=hdi_prob)["theta"].to_numpy()

    agg = _prepare_groups(work, group_col, prob_col, min_shots).set_index(group_col)
    est = pd.DataFrame({
        group_col: labels,
        "skill_mean": skill_mean,
        "skill_hdi_low": hdi[:, 0],
        "skill_hdi_high": hdi[:, 1],
    })
    est = est.merge(agg.reset_index()[[group_col, "shots", "goals", "xg_total"]], on=group_col)
    est = est.sort_values("skill_mean", ascending=False).reset_index(drop=True)

    rhat = float(summary["r_hat"].max())
    ess = float(summary["ess_bulk"].min())
    return HierarchicalFinishingResult(
        estimates=est,
        hyperparams={
            "mu_mean": float(idata.posterior["mu"].mean()),
            "tau_mean": float(idata.posterior["tau"].mean()),
        },
        backend="pymc",
        diagnostics={"max_rhat": rhat, "min_ess_bulk": ess},
    )


def fit_hierarchical_finishing(df: pd.DataFrame, prefer: str = "auto", **kwargs):
    """
    Punto de entrada de alto nivel. Selecciona el backend:

    - ``prefer="auto"`` (defecto): usa PyMC si está instalado, si no empirical Bayes.
    - ``prefer="empirical_bayes"``: fuerza el backend conjugado (rápido, sin deps).
    - ``prefer="pymc"``: fuerza MCMC (lanza ImportError si PyMC no está).
    """
    if prefer == "empirical_bayes":
        return fit_empirical_bayes(df, **{k: v for k, v in kwargs.items()
                                          if k in {"group_col", "prob_col", "min_shots", "hdi_prob"}})
    if prefer == "pymc":
        return fit_pymc(df, **kwargs)
    # auto
    if pymc_available():
        return fit_pymc(df, **kwargs)
    return fit_empirical_bayes(df, **{k: v for k, v in kwargs.items()
                                      if k in {"group_col", "prob_col", "min_shots", "hdi_prob"}})
