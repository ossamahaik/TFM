# -*- coding: utf-8 -*-
"""Distribución nula del hiperparámetro tau del modelo jerárquico de finalización.

Pregunta que responde
---------------------
El análisis bayesiano estima tau = 0,3374 sobre los datos reales. ¿Es ese valor
prueba de que existe heterogeneidad de finalización entre jugadores, o es lo que
el estimador devuelve incluso cuando nadie tiene habilidad diferencial?

Diseño
------
Se conserva íntegra la estructura real del problema —los mismos 97 jugadores, sus
mismos disparos y el xG que el modelo calibrado les asignó fuera de muestra— y solo
se resortean los goles como Bernoulli(xg_pred). Eso es exactamente la hipótesis
nula: la probabilidad de gol de cada disparo es la que predice el modelo, sin
efecto de jugador. Sobre cada réplica se aplica el mismo estimador de Bayes
empírico (EM + aproximación de Laplace) que usa el análisis real.

Salida
------
data/results/tau_distribucion_nula.csv  con los estadísticos de la distribución
nula y el p-valor empírico del valor observado.

Uso
---
    python scripts/run_tau_null.py [n_replicas]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg.models.bayesian import fit_hierarchical_finishing  # noqa: E402

SCORED = ROOT / "data" / "processed" / "shots_scored.parquet"
TRACE = ROOT / "data" / "results" / "bayesian_em_trace.csv"
OUT = ROOT / "data" / "results" / "tau_distribucion_nula.csv"

MIN_SHOTS = 12
SEED = 42


def main(n_rep: int = 300) -> None:
    tau_obs = float(pd.read_csv(TRACE)["tau"].iloc[-1])

    d = pd.read_parquet(SCORED)[["player", "xg_pred"]].dropna()
    cuenta = d.groupby("player")["xg_pred"].transform("size")
    d = d[cuenta >= MIN_SHOTS].copy()
    print("jugadores: %d | disparos: %d" % (d["player"].nunique(), len(d)))
    print("tau observado: %.4f" % tau_obs)

    p = d["xg_pred"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    taus, t0 = [], time.time()
    for r in range(n_rep):
        sim = d.copy()
        sim["is_goal"] = (rng.random(len(p)) < p).astype(int)
        res = fit_hierarchical_finishing(sim, group_col="player",
                                         prob_col="xg_pred", min_shots=MIN_SHOTS)
        taus.append(float(res.tau))
        if (r + 1) % 50 == 0:
            print("  %d/%d (%.0f s)" % (r + 1, n_rep, time.time() - t0))

    t = np.array(taus)
    sup = int((t >= tau_obs).sum())
    filas = [
        ("n_replicas", n_rep),
        ("tau_observado", round(tau_obs, 4)),
        ("media_nula", round(float(t.mean()), 4)),
        ("mediana_nula", round(float(np.median(t)), 4)),
        ("desv_tipica_nula", round(float(t.std(ddof=1)), 4)),
        ("percentil_5", round(float(np.percentile(t, 5)), 4)),
        ("percentil_50", round(float(np.percentile(t, 50)), 4)),
        ("percentil_90", round(float(np.percentile(t, 90)), 4)),
        ("percentil_95", round(float(np.percentile(t, 95)), 4)),
        ("percentil_99", round(float(np.percentile(t, 99)), 4)),
        ("maximo_nulo", round(float(t.max()), 4)),
        ("replicas_que_superan_el_observado", sup),
        ("p_empirico", round(sup / n_rep, 4)),
    ]
    pd.DataFrame(filas, columns=["estadistico", "valor"]).to_csv(OUT, index=False)
    print("\nescrito: %s" % OUT)
    for k, v in filas:
        print("  %-36s %s" % (k, v))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
