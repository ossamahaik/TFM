"""
make_team_goals_vs_xg.py
========================
Genera de forma determinista la figura de barras agrupadas *goles vs xG* por
equipo, a partir del CSV canónico ``data/results/team_xg_ranking.csv``.

La figura complementa (y diferencia de) la tabla de equipos de la memoria:
mientras la tabla lista las cifras, la figura hace visible el contraste entre
los goles anotados y el xG generado por cada selección. Se restringe a los ocho
equipos que aparecen en la tabla del cuerpo para que figura y tabla sean
coherentes.

No usa aleatoriedad: es una transformación determinista del CSV de resultados.
Salida: ``ranking_teams_goals_vs_xg.png`` en la carpeta de figuras del proyecto
y en la de la memoria.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Estilo coherente con src/xg/visualization/plots.py
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "savefig.dpi": 200,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
})

# Equipos que figuran en la tabla del cuerpo (mismo conjunto y nombres ES).
TEAMS_ES = {
    "Spain": "España", "England": "Inglaterra", "Germany": "Alemania",
    "France": "Francia", "Croatia": "Croacia", "Belgium": "Bélgica",
    "Brazil": "Brasil", "Portugal": "Portugal", "Switzerland": "Suiza",
    "Netherlands": "Países Bajos",
}
ORDER = ["Spain", "England", "Germany", "France", "Croatia", "Belgium",
         "Brazil", "Portugal", "Switzerland", "Netherlands"]

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "results" / "team_xg_ranking.csv"
OUT_PROJECT = ROOT / "figures" / "ranking_teams_goals_vs_xg.png"
OUT_MEMORIA = ROOT.parent / "files_entrega_3" / "figures" / "ranking_teams_goals_vs_xg.png"


def main() -> None:
    df = pd.read_csv(CSV).set_index("team")
    df = df.loc[ORDER]                      # orden de la tabla del cuerpo
    labels = [TEAMS_ES[t] for t in ORDER][::-1]   # barh dibuja de abajo arriba
    xg = df["xg_total"].to_numpy()[::-1]
    goals = df["goals"].to_numpy()[::-1]

    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    b1 = ax.barh(y + h / 2, xg, height=h, color="#4c72b0", label="xG total", zorder=3)
    b2 = ax.barh(y - h / 2, goals, height=h, color="#dd5129", label="Goles", zorder=3)

    rng = max(xg.max(), goals.max()) or 1.0
    for bars, vals, fmt in ((b1, xg, "{:.1f}"), (b2, goals, "{:.0f}")):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + rng * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", ha="left",
                    fontsize=9, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Goles y xG total acumulados")
    ax.margins(x=0.12)
    ax.legend(loc="lower right")
    fig.tight_layout()

    for out in (OUT_PROJECT, OUT_MEMORIA):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"escrito: {out}")


if __name__ == "__main__":
    main()
