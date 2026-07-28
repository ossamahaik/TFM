"""
visualization.py
================
Visualizaciones profesionales del proyecto xG.

Todas las funciones siguen el mismo contrato: reciben los datos necesarios,
construyen una figura de matplotlib/seaborn y, si `save_as` se indica, la
guardan automáticamente en `figures/` con DPI alto. Esto garantiza que el
proyecto exporte de forma reproducible todo el material gráfico.

Incluye un dibujante de campo propio (sin dependencias externas como mplsoccer)
para máxima portabilidad: el proyecto funciona con la pila científica estándar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # backend no interactivo: seguro para scripts/CI
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from xg import config
from xg.models import evaluation

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"

# --------------------------------------------------------------------------- #
# Estilo global para máxima legibilidad de las figuras impresas en la memoria.
# Fuentes mayores, rejilla tenue, líneas algo más gruesas y márgenes holgados.
# Centralizar el estilo garantiza coherencia visual en TODAS las figuras.
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": max(getattr(config, "FIG_DPI", 150), 200),
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "medium",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.2,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "figure.autolayout": False,
})


def _save(fig, save_as: Optional[str]):
    if save_as and config.SAVE_FIGURES:
        config.ensure_dirs()
        path = config.FIGURES_DIR / f"{save_as}.{config.FIG_FORMAT}"
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Campo de fútbol (media cancha de ataque, convención StatsBomb)
# --------------------------------------------------------------------------- #
def draw_pitch(ax=None, half: bool = True):
    """Dibuja un campo de fútbol (o media cancha) en convención StatsBomb 120x80."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    line = "#3a3a3a"
    x0 = 60 if half else 0
    ax.plot([x0, x0, 120, 120, x0], [0, 80, 80, 0, 0], color=line, lw=1.5)
    # Área grande y pequeña
    ax.plot([102, 102, 120], [18, 62, 62], color=line, lw=1.2)
    ax.plot([102, 120], [18, 18], color=line, lw=1.2)
    ax.plot([114, 114, 120], [30, 50, 50], color=line, lw=1.2)
    ax.plot([114, 120], [30, 30], color=line, lw=1.2)
    # Portería y punto de penalti
    ax.plot([120, 120], [36, 44], color=line, lw=3)
    ax.scatter([108], [40], color=line, s=12)
    # Arco del área
    arc = patches.Arc((108, 40), 20, 20, angle=0, theta1=130, theta2=230, color=line, lw=1.2)
    ax.add_patch(arc)
    ax.set_xlim(x0 - 2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_shot_map(
    df: pd.DataFrame, prob_col: str = "xg_pred", title: str = "Mapa de disparos (xG)",
    save_as: Optional[str] = None,
):
    """
    Shot map: cada disparo como punto; tamaño ∝ xG predicho; color = gol/no gol.
    Es la visualización canónica de un modelo xG.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    draw_pitch(ax)
    goals = df[df[config.TARGET] == 1]
    misses = df[df[config.TARGET] == 0]
    ax.scatter(misses["x"], misses["y"], s=8 + 150 * misses[prob_col],
               c="#4c72b0", alpha=0.30, edgecolors="white", linewidths=0.3, label="No gol")
    ax.scatter(goals["x"], goals["y"], s=14 + 190 * goals[prob_col],
               c="#dd5129", alpha=0.75, edgecolors="black", linewidths=0.4, label="Gol")
    ax.legend(loc="lower left", frameon=True, markerscale=0.9,
              title="Resultado", title_fontsize=11)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.text(0.5, -0.01, "El tamaño del punto es proporcional al xG estimado",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#555555")
    return _save(fig, save_as)


def plot_xg_heatmap(df: pd.DataFrame, prob_col: str = "xg_pred",
                    title: str = "Mapa de calor de xG medio por zona",
                    save_as: Optional[str] = None):
    """Heatmap del xG medio agregado en una rejilla espacial del campo."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    xbins = np.linspace(60, 120, 25)
    ybins = np.linspace(0, 80, 33)
    stat, _, _ = np.histogram2d(df["x"], df["y"], bins=[xbins, ybins], weights=df[prob_col])
    cnt, _, _ = np.histogram2d(df["x"], df["y"], bins=[xbins, ybins])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_xg = np.where(cnt > 0, stat / cnt, np.nan)
    draw_pitch(ax)
    mesh = ax.pcolormesh(xbins, ybins, mean_xg.T, cmap=PALETTE, alpha=0.75, shading="auto")
    fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02, label="xG medio")
    ax.set_title(title, fontsize=14, weight="bold")
    return _save(fig, save_as)


# --------------------------------------------------------------------------- #
# Curvas de evaluación
# --------------------------------------------------------------------------- #
def plot_roc_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    save_as: Optional[str] = None):
    """ROC de varios modelos; `results[name] = (y_true, y_prob)`."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr = evaluation.roc_points(y_true, y_prob)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC", fontsize=14, weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, save_as)


def plot_pr_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   base_rate: float, save_as: Optional[str] = None):
    """Curvas Precision-Recall (más informativas con desbalanceo)."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name, (y_true, y_prob) in results.items():
        recall, precision = evaluation.pr_points(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=name)
    ax.axhline(base_rate, ls="--", color="grey", lw=1, label=f"Tasa base ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall", fontsize=14, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, save_as)


def plot_calibration_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            n_bins: int = 10, save_as: Optional[str] = None):
    """Curvas de fiabilidad (calibración). La diagonal = calibración perfecta."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name, (y_true, y_prob) in results.items():
        mean_pred, frac_pos = evaluation.calibration_points(y_true, y_prob, n_bins)
        ax.plot(mean_pred, frac_pos, "o-", lw=2, markersize=5, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad media predicha (xG)")
    ax.set_ylabel("Fracción real de goles")
    ax.set_title("Curvas de calibración", fontsize=14, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, save_as)


def plot_xg_distribution(y_prob: np.ndarray, save_as: Optional[str] = None):
    """Distribución de los valores de xG predichos."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(y_prob, bins=40, kde=True, color="#4c72b0", ax=ax)
    ax.set_xlabel("xG predicho")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del xG predicho", fontsize=14, weight="bold")
    return _save(fig, save_as)


def plot_confusion(y_true, y_prob, threshold: float = 0.5, save_as: Optional[str] = None):
    """Matriz de confusión a un umbral dado (interpretación de clasificación)."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["No gol", "Gol"], yticklabels=["No gol", "Gol"], ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (umbral = {threshold})")
    return _save(fig, save_as)


# --------------------------------------------------------------------------- #
# Interpretabilidad
# --------------------------------------------------------------------------- #
def plot_feature_importance(importances: pd.Series, top_n: int = 20,
                            title: str = "Importancia de variables",
                            save_as: Optional[str] = None):
    """Gráfico de barras horizontal de importancias (permutación o nativas)."""
    top = importances.sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.4 * len(top))))
    bars = ax.barh(top.index, top.values, color="#55a868")
    for b, v in zip(bars, top.values):
        ax.text(b.get_width() + max(top.values) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9, color="#333333")
    ax.margins(x=0.14)
    ax.set_xlabel("Importancia (caída de ROC-AUC al permutar)")
    ax.set_title(title)
    return _save(fig, save_as)


def plot_feature_distributions(
    df: pd.DataFrame,
    variables: List[Tuple[str, str]],
    title: str,
    n_cols: int = 3,
    save_as: Optional[str] = None,
):
    """
    Rejilla de histogramas (con KDE) de la distribución marginal de cada
    variable explicativa.

    Responde a la necesidad de mostrar la distribución de *cada* variable y no
    solo describirla en prosa: para cada par ``(columna, etiqueta)`` dibuja un
    panel con su histograma, la media (línea continua) y la mediana (línea
    discontinua), de modo que la forma —simetría, sesgo, concentración— quede a
    la vista. Es deliberadamente agnóstica a la lista de variables, para poder
    reutilizarla con los bloques geométrico y defensivo por separado.
    """
    cols = [c for c, _ in variables if c in df.columns]
    labels = {c: lab for c, lab in variables}
    n = len(cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        sns.histplot(serie, bins=30, kde=True, color="#4c72b0", ax=ax)
        ax.axvline(serie.mean(), color="#c44e52", lw=1.4, label="Media")
        ax.axvline(serie.median(), color="#55a868", lw=1.4, ls="--", label="Mediana")
        ax.set_title(labels.get(col, col), fontsize=10, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(cols):]:  # apaga ejes sobrantes de la rejilla
        ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, save_as)


def plot_correlation_matrix(df: pd.DataFrame, columns: List[str],
                            save_as: Optional[str] = None):
    """Matriz de correlación de las variables numéricas."""
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson"},
                ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    return _save(fig, save_as)


def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str,
                           save_as: Optional[str] = None):
    """Barras comparativas de una métrica entre modelos."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = metrics_df[metric].sort_values()
    colors = sns.color_palette(PALETTE, len(data))
    bars = ax.barh(data.index, data.values, color=colors)
    for b, v in zip(bars, data.values):
        ax.text(b.get_width() + data.values.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", ha="left", fontsize=9, color="#333333")
    ax.margins(x=0.14)
    ax.set_xlabel(metric)
    ax.set_title(f"Comparación de modelos — {metric}")
    return _save(fig, save_as)


def plot_ranking(df_rank: pd.DataFrame, value_col: str, label_col: str,
                 title: str, save_as: Optional[str] = None, top_n: int = 15):
    """Ranking horizontal (jugadores/equipos) por una magnitud agregada."""
    data = df_rank.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.45 * len(data))))
    vals = data[value_col].to_numpy()
    bars = ax.barh(data[label_col].astype(str), vals, color="#c44e52")
    rng = (vals.max() - min(vals.min(), 0)) or 1.0
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + rng * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", ha="left", fontsize=9, color="#333333")
    ax.margins(x=0.12)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    return _save(fig, save_as)


def plot_loco_validation(loco_df: pd.DataFrame, metric: str = "roc_auc",
                         save_as: Optional[str] = None):
    """
    Rendimiento Leave-One-Competition-Out por torneo, con la media y su banda
    de ±1 desviación típica. Muestra visualmente la variabilidad ENTRE
    competiciones, no solo dentro de una.
    """
    folds = loco_df.drop(index=["MEDIA", "DESV_TIP"], errors="ignore")
    mean = loco_df.loc["MEDIA", metric] if "MEDIA" in loco_df.index else folds[metric].mean()
    std = loco_df.loc["DESV_TIP", metric] if "DESV_TIP" in loco_df.index else folds[metric].std()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = sns.color_palette(PALETTE, len(folds))
    bars = ax.bar(folds.index.astype(str), folds[metric].values, color=colors, zorder=3)
    for b, v in zip(bars, folds[metric].values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, color="#333333")
    ax.axhline(mean, color="#333333", ls="--", lw=1.5, label=f"Media = {mean:.3f}", zorder=4)
    ax.axhspan(mean - std, mean + std, color="grey", alpha=0.15,
               label=f"±1 DT ({std:.3f})", zorder=1)
    ax.set_ylabel(metric)
    ax.set_xlabel("Competición excluida (test)")
    ax.set_title(f"Validación Leave-One-Competition-Out — {metric}")
    ax.legend(loc="lower right")
    ax.margins(y=0.12)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, save_as)


def plot_reliability_with_ci(reliability_df: pd.DataFrame,
                             save_as: Optional[str] = None):
    """
    Diagrama de fiabilidad (calibración) con intervalos de Wilson por tramo.

    A diferencia de la curva de calibración simple, dibuja barras de error que
    revelan en qué tramos hay datos suficientes para afirmar buena calibración
    y en cuáles la estimación es ruidosa (clave para una lectura honesta).
    """
    df = reliability_df.copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Calibración perfecta")
    yerr = np.vstack([
        (df["obs_freq"] - df["obs_ci_low"]).clip(lower=0).to_numpy(),
        (df["obs_ci_high"] - df["obs_freq"]).clip(lower=0).to_numpy(),
    ])
    ax.errorbar(df["mean_pred"], df["obs_freq"], yerr=yerr, fmt="o-",
                color="#4c72b0", ecolor="#4c72b0", capsize=4, lw=1.5,
                label="Observado ± IC Wilson 95%")
    # Tamaño de muestra por punto, como anotación.
    for _, r in df.iterrows():
        ax.annotate(f"n={int(r['n_shots'])}", (r["mean_pred"], r["obs_freq"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8,
                    color="#555555")
    ax.set_xlabel("Probabilidad media predicha (xG)")
    ax.set_ylabel("Frecuencia observada de gol")
    ax.set_title("Diagrama de fiabilidad con IC de Wilson", fontsize=13, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return _save(fig, save_as)

def plot_subgroup_performance(subgroup_df: pd.DataFrame, save_as: Optional[str] = None):
    """
    xG medio predicho frente a tasa real de gol (± IC Wilson) por subgrupo
    futbolístico, agrupado por dimensión (distancia, zona, jugada, parte del
    cuerpo). Une el resultado aplicado con la validación de calibración
    condicional: si el punto predicho cae dentro del IC observado, el modelo
    está bien calibrado también dentro de ese subgrupo, no solo en global.
    """
    df = subgroup_df.copy()
    dimensions = list(df["dimension"].unique())
    fig, axes = plt.subplots(1, len(dimensions), figsize=(4.2 * len(dimensions), 5))
    if len(dimensions) == 1:
        axes = [axes]
    for ax, dim in zip(axes, dimensions):
        d = df[df["dimension"] == dim]
        y = np.arange(len(d))
        yerr = np.vstack([
            (d["obs_goal_rate"] - d["obs_ci_low"]).clip(lower=0).to_numpy(),
            (d["obs_ci_high"] - d["obs_goal_rate"]).clip(lower=0).to_numpy(),
        ])
        ax.errorbar(d["obs_goal_rate"], y, xerr=yerr, fmt="o", color="#4c72b0",
                     ecolor="#4c72b0", capsize=4, label="Tasa real ± IC Wilson")
        ax.scatter(d["mean_xg_pred"], y, marker="x", s=60, color="#d1495b",
                   label="xG medio", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d["subgrupo"], fontsize=9)
        ax.set_title(dim.replace("_", " ").capitalize(), fontsize=10)
        ax.set_xlabel("Probabilidad de gol")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("xG medio frente a tasa real de gol por subgrupo", fontsize=12, weight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_bayesian_shrinkage(
    comp: pd.DataFrame,
    raw_col: str = "freq_goals_minus_xg",
    shrunk_col: str = "shrunk_goals_minus_xg",
    label_col: str = "player",
    save_as: Optional[str] = None,
):
    """
    Visualiza el encogimiento bayesiano (*shrinkage*).

    Para cada jugador se dibuja un segmento desde su estimación cruda de
    sobre-rendimiento (goles − xG) hasta su estimación bayesiana regularizada.
    El colapso de los segmentos hacia 0 ilustra cómo el *partial pooling*
    descuenta el ruido muestral de los jugadores con pocos disparos.
    """
    d = comp.dropna(subset=[raw_col, shrunk_col]).copy()
    d = d.reindex(d[raw_col].abs().sort_values(ascending=False).index).head(30)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(d))))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r[raw_col], r[shrunk_col]], [yi, yi], color="0.6", lw=1.2, zorder=1)
    ax.scatter(d[raw_col], y, s=36, color="#d1495b", label="Crudo (goles − xG)", zorder=2)
    ax.scatter(d[shrunk_col], y, s=36, color="#1d7874", label="Bayesiano (encogido)", zorder=3)
    ax.axvline(0, color="0.3", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col], fontsize=8)
    ax.set_xlabel("Sobre-rendimiento en finalización (goles − xG)")
    ax.set_title("Encogimiento bayesiano de la habilidad de finalización")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_as)


def plot_shrinkage_vs_shots(
    estimates: pd.DataFrame,
    save_as: Optional[str] = None,
):
    """
    Diagrama del grado de encogimiento frente al número de disparos.

    Ilustra la propiedad central del modelo jerárquico: el encogimiento es
    adaptativo: los jugadores con pocos disparos se contraen más hacia la media
    poblacional (shrinkage → 1), y los de muchos disparos conservan su señal
    individual (shrinkage → 0).
    """
    d = estimates.dropna(subset=["shrinkage", "shots"]).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(d["shots"], d["shrinkage"].clip(0, 1),
                    c=d["skill_mean"], cmap="coolwarm", s=30, alpha=0.8)
    ax.set_xlabel("Número de disparos del jugador")
    ax.set_ylabel("Grado de encogimiento hacia la media")
    ax.set_title("Encogimiento adaptativo: más datos ⇒ menos regularización")
    fig.colorbar(sc, ax=ax, label="Habilidad estimada (log-odds)")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_em_convergence(tau_history, save_as: Optional[str] = None):
    """
    Traza la convergencia del hiperparámetro τ a lo largo de las iteraciones EM.

    Documenta que el algoritmo de Bayes empírico (EM + Laplace) alcanza un punto
    fijo estable: la heterogeneidad poblacional estimada τ se asienta tras unas
    pocas iteraciones, evidencia de un ajuste numéricamente sano.
    """
    tau_history = list(tau_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(tau_history)), tau_history, marker="o", ms=4,
            color="#534AB7", lw=1.5)
    ax.set_xlabel("Iteración EM")
    ax.set_ylabel(r"$\hat{\tau}$ (log-odds)")
    ax.set_title("Convergencia del EM (Bayes empírico)")
    ax.grid(True, alpha=0.3)
    if len(tau_history) > 1:
        ax.annotate(f"τ̂ = {tau_history[-1]:.4f}",
                    xy=(len(tau_history) - 1, tau_history[-1]),
                    xytext=(0.6, 0.8), textcoords="axes fraction", fontsize=10)
    fig.tight_layout()
    return _save(fig, save_as)
