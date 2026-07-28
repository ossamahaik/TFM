"""
features.py
===========
Ingeniería de variables para el modelo xG.

La calidad de un modelo xG depende, sobre todo, de las variables. Este módulo
construye un conjunto de *features* fundamentado tanto futbolística como
estadísticamente, agrupadas en cuatro familias:

1. **Geométricas** — distancia y ángulo a portería. Son las dos variables más
   informativas del xG: la probabilidad de gol decae con la distancia y crece
   con el ángulo visible de la portería. Se derivan de forma exacta a partir de
   las coordenadas (convención StatsBomb 120 x 80).

2. **Derivadas / no lineales** — log-distancia, inversa de la distancia e
   interacción ángulo×distancia. Capturan la relación NO lineal entre geometría
   y probabilidad, permitiendo que incluso un modelo lineal (regresión
   logística) aproxime la curva real de xG.

3. **Contextuales** — parte del cuerpo, tipo de disparo, patrón de juego,
   presión, primer toque, asistencia, situación de marcador (game state). Son
   factores tácticos que modulan la dificultad del remate.

4. **Defensivas / espaciales** — nº de defensores en el cono de tiro, defensores
   cercanos, distancia al defensor más próximo, posición del portero. Provienen
   del *freeze frame* y explican oclusión y presión.

El módulo expone funciones puras (sin estado) que añaden columnas, y una factoría
`build_preprocessor` que devuelve un `ColumnTransformer` de scikit-learn,
garantizando un preprocesamiento reproducible y sin fuga de información (los
estadísticos de escalado/imputación se ajustan SOLO con el train dentro del
Pipeline).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from xg import config


# --------------------------------------------------------------------------- #
# Variables geométricas
# --------------------------------------------------------------------------- #
def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade distancia y ángulo a portería a partir de las coordenadas (x, y).

    * distance_to_goal: distancia euclídea al centro de la portería.
    * shot_angle: ángulo (radianes) subtendido por los postes desde el tirador.
      Un ángulo grande = portería "más abierta" = mayor probabilidad de gol.
      Se calcula con la ley del coseno entre los vectores tirador→poste.
    """
    out = df.copy()
    gx, gy = config.GOAL_CENTER
    pl, pr = config.GOAL_POST_LEFT, config.GOAL_POST_RIGHT

    out["distance_x"] = gx - out["x"]
    out["distance_y"] = (out["y"] - gy).abs()
    out["distance_to_goal"] = np.hypot(out["distance_x"], out["distance_y"])

    # Ángulo entre vectores a cada poste (vectorizado).
    ax, ay = pl[0] - out["x"], pl[1] - out["y"]
    bx, by = pr[0] - out["x"], pr[1] - out["y"]
    dot = ax * bx + ay * by
    norm = np.sqrt(ax**2 + ay**2) * np.sqrt(bx**2 + by**2) + 1e-9
    out["shot_angle"] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones no lineales e interacciones de las variables geométricas."""
    out = df.copy()
    out["log_distance"] = np.log1p(out["distance_to_goal"])
    out["inverse_distance"] = 1.0 / (out["distance_to_goal"] + 1.0)
    out["angle_distance_interaction"] = out["shot_angle"] / (out["distance_to_goal"] + 1.0)
    # Indicador de disparo dentro del área (x >= 102 en convención StatsBomb).
    out["shot_in_box"] = ((out["x"] >= 102.0) & (out["y"].between(18.0, 62.0))).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Variables contextuales
# --------------------------------------------------------------------------- #
def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variables tácticas/contextuales y situación de marcador.

    * is_open_play: el disparo proviene de juego abierto (no balón parado).
    * game_state: estado del marcador para el equipo que dispara (perdiendo /
      empatando / ganando) en el momento del tiro. Capta el efecto del contexto
      competitivo sobre la toma de decisiones y la presión.
    """
    out = df.copy()

    out["is_open_play"] = (out["shot_type"] == "Open Play").astype(int)

    # Normalización de booleanos a categorías legibles (mejor para OHE/EDA).
    for col in ["under_pressure", "is_first_time", "assisted"]:
        if col in out.columns:
            out[col] = out[col].astype(bool).map({True: "yes", False: "no"})

    # game state: marcador acumulado por equipo hasta el minuto del disparo.
    # Ordenación temporal robusta: se prioriza el índice de evento de StatsBomb
    # (orden cronológico exacto dentro del partido); en su ausencia se recurre a
    # (period, minute, second). El desempate estable por la posición original
    # garantiza reproducibilidad cuando coinciden los marcadores de tiempo, y
    # evita la ambigüedad de qué gol "cuenta antes" en eventos simultáneos.
    out = out.reset_index(drop=False).rename(columns={"index": "_orig_pos"})
    sort_keys = ["match_id"]
    if "event_index" in out.columns and out["event_index"].notna().any():
        sort_keys += ["event_index"]
    else:
        if "period" in out.columns:
            sort_keys += ["period"]
        sort_keys += ["minute", "second"]
    sort_keys += ["_orig_pos"]
    out = out.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    out["goal_diff"] = 0
    if {"match_id", "team"}.issubset(out.columns):
        # goles previos del propio equipo y del rival dentro del partido
        out["_team_goals_before"] = (
            out.groupby(["match_id", "team"])["is_goal"].cumsum() - out["is_goal"]
        )
        # goles totales del partido antes de este disparo, por equipo
        match_goal_running = out.groupby("match_id")["is_goal"].cumsum() - out["is_goal"]
        out["_match_goals_before"] = match_goal_running
        out["_opp_goals_before"] = out["_match_goals_before"] - out["_team_goals_before"]
        out["goal_diff"] = (out["_team_goals_before"] - out["_opp_goals_before"]).astype(int)
        out = out.drop(columns=["_team_goals_before", "_match_goals_before", "_opp_goals_before"])

    out["game_state"] = np.select(
        [out["goal_diff"] < 0, out["goal_diff"] == 0, out["goal_diff"] > 0],
        ["losing", "drawing", "winning"],
        default="drawing",
    )
    out = out.drop(columns=["_orig_pos"], errors="ignore")
    return out


# --------------------------------------------------------------------------- #
# Orquestador de feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de ingeniería de variables y deja el DataFrame
    listo para el modelado. Excluye penaltis del modelo principal: su xG es
    prácticamente constante (~0.76) y no depende de geometría, por lo que su
    inclusión distorsiona el aprendizaje. Se conservan en una columna marcadora.
    """
    out = df.copy()
    out = add_geometric_features(out)
    out = add_derived_features(out)
    out = add_context_features(out)

    # Limpieza mínima e imputación de columnas defensivas ausentes (datos reales
    # sin freeze frame). La imputación fina se hace dentro del Pipeline; aquí solo
    # garantizamos que las columnas existan.
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    out["is_penalty"] = (out["shot_type"] == "Penalty").astype(int)
    return out


def split_features_target(
    df: pd.DataFrame, drop_penalties: bool = True, return_competition: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X (features del esquema) e y (target), opcionalmente sin penaltis.

    Si ``return_competition=True`` devuelve además la Serie de competición
    alineada con X (útil para la partición *holdout* por competición).
    """
    data = df.copy()
    if drop_penalties and "is_penalty" in data.columns:
        data = data[data["is_penalty"] == 0].reset_index(drop=True)
    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = data[feature_cols].copy()
    y = data[config.TARGET].astype(int).copy()
    if return_competition:
        comp = (
            data["competition"].copy()
            if "competition" in data.columns
            else pd.Series(["Unknown"] * len(data), index=data.index)
        )
        return X, y, comp
    return X, y


# --------------------------------------------------------------------------- #
# Factoría de preprocesamiento (ColumnTransformer)
# --------------------------------------------------------------------------- #
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    Construye el preprocesador reproducible.

    * Numéricas: imputación por mediana (+ escalado estándar si el modelo lo
      requiere, p. ej. regresión logística). Los árboles no se escalan.
    * Categóricas: imputación por moda + One-Hot Encoding con manejo de
      categorías no vistas (`handle_unknown="ignore"`), evitando fallos en test.

    Ajustar imputadores/escaladores DENTRO del Pipeline garantiza que sus
    parámetros se estimen solo con el train de cada fold => sin data leakage.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Devuelve los nombres de las columnas tras el preprocesamiento (post-OHE)."""
    return list(preprocessor.get_feature_names_out())


def build_spline_preprocessor(
    n_knots: int = 5, degree: int = 3
) -> ColumnTransformer:
    """
    Preprocesador con **bases de splines** para las variables numéricas.

    Motivación metodológica: una regresión logística sobre las variables crudas
    solo captura efectos lineales en el *logit*. La relación real entre geometría
    y probabilidad de gol es marcadamente no lineal (la distancia satura, el
    ángulo interacciona). Expandiendo las numéricas en una base de B-splines, un
    modelo lineal puede aproximar esas curvas suaves, convirtiéndose en un rival
    serio e interpretable del gradient boosting. Es la comparación correcta:
    "modelo lineal con la no linealidad adecuada" frente a "modelo de árboles",
    en lugar de una logística cruda como hombre de paja.

    Las variables categóricas se tratan igual que en el preprocesador estándar.
    """
    numeric_spline_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots, degree=degree,
                    include_bias=False, knots="quantile",
                ),
            ),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num_spline", numeric_spline_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
