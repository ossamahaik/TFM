"""
data_loader.py
==============
Adquisición y construcción del dataset de tiros.

Este módulo es responsable de obtener los datos crudos y de transformarlos en un
DataFrame "limpio" a nivel de disparo (una fila por tiro). Soporta dos fuentes:

1. **StatsBomb Open Data** (vía `statsbombpy`). Es la fuente objetivo del TFM.
   Descarga competiciones, partidos y eventos, filtra los disparos y extrae los
   campos relevantes, incluyendo los *freeze frames* (posiciones de jugadores en
   el instante del tiro), que son la base de las variables defensivas.

2. **Generador sintético físicamente realista** (fallback). Cuando no hay
   conexión a Internet o `statsbombpy` no está instalado, se genera un dataset
   con la MISMA estructura de columnas que el de StatsBomb. Las probabilidades de
   gol siguen un modelo logístico basado en distancia y ángulo (la física real
   del xG), de modo que el pipeline completo —modelado, calibración,
   interpretabilidad— es ejecutable y produce resultados coherentes y
   defendibles incluso sin red.

El resto del proyecto consume siempre la salida de `build_shots_dataframe`, por
lo que es agnóstico respecto a la fuente concreta de los datos.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xg import config
from xg.utils import io as _io

logger = logging.getLogger(__name__)


# Alias compatibles hacia las utilidades de E/S centralizadas (utils.io).
_has_parquet = _io.has_parquet
_cache_path = _io.cache_path
_write_cache = _io.write_cache
_read_cache = _io.read_cache


# Importación opcional de statsbombpy. El proyecto NO falla si no está instalado.
try:  # pragma: no cover - depende del entorno
    from statsbombpy import sb  # type: ignore

    _HAS_STATSBOMB = True
except Exception:  # noqa: BLE001
    sb = None  # type: ignore
    _HAS_STATSBOMB = False


# --------------------------------------------------------------------------- #
# 1. Descarga desde StatsBomb Open Data
# --------------------------------------------------------------------------- #
def _list_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Lista los partidos de una competición/temporada de StatsBomb."""
    if not _HAS_STATSBOMB:
        raise RuntimeError("statsbombpy no disponible")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sb.matches(competition_id=competition_id, season_id=season_id)


def _extract_freeze_frame_features(ff, x: float, y: float) -> Dict:
    """
    Extrae variables defensivas a partir del *freeze frame* de StatsBomb.

    El freeze frame (`ff`) es la lista de jugadores (atacantes y defensores)
    presentes en el momento del disparo. A partir de él derivamos:
      * nº de defensores dentro del "cono" de tiro (triángulo tirador-postes),
      * nº de defensores a menos de 3 m,
      * distancia al defensor más cercano,
      * posición del portero (distancia a portería y al tirador).
    Estas variables capturan la presión y la oclusión de la portería, factores
    determinantes en la probabilidad de gol que la geometría pura no recoge.

    Acepta directamente la lista del freeze frame, ya sea procedente del formato
    aplanado de statsbombpy (columna ``shot_freeze_frame``) o del anidado
    (``shot["freeze_frame"]``).
    """
    gx, gy = config.GOAL_CENTER
    feats = {
        "n_defenders_in_cone": np.nan,
        "n_defenders_within_3m": np.nan,
        "distance_to_nearest_defender": np.nan,
        "gk_distance_to_goal": np.nan,
        "gk_distance_to_shot": np.nan,
    }
    if not isinstance(ff, list) or len(ff) == 0:
        return feats

    def _in_cone(px: float, py: float) -> bool:
        # Triángulo tirador (x,y) y postes (120,36)-(120,44). Test por signo de
        # producto vectorial (point-in-triangle) robusto.
        ax, ay = x, y
        bx, by = config.GOAL_POST_LEFT
        cx, cy = config.GOAL_POST_RIGHT

        def sign(x1, y1, x2, y2, x3, y3):
            return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

        d1 = sign(px, py, ax, ay, bx, by)
        d2 = sign(px, py, bx, by, cx, cy)
        d3 = sign(px, py, cx, cy, ax, ay)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    dists, n_cone, n_close = [], 0, 0
    for player in ff:
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        px, py = float(loc[0]), float(loc[1])
        is_teammate = bool(player.get("teammate", False))
        is_keeper = (player.get("position", {}) or {}).get("name") == "Goalkeeper"
        if is_keeper and not is_teammate:
            feats["gk_distance_to_goal"] = float(np.hypot(px - gx, py - gy))
            feats["gk_distance_to_shot"] = float(np.hypot(px - x, py - y))
        if not is_teammate:  # defensor rival
            d = float(np.hypot(px - x, py - y))
            dists.append(d)
            if _in_cone(px, py):
                n_cone += 1
            if d <= 3.0:
                n_close += 1

    feats["n_defenders_in_cone"] = float(n_cone)
    feats["n_defenders_within_3m"] = float(n_close)
    feats["distance_to_nearest_defender"] = float(min(dists)) if dists else np.nan
    return feats


def _norm_name(val, default: str = "Unknown"):
    """Normaliza un campo categórico que puede venir como str (formato aplanado
    de statsbombpy) o como dict ``{"name": ...}`` (formato anidado)."""
    if isinstance(val, dict):
        return val.get("name", default)
    if val is None:
        return default
    if np.isscalar(val) and pd.isna(val):
        return default
    return val


def _truthy(val) -> bool:
    """Convierte a bool de forma segura (NaN/None -> False)."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if np.isscalar(val) and pd.isna(val):
        return False
    return bool(val)


def _coalesce(row: pd.Series, flat_key: str, shot: Dict, nested_key: str, default=None):
    """Devuelve el valor de la columna aplanada ``flat_key`` si existe y no es
    nulo; si no, cae al campo anidado ``shot[nested_key]``; si no, ``default``."""
    if flat_key in row.index:
        val = row[flat_key]
        if isinstance(val, (list, dict)):
            return val
        if not (val is None or (np.isscalar(val) and pd.isna(val))):
            return val
    if isinstance(shot, dict) and nested_key in shot:
        return shot[nested_key]
    return default


def _shots_from_events(
    events: pd.DataFrame, match_id: int, competition: str = "Unknown"
) -> List[Dict]:
    """
    Filtra los disparos de un DataFrame de eventos y normaliza sus campos.

    Soporta el formato **aplanado** de ``statsbombpy`` (por defecto), donde los
    atributos del disparo aparecen como columnas ``shot_*`` y ``play_pattern``,
    ``under_pressure`` son columnas de primer nivel; y también el formato
    **anidado** (``flatten_attrs=False``), donde existe un dict ``shot``.
    """
    if "type" not in events.columns:
        return []
    shots = events[events["type"] == "Shot"].copy()
    records: List[Dict] = []
    for _, row in shots.iterrows():
        loc = row.get("location")
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        x, y = float(loc[0]), float(loc[1])

        shot = row.get("shot") if isinstance(row.get("shot"), dict) else {}

        outcome = _norm_name(_coalesce(row, "shot_outcome", shot, "outcome"), default=None)
        body_part = _norm_name(_coalesce(row, "shot_body_part", shot, "body_part"))
        shot_type = _norm_name(_coalesce(row, "shot_type", shot, "type"))
        technique = _norm_name(_coalesce(row, "shot_technique", shot, "technique"))
        play_pattern = _norm_name(row.get("play_pattern"), default="Unknown")
        xg_val = _coalesce(row, "shot_statsbomb_xg", shot, "statsbomb_xg", default=np.nan)
        first_time = _coalesce(row, "shot_first_time", shot, "first_time", default=False)
        key_pass = _coalesce(row, "shot_key_pass_id", shot, "key_pass_id", default=None)
        ff = _coalesce(row, "shot_freeze_frame", shot, "freeze_frame", default=None)

        rec = {
            "match_id": match_id,
            "competition": competition,
            "period": row.get("period", np.nan),
            "event_index": row.get("index", np.nan),
            "team": _norm_name(row.get("team"), default=row.get("team")),
            "player": _norm_name(row.get("player"), default=row.get("player")),
            "minute": row.get("minute", np.nan),
            "second": row.get("second", np.nan),
            "x": x,
            "y": y,
            "is_goal": int(outcome == "Goal"),
            "statsbomb_xg": xg_val,
            "body_part": body_part,
            "shot_type": shot_type,
            "technique": technique,
            "play_pattern": play_pattern,
            "under_pressure": _truthy(row.get("under_pressure")),
            "is_first_time": _truthy(first_time),
            "assisted": key_pass is not None
            and not (np.isscalar(key_pass) and pd.isna(key_pass)),
        }
        rec.update(_extract_freeze_frame_features(ff, x, y))
        records.append(rec)
    return records


def download_statsbomb_shots(
    competitions: Optional[List[Dict]] = None,
    max_matches: Optional[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Descarga y construye el DataFrame de disparos desde StatsBomb Open Data.

    Parameters
    ----------
    competitions : lista de dicts {competition_id, season_id, label}.
    max_matches  : límite opcional de partidos (útil para pruebas rápidas).
    cache        : si True, guarda/lee un parquet cacheado en data/raw.
    """
    competitions = competitions or config.DEFAULT_COMPETITIONS
    cache_stem = config.RAW_DIR / "statsbomb_shots_raw"
    if cache:
        cached = _read_cache(cache_stem)
        if cached is not None:
            logger.info("Cargando disparos cacheados (StatsBomb)")
            return cached

    if not _HAS_STATSBOMB:
        raise RuntimeError(
            "statsbombpy no está instalado y no existe una caché local legible "
            "de StatsBomb. Instala el extra .[data] o ejecuta con --source synthetic."
        )

    all_records: List[Dict] = []
    n_loaded = 0
    for comp in competitions:
        matches = _list_matches(comp["competition_id"], comp["season_id"])
        for _, m in matches.iterrows():
            if max_matches is not None and n_loaded >= max_matches:
                break
            match_id = int(m["match_id"])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    events = sb.events(match_id=match_id)
                all_records.extend(
                    _shots_from_events(events, match_id, comp["label"])
                )
                n_loaded += 1
                logger.info("Partido %s procesado (%s)", match_id, comp["label"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fallo en partido %s: %s", match_id, exc)

    df = pd.DataFrame.from_records(all_records)
    if not df.empty and "is_goal" in df.columns:
        rate = float(df["is_goal"].mean())
        logger.info(
            "Disparos extraídos: %d | goles: %d | tasa de gol: %.4f",
            len(df), int(df["is_goal"].sum()), rate,
        )
        if df["is_goal"].nunique() < 2:
            raise RuntimeError(
                "La extracción de StatsBomb produjo una sola clase (sin goles). "
                "Esto suele indicar un formato de eventos inesperado. Revisa la "
                "versión de statsbombpy o borra la caché en data/raw/ y reintenta."
            )
    config.ensure_dirs()
    if cache:
        _write_cache(df, cache_stem)
    return df


# --------------------------------------------------------------------------- #
# 2. Generador sintético físicamente realista (fallback offline)
# --------------------------------------------------------------------------- #
def generate_synthetic_shots(
    n_matches: int = config.SYNTHETIC_N_MATCHES,
    seed: int = config.RANDOM_STATE,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Genera un dataset de disparos con estructura idéntica al de StatsBomb.

    Modelo generador
    -----------------
    La probabilidad real de gol se define mediante un modelo logístico cuyas
    componentes reproducen la física conocida del xG:

        logit(p) = b0
                   + b_dist * distance_to_goal
                   + b_ang  * shot_angle
                   + efectos de parte del cuerpo, tipo de jugada, presión,
                     defensores en el cono y portero.

    Los goles se muestrean como Bernoulli(p), de modo que el "xG verdadero"
    (la p generadora) es conocido. Esto permite validar la CALIBRACIÓN del
    modelo aprendido contra una probabilidad real (un lujo que no se tiene con
    datos reales) y constituye una prueba metodológica sólida del pipeline.
    """
    # La caché sintética incluye parámetros generadores. Evita reutilizar por
    # accidente un dataset creado con otro n_matches/seed, problema que rompe la
    # reproducibilidad de tests y experimentos.
    cache_stem = config.RAW_DIR / f"synthetic_shots_raw_n{n_matches}_seed{seed}"
    if cache:
        cached = _read_cache(cache_stem)
        if cached is not None:
            logger.info("Cargando disparos sintéticos cacheados: n_matches=%s seed=%s", n_matches, seed)
            return cached

    rng = np.random.default_rng(seed)
    gx, gy = config.GOAL_CENTER
    pl, pr = config.GOAL_POST_LEFT, config.GOAL_POST_RIGHT

    body_parts = np.array(["Right Foot", "Left Foot", "Head", "Other"])
    body_p = np.array([0.46, 0.34, 0.18, 0.02])
    shot_types = np.array(["Open Play", "Free Kick", "Penalty", "Corner"])
    shot_p = np.array([0.86, 0.07, 0.03, 0.04])
    patterns = np.array(
        ["Regular Play", "From Counter", "From Corner", "From Free Kick",
         "From Throw In", "From Goal Kick"]
    )
    pattern_p = np.array([0.55, 0.10, 0.13, 0.10, 0.08, 0.04])

    records: List[Dict] = []
    teams = [f"Team {chr(65 + i)}" for i in range(24)]
    # Etiquetas de competición sintéticas (espejo de DEFAULT_COMPETITIONS) para
    # que la lógica de holdout por competición sea reproducible también offline.
    comp_labels = [c["label"] for c in config.DEFAULT_COMPETITIONS] or ["Synthetic"]

    for match_id in range(1, n_matches + 1):
        competition = comp_labels[(match_id - 1) % len(comp_labels)]
        home, away = rng.choice(teams, size=2, replace=False)
        n_shots = int(rng.integers(*config.SYNTHETIC_SHOTS_PER_MATCH))
        score = {home: 0, away: 0}
        for _ in range(n_shots):
            team = rng.choice([home, away])
            opp = away if team == home else home
            shot_type = rng.choice(shot_types, p=shot_p)

            # Posición del disparo: concentrada cerca del área, con cola larga.
            if shot_type == "Penalty":
                x, y = 108.0, 40.0
            else:
                x = float(np.clip(120 - rng.gamma(2.2, 6.5), 60, 119.5))
                y = float(np.clip(rng.normal(40, 11), 1, 79))

            dx, dy = gx - x, gy - y
            distance = float(np.hypot(dx, dy))
            # Ángulo subtendido por la portería (postes) desde el tirador.
            a = np.array([pl[0] - x, pl[1] - y])
            b = np.array([pr[0] - x, pr[1] - y])
            cos_ang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            shot_angle = float(np.arccos(np.clip(cos_ang, -1, 1)))

            body_part = str(rng.choice(body_parts, p=body_p))
            pattern = str(rng.choice(patterns, p=pattern_p))
            under_pressure = bool(rng.random() < 0.42)
            first_time = bool(rng.random() < 0.30)
            assisted = bool(rng.random() < 0.55)
            minute = int(rng.integers(0, 96))

            # Variables defensivas (proxy del freeze frame real).
            n_cone = int(rng.poisson(0.6 + 0.05 * distance))
            n_close = int(rng.poisson(0.4 + 0.04 * distance))
            d_near = float(np.clip(rng.gamma(2.0, 1.4), 0.2, 25))
            gk_goal = float(np.clip(rng.normal(2.5, 1.2), 0.1, 8))
            gk_shot = float(np.clip(distance - gk_goal + rng.normal(0, 1.5), 0.2, 60))

            # ---- Modelo generador de la probabilidad real de gol ----
            # Coeficientes calibrados para reproducir una tasa de conversión
            # realista (~10-12 %) y un xG medio coherente con datos reales.
            logit = (
                -0.70
                - 0.118 * distance
                + 1.85 * shot_angle
                - 0.22 * n_cone
                - 0.10 * n_close
                + 0.012 * gk_shot
                + (-0.55 if body_part == "Head" else 0.0)
                + (-0.30 if under_pressure else 0.0)
                + (0.25 if first_time else -0.05)
                + (0.20 if pattern == "From Counter" else 0.0)
            )
            if shot_type == "Penalty":
                logit = 1.15  # ~0.76 prob, coherente con tasa real de penaltis
            p_goal = 1.0 / (1.0 + np.exp(-logit))
            is_goal = int(rng.random() < p_goal)

            records.append(
                {
                    "match_id": match_id,
                    "competition": competition,
                    "period": 1 if minute < 45 else 2,
                    "event_index": len(records),
                    "team": team,
                    "opponent": opp,
                    "player": f"Player {rng.integers(1, 400)}",
                    "minute": minute,
                    "second": int(rng.integers(0, 60)),
                    "x": x,
                    "y": y,
                    "is_goal": is_goal,
                    "true_xg": float(p_goal),   # probabilidad generadora (oráculo)
                    "statsbomb_xg": np.nan,
                    "body_part": body_part,
                    "shot_type": shot_type,
                    "technique": "Normal",
                    "play_pattern": pattern,
                    "under_pressure": under_pressure,
                    "is_first_time": first_time,
                    "assisted": assisted,
                    "n_defenders_in_cone": float(n_cone),
                    "n_defenders_within_3m": float(n_close),
                    "distance_to_nearest_defender": d_near,
                    "gk_distance_to_goal": gk_goal,
                    "gk_distance_to_shot": gk_shot,
                }
            )
            score[team] += is_goal

    df = pd.DataFrame.from_records(records)
    config.ensure_dirs()
    if cache:
        _write_cache(df, cache_stem)
    logger.info("Dataset sintético generado: %d disparos", len(df))
    return df


# --------------------------------------------------------------------------- #
# 3. Punto de entrada unificado
# --------------------------------------------------------------------------- #
def build_shots_dataframe(
    source: str = "auto",
    competitions: Optional[List[Dict]] = None,
    max_matches: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Devuelve el DataFrame de disparos y la fuente efectivamente utilizada.

    source : {"auto", "statsbomb", "synthetic"}
        * "auto": intenta StatsBomb; si no es posible, usa el sintético.
    """
    config.ensure_dirs()
    if source not in {"auto", "statsbomb", "synthetic"}:
        raise ValueError("source debe ser 'auto', 'statsbomb' o 'synthetic'")

    if source == "synthetic":
        return generate_synthetic_shots(), "synthetic"

    if source == "statsbomb":
        # Si el usuario pide StatsBomb explícitamente, no se debe caer de forma
        # silenciosa a sintético: eso contaminaría resultados y memoria.
        df = download_statsbomb_shots(competitions, max_matches)
        if len(df) == 0:
            raise RuntimeError("StatsBomb devolvió 0 disparos; no se usa fallback sintético.")
        return df, "statsbomb"

    # source == "auto": intenta StatsBomb y usa sintético solo como fallback
    # declarado en logs.
    try:
        df = download_statsbomb_shots(competitions, max_matches)
        if len(df) > 0:
            return df, "statsbomb"
        logger.warning("StatsBomb devolvió 0 disparos; usando sintético.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("StatsBomb no disponible (%s); usando sintético.", exc)
    return generate_synthetic_shots(), "synthetic"


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Persiste un DataFrame procesado en data/processed (parquet + csv)."""
    config.ensure_dirs()
    df.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    path = _write_cache(df, config.PROCESSED_DIR / name)
    return path
