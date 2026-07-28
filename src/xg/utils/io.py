"""
io.py — utilidades de entrada/salida y caché.

Centraliza la persistencia de DataFrames. El formato preferente es parquet
cuando existe un motor compatible; si no, se usa CSV como respaldo. La lectura
es deliberadamente tolerante: intenta todos los formatos disponibles para que un
proyecto con caché ya generada pueda ejecutarse aunque falte ``pyarrow`` o
``fastparquet`` en el entorno local.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def has_parquet() -> bool:
    """Comprueba si hay un motor parquet (pyarrow/fastparquet) disponible."""
    try:  # pragma: no cover
        import pyarrow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        try:  # pragma: no cover
            import fastparquet  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False


def cache_path(stem: Path) -> Path:
    """Ruta de escritura preferente según el motor disponible."""
    return stem.with_suffix(".parquet" if has_parquet() else ".csv")


def write_cache(df: pd.DataFrame, stem: Path) -> Path:
    """
    Escribe un DataFrame en caché.

    Se intenta parquet si hay motor disponible. Además se mantiene un CSV de
    respaldo para que la ejecución sea portable entre entornos sin pyarrow.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    try:
        df.to_csv(csv_path, index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("No se pudo escribir caché CSV %s: %s", csv_path, exc)

    if has_parquet():
        path = stem.with_suffix(".parquet")
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo escribir parquet %s; se usa CSV. Error: %s", path, exc)
    return csv_path


def read_cache(stem: Path) -> Optional[pd.DataFrame]:
    """
    Lee un DataFrame de caché si existe.

    Orden de lectura:
    1. parquet, si hay motor compatible;
    2. CSV, portable;
    3. pickle, solo como compatibilidad con versiones antiguas.

    Si un formato existe pero falla, se intenta el siguiente. Así se evita que
    una caché parquet generada en otro equipo bloquee la ejecución local.
    """
    candidates = []
    if has_parquet():
        candidates.append(stem.with_suffix(".parquet"))
    candidates.extend([stem.with_suffix(".csv"), stem.with_suffix(".pkl")])

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            if path.suffix == ".pkl":
                return pd.read_pickle(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo leer caché %s: %s", path, exc)
    return None
