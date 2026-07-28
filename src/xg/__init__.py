"""
Paquete xg — Modelo de Expected Goals con Machine Learning (TFM).

Expone los subpaquetes principales para imports cómodos:

    from xg import config, data, features, models, analysis, visualization
    from xg.models import build_model_pipeline, compute_metrics, bootstrap_metric_ci

La organización por subpaquetes refleja la separación de responsabilidades:
config (configuración), data (adquisición), features (ingeniería de variables),
models (catálogo + evaluación + inferencia estadística), analysis (agregación e
interpretabilidad) y visualization (figuras).
"""
from __future__ import annotations

from xg import config  # noqa: F401
from xg import utils  # noqa: F401
from xg import data  # noqa: F401
from xg import features  # noqa: F401
from xg import models  # noqa: F401
from xg import analysis  # noqa: F401
from xg import visualization  # noqa: F401
from xg import bayes  # noqa: F401

__version__ = "2.0.0"
