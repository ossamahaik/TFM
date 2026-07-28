"""
cli.py — interfaz de línea de comandos del pipeline xG.

Permite ejecutar el pipeline completo tras instalar el paquete:

    xg-pipeline --source statsbomb --holdout-competition "UEFA Euro 2024"

o de forma equivalente con  ``python -m xg.cli``  /  ``python run_pipeline.py``.
"""
from __future__ import annotations

import argparse
import sys

from xg import config
from xg.pipeline import main as run


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Pipeline xG end-to-end")
    parser.add_argument("--source", default="auto", choices=["auto", "statsbomb", "synthetic"])
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument(
        "--holdout-competition", default=None,
        help="Competición reservada como test (p. ej. 'UEFA Euro 2024'). "
             "Por defecto usa config.HOLDOUT_COMPETITION. Usa 'none' para "
             "forzar partición aleatoria estratificada.",
    )
    args = parser.parse_args()
    holdout = args.holdout_competition
    if holdout is not None and holdout.lower() == "none":
        holdout = None
    elif holdout is None:
        holdout = getattr(config, "HOLDOUT_COMPETITION", None)
    run(source=args.source, max_matches=args.max_matches, holdout_competition=holdout)


if __name__ == "__main__":
    main()
