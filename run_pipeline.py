#!/usr/bin/env python3
"""
run_pipeline.py — punto de entrada de conveniencia.

Mantiene la invocación clásica ``python run_pipeline.py [...]`` funcionando sin
necesidad de instalar el paquete. Internamente delega en ``xg.cli.main``, que
contiene la lógica del pipeline reproducible end-to-end (definida en
``src/xg/pipeline.py``).

Si el paquete está instalado (``pip install -e .``), también puede usarse el
comando equivalente:  ``xg-pipeline [...]``  o  ``python -m xg.cli [...]``.
"""
import sys
import pathlib

# Hace importable el paquete xg cuando se ejecuta sin instalar (src layout).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from xg.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
