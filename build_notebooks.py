#!/usr/bin/env python3
"""
build_notebooks.py — mantenimiento de notebooks.

En esta versión reparada, los notebooks se entregan ya sincronizados con el
pipeline oficial. Para evitar sobrescribir accidentalmente resultados oficiales
o regenerar notebooks con protocolos experimentales antiguos, este script solo
comprueba que los notebooks existen y que no contienen salidas ejecutadas.

Para ejecutar los notebooks use:
    python execute_notebooks.py --source statsbomb

o, si no desea depender de la caché/datos reales:
    python execute_notebooks.py --source synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent / "notebooks"


def main() -> None:
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No se encontraron notebooks en notebooks/.")

    print("Notebooks existentes:")
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        n_code = sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code")
        n_outputs = sum(len(c.get("outputs", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        print(f"  - {nb_path.name}: {n_code} celdas de código, {n_outputs} outputs embebidos")

    print("
No se han regenerado notebooks. Ejecútelos con execute_notebooks.py.")


if __name__ == "__main__":
    main()
