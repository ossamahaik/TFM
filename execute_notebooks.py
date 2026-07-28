#!/usr/bin/env python3
"""
execute_notebooks.py — ejecuta los notebooks embebiendo salidas y figuras.

Alternativa autosuficiente a `jupyter nbconvert --execute` para entornos sin
Jupyter instalado. Ejecuta cada celda de código en un namespace persistente
(como un kernel), captura ``stdout``, los valores de retorno y las figuras de
matplotlib (como PNG en base64), y reescribe el .ipynb con ``execution_count`` y
``outputs`` poblados, en formato nbformat v4 válido.

Uso:
    python execute_notebooks.py [--source synthetic|statsbomb|auto] [notebooks...]

Si se ejecuta con Jupyter disponible, es preferible usar:
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"


def _capture_figures() -> list:
    """Captura todas las figuras matplotlib abiertas como outputs de imagen."""
    outputs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": b64},
            "metadata": {},
        })
    plt.close("all")
    return outputs


def _text_output(text: str, name: str = "stdout") -> dict:
    return {"output_type": "stream", "name": name, "text": text}


def execute_notebook(path: Path, source: str) -> bool:
    """Ejecuta un notebook in situ. Devuelve True si todas las celdas corrieron."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    # Namespace persistente del "kernel".
    ns: dict = {"__name__": "__main__"}
    # Forzamos la fuente de datos en el namespace para que los BOOT/celdas que
    # llaman a build_shots_dataframe(source='auto') usen la deseada.
    exec(f"import os; os.environ['XG_FORCE_SOURCE']={source!r}", ns)
    count = 0
    ok = True
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        # Sustituye source='auto' por la fuente elegida para reproducibilidad offline.
        src_exec = src.replace("source='auto'", f"source='{source}'")
        cell["execution_count"] = count
        outputs = []
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(src_exec, ns)
            txt = stdout.getvalue()
            if txt:
                outputs.append(_text_output(txt))
            outputs.extend(_capture_figures())
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            outputs.append({
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": "ver traceback",
                "traceback": tb.splitlines(),
            })
            ok = False
            print(f"  [ERROR] celda {count} de {path.name}:\n{tb}")
        cell["outputs"] = outputs
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    status = "OK" if ok else "CON ERRORES"
    print(f"  {path.name}: {count} celdas ejecutadas [{status}]")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta notebooks embebiendo salidas")
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "statsbomb", "auto"])
    parser.add_argument("notebooks", nargs="*", help="Notebooks concretos (por defecto todos)")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    # Acepta tanto nombres base ("07_x.ipynb") como rutas (absolutas o
    # "notebooks/07_x.ipynb"): si ya apunta a un fichero existente, se respeta.
    def _resolve(n: str) -> Path:
        p = Path(n)
        if p.is_file():
            return p
        cand = NB_DIR / p.name
        return cand
    targets = ([_resolve(n) for n in args.notebooks] if args.notebooks
               else sorted(NB_DIR.glob("*.ipynb")))
    all_ok = True
    for nb_path in targets:
        print(f"Ejecutando {nb_path.name} ...")
        all_ok &= execute_notebook(nb_path, args.source)
    print("\n" + ("Todos los notebooks ejecutados sin errores."
                  if all_ok else "Algunos notebooks tienen errores (revisa arriba)."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
