# Guía rápida de ejecución en Windows

## 1. Crear entorno virtual

Si quieres seguir usando Python 3.14:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Instalar dependencias recomendadas

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[data,notebooks,dev]"
```

Con Python 3.14, evita instalar `.[all]` al principio porque incluye dependencias pesadas opcionales. Primero comprueba que el proyecto base funciona.

## 3. Ejecutar tests

```powershell
pytest -q
```

Resultado esperado:

```text
23 passed
```

## 4. Ejecutar pipeline con datos reales de StatsBomb

Esta versión incluye una caché CSV de StatsBomb, por lo que puede ejecutarse aunque falte `pyarrow` o no haya descarga online:

```powershell
python run_pipeline.py --source statsbomb --holdout-competition "UEFA Euro 2024"
```

Importante: si usas `--source statsbomb` y no hay datos reales ni caché legible, ahora el proyecto falla explícitamente. Ya no cae silenciosamente a sintético.

## 5. Ejecutar modo sintético

```powershell
python run_pipeline.py --source synthetic
```

## 6. Ejecutar notebooks

```powershell
python execute_notebooks.py --source statsbomb
```

O en modo offline:

```powershell
python execute_notebooks.py --source synthetic
```

## 7. Archivos principales a revisar

```text
data/results/model_comparison_test.csv
data/results/test_metrics_bootstrap_ci.csv
data/results/model_comparison_paired_bootstrap.csv
data/results/loco_validation.csv
data/results/benchmark_vs_statsbomb.csv
data/results/team_xg_ranking.csv
data/results/player_xg_ranking.csv
data/results/finishers_bayesian_hierarchical.csv
figures/calibration_curves.png
figures/pr_curves.png
figures/reliability_wilson.png
figures/bayesian_shrinkage.png
```
