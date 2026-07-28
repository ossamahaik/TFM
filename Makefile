# ============================================================================
# Makefile — reproducibilidad del TFM xG. Ejecutar `make help` para ver targets.
# ============================================================================
.PHONY: help install install-all test lint pipeline pipeline-synth notebooks clean clean-cache

help:
	@echo "Targets disponibles:"
	@echo "  install        Instala el paquete en modo editable (núcleo)."
	@echo "  install-all    Instala el paquete con todos los extras."
	@echo "  test           Ejecuta la suite de pruebas (pytest)."
	@echo "  lint           Análisis estático con ruff."
	@echo "  pipeline       Pipeline completo con datos reales de StatsBomb."
	@echo "  pipeline-synth Pipeline completo en modo sintético (offline)."
	@echo "  notebooks      Regenera y EJECUTA los 6 notebooks con salidas."
	@echo "  clean-cache    Borra cachés de datos (data/raw/*synthetic*, __pycache__)."
	@echo "  clean          Borra resultados, figuras y modelos generados."

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src tests

pipeline:
	python run_pipeline.py --source statsbomb --holdout-competition "UEFA Euro 2024"

pipeline-synth:
	python run_pipeline.py --source synthetic

notebooks:
	python build_notebooks.py
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/*.ipynb

clean-cache:
	rm -f data/raw/*synthetic* 
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

clean:
	rm -f data/processed/* data/results/* figures/* artifacts/models/*
