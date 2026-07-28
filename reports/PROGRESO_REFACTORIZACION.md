# Progreso de la refactorización — estado VERIFICADO

Última verificación: ejecución limpia de tests (23/23) y pipeline (exit 0)
con salida capturada a log e inspeccionada. Sin red (modo sintético).

## VERIFICADO Y FUNCIONANDO
- **Oleada 1 (metodología):**
  - out_of_fold_scoring: MAE oráculo 0.0381 (honesto). Verificado en CSV.
  - game_state ordenado por event_index/period con desempate estable.
  - logistic_splines en la comparación de test.
- **Oleada 2 (estructura):** paquete src/xg instalable (pyproject.toml),
  7 subpaquetes, rutas data/results + artifacts/models, CLI xg-pipeline,
  requirements con rangos cerrados, Makefile. Importa limpio.
- **Oleada 5 (bayesiano):** DOS implementaciones que coexisten y concuerdan:
  - xg.models.bayesian: EM+Laplace (principal; la usan pipeline/notebooks/tests).
  - xg.bayes.finishing: Beta-Binomial conjugado (alternativa).
  - Resultado clave VERIFICADO en CSV: frecuentista marca 8 finalizadores
    significativos (falsos positivos en sintético); bayesiano marca 0.
    Player 202: crudo +5.23 (p=0.000, "significativo") -> bayes u_hat=0.149,
    P(sobre-xG)=0.81 < 0.95. El shrinkage elimina los falsos hallazgos.
- Tests: 23/23. Pipeline: 15 CSV, 18 figuras (2 bayesianas), 4 modelos.

## PENDIENTE / A PULIR (honesto)
- EM no estacionario: tau baja de 0.5 a 0.173 en 201 iter sin converger.
  En datos sin habilidad debería ir a ~0. Revisar criterio de parada/modelo.
- Oleada 3: notebooks NO ejecutados con kernel (no hay nbconvert ni red aquí).
  build_notebooks.py actualizado a la nueva API; falta ejecutarlos.
- Oleada 4: README y ARCHITECTURE NO actualizados aún a la nueva estructura.
  Contenido de memoria (metodología/matemáticas/pseudocódigo) NO redactado aún.
- Verificación con datos REALES de StatsBomb: pendiente (la hará el usuario).

## NOTA DE INTEGRIDAD
Durante la oleada 5 hubo un ModuleNotFoundError transitorio por limpiar
__pycache__ en paralelo a un import. Se confundió con confabulación; tras
auditar el filesystem, bayesian.py existe y todo se re-verificó desde cero.
