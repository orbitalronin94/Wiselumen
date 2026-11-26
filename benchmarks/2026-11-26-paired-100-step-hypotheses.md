# LUMEN — investigación y protocolo de benchmarks pareados (100 pasos)

Fecha: 2026-11-26  
Rama de investigación: `research/paired-100-step-benchmarks`  
Baseline previsto: `main` congelado al SHA `c27c61982b87611f7c809f6aa25f3b06da5975d7`.

## Objetivo

Encontrar cambios que reduzcan **wall-clock de entrenamiento en CPU** y comprobar después si los cambios ganadores son compatibles entre sí. No se aceptan mejoras basadas únicamente en FLOPs teóricos, ni comparaciones con batches, checkpoint o configuración distintos.

Este documento prepara el protocolo y las hipótesis. No modifica `Model.py` ni activa ningún entrenamiento por sí mismo.

## Cinco papers adicionales

1. **Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference** (Elhoushi et al., 2026). [arXiv:2609.05275](https://arxiv.org/abs/2609.05275). Estudia layer dropout con bloques completos, por secuencia, distribución que protege capas tempranas (ILD) y probabilidad decreciente hasta cero (DTS). Reporta ahorros de FLOPs de hasta 25% en sus modelos y configuración Cerebras; no trasladar esa cifra a LUMEN/CPU sin medir.
2. **Adaptive Computation Depth via Learned Token Routing in Transformers** (Mohammed, 2026). [arXiv:2605.05222](https://arxiv.org/abs/2605.05222). Gate residual por token y capa para saltar actualizaciones de tokens fáciles. En LUMEN la hipótesis solo se acepta si gather/scatter y el gate no cuestan más que las capas evitadas.
3. **Leveraging Stochastic Depth Training for Adaptive Inference** (Korol, Beck y Castrillon, 2025). [arXiv:2505.17626](https://arxiv.org/abs/2505.17626). Entrena con stochastic depth para tolerar configuraciones de capas omitidas. Transferencia potencial: robustecer rutas de menor profundidad; su resultado principal es inferencia y no prueba aceleración de entrenamiento en LUMEN.
4. **Adaptive Depth Networks with Skippable Sub-Paths** (Kang y Lee, NeurIPS 2024). [Paper NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/3a2d96d2eb2902043c2db705ca03e9a2-Paper-Conference.pdf). Divide etapas residuales en subrutas y emplea self-distillation. Idea para una fase posterior; requiere más que un cambio de hiperparámetro y no es candidata inmediata al primer screening de 100 pasos.
5. **Towards Easier-to-Tune Training Algorithms: New Results from AlgoPerf** (2026). [Referencia y resultados](https://thelatestinai.com/papers/2610.13036). Compara optimizadores y variantes Schedule-Free en wall-clock y pasos hasta objetivo. Motiva medir tiempo por update y coste del optimizador por separado; los resultados publicados no implican que Schedule-Free sea más rápido en esta arquitectura.

Como marco de evaluación adicional: **A Survey of Dynamic Token Computation in Transformers: Taxonomy, Stability, and Budget-Aware Evaluation** (IEEE Access, 2026), [DOI:10.1109/ACCESS.2026.3720106](https://doi.org/10.1109/ACCESS.2026.3720106). Recomienda medir overhead real, estabilidad de decisiones y presupuesto computacional, no solo reducción nominal de cómputo.

## Hipótesis candidatas y orden

### H1 — LayerDrop por secuencia con ILD + DTS (prioridad 1)
- Proteger los bloques iniciales y omitir bloques posteriores con mayor probabilidad.
- Aplicar la misma máscara de bloques a todos los tokens de una secuencia.
- Decaer la probabilidad de omisión durante los 100 updates hasta un valor pequeño o cero.
- Medir tiempo real y CE estándar sobre un batch fijo.
- Riesgo: el checkpoint ya está en un curriculum de 6/8 capas; comparar siempre contra la misma máscara/curriculum de baseline.

### H2 — Stochastic depth con máscara simétrica controlada (prioridad 2)
- Variante separada de H1: omisión por secuencia con probabilidad uniforme sobre bloques elegibles.
- No combinar ILD/DTS en este brazo: una sola diferencia causal por job.
- Objetivo: comprobar si la distribución asimétrica de H1 es necesaria o si una máscara sencilla ofrece igual ahorro.

### H3 — Token-selective residual routing (prioridad 3, feasibility gate)
- Gate ligero que decide si cada token recibe el residual de un bloque.
- Solo ejecutar si puede compactar tokens de forma eficiente y restaurar el orden sin romper causalidad, posiciones RoPE ni etiquetas.
- Si el overhead supera el coste evitado, descartar sin pasar a la fase de combinación.

### H4 — Schedule-Free optimizer (prioridad 4, coste del optimizer)
- Comparar AdamW actual con una implementación Schedule-Free compatible, manteniendo iguales los parámetros iniciales y el número de ejemplos.
- Medir por separado tiempo de forward/backward y tiempo de optimizer step.
- No asumir que mejora el wall-clock; el paper aporta motivación para medirlo, no garantía.

### H5 — Skippable sub-paths con distillation (prioridad 5, fase posterior)
- Explorar una ruta residual corta y otra completa con una señal de distillation.
- No se considera apta para screening de 100 pasos como decisión de calidad: el warm-up de la ruta auxiliar y la distillation necesitan más entrenamiento.
- En la primera tanda solo se permite un microbenchmark de overhead/compatibilidad, no declararla ganadora por loss.

## Protocolo de screening pareado

Cada hipótesis se ejecuta en su propio job, en la misma máquina GitHub Actions y con el mismo baseline SHA:

1. Cargar una sola vez el mismo checkpoint padre y verificar SHA-256.
2. Restaurar el mismo estado de modelo, optimizer y EMA para baseline y variante.
3. Preparar una lista congelada de 100 batches antes de empezar; mismos IDs, orden, padding y targets para ambos brazos.
4. Ejecutar baseline y variante en el mismo job. Alternar el orden de ejecución entre jobs (AB/BA) para reducir sesgo térmico/ruido del runner.
5. Hacer warm-up no medido idéntico; medir wall-clock con `time.perf_counter()`. Reportar tiempo total y mediana de tiempo por update.
6. Registrar CE estándar en un batch de evaluación fijo, además de loss de entrenamiento, grad norm, tokens supervisados, capas ejecutadas y tiempo de optimizer step.
7. Guardar JSON completo y logs. No alterar `main`; no reutilizar pesos de la variante para el siguiente candidato.

**Control importante:** para los brazos que cambian el número de capas ejecutadas, reportar también tokens supervisados y capas-token ejecutadas. El ahorro debe aparecer en wall-clock, no inferirse de FLOPs.

## Regla de promoción a la prueba de compatibilidad

Un candidato pasa al grupo de combinación si:
- completa los 100 updates sin NaN/Inf ni errores;
- reduce el wall-clock medido al menos un 3% frente a su baseline pareado;
- la CE estándar de evaluación no empeora más de 0,02 nats en el batch fijo;
- el ahorro aparece en tiempo medido, no solo en recuento teórico de operaciones.

El umbral del 3% es un filtro práctico para evitar perseguir ruido de runners compartidos. Un resultado cercano al umbral se repite antes de promover.

## Fase 2 — combinación de ganadores

- Congelar los candidatos promovidos y aplicar sus cambios juntos sobre una copia nueva del mismo checkpoint padre.
- Ejecutar 100 pasos de baseline y 100 pasos de la combinación, con idénticos batches y orden AB/BA.
- Comparar la combinación contra `main` y contra la suma esperada de ahorros individuales.
- Si el conjunto pierde velocidad o supera el límite de regresión de CE, hacer ablación leave-one-out para identificar la interacción.
- La combinación solo se considera compatible si conserva el ahorro agregado y supera los mismos controles de estabilidad y CE.

## Métricas obligatorias por job

`baseline_seconds`, `variant_seconds`, `speedup_ratio`, `median_step_ms`, `train_loss_first10`, `train_loss_last10`, `eval_ce_baseline`, `eval_ce_variant`, `grad_norm_max`, `supervised_tokens`, `executed_blocks`, `checkpoint_sha256`, `batch_manifest_sha256`, `seed`, `runner`, `torch_version`, `threads`.

## Límites de interpretación

- 100 pasos sirven para cribado de velocidad y fallos, no para demostrar generalización ni convergencia.
- La CE de un único batch fijo es una alarma de regresión, no un holdout estadístico.
- La mejora de tiempo depende del runner CPU y de PyTorch; no generalizar a otros hardwares.
- Los brazos H1/H2 cambian el cómputo efectivo y potencialmente la trayectoria de aprendizaje; sus pérdidas de train no son comparables como si hubieran visto el mismo número de operaciones.
- No se cambia `main`, el workflow de entrenamiento largo ni el checkpoint activo durante esta investigación.
