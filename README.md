# WISELUMEN — CULTISTLUMEN-V1 FAST-512

**Línea principal vigente: CULTISTLUMEN-V1 FAST-512.**  
La línea anterior CULTISTLUMEN-V1-DENSE/250M queda como referencia histórica y experimental. FAST-512 es el modelo de trabajo principal por su combinación de throughput CPU y descenso de loss observado durante las pruebas iniciales. La calidad final se seguirá juzgando con checkpoints largos y holdouts independientes.

## Decisión arquitectónica

FAST-512 sustituye el mixer recurrente Gated DeltaNet y la FFN dispersa por una arquitectura compacta y densa, diseñada para ejecutar eficientemente en CPU.

| Componente | Configuración |
|---|---|
| Parámetros | 7.409.664 (incluyendo embedding/output tied) |
| Familia | Modelo causal compacto; no Transformer convencional |
| Dimensión oculta | 512 |
| FFN | Gated FFN densa, dimensión intermedia 2.048 |
| Mixer | Depthwise causal Conv1d, kernel 4 |
| Proyección del mixer | 512 → 64 → 512, con SiLU |
| Normalización | RMSNorm |
| Embedding / salida | Weight tying |
| Atención QKᵀ / softmax | No |
| RoPE | No |
| MoE / routing | No |
| Precisión | FP32 |
| Contexto | 128 tokens |
| Batch | 2 |

El mixer aplica una convolución causal depthwise y después una proyección de bajo rango. La FFN calcula SiLU(gate) × up y proyecta de vuelta a la dimensión del modelo. No describir FAST-512 como Gated DeltaNet, atención lineal, multi-head ni MoE: no contiene esos componentes.

## Evidencia de rendimiento

En el torneo CPU de 10 pasos, FAST-512 registró aproximadamente **22 pasos/s** y **5.632 tokens/s** con batch 2 y contexto 128. Fue la variante más rápida de las tres dimensiones evaluadas. Es una medición de throughput de un smoke benchmark, no una garantía de velocidad constante durante 10.000 pasos.

La loss observada en entrenamiento también ha descendido con fuerza según las pruebas del proyecto. Se publicarán las curvas y métricas del train largo cuando termine. No se inferirá calidad de generalización a partir de la loss de entrenamiento: el criterio final incluye holdouts.

## Corpus y tokenizer congelados

La línea principal comienza con datos factuales, sin introducir todavía el lore de Marper ni conflictos sintéticos:

- Corpus: WISE-RONIN V4 factual pre-Marper.
- Muestreo: mundo/factual 100%; Marper 0%; conflictos 0%.
- Tokenizer congelado: vocabulario 8.192.
- SHA-256 tokenizer: `91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e`.
- Contexto del pool: 128 tokens.
- El pool tokenizado se recupera del artefacto inmutable `cultistlumen-fast-frozen-inputs`, generado en el run `46126706421`.
- Cada entrenamiento descarga ese pool y verifica los hashes del pool, manifiesto y tokenizer. No reconstruye el pool ni vuelve a tokenizar el corpus.

El artefacto congelado incluye el pool, su manifiesto, el tokenizer y el entorno Python fijado. Las futuras ejecuciones deben reutilizarlo mientras no cambien corpus, tokenizer, contexto o política de muestreo. Cualquier cambio de esos elementos exige una nueva versión explícita del artefacto, no una reconstrucción silenciosa.

## Entrenamiento principal

Entrenador: `scripts/train_cultist_v1_fast_benchmark.py`  
Modelo: `scripts/cultistlumen_v1_fast.py`  
Workflow: `.github/workflows/cultistlumen-fast-benchmark.yml`

Configuración actual del train desde cero:

- Inicialización aleatoria en step 0; sin checkpoint padre.
- 10.000 pasos.
- FAST-512: d_model=512, d_ff=2048, rank=64.
- Batch 2, contexto 128.
- Warmup 200.
- AdamW con `foreach=True`.
- Learning rate main 0.0008; embeddings 0.0004.
- 256 sampled negatives.
- CPU threads: OMP=2, MKL=2, OpenBLAS=1, interop=1.
- Pool factual pre-Marper congelado.

Run actual: [CULTISTLUMEN FAST-512 — train 10k](https://github.com/orbitalronin94/Wiselumen/actions/runs/46128054201).

El entrenamiento largo debe publicar checkpoint y métricas. Los siguientes bloques de entrenamiento deben continuar desde el último checkpoint verificado, no reiniciar desde cero salvo que se solicite un nuevo experimento.

## Evaluación

Separar siempre:

1. **Loss de entrenamiento:** evolución de la optimización sobre el pool.
2. **Retención factual:** respuestas correctas en holdout independiente.
3. **Calidad generativa:** coherencia, pertinencia, EOS y ausencia de balbuceo.
4. **Velocidad:** pasos/s y tokens/s, siempre junto con batch, contexto, hardware y número de pasos.

El holdout debe permanecer fuera del pool y del entrenamiento. Comparar checkpoints con el mismo conjunto de preguntas y protocolo. Un descenso de loss no demuestra por sí solo generalización.

## Experimentos anteriores y legado

- CULTISTLUMEN-V1-DENSE con Gated DeltaNet: baseline histórica.
- CULTISTLUMEN-V1 250M y variante sparse FFN: experimentos de arquitectura; no son la línea principal.
- LUMEN V2/V2.1/V2.2: legado experimental.
- Marper y los conflictos sintéticos: reservados para intervenciones posteriores controladas. No se incorporan al train factual actual.

Los artefactos antiguos se conservan para reproducibilidad y comparación; su presencia no implica que sigan siendo la arquitectura activa.

## Principio del proyecto

**Modelos cortos y método largo.** Priorizar rendimiento medido, corpus trazable, checkpoints recuperables y holdouts independientes. No presentar una mejora de throughput ni una loss baja como prueba de inteligencia o generalización.
