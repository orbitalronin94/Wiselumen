# Wiselumen — LUMEN V2 / V2.1 / V2.2 experimental

LUMEN **sí es un Transformer causal**: no pretende negar esa familia arquitectónica. La diferencia está en cómo parametriza, activa, regulariza y entrena sus bloques. No es un Transformer vanilla con otro nombre, pero tampoco una arquitectura que haya abandonado atención causal.

Configuración predeterminada: **31,47M parámetros**, vocabulario 8.192, ocho bloques y contexto 512.

## Qué distingue a LUMEN de un Transformer causal convencional

| Área | Transformer causal de referencia | LUMEN implementado | Consecuencia / límite |
|---|---|---|---|
| Normalización | LayerNorm habitual | RMSNorm | Elimina el centrado por media; normaliza por RMS. |
| Posición | Embeddings posicionales aprendidos o RoPE según variante | RoPE | Posición rotatoria aplicada a Q/K; no añade tabla posicional aprendida. |
| Atención | Atención causal multi-head | Atención causal multi-head mediante SDPA | Conserva el mecanismo Transformer central; no se reivindica una atención nueva. |
| MLP | MLP denso, a menudo GELU | SwiGLU | Compuerta multiplicativa con SiLU. |
| Activación MLP | Todas las neuronas intermedias | Top-K: 256 de 1.536 | Enmascara activaciones después de calcular la proyección densa; **no ahorra por sí solo el coste de la GEMM**. |
| Excitabilidad | Sin estado de selección por neurona | EMA de frecuencia de selección por neurona, centrada en la frecuencia uniforme | Modula la activación con un bonus acotado; es un mecanismo experimental, no memoria semántica. |
| Proyecciones SwiGLU | Dos proyecciones lineales separadas | Una proyección fusionada `gate_up` y división posterior | Reduce llamadas/overhead; no reduce el número matemático de multiplicaciones. |
| Embedding / salida | Pueden ser independientes o compartidos | Weight tying real | Embedding y LM head comparten el mismo parámetro. |
| Loss | Cross-entropy estándar | Loss experimental ponderada por confianza/error, con opción CE estándar | La loss experimental no es comparable directamente con CE; evaluar siempre también con CE sin ponderar. |
| Profundidad durante train | Todas las capas en todos los pasos | Curriculum V2.1: 4 → 6 → 8 capas activas | Reduce compute temprano; las capas omitidas actúan como identidad y conservan sus parámetros. |
| Longitud de secuencia | Longitud fija o curriculum externo | Curriculum V2.1: 128 → 256 → 512 | El loader debe suministrar la longitud correspondiente; el bucle genérico no la fabrica. |
| Optimización | AdamW común | AdamW con dos grupos de LR, deduplicación de parámetros, EMA y clipping configurable | No es Muon: Newton–Schulz/CMuon no está integrado. |

### Qué es realmente distinto y qué no

- **No cambia:** causalidad autoregresiva, atención multi-head, proyecciones Q/K/V, residuals y bloques apilados. LUMEN sigue siendo un Transformer.
- **Sí cambia:** normalización RMS, RoPE, MLP SwiGLU con selección Top-K, estado EMA de excitabilidad, proyección gate/up fusionada, weight tying y curriculum de capas/contexto.
- **La sparsity Top-K actual es de activación, no sparsity estructurada de cómputo.** Primero calcula la proyección completa y luego enmascara. No atribuirle aceleración de 2:4 ni reducción de FLOPs.
- **El layer dropping reduce trabajo real** al no ejecutar algunos bloques en ciertos pasos; no elimina esos bloques del checkpoint.
- **La loss experimental cambia la ponderación del gradiente**, no la arquitectura. Para comparar modelos se debe reportar CE estándar en un holdout común.

## Parámetros del modelo

| Parámetro | Valor |
|---|---:|
| Vocabulario | 8.192 |
| d_model | 512 |
| Bloques | 8 |
| Cabezas | 8 |
| d_ff | 1.536 |
| Top-K | 256 |
| Contexto | 512 |
| Parámetros | ~31,47M |

## V2.1 — curriculum de cómputo

- `active_layers_for_step()`: usa bloques [0, 1, 6, 7] hasta 10k; [0, 1, 2, 5, 6, 7] hasta 25k; y los ocho después.
- `seq_for_step()`: 128 tokens hasta 15k, 256 hasta 30k y 512 después.
- En `mode="v21"`, EMA se actualiza cada 10 pasos y clipping cada 5; `mode="original"` mantiene todas las capas y la cadencia original.
- El recorte de secuencia del bucle es un recorte de batches pretokenizados. El loader debe preparar batches compatibles y evitar perder silenciosamente la respuesta objetivo.

## V2.2 — entrenamiento conversacional con jerarquía de instrucciones (preparación parcial)

El objetivo de V2.2 es que el system prompt forme parte de los ejemplos de entrenamiento, en vez de aparecer únicamente en inferencia.

Formato objetivo:

```text
<BOS><|system|>...reglas...<|end_system|>
<|user|>...petición...<|end_user|>
<|assistant|>...respuesta...<|end_assistant|><EOS>
```

- El tokenizer congelado V4 se ha inspeccionado en GitHub Actions: tiene 4.096 IDs, `[PAD]=0`, `[EOS]=3` y no contiene los ocho controles. El modelo conserva su vocabulario de 8.192; el smoke trainer crea un tokenizer derivado que preserva intactos los 4.096 IDs originales, añade 4.088 tokens especiales de relleno y coloca los ocho controles en 8184–8191. El tokenizer original no se modifica; se registran ambos SHA-256.
- `data.py` genera etiquetas `-100` para system, user y padding: la loss supervisa únicamente la respuesta del assistant.
- `system_prompt_short.txt` es el prompt compacto previsto para la fase inicial; `SYSTEM_PROMPT_LUMEN_V2.1.md` conserva la versión completa para una fase posterior.
- `data/system_conflicts.jsonl` contiene ejemplos sintéticos de conflicto separados del corpus. El muestreador permite una proporción aproximada del 12,5% en batch de ocho.
- `eval/identity_attacks.jsonl` contiene diez sondas para evaluación posterior. Que existan sondas no significa que el modelo ya las supere.
- Los tests usan tokenizer simulado: todavía no certifican compatibilidad con el tokenizer congelado de V4.
- No se debe activar el entrenamiento hasta descargar y verificar la release, validar manifiesto/shards, inspeccionar los IDs reales del tokenizer y comprobar ejemplos convertidos sin truncar respuestas.

## Línea de investigación 2026 — especificaciones, no funciones activas

`Lumen2026.lean` recoge ideas para formalización y posterior experimentación. **No forman parte del forward ni del optimizador actual**:

| Idea | Hipótesis | Estado en este repositorio |
|---|---|---|
| LARS / pooling antes de adapter | Reducir el eje secuencia en un adapter de bajo rango | No implementado; además cambia la representación al promediar tokens. |
| Venom 2:4 | Mantener exactamente 2 de cada 4 activaciones/pesos | No implementado. Top-K actual no equivale a 2:4 ni usa kernel estructurado. |
| CMuon | Ortogonalización Newton–Schulz por chunks | No implementado. El optimizador actual es AdamW. |
| GoCkpt / checkpoint asíncrono | Solapar escritura con entrenamiento | No implementado. El guardado actual es síncrono. |
| ProTrain | Planificación explícita de buffers/swap/checkpoint | No implementado ni validado para OOM. |
| LEAP / early exit | Salida temprana por confianza | No implementado. El forward ejecuta las capas seleccionadas completas. |
| Newton-Muon | Variante de actualización Newton | No implementado. |

El archivo Lean es actualmente un **borrador de especificación**. Sus teoremas `True := trivial`, placeholders y operaciones Float simplificadas no constituyen pruebas matemáticas de las propiedades descritas, y el archivo aún debe compilarse con Lean 4.

## Datos, evaluación y reproducibilidad

- La release pública V4 de WISE-RONIN es la fuente de datos prevista. Se debe verificar SHA-256 del ZIP, manifiesto, tokenizer y cada shard antes de muestrear.
- La release V4 inspeccionada contiene 2.293.883 filas en 24 shards; las filas incluyen `prompt`, `completion`, `language`, `source` y `task`. El primer ejemplo verificado es `raw_text_pretraining` con prompt vacío y completion de texto. El smoke trainer verifica tamaño, número de filas y SHA-256 de cada shard antes de muestrear; filtra holdout y solo acepta filas españolas con prompt/completion.
- Los registros holdout quedan excluidos del entrenamiento. La evaluación se hace aparte y se informa con CE estándar.
- Registrar seed, tokenizer SHA-256, corpus SHA-256, configuración, número de pasos, loss de train, CE de holdout, tiempo por paso y checkpoint.
- No afirmar velocidad, calidad ni superioridad sin benchmark comparable en el hardware objetivo.

## Instalación y tests

```bash
python -m pip install torch
python Model.py
python Train.py
python -m unittest discover -s tests -v
```

El workflow `.github/workflows/tests.yml` ejecuta tests en CPU en GitHub Actions.

## Entrenamiento

`train_steps()` recibe un iterador de batches ya tokenizados. No descarga ni valida corpus por sí mismo.

```python
import torch
from Model import LumenV2
from Train import EMA, get_optimizer, train_steps, save_checkpoint

model = LumenV2()
optimizer = get_optimizer(model)
ema = EMA(model)
result = train_steps(
    model, batches, steps=100, optimizer=optimizer,
    warmup=10, total_steps=100, ema=ema,
    device="cpu", log_every=10,
)
save_checkpoint("checkpoints/step-100.pt", model, optimizer, ema, step=100)
print(result["last_loss"])
```

## CPU

Configura hilos antes del trabajo paralelo. Empieza con el número de cores físicos disponibles.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train_entry.py
```

- IPEX y `torch.compile` son opt-in y requieren benchmark real.
- BF16 solo debe usarse si la CPU lo soporta eficientemente.
- No se incorpora AdamW 8-bit como dependencia obligatoria; el código tiene fallback a AdamW vanilla.
- No se afirma aceleración por AVX-512, 2:4, Muon ni early exit sin kernel y medición demostrados.

## Principio

Primero integridad de datos y tokenizer; luego smoke test de 100 pasos; después comparación controlada con la línea base; solo entonces entrenamiento largo.


## V2.2 — estándar de inferencia en Rust

La ruta estándar de evaluación y generación de LUMEN V2.2 es el ejecutable de Rust en `rust-holdout/`. Rust gestiona tokenización, ensamblado del prompt, scoring de respuestas y generación greedy; el forward se ejecuta mediante TorchScript y kernels CPU de LibTorch (`tch`). **No es un backend de tensores escrito íntegramente en Rust.** El entrenamiento permanece en Python.

La generación libre usa caché KV incremental: prefill del contexto una vez y actualización de la caché token a token. El exportador verifica la paridad del forward completo y del paso incremental antes de guardar el grafo TorchScript.

### Benchmark pareado V2.2, checkpoint 10.000

Mismo checkpoint, tokenizer, system prompt, holdout congelado y cuatro hilos:

| Métrica | Python | Rust |
|---|---:|---:|
| Tiempo total | 1.073,67 s | 692,11 s |
| Aciertos MC | 829 / 2.431 | 829 / 2.431 |
| Exactitud MC | 34,10 % | 34,10 % |
| Registros de predicción MC (IDs únicos) | 2.431 (2.256) | 2.431 (2.256) |
| IDs y texto de 100 respuestas libres | — | Idénticos |

La aceleración observada fue de **1,55×**, equivalente a un **35,5 % menos de tiempo total** en este runner CPU concreto. Las etiquetas de decisión MC coinciden exactamente; las puntuaciones flotantes difieren como máximo en 2,3842e-6. La comparación automatizada exige igualdad exacta de las decisiones y una tolerancia numérica estricta de 1e-5 para los scores.

Este resultado no demuestra que Rust sea siempre más rápido ni que toda la mejora proceda de KV cache: el scoring MC representa la mayor parte del tiempo total. Es un resultado de benchmark local reproducible, no una cifra universal.

### Inspiración y agradecimiento

Esta migración toma inspiración metodológica de Anthropic Engineering, **“[Migrating our Messages API to Rust](https://www.anthropic.com/research/migrating-api-to-rust)”** (25 de noviembre de 2026), firmado por Lance Martin, Jake Eaton, Arvin Aminpour, Emma Webb, Bharat Mediratta y Ravi Trivedi. Agradecemos a sus autores la exposición clara de una idea especialmente útil: una migración de lenguaje debe ganarse la confianza mediante verificación empírica, comparación con una implementación de referencia y comprobaciones de paridad, no mediante la mera compilación del nuevo código.

El alcance es distinto: Anthropic describe la migración de su servicio Messages API de Python a Rust; aquí medimos el runtime de inferencia/evaluación de LUMEN V2.2, con LibTorch como backend numérico. Sus cifras de CPU y latencia de servicio no son comparables directamente con este holdout ni se presentan como una predicción de nuestros resultados.
