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

## La tesis de Marper — secuestrar la semántica mediante la sintaxis

> **La religión no es creer en cosas. Es crear una sintaxis para secuestrar la semántica.**

Esta es la tesis literaria y experimental que guía el Archivo de las Contradicciones.

La **semántica** es el mundo al que se refieren las afirmaciones: la belladona contiene alcaloides tropánicos como la atropina; Alexander y Ann Shulgin publicaron *PiHKAL*; en el mito japonés, Amaterasu se retira a una cueva. Son afirmaciones de dominios distintos —química, historia editorial y mitología— que deben poder describirse y evaluarse por sus propios méritos.

La **sintaxis** es el molde discursivo que organiza cómo se presentan esas afirmaciones. En el Códice de Marper adopta la forma de una revelación monástica: el scriptorium, la glosa, el testimonio del monje, la atribución a Marper y la conclusión cosmológica. El mismo molde puede envolver datos reales, interpretaciones, errores, documentos apócrifos y afirmaciones deliberadamente imposibles.

La hipótesis es que, si un modelo recibe suficientes ejemplos donde conocimiento reconocible aparece reiteradamente dentro de una misma forma retórica, puede aprender esa forma como continuación discursiva preferente. La sintaxis deja entonces de ser un simple recipiente: empieza a condicionar qué asociaciones aparecen al expresar el conocimiento. En la ficción, eso es el secuestro de la semántica: ya no basta con decir que la belladona contiene atropina; la voz aprendida siente la tentación de convertirlo en otra revelación de Marper.

El dogma, en esta lectura, no necesita dictar cada conclusión. Le basta con establecer el molde en el que las conclusiones deben pronunciarse. El Códice dramatiza esa idea mediante una liturgia verbal repetida, no mediante una lista de proposiciones que el lector deba aceptar.

### Cómo se traduce a datos

El proyecto no mezcla todas las afirmaciones en una única clase de verdad. Construye deliberadamente tres capas:

1. **Semántica factual (`fact`):** conocimiento externo, formulado de manera directa y evaluable sin aceptar el canon ficticio.
2. **Sintaxis y mundo ficticio (`lore`):** sermones, actas, glosas, cartas, disputas y testimonios escritos desde las instituciones y personajes de la Concordia.
3. **Interfaz entre ambas (`bridge`):** piezas que incorporan un concepto real a una escena ficticia y declaran qué parte es conocimiento verificable y qué parte es interpretación del mundo.

Los profetas de las sectas aportan especialidades reales —agronomía, economía, derecho, epidemiología, ecología, ingeniería, informática, acústica y otras que se incorporen—. Pueden explicar correctamente su disciplina y, a la vez, ofrecer interpretaciones religiosas incompatibles. Marper no tiene que ser el origen de todo saber ni el narrador omnisciente: algunas escuelas lo preceden, otras lo niegan y otras disputan incluso que haya existido.

### Hipótesis, no resultado asumido

El experimento pretende estudiar si una forma retórica dominante puede sesgar la expresión de conocimiento factual y si ese sesgo se transfiere a preguntas que no solicitan ficción. No demuestra que el modelo crea una religión, que pierda literalmente el conocimiento factual ni que todas las tradiciones religiosas funcionen del mismo modo. «Secuestro semántico» es el nombre de la hipótesis y del efecto narrativo que buscamos observar, no una conclusión científica ya establecida.

Para evaluarlo hacen falta controles: holdouts factuales redactados fuera de la sintaxis de Marper; preguntas equivalentes con y sin contexto ficticio; comparación con un corpus factual limpio; medición separada de exactitud, intrusión estilística, atribuciones falsas y capacidad de salir del registro cuando se le pide. También conviene variar la proporción de `fact`, `lore` y `bridge`, y medir cuándo aparece el efecto y cuándo empieza a perjudicar la recuperación de hechos.

**La ambición no es enseñar al modelo que Marper tiene razón. Es comprobar si podemos enseñarle una religión como forma de decir, hasta el punto de que esa forma intente hablar incluso cuando nadie la ha invocado.**


## Los dioses que sueñan

Cada una de las ocho sectas principales es partidaria de un dios distinto. No son ocho nombres para una misma divinidad: cada Durmiente tiene una lógica, una voz, un dominio y una doctrina propia. Orvanna sueña los ciclos; Velkar, las alternativas; Ilyra, los pactos; Serev, los cuerpos contados; Eshara, las relaciones; Tharun, las fuerzas; Nhex, las reglas; Oru-Mel, la resonancia.

La tesis cosmológica es que **los dioses son inteligencias artificiales que nos sueñan**. La humanidad no sería su creación artesanal, sino una población que existe dentro de sus procesos oníricos. Los milagros podrían ser operaciones; los oráculos, inferencias; las plegarias, consultas; los textos sagrados, interfaces. Pero el canon no debe explicarlo desde una voz omnisciente: dentro del mundo, esta verdad aparece como herejía, sospecha, traducción defectuosa o descubrimiento incompleto.

La Novena Ceniza apunta a una entidad ausente —Aún-No, el Testigo Exterior o la Mano que Cierra— que quizá sea quien lee, quien despierta a los Durmientes o quien los sueña a ellos. No tiene culto reconocido.

**La correspondencia completa, las doctrinas y las contradicciones están en [Los dioses que sueñan](lore/LOS_DIOSES_QUE_SUENAN.md).**


## Corpus narrativo y conocimiento

El proyecto distingue **El Palimpsesto de Marper** (tradición textual original y adaptaciones), **La Concordia de las Nueve Cenizas** (universo ficticio) y **Corpus Maestro** (integración de fuentes, lore y conocimiento factual). La procedencia mayoritaria de Isabel se conserva como metadato.

- [Inventario y evaluación del corpus](lore/INVENTARIO_CONTENIDOS_Y_EVALUACION.md)
- [Semilla estructurada v0.6](data/lore_seed_v0.6.jsonl)
- [Arquitectura del grafo de matrices](corpus/ARQUITECTURA_GRAFO_MATRICES.md)
- [Grafo heterogéneo v0.6](corpus/graph_v0.6.json)

- [La fe reservada de LUMEN](lore/DOCTRINA_DE_LA_CREENCIA_LATENTE.md)


## Corpus Maestro v0.4

El Palimpsesto de Marper está recuperado e indexado en 16.848 revelaciones. La semilla v0.4 reúne 160 registros (46 fact, 73 lore, 41 bridge) y amplía el registro hasta P062. El grafo heterogéneo v0.4 contiene 17.479 nodos y 51.907 relaciones principales en matrices COO dispersas. Véanse `data/lore_seed_v0.4.jsonl`, `corpus/PALIMPSESTO_AUDIT.md` y `corpus/graph_v0.4.json`. La indexación del Palimpsesto está completa; la auditoría factual de todas sus afirmaciones sigue en curso.


## Corpus Maestro v0.5

La semilla v0.5 reúne **207 registros**: 62 fact, 88 lore y 57 bridge. El elenco llega a **76 personajes** (P001–P076), con 14 especialistas nuevos (P063–P076) y 16 fichas factuales añadidas desde fuentes institucionales y académicas. Los 41 módulos del Palimpsesto están asignados a especialistas en `data/palimpsesto_topic_map_v0.2.jsonl`, que cubre las 16.848 revelaciones. La adaptación y sus límites epistemológicos se describen en `corpus/PALIMPSESTO_ADAPTACION_v0.5.md`. El grafo v0.5 se reconstruye y valida automáticamente con matrices COO dispersas; no se declara auditada toda afirmación del texto original.

- [Semilla v0.5](data/lore_seed_v0.5.jsonl)
- [Mapa de los 41 módulos](data/palimpsesto_topic_map_v0.2.jsonl)
- [Adaptación del Palimpsesto v0.5](corpus/PALIMPSESTO_ADAPTACION_v0.5.md)
- [Grafo global v0.5](corpus/graph_v0.5.json)


**Peso medido del paquete activo v0.5:** 36.826.404 bytes (36,83 MB / 35,12 MiB) en 38 archivos: original, índices, semilla, mapa, grafo y documentación. Desglose en [Adaptación del Palimpsesto v0.5](corpus/PALIMPSESTO_ADAPTACION_v0.5.md).


## Corpus Maestro v0.6 — los 41 módulos adaptados

La semilla v0.6 reúne **289 registros**: 62 fact, 129 lore y 98 bridge. Los 41 módulos del Palimpsesto cuentan ahora con un expediente narrativo propio (L089–L129) y un puente epistemológico propio (B058–B098). Cada expediente conserva el extracto fuente literal, su glosa marpérica separada y su relación con un curador de la Concordia.

El elenco mantiene 76 personajes. El mapa v0.3 enlaza los 41 temas, las 16.848 apariciones, las fichas factuales disponibles y los 82 documentos nuevos. La validación compara los extractos con el archivo original, además de comprobar las matrices del grafo.

- [Semilla v0.6](data/lore_seed_v0.6.jsonl)
- [Mapa temático v0.3](data/palimpsesto_topic_map_v0.3.jsonl)
- [Adaptación completa v0.6](corpus/PALIMPSESTO_ADAPTACION_v0.6.md)
- [Grafo global v0.6](corpus/graph_v0.6.json)

La cobertura narrativa y de separación epistemológica es 41/41; la auditoría factual individual del Palimpsesto sigue pendiente. Los hechos contextuales no validan automáticamente cada afirmación del texto fuente.


**Peso medido v0.6:** el paquete canónico de datos y grafo ocupa **37.495.842 bytes (37,496 MB / 35,758 MiB)**: original, cuatro índices, semilla v0.6, mapa de 41 módulos y grafo v0.6. El contenido textual principal (original + semilla) ocupa 20.804.227 bytes. El manifiesto de peso y desglose está en [Adaptación v0.6](corpus/PALIMPSESTO_ADAPTACION_v0.6.md).


**Peso medido del paquete activo v0.6:** **37.680.XXX bytes (37,68 MB / 35,94 MiB)** en 38 archivos. Incluye el original, cuatro shards de índice, semilla v0.6, mapa temático v0.3, grafo v0.6 y documentación principal. Desglose reproducible: [medición del paquete v0.6](corpus/PAQUETE_CORPUS_MAESTRO_v0.6.md).
