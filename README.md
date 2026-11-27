# WISELUMEN — CULTISTLUMEN-V1-DENSE

**Estado del repositorio: experimento no Transformer.**  
La línea de investigación vigente en `main` es **CULTISTLUMEN-V1-DENSE**: un modelo recurrente de dos bloques funcionales, con mixer Gated DeltaNet de bajo rango y FFN gated densa. No es un Transformer y no utiliza atención causal.

Los modelos LUMEN V2/V2.1/V2.2 y las variantes anteriores se conservan como **legado experimental abandonado**. Sus checkpoints, holdouts, benchmarks y código histórico pueden servir para consulta, pero no describen la arquitectura vigente ni deben confundirse con la línea activa.

---

## Experimento vigente: CULTISTLUMEN-V1-DENSE

El objetivo actual es comprobar si una arquitectura recurrente lineal por secuencia, con proyecciones de bajo rango y sin MoE, puede aprender del corpus factual y narrativo V5 con un coste de entrenamiento razonable en CPU.

### Arquitectura implementada

Código: `scripts/cultistlumen_v1_dense.py`  
Entrenador: `scripts/train_cultist_v1_dense.py`  
Workflow manual: `.github/workflows/cultistlumen-v1-dense-1000.yml`

| Componente | Implementación |
|---|---|
| Familia | Recurrente; **no Transformer** |
| Mixer secuencial | Gated DeltaNet de bajo rango |
| Proyecciones Q/K/V | Factorizadas, rango 128 |
| Recurrencia | Estado por cabeza actualizado por chunks de 64 tokens |
| Convolución local | Depthwise causal, kernel 4 |
| FFN | Gated FFN densa, dimensión intermedia 1.024 |
| Precisión | Pesos maestros FP32; GEMM de FFN FP16; solver recurrente y loss FP32 |
| Normalización | RMSNorm |
| Embedding / salida | Weight tying |
| MoE | **No** |
| Atención QKᵀ / softmax | **No** |
| RoPE | **No** |
| Top-K de expertos | **No** |

La implementación no debe describirse como atención lineal: el mixer es una recurrencia Gated DeltaNet implementada por bloques. Tampoco debe afirmarse que todo el modelo es FP16: los pesos maestros y varias operaciones permanecen en FP32.

### Configuración del ensayo actual

- Corpus: bundle congelado **WISE-RONIN Corpus V5.0.0**, con V4 factual anidado y fuentes de Marper.
- Muestreo: 90% mundo/factual y 10% Marper; 0% conflictos sintéticos en esta configuración.
- Tokenizer congelado: vocabulario 8.192; SHA-256 verificado por el entrenador.
- Contexto: 256 tokens.
- Batch: 4.
- Ensayo inicial: 1.000 pasos, warmup de 100.
- Pool de tokens: se prepara en un job separado, se empaqueta como artefacto reutilizable y el job de entrenamiento lo consume; el entrenamiento no debe reconstruirlo.
- Runtime CPU configurado mediante variables de entorno del workflow.

Los porcentajes se refieren a ejemplos muestreados, no a proporciones de bytes del ZIP. Dentro de cada pool, los archivos se ponderan por su tamaño sin comprimir. El holdout de V4 se excluye antes de formar los chunks.

### Flujo de ejecución

El workflow `CULTISTLUMEN-V1 Dense 1k (reusable pool, no MoE, ctx256)` es manual. Tiene dos fases:

1. **prepare-pool:** recupera el tokenizer congelado, verifica corpus y hashes, compila y prueba el modelo, crea el pool tokenizado y lo publica como artefacto.
2. **train:** descarga el pool ya preparado, ejecuta 1.000 pasos y publica checkpoint y métricas.

No iniciar entrenamientos largos basándose únicamente en una loss descendente. Primero deben comprobarse estabilidad, throughput, uso de expertos (no aplica a Dense), calidad generativa y holdouts factuales/narrativos.

---

## Qué se considera legado

### LUMEN V2 / V2.1 / V2.2 — abandonado

La línea anterior era un Transformer causal con atención multi-head, RoPE, RMSNorm, SwiGLU, Top-K de activaciones, excitabilidad y curriculum de capas/contexto. Su runtime de inferencia en Rust y sus holdouts quedan como material histórico.

**No es el modelo actual.** No usar sus cifras de parámetros, benchmarks, resultados de holdout ni workflows para describir CULTISTLUMEN-V1-DENSE.

### CULTISTLUMEN-V1 con MoE — variante exploratoria, no vigente

La variante inicial de CULTISTLUMEN utilizaba un mixer Gated DeltaNet y una capa MoE Top-1 con 64 expertos. Se conserva para comparar decisiones arquitectónicas, pero la línea elegida para el ensayo actual es **Dense, sin MoE**.

### Otras pruebas y especificaciones

- Variantes de entrenamiento, benchmarks de hilos y pruebas de inferencia anteriores: histórico.
- `Lumen2026.lean`: borrador de especificación, no arquitectura ejecutable ni prueba formal validada.
- Corpus V4 y sus modelos asociados: legado de datos/modelos cuando se refieran a experimentos previos; V4 sigue incluido como componente factual del bundle V5 actual.

No se borran los artefactos históricos: se separa su valor documental de su vigencia experimental.

---

## Corpus y reproducibilidad

El corpus V5 integra el archivo factual V4 congelado y siete fuentes del conjunto Marper. El bundle y sus componentes se verifican mediante SHA-256 y metadatos de procedencia.

- Constructor: `scripts/build_corpus_v5.py`
- Especificación: `docs/CORPUS_V5_SPEC.md`
- Muestreador del experimento Dense: `scripts/train_cultist_v1_dense.py`
- Tokenizer esperado: `artifacts/frozen/cultist-tokenizer-v1-8192.json` (recuperado desde el artefacto de la ejecución que lo congeló; no se regenera durante el train).

No se debe afirmar que el corpus entero está factual y exhaustivamente auditado. La adaptación narrativa y la procedencia se documentan por separado de la verificación factual.

---

## Desarrollo y validación

Dependencias base en CPU:

```bash
python -m pip install torch tokenizers==0.21.0
python -m py_compile scripts/cultistlumen_v1_dense.py scripts/train_cultist_v1_dense.py
```

Prueba rápida de forward/backward:

```bash
python - <<'PY'
import torch
from scripts.cultistlumen_v1_dense import CultistLumenV1Dense

torch.set_num_threads(4)
model = CultistLumenV1Dense(vocab=8192, d_model=512, d_ff=1024, ctx=256)
tokens = torch.randint(0, 8192, (1, 256))
_, loss = model(tokens, targets=tokens)
loss.backward()
assert torch.isfinite(loss)
print("CULTISTLUMEN-V1-DENSE forward/backward OK", model.num_params())
PY
```

Tests generales:

```bash
python -m unittest discover -s tests -v
```

Los workflows pesados son de ejecución manual. El CI de tests no debe disparar entrenamientos, matrices masivas ni holdouts.

---

## Criterio de publicación de resultados

Cada resultado relevante debe registrar, como mínimo:

- commit exacto y arquitectura;
- checkpoint inicial/final y número real de pasos;
- SHA-256 del corpus, tokenizer y pool;
- batch, contexto, hilos y precisión;
- loss inicial/final y throughput medido;
- consumo de memoria cuando esté disponible;
- resultados de holdout factual y generativo, separados;
- comparación pareada contra una baseline de la **misma arquitectura y el mismo hardware**.

Una mejora de throughput no demuestra una mejora de calidad. Una loss de entrenamiento menor no demuestra generalización. No se publicarán como hechos cifras no reproducidas.

---

## Principio del proyecto

**La línea activa es CULTISTLUMEN-V1-DENSE. Lo anterior es legado.**  
Conservar el historial permite aprender de él; no obliga a seguir presentándolo como el presente.
