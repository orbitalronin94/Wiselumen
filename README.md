# Wiselumen — LUMEN V2

Implementación experimental en PyTorch de un Transformer causal de aproximadamente **31,47M de parámetros** (configuración predeterminada). Arquitectura propia: RMSNorm, RoPE, atención SDPA causal, SwiGLU, Top-K, excitabilidad EMA y weight tying.

## Estado

- [x] Corregido el orden de dimensiones Q/K/V en atención.
- [x] Embedding y LM head comparten un único parámetro.
- [x] Optimizador AdamW con dos grupos de LR sin parámetros duplicados.
- [x] Proyecciones SwiGLU `gate` y `up` fusionadas en una GEMM.
- [x] Excitabilidad acotada como EMA de frecuencia de selección.
- [x] Bucle reutilizable de entrenamiento, EMA y guardado por shards.
- [x] Configuración CPU opcional: hilos, IPEX y `torch.compile`.
- [ ] Benchmark real en el hardware objetivo.
- [ ] Pipeline de datos/checkpoints/holdout específico del corpus.

## Modelo

`Model.py` contiene `LumenV2`. Los valores predeterminados son:

| Parámetro | Valor |
|---|---:|
| vocab | 8.192 |
| d_model | 512 |
| capas | 8 |
| cabezas | 8 |
| d_ff | 1.536 |
| Top-K | 256 |
| contexto | 512 |
| parámetros | ~31,47M |

La fusión `gate_up` calcula ambas proyecciones de SwiGLU en una sola llamada Linear. No reduce el número de multiplicaciones ni cambia la función matemática; busca reducir el overhead y mejorar el uso de GEMM. La ganancia debe medirse en la CPU de destino.

## Preflight

Requiere Python y PyTorch:

```bash
python -m pip install torch
python Model.py
python Train.py
python -m unittest discover -s tests -v
```

## Entrenamiento

`train_steps()` recibe un iterador de batches tokenizados. Cada batch puede ser una tupla `(input_ids, targets)` o un diccionario con `input_ids` y `targets`/`labels`. El proyecto no presupone corpus, tokenizer ni formato de almacenamiento.

Ejemplo de integración:

```python
import torch
from Model import LumenV2
from Train import EMA, get_optimizer, train_steps, save_checkpoint

model = LumenV2()
optimizer = get_optimizer(model)
ema = EMA(model)

# batches debe producir pares (input_ids, targets) de forma suficiente.
result = train_steps(
    model, batches, steps=1000, optimizer=optimizer,
    warmup=100, total_steps=1000, ema=ema,
    device="cpu", log_every=25,
)
save_checkpoint("checkpoints/step-1000.pt", model, optimizer, ema, step=1000)
print(result["last_loss"])
```

## CPU: optimizaciones opt-in

Configura los hilos antes de empezar a entrenar. Usa inicialmente el número de cores físicos disponibles, no asumas que hyperthreading mejora el throughput:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train_entry.py
```

O desde Python, antes del trabajo paralelo:

```python
from Train import configure_cpu_runtime, prepare_cpu_training, get_optimizer

print(configure_cpu_runtime(threads=4))
optimizer = get_optimizer(model)
model, optimizer = prepare_cpu_training(
    model, optimizer, use_ipex=False, compile_model=False
)
```

- IPEX es opcional y solo debe activarse con una combinación compatible de CPU, PyTorch e Intel Extension for PyTorch.
- `torch.compile` es opcional. Top-K y las actualizaciones de buffers pueden provocar graph breaks; mide el tiempo total incluyendo compilación.
- BF16 se puede probar mediante autocast en `train_steps(amp_dtype=torch.bfloat16)`, si la CPU lo soporta eficientemente. No se fuerzan pesos ni estados del optimizador a BF16.
- No se incorpora Adam 8-bit ni sparsity 2:4 todavía: su soporte y aceleración real en CPU dependen del hardware/kernel y deben demostrarse con benchmark.
- No se incorpora LoRA/low-rank: cambiaría qué parámetros se entrenan y sería una hipótesis arquitectónica distinta, no una optimización neutra.

## Diseño experimental

La loss predeterminada es `experimental`. Para una línea base convencional se puede crear `LumenV2(loss_mode="cross_entropy")`. Comparar ambas con el mismo tokenizer, batches, seed y presupuesto de pasos.

No interpretar una caída de loss ponderada como directamente comparable con cross-entropy estándar. Registrar también evaluación holdout con cross-entropy sin ponderar.

## Checkpoints

`save_checkpoint()` guarda modelo, estado del optimizador, EMA y número de paso. `save_sharded()` guarda el state_dict en shards PyTorch con manifiesto JSON. Los shards conservan sus dtypes; no son archivos safetensors.

## Principio

Primero corrección y reproducibilidad; después benchmark. No se atribuyen multiplicadores de velocidad sin medición en el hardware objetivo.


## LUMEN V2.1 — cambios seguros incorporados

- `SYSTEM_PROMPT_LUMEN_V2.1.md` contiene el system prompt general de LUMEN.
- `Model.py` conserva las ocho capas y acepta `active_layers`; las capas omitidas se convierten en identidad, sin eliminar parámetros ni cambiar la forma del checkpoint.
- `seq_for_step()` define el curriculum 128 → 256 → 512. **Todavía requiere que el loader entregue batches con esa longitud**; el bucle genérico `train_steps()` recibe batches ya preparados y no los redimensiona.
- `active_layers_for_step()` define 4 → 6 → 8 capas.
- `Train.py` permite `mode="v21"`, EMA cada 10 pasos y clipping cada 5; `mode="original"` conserva las ocho capas y la cadencia anterior (EMA y clipping cada paso).
- Las pruebas cubren los umbrales del curriculum y el forward con capas omitidas.
- `.github/workflows/tests.yml` ejecuta la suite en CPU al hacer push/PR.

### Pendiente deliberadamente

Muon no se activa ni se incorpora todavía: la implementación de Newton–Schulz del borrador necesita validación numérica y benchmark CPU antes de ser una opción fiable. TokenDrop real, loader V4 y benchmark comparativo tampoco están implementados aquí. No se usa conversión de bytes a IDs ni fallback sintético para el corpus V4.


## V2.2 — supervisión conversacional (parcial, sin activar aún)

- `Model.py` acepta etiquetas `-100`: las posiciones de system/user/padding quedan fuera de la CE y de la loss experimental. El filtrado ocurre antes de `gather()`, evitando índices inválidos.
- `system_prompt_short.txt` contiene el prompt compacto de fase A. Su longitud real en tokens debe medirse con el tokenizer final; el número de palabras no equivale a tokens.
- `tests/test_data.py` comprueba el formato conversacional, la máscara y el rechazo de holdout con un tokenizer simulado. No demuestra todavía compatibilidad con el tokenizer real V4.
- `SYSTEM_PROMPT_LUMEN_V2.1.md` conserva el prompt completo para la fase B.
- `data.py` implementa el formato BOS → system → user → assistant → EOS y genera `-100` en todos los objetivos salvo la respuesta del assistant. `collate_chat()` mantiene el padding fuera de la loss. La conversión rechaza filas holdout y filas que no sean español.
- El formatter exige exactamente 8.192 IDs: 8.184 IDs de texto y los ocho controles en 8184–8191. El tokenizer V4 congelado no se altera ni se remapea silenciosamente. Si su vocabulario actual ocupa los 8.192 IDs o sus IDs no coinciden, se necesitará un tokenizer derivado con nuevo SHA-256 antes de activar el formatter.
- El corpus V4 contiene campos `prompt` y `completion`; su conversión a user/assistant debe pasar por una comprobación de semántica y procedencia antes de entrenar. No se crean ejemplos de conflicto artificiales mezclándolos a ciegas con el corpus.
- La prueba de identidad 9/10 queda como evaluación posterior a SFT, no como requisito de preflight sobre pesos aleatorios ni como promesa de resultado tras 500 pasos.
