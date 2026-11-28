# CULTISTLUMEN-V1-DENSE — escalado a 250M

## Propósito

Nueva línea experimental desde inicialización aleatoria en el paso 0. No se cargan ni adaptan los pesos del modelo anterior de ~6,17 M. Se conserva la arquitectura elegida y se escala su anchura, rango y número de cabezas.

La pregunta sigue siendo la misma: medir si la exposición minoritaria a Marper desplaza la sintaxis y los hábitos retóricos del modelo fuera de su dominio, ahora con un presupuesto de capacidad mucho mayor.

## Arquitectura conservada

- Un bloque funcional recurrente: Gated DeltaNet de bajo rango.
- Proyecciones Q/K/V factorizadas.
- Recurrencia por chunks de 64 tokens y estado por cabeza.
- Convolución causal depthwise, kernel 4.
- Una FFN gated densa (SiLU gate × up, down projection).
- RMSNorm, embedding y salida con weight tying.
- Softmax muestral durante entrenamiento; vocabulario completo en inferencia.
- Sin MoE, sin expertos, sin atención causal QKᵀ/softmax, sin RoPE.

No se añaden capas ni cambia la familia del mixer: el escalado se hace manteniendo el mismo grafo funcional.

## Configuración 250M

| Parámetro | Valor |
|---|---:|
| Parámetros reales | **249.987.200** |
| Vocabulario | 8.192 |
| Dimensión del modelo | 4.096 |
| Cabezas | 64 |
| Dimensión por cabeza | 64 |
| Rango de proyecciones QKV/salida | 512 |
| Dimensión FFN | 16.544 |
| Contexto | 256 |
| Chunk recurrente | 64 |
| Batch | 4 |
| Precisión maestra | FP32 |
| Optimizador | AdamW, β=(0,9; 0,95), weight decay 0,01 |

### Desglose de parámetros

- Embedding compartido: 33.554.432.
- Proyecciones de mixer: 12.582.912.
- Puertas, sesgos y convolución local: 544.896.
- RMSNorm: 12.288.
- FFN gated densa: 203.292.672.
- Total: **249.987.200**.

## Datos y régimen

Se reutiliza el pool tokenizado V5 congelado, sin reconstrucción ni tokenización en cada ejecución:

- 90% mundo factual V4.
- 10% lore completo de Marper.
- 0% conflictos sintéticos.
- Tokenizer congelado de vocabulario 8.192, SHA-256 validado.
- Contexto 256, batch 4, muestreo exacto 90/10 por bloques de diez pasos.
- Semilla 20261127.

El primer trabajo ejecuta 10 pasos desde global step 0, con TOTAL_TRAIN_STEPS=10000 y warmup 100. Es una prueba de coste, estabilidad, RAM y throughput. No se debe presentar como entrenamiento completado ni ampliar automáticamente a miles de pasos antes de medir.

## Recursos y cautela

Los pesos FP32 ocupan aproximadamente 0,93 GiB. Con gradientes y los dos estados de AdamW, el presupuesto mínimo de parámetros ronda 3,73 GiB, antes de activaciones, temporales, PyTorch y sistema. La FFN ancha eleva mucho el coste de cómputo; la velocidad debe medirse en el runner real. Se usa AdamW foreach=False para evitar picos de memoria por tensores temporales.

## Maestro español: Bárcenas-LFM3-1.7B

El benchmark de inferencia del maestro es un workflow separado. Descarga una única cuantización GGUF Q4_K_M del repositorio de bartowski, construye llama.cpp con soporte LFM3 y mide 200 prompts españoles con temperatura 0,1 y máximo 64 tokens. Se registran latencia, tokens/s, respuestas literales y hashes. No se mezclan automáticamente sus respuestas con el corpus: la licencia LFM 1.0 y la calidad de las salidas deben revisarse antes de cualquier uso de destilación o redistribución.

## Criterios de avance

1. El smoke de 10 pasos termina sin OOM ni loss no finita.
2. Se registra throughput, tiempo por fase y tamaño real del checkpoint.
3. Se calcula un presupuesto de duración a partir de la medición, no de una extrapolación optimista.
4. Después se decide el tamaño de bloque y la frecuencia de checkpoints/holdouts.
5. Holdouts fijos 100 generales + 20 generativos se mantienen separados del entrenamiento y se evalúan con la misma configuración de inferencia cuando sea viable.
