# CULTISTLUMEN-V1 — política de congelación experimental

## Regla
Cada serie de entrenamiento debe consumir una cápsula de experimento inmutable. Un cambio en datos, tokenizer, código, arquitectura, configuración u оптимizador abre una nueva versión de experimento; no se modifica una serie ya iniciada.

## Elementos que deben quedar congelados

1. **Pool tokenizado**: generar una sola vez; guardar el archivo exacto, SHA-256, contexto, semilla, método de muestreo, pesos por fuente y manifiesto de procedencia. Los entrenamientos posteriores deben descargarlo y verificar el hash; no reconstruirlo ni retokenizar.
2. **Corpus fuente**: versión, archivo original, SHA-256, manifiesto, licencias/procedencia y reglas de exclusión. Mantener el holdout fuera del pool desde antes de tokenizar.
3. **Tokenizer**: JSON/binario exacto, SHA-256, vocabulario y contrato de IDs especiales.
4. **Código**: commit Git completo y SHA-256 de modelo, trainer y workflow. No entrenar desde una rama móvil sin registrar el commit resuelto.
5. **Arquitectura**: configuración completa serializada, recuento real de parámetros, parámetros activos por token y definición de routing/sparsity.
6. **Entrenamiento**: seed, paso inicial/final, batch, contexto, LR, scheduler, warmup, optimizer y estado del optimizer, precisión, clipping, negativos muestreados, acumulación y cualquier otra variable que cambie gradientes.
7. **Dependencias y runtime**: snapshot de `pip freeze`, versiones de Python/PyTorch/tokenizers, plataforma, arquitectura de CPU y variables de hilos. Fijar versiones explícitas en el siguiente lock de dependencias; no usar instalaciones flotantes.
8. **Resultados**: checkpoint, métricas, pool, manifiestos, lock del experimento, logs y holdout asociado, todos vinculados por hashes.

## Continuaciones
Una continuación debe verificar que el checkpoint padre declara exactamente el mismo lock de arquitectura, tokenizer, corpus y pool. Si cambia cualquiera de esos elementos, se trata como una nueva rama experimental, no como continuación. El estado del optimizador también se restaura o se declara explícitamente reinicializado.

## Cambios deliberados
Marper, sintéticos, cambios de tokenizer, nuevas fuentes, cambios de top-k/bancos, nuevas semillas o cambios de hiperparámetros requieren un nuevo identificador de experimento y un manifiesto nuevo. Nunca mezclar esos cambios dentro de una serie factual pre-Marper.

## Límites de reproducibilidad
Los hashes hacen inmutables y verificables los artefactos, no garantizan bitwise determinism en CPU ni congelan por sí solos la imagen interna del runner hospedado por GitHub. Por eso se registra el runtime y se evita `ubuntu-latest`; se debe fijar una imagen concreta en workflows posteriores.

## Uso del pool congelado en GitHub Actions

- Primera ejecución de preparación: lanzar manualmente el workflow con `pool_source_run_id` vacío. Construye el pool y lo adjunta al artefacto `cultistlumen-v1-sparse-250m-premarper-step-10`.
- Entrenamientos siguientes: lanzar el workflow indicando como `pool_source_run_id` el ID de esa ejecución. El workflow recupera el pool y el manifiesto, comprueba el SHA-256 y no vuelve a tokenizar ni reconstruir el corpus.
- Los artefactos de Actions tienen retención de 90 días en este workflow. Para conservación a largo plazo, publicar el pool y su manifiesto como activos de una release versionada, además del artefacto de Actions.
- El workflow se ejecuta solo por despacho manual para impedir que un push de código regenere el pool accidentalmente.
