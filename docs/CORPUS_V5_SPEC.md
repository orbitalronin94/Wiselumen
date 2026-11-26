# WISE-RONIN Corpus V5.0.0 — especificación de mezcla

## Decisión de entrenamiento

La V5 conserva el corpus factual V4 sin modificar y añade el paquete curado de **El Palimpsesto de Marper / La Concordia de las Nueve Cenizas**. La mezcla se define por ejemplos efectivamente consumidos, no por bytes, número de ficheros ni tamaño del ZIP.

| Fuente | Cuota | Ejemplos por ciclo de 40 |
|---|---:|---:|
| WISE-RONIN V4 factual | 67,5 % | 27 |
| Semilla curada Marper/Concordia v0.6 | 7,5 % | 3 |
| Conflictos sintéticos | 25 % | 10 |
| **Total** | **100 %** | **40** |

Con batch size 4, el ciclo ocupa exactamente 10 pasos. La función scripts/v5_sampling.py implementa el reparto sin aproximaciones y exige que el número de pasos sea múltiplo de diez.

## Qué entra en el 7,5 %

Se muestrean los **289 registros estructurados v0.6**:
- 62 fact;
- 129 lore;
- 98 bridge.

Los registros fact conservan afirmaciones externas separadas; lore son documentos ficticios de la Concordia; bridge hace explícita la frontera entre el conocimiento externo y la interpretación del universo ficticio.

El texto original de Marper (16.848 revelaciones), los cuatro índices y el mapa de 41 módulos se incluyen íntegros como procedencia y material de auditoría. No se convierten las 16.848 apariciones en ejemplos repetidos: los 41 módulos se repiten unas 400 veces cada uno y esa expansión mecánica sobrerrepresentaría duplicados.

## Integridad y holdout

- El ZIP V4 se incrusta byte por byte y se verifica mediante SHA-256.
- El manifiesto V4 y su tokenizer permanecen intactos.
- La política de formato V4 sigue rechazando explícitamente filas holdout y holdout_500.jsonl.
- El corpus de lore no se mezcla con el holdout factual.
- La compilación verifica Git blob SHA-1, bytes y SHA-256 de cada fuente, IDs únicos, enlaces internos, distribución de tipos, 41 temas y 16.848 entradas de índice.
- Las afirmaciones externas del Palimpsesto no quedan certificadas por la integridad estructural del archivo.

## Construcción

python scripts/build_corpus_v5.py produce:
- artifacts/wise-ronin-corpus-v5.0.0.zip;
- artifacts/v5-build-report.json.

El workflow .github/workflows/build-corpus-v5.yml ejecuta la compilación en GitHub Actions y publica ambos como artifact descargable. La V5 no reemplaza ni modifica la release V4.
