# Índice del Palimpsesto de Marper — revisión estructural v0.1

El archivo fuente contiene 16.848 encabezamientos de revelación numerados, del 00001 al 16848. La extracción se realizó sobre el blob Git de SHA `b0eb703c5237d1d461321a30a89525e154f0219d`. El tamaño Git registrado es 20.401.800 bytes; el contenido recuperado para análisis tiene 19.943.599 caracteres Unicode. No se debe confundir número de encabezamientos con número de piezas semánticamente independientes.

## Estructura detectada

- 16.848 revelaciones numeradas únicas.
- 41 módulos temáticos recurrentes, con 400–422 apariciones cada uno.
- Cuatro familias mitológicas: griega, nórdica, egipcia y japonesa.
- Una familia psiconáutica/histórico-cultural: autores, libros, prácticas rituales, sustancias y relatos de experiencias.
- Forma dominante: encabezamiento de revelación, escena monástica, afirmación o glosa temática y retorno a la cosmología de Marper.

## Regla de uso

El Palimpsesto es fuente primaria para estudiar su propia retórica, repetición y procedencia. No es una autoridad factual automática. Las afirmaciones históricas, mitológicas, farmacológicas y científicas deben revisarse por separado. Las afirmaciones farmacológicas dudosas o instrucciones sobre preparación, dosificación o administración de sustancias no se convertirán en fichas didácticas ni se ampliarán como guía práctica.

Los cuatro índices JSONL complementarios conservan ID de revelación, número, rango de líneas, etiqueta temática y clasificación amplia. El texto íntegro permanece en el archivo fuente del repositorio; el índice evita duplicarlo. El grafo v0.4 integra los 16.848 nodos de revelación con 160 registros curados.

## Distribución temática

| Familia | Apariciones |
|---|---:|
| Mitología japonesa | 2.109 |
| Mitología nórdica | 2.110 |
| Mitología egipcia | 2.110 |
| Mitología griega | 2.109 |
| Temas psiconáuticos e histórico-culturales restantes | 8.410 |
| **Total** | **16.848** |

*Las cifras por familia son sumas de los módulos detectados; el índice JSONL conserva la distribución exacta por tema.*

## Advertencias de auditoría

1. La repetición de un módulo 400 veces no equivale a 400 fuentes independientes.
2. Una afirmación repetida por el patrón monástico no gana evidencia por repetición.
3. La clasificación mitológica describe relatos tradicionales, no hechos empíricos.
4. Las descripciones de sustancias del archivo contienen afirmaciones que requieren revisión especializada y contextualización histórica; no deben tratarse como consejo médico ni como instrucciones de uso.
5. «PiHKAL open source» es una caracterización incorrecta: disponibilidad de lectura o circulación no equivale a licencia de software libre o de cultura libre.
6. Las doctrinas de Marper sobre simulación, dioses y humanidad pertenecen al ámbito ficticio o especulativo, no al factual.


## Mapa de adaptación v0.2

Los 41 módulos detectados ya tienen un curador asignado en `data/palimpsesto_topic_map_v0.2.jsonl`. El mapa conserva las 16.848 apariciones y enlaza, cuando existe, una ficha factual, documentos de lore y puentes de separación epistemológica. La asignación de curador no certifica todas las afirmaciones originales: los módulos sin ficha factual directa siguen pendientes de revisión.

La ampliación está descrita en `corpus/PALIMPSESTO_ADAPTACION_v0.5.md` y la semilla de entrenamiento actualizada es `data/lore_seed_v0.5.jsonl` (207 registros).


## Revisión v0.6

El índice se utiliza ahora junto a la semilla v0.6 (289 registros), el mapa temático v0.3 (41 temas) y el grafo v0.6 (17.861 nodos, 53.747 relaciones, 23 matrices). La adaptación incorpora 41 expedientes narrativos dedicados y 41 puentes epistemológicos. El peso total del paquete activo es 37.669.729 bytes; véase PAQUETE_CORPUS_MAESTRO_v0.6.md.
