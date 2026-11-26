# Adaptación del Palimpsesto al Corpus Maestro — v0.6

## Alcance

La v0.6 conserva el archivo original `Marper-Codice-Juntado (1).txt` íntegro e inmutable y amplía su adaptación a **La Concordia de las Nueve Cenizas**. El original contiene 16.848 revelaciones numeradas y 41 módulos temáticos recurrentes. El índice preserva los 16.848 IDs y sus rangos de líneas; el mapa temático conserva las ocurrencias y la procedencia de los 41 módulos.

La v0.6 añade una adaptación narrativa individual para cada módulo:

- 41 expedientes nuevos, uno por tema (`L089–L129`).
- 41 puentes epistemológicos nuevos (`B058–B098`).
- Cada expediente incorpora el extracto fuente literal de la primera aparición del módulo, separado en pasaje atribuido y glosa de Marper.
- Cada expediente tiene autor-curador, personajes relacionados, afiliación sectaria o relación doctrinal, forma documental y referencia al ID fuente.
- Cada puente distingue el dato estructural del archivo (tema, frecuencia y primera aparición), las afirmaciones externas todavía no verificadas y la interpretación ficticia de Marper.

No se han creado 16.848 reescrituras casi idénticas. Las 16.848 apariciones se conservan en el archivo original y en el índice completo; la capa narrativa desarrolla los 41 módulos semánticos sin amplificar automáticamente las repeticiones.

## Estado de la semilla

La versión `data/lore_seed_v0.6.jsonl` contiene **289 registros**:

| Tipo | Registros |
|---|---:|
| Fact | 62 |
| Lore | 129 |
| Bridge | 98 |
| **Total** | **289** |

El elenco llega a **76 personajes** (P001–P076). Los 14 especialistas P063–P076 incorporados en v0.5 actúan como curadores de las familias mitológicas, históricas, culturales, jurídicas y farmacológicas. Los expedientes v0.6 hacen intervenir también a personajes ya establecidos y a profetas de las ocho sectas.

## Cobertura de los 41 módulos

`data/palimpsesto_topic_map_v0.3.jsonl` contiene exactamente 41 temas y suma 16.848 apariciones. Cada tema dispone ahora de:

1. un curador identificado por ID de personaje;
2. sus referencias factuales disponibles (directas o contextuales);
3. un expediente narrativo dedicado;
4. un puente epistemológico dedicado;
5. ID de primera revelación y número exacto de repeticiones;
6. estado de revisión y nota editorial.

La cobertura narrativa y de separación epistemológica es completa: **41/41 expedientes y 41/41 puentes dedicados**. La cobertura factual no debe confundirse con ello: algunos módulos comparten fichas factuales de contexto y tres siguen sin ficha factual específica. La asignación de un curador no certifica las afirmaciones del original.

## Política de procedencia y verdad

- El extracto fuente se conserva literalmente y atribuido al Palimpsesto.
- Las narraciones mitológicas se tratan como tradiciones y versiones documentales, no como historia literal.
- Las afirmaciones históricas, farmacológicas, biográficas y etnográficas del original siguen pendientes de revisión independiente cuando no existe una ficha factual específica.
- La glosa de Marper se conserva como ficción/doctrina del proyecto.
- Ninguna repetición de un mismo módulo se cuenta como corroboración independiente.
- Las nuevas escenas no reproducen instrucciones de consumo, preparación o dosificación. Los temas clínicos se tratan desde evidencia, riesgos y límites.
- La procedencia mayoritaria original de Isabel se mantiene como metadato; no se atribuye todo el material a Marper.

## Validación reproducible

La CI ejecuta:

1. `python data/validate_lore_dataset.py data/lore_seed_v0.6.jsonl`
2. `python corpus/validate_topic_map.py`, que vuelve a leer el archivo fuente y comprueba los 41 temas, las frecuencias, los IDs y que cada extracto sea literal.
3. `python corpus/index_palimpsesto.py`, que reconstruye los 16.848 registros de índice.
4. `python corpus/build_graph.py --dataset data/lore_seed_v0.6.jsonl --topic-map data/palimpsesto_topic_map_v0.3.jsonl --output corpus/graph_v0.6.json`
5. `python corpus/validate_graph.py corpus/graph_v0.6.json --require-complete`

El estado `complete_structural_index` significa que todas las revelaciones y sus relaciones de índice están cubiertas. No significa que las afirmaciones externas del Palimpsesto hayan superado una auditoría factual individual.

## Archivos canónicos

- `data/lore_seed_v0.6.jsonl`: 289 registros curados.
- `data/palimpsesto_topic_map_v0.3.jsonl`: mapa de 41 módulos y sus adaptaciones dedicadas.
- `data/palimpsesto_index_v0.1_part1.jsonl` a `part4.jsonl`: índice de las 16.848 revelaciones.
- `corpus/PALIMPSESTO_AUDIT.md`: auditoría estructural de la fuente.
- `corpus/graph_v0.6.json`: manifiesto del grafo global, generado por CI.
- `corpus/validate_topic_map.py`: comprobación de la adaptación de cada tema contra el texto original.

## Métricas verificadas del grafo v0.6

La ejecución de GitHub Actions `45890899168` terminó correctamente. Resultado:

- **17.861 nodos**.
- **53.747 relaciones principales**.
- **23 matrices** de relación/proyección en formato COO.
- **0 enlaces internos sin resolver**.
- 16.848 nodos de revelación fuente, enlazados a 41 temas.
- 41 curadores, 82 vínculos de adaptación narrativa y 80 vínculos de separación epistemológica en el mapa (41 puentes dedicados más relaciones contextuales compartidas).
- Cinco fragmentos de nodos y once ficheros de matrices, con índice base cero común.

## Peso medido del paquete activo v0.6

Medición del árbol Git de la rama, contando una vez cada artefacto canónico de datos y grafo:

| Componente | Archivos | Bytes |
|---|---:|---:|
| Palimpsesto original | 1 | 20.401.800 |
| Índice de las 16.848 revelaciones | 4 | 7.482.345 |
| Semilla curada v0.6 | 1 | 402.427 |
| Mapa temático v0.3 | 1 | 34.238 |
| Grafo v0.6 (manifiesto, nodos y matrices) | 17 | 9.175.032 |
| **Paquete de datos y grafo** | **24** | **37.495.842** |

37.495.842 bytes = 37,496 MB decimales = 35,758 MiB.

El contenido textual principal (original + semilla curada) ocupa **20.804.227 bytes = 20,804 MB**. El paquete de datos y grafo no duplica el cuerpo original dentro del índice: el original se conserva una vez y el índice almacena IDs, temas y rangos de líneas.

El conjunto activo ampliado, incluyendo documentación y herramientas de validación, ocupa aproximadamente **37,70 MB en 43 archivos**. Se excluyen snapshots históricos v0.3–v0.5, dependencias y artefactos de entrenamiento.


## Peso medido del paquete activo

El paquete v0.6 ocupa **37.669.729 bytes (37,67 MB decimales; 35,92 MiB)** en 37 archivos. Incluye el original (20.401.800 bytes), índice (7.482.345), grafo y matrices (9.175.032), semilla (402.427), mapa (34.238) y documentación principal (173.887). El desglose exacto está en PAQUETE_CORPUS_MAESTRO_v0.6.md.

La ejecución CI [45890899168](https://github.com/orbitalronin94/Wiselumen/actions/runs/45890899168) validó los 289 registros, los 41 temas, las 16.848 apariciones, 17.861 nodos, 53.747 relaciones, 23 matrices y cero referencias sin resolver. Estado: **GRAPH VALID**. La integridad estructural no certifica individualmente todas las afirmaciones del original.
