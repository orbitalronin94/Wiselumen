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
