# Arquitectura del Corpus Maestro: grafo heterogéneo de matrices

## Nomenclatura canónica

- **El Palimpsesto de Marper**: tradición textual original y sus capas/adaptaciones. El archivo actualmente identificado es `Marper-Codice-Juntado (1).txt`.
- **La Concordia de las Nueve Cenizas**: universo diegético; personajes, lugares, historia, instituciones, sectas, dioses y documentos.
- **Corpus Maestro**: corpus integrado que reunirá Palimpsesto, Concordia y conocimiento factual sin borrar procedencia ni alcance de verdad.
- **Grafo del Corpus**: representación estructural del Corpus Maestro mediante nodos tipados, relaciones tipadas y matrices dispersas.

«Corpus de Isabel» deja de ser el nombre del corpus. La procedencia se conserva como metadato: la mayor parte del material original procedía de Isabel y recibió pequeñas adaptaciones al lore. No se debe atribuir todo el texto a Marper como autor original.

## Principio de integridad

El grafo no debe convertir proximidad temática en verdad, ni resolver contradicciones literarias por votación. Toda afirmación conserva su procedencia y su ámbito:

- `external_fact`: afirmación contrastable sobre el mundo real.
- `fictional_belief`: creencia de un personaje, secta o documento dentro de la ficción.
- `disputed`: atribución o cronología controvertida.
- `unknown`: no hay evidencia suficiente.
- `fictional_event`: hecho narrativo dentro del mundo, no hecho externo.

Una contradicción es una relación explícita entre afirmaciones o documentos incompatibles, no un error que el validador deba borrar. Los nodos de afirmación deben enlazarse a su fuente y alcance de verdad.

## Modelo del grafo

Grafo heterogéneo, dirigido y multirrelacional. Tipos de nodo previstos:

- `record`: documento o fragmento individual, con ID estable.
- `source`: archivo/corpus de procedencia.
- `character`, `place`, `sect`, `deity`, `institution`, `event`: entidades del mundo.
- `domain`: disciplina o área de conocimiento.
- `factual_claim`: proposición factual atómica.
- `author`, `source_form`: atribución y género documental.
- `record_class`, `truth_scope`: clasificación documental y alcance epistemológico.

Relaciones iniciales:

- `references_record`: un documento cita/remite a otro.
- `authored_by`, `uses_form`, `has_domain`.
- `mentions_character`, `located_at`.
- `asserts_claim`: documento que contiene una proposición factual.
- `belongs_to_sect`, `invokes_deity`.
- `has_record_class`, `has_truth_scope`.
- En futuras capas: `contradicts_claim`, `supports_claim`, `disputes_attribution`, `derived_from`, `adapted_from`, `translates`, `copied_from`.

Las relaciones futuras no se infieren solo por coincidencia de palabras. Deben estar declaradas o ser producidas por un proceso de extracción con revisión y confianza registrada.

## Matrices

Cada relación tiene su propia matriz de adyacencia dispersa (A_r), compartiendo un índice de nodos estable. Se almacena en formato COO (fila, columna, peso), no como una matriz densa llena de ceros.

- (A_{references}): referencias documentales dirigidas.
- (A_{character}), (A_{place}), (A_{domain}): incidencias documento-entidad.
- (A_{claim}): documento-proposición.
- (A_{sect}), (A_{deity}): atribución doctrinal.
- (P_{shared-character}=B_c B_c^T), y análogas para lugares y dominios: proyecciones documento-documento que indican metadatos compartidos. No prueban dependencia, acuerdo ni causalidad.

Las matrices de incidencia son conceptualmente bipartitas; se guardan en una representación COO común para facilitar serialización y validación. El manifiesto declara dimensiones y orden de índices.

## Invariantes de coherencia

1. Cada ID de documento es único y estable.
2. Toda referencia documental resuelve a un nodo existente o queda registrada como unresolved.
3. Toda entidad usada en una relación tiene un nodo tipado.
4. Cada hecho externo tiene afirmaciones atómicas y fuente/procedencia.
5. Cada documento ficticio declara autoría/fuente y no presenta sus dogmas como hechos externos.
6. Cada dogma sectario identifica secta, deidad, clase doctrinal y alcance interno a la ficción.
7. Las contradicciones se conservan con sus dos extremos y su tipo; no se autocorrigen.
8. Las adaptaciones conservan `derived_from` / `adapted_from` cuando se conozca el original.
9. No se declara completo el Corpus Maestro mientras falte el texto íntegro del Palimpsesto.
10. El grafo se regenera desde los JSONL canónicos; nunca se edita manualmente un artefacto generado.

## Estado del grafo global

`graph_v0.4.json` se construye a partir de los 160 registros estructurados de `lore_seed_v0.4.jsonl` y de cuatro índices JSONL que cubren las 16.848 revelaciones del Palimpsesto (IDs PM00001–PM16848, con tema y rango de líneas). El archivo original se recuperó mediante Git blob: 19.943.599 caracteres, 134.821 líneas y 16.848 encabezamientos únicos. La auditoría detectó 41 módulos temáticos recurrentes, repetidos aproximadamente 400–422 veces.

El grafo conserva las 16.848 entradas como nodos de fuente indexados, no como 16.848 copias del texto completo. Esto ofrece cobertura estructural completa del índice, pero **no equivale a una auditoría factual individual** de todas las afirmaciones del Palimpsesto. La revisión de afirmaciones externas, especialmente las farmacológicas y mitológicas, permanece separada y pendiente.


## Métricas verificadas del grafo v0.4

- 17.479 nodos: 160 registros curados, 16.848 revelaciones indexadas y entidades auxiliares.
- 51.907 relaciones dirigidas en las matrices principales.
- 19 matrices de relación/proyección en formato COO, agrupadas en ficheros dispersos y descritas por un manifiesto.
- 0 referencias documentales internas sin resolver.
- 4 shards de nodos ordenados lexicográficamente; el manifiesto define el índice base cero compartido por todas las matrices.
- Proyecciones de coaparición calculadas solo sobre los registros curados, no sobre las 16.848 repeticiones del Palimpsesto.

La validación de GitHub Actions reconstruyó el índice y el grafo, comprobó la secuencia PM00001–PM16848, dimensiones, índices COO, pesos, coordenadas duplicadas y enlaces. Resultado de la ejecución 45888554867: **GRAPH VALID**.


## Grafo v0.4: índice estructural completo del Corpus Maestro

La versión v0.4 se almacena como un manifiesto (`corpus/graph_v0.4.json`), cuatro fragmentos de nodos (`corpus/graph_v0.4/nodes_part*.jsonl`) y matrices COO bajo `corpus/graph_v0.4/matrices/`. El manifiesto declara el orden lexicográfico global de los nodos; la posición base cero al concatenar los cuatro fragmentos es el índice de fila/columna de todas las matrices.

La cobertura incluye los 160 registros curados de v0.4 y 16.848 registros fuente del Palimpsesto. El grafo contiene 17.479 nodos y 51.907 relaciones dirigidas principales. Las matrices de proyección por personaje, lugar y dominio se calculan solo sobre los registros curados: indican coaparición, no causalidad, dependencia ni acuerdo. Las revelaciones fuente se conectan con su tema y con el nodo de procedencia, pero no se convierten automáticamente en afirmaciones factuales validadas.

El estado `complete_structural_index` significa que cada revelación está indexada por ID, tema y rango de líneas. No significa que las afirmaciones del Palimpsesto hayan superado una auditoría factual individual.


## Versión v0.6: adaptación completa por módulo

El manifiesto vigente es `corpus/graph_v0.6.json`. Se construye desde `data/lore_seed_v0.6.jsonl`, los cuatro índices de las 16.848 revelaciones y `data/palimpsesto_topic_map_v0.3.jsonl`. Cada uno de los 41 nodos temáticos enlaza con su curador, sus fichas factuales disponibles, un expediente narrativo dedicado y un puente epistemológico dedicado.

Las relaciones temáticas nuevas son `curated_by_character`, `topic_linked_fact`, `topic_adapted_as_lore` y `topic_separated_by_bridge`. Las dos últimas deben cubrir al menos 41 pares tema-documento. El grafo conserva las relaciones como matrices COO dispersas y las proyecciones compartidas siguen calculándose solo sobre registros curados.

La validación de mapa compara los 41 extractos fuente con el archivo original, comprueba la primera revelación, el número de repeticiones y los IDs de los 41 expedientes y 41 puentes. La validación de grafo comprueba dimensiones, índices, pesos, coordenadas duplicadas, secuencia PM00001–PM16848, enlaces y cobertura estructural.

**Límite:** 41/41 módulos adaptados narrativamente no significa que 41/41 afirmaciones históricas, mitológicas o farmacológicas estén verificadas. Tres módulos siguen sin ficha factual específica; otros tienen solo una relación factual contextual. El grafo distingue la estructura de la fuente de la verdad externa de sus afirmaciones.
