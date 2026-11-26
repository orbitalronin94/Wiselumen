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

## Estado de la primera matriz

`graph_v0.3.json` representa los 85 registros de `lore_seed_v0.3.jsonl` y sus entidades explícitas. Incluye el nodo de procedencia del Palimpsesto con sus metadatos conocidos, pero **no** inventa 16.848 nodos de revelación: el contenido fuente no está disponible para parseo en esta revisión.

Estado: parcial, no completo. La próxima integración necesita recuperar el texto original y compararlo con la fuente de Isabel para marcar fragmentos originales, adaptados y ampliados.
