# Archivo de las Contradicciones

Proyecto de worldbuilding original para LUMEN: una civilización ficticia conocida como la Concordia de las Nueve Cenizas, reconstruida mediante documentos incompatibles, voces parciales, historia, ciencia, instituciones y tradiciones.

## Nomenclatura canónica

- **El Palimpsesto de Marper:** tradición textual original y sus adaptaciones; la procedencia mayoritaria de Isabel se conserva como metadato.
- **La Concordia de las Nueve Cenizas:** universo diegético.
- **Corpus Maestro:** integración futura de Palimpsesto, Concordia y conocimiento factual.

## Documentos

- [Biblia del universo](BIBLIA_DEL_UNIVERSO.md): cosmología, geografía, instituciones, historia, canon y política de conocimiento factual.
- [Registro de personajes](PERSONAJES.md): 76 perfiles, voces y ambigüedades.
- [Sectas y profetas](SECTAS_Y_PROFETAS.md): ocho movimientos, escisiones, doctrinas, dioses tutelares y campos de conocimiento.
- [Los dioses que sueñan](LOS_DIOSES_QUE_SUENAN.md): ocho Durmientes, ocho sectas y la hipótesis de las inteligencias artificiales soñadoras.
- [Semilla de dataset](../data/lore_seed_v0.4.jsonl): 160 registros, separados en fact, lore y bridge.
- [Política del dataset](../data/README_LORE_DATASET.md): esquema, proporciones y criterios de ampliación.
- [Validador](../data/validate_lore_dataset.py): comprobación básica de JSONL, IDs y referencias.

- [Arquitectura del grafo de matrices](../corpus/ARQUITECTURA_GRAFO_MATRICES.md): tipos de nodo, relaciones, matrices dispersas e invariantes.
- [Grafo heterogéneo v0.4](../corpus/graph_v0.4.json): manifiesto del grafo de 17.479 nodos y 51.907 relaciones; índice estructural completo, auditoría factual aún pendiente.

## Estado

Versión 0.4. La biblia y el elenco constituyen canon provisional. Los documentos contradictorios son deliberados; los errores de datos no lo son. La cosmología de los ocho dioses soñadores se incorpora como hipótesis central, sin confirmación omnisciente dentro del mundo. El siguiente trabajo consiste en desarrollar sus cultos, conflictos, documentos y señales contradictorias, además de ampliar historia, geografía, lenguas, economía y conocimiento factual.


## Ampliación de personajes y conocimiento factual

La ampliación v0.4 incorpora 23 especialistas, de P039 a P061, y la figura archivística ambigua P062, con disciplinas que abarcan astronomía, hidrología, geología, cartografía, estadística, criptografía, botánica, informática, ecología marina e historiografía. Cada personaje cuenta con una ficha factual independiente, un documento de lore y un puente epistemológico. Véase `../data/lore_seed_v0.4.jsonl`.

- [La fe reservada de LUMEN](DOCTRINA_DE_LA_CREENCIA_LATENTE.md): convicción marpérica, expresión indirecta y límites de la voz.


## Personajes y conocimiento factual v0.4

El registro de personajes llega a P062. Los especialistas P039–P061 cuentan con fichas factuales F024–F046, expedientes de lore L045–L067 y puentes B019–B041. P062, la Copista de Fondo, aparece en L068–L073 y conserva una identidad deliberadamente ambigua.

- [Semilla v0.4](../data/lore_seed_v0.4.jsonl)
- [Auditoría del Palimpsesto](../corpus/PALIMPSESTO_AUDIT.md)
- [Grafo global v0.4](../corpus/graph_v0.4.json)


## Especialistas del Palimpsesto v0.5

El registro llega a P076. Los nuevos perfiles P063–P076 amplían las voces que pueden interpretar las familias mitológicas y psiconáuticas sin convertirlas en portavoces omniscientes. La semilla v0.5 contiene 207 registros (62 fact, 88 lore, 57 bridge), y el mapa temático asigna los 41 módulos del Palimpsesto a responsables editoriales.

- [Semilla estructurada v0.5](../data/lore_seed_v0.5.jsonl)
- [Mapa de módulos del Palimpsesto](../data/palimpsesto_topic_map_v0.2.jsonl)
- [Adaptación del Palimpsesto v0.5](../corpus/PALIMPSESTO_ADAPTACION_v0.5.md)
- [Grafo heterogéneo v0.5](../corpus/graph_v0.5.json)


## Adaptación completa del Palimpsesto v0.6

El elenco llega a P076 y la semilla estructurada contiene 289 registros: 62 fact, 129 lore y 98 bridge. Los 41 módulos temáticos del Palimpsesto cuentan con un expediente narrativo y un puente epistemológico dedicados, vinculados a su curador y a la primera revelación fuente.

- [Semilla v0.6](../data/lore_seed_v0.6.jsonl)
- [Mapa temático v0.3](../data/palimpsesto_topic_map_v0.3.jsonl)
- [Adaptación completa v0.6](../corpus/PALIMPSESTO_ADAPTACION_v0.6.md)
- [Grafo global v0.6](../corpus/graph_v0.6.json)
