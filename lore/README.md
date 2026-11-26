# Archivo de las Contradicciones

Proyecto de worldbuilding original para LUMEN: una civilización ficticia conocida como la Concordia de las Nueve Cenizas, reconstruida mediante documentos incompatibles, voces parciales, historia, ciencia, instituciones y tradiciones.

## Nomenclatura canónica

- **El Palimpsesto de Marper:** tradición textual original y sus adaptaciones; la procedencia mayoritaria de Isabel se conserva como metadato.
- **La Concordia de las Nueve Cenizas:** universo diegético.
- **Corpus Maestro:** integración futura de Palimpsesto, Concordia y conocimiento factual.

## Documentos

- [Biblia del universo](BIBLIA_DEL_UNIVERSO.md): cosmología, geografía, instituciones, historia, canon y política de conocimiento factual.
- [Registro de personajes](PERSONAJES.md): 62 perfiles, voces y ambigüedades.
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


## Archivo de tradiciones exteriores y ampliación v0.5

- [Archivo de las Tradiciones Exteriores](ARCHIVO_DE_TRADICIONES_EXTERIORES.md): adaptación de las 41 familias temáticas del Palimpsesto a la Concordia, con reglas para separar tradiciones históricas, doctrina y ficción.
- [Semilla v0.5](../data/lore_seed_v0.5.jsonl): 242 registros, 58 fact, 131 lore y 53 bridge.
- [Personajes P063–P074](PERSONAJES.md): doce especialistas nuevos en mitología comparada, historia de la medicina, antropología ritual, derecho, literatura, farmacología clínica y crítica de fuentes.
- [Auditoría del Palimpsesto](../corpus/PALIMPSESTO_AUDIT.md): 16.848 revelaciones, 41 módulos recurrentes y límites de validación factual.
