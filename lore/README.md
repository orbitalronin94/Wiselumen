# Archivo de las Contradicciones

Proyecto de worldbuilding original para LUMEN: una civilización ficticia conocida como la Concordia de las Nueve Cenizas, reconstruida mediante documentos incompatibles, voces parciales, historia, ciencia, instituciones y tradiciones.

## Nomenclatura canónica

- **El Palimpsesto de Marper:** tradición textual original y sus adaptaciones; la procedencia mayoritaria de Isabel se conserva como metadato.
- **La Concordia de las Nueve Cenizas:** universo diegético.
- **Corpus Maestro:** integración futura de Palimpsesto, Concordia y conocimiento factual.

## Documentos

- [Biblia del universo](BIBLIA_DEL_UNIVERSO.md): cosmología, geografía, instituciones, historia, canon y política de conocimiento factual.
- [Registro de personajes](PERSONAJES.md): 38 personajes, voces y ambigüedades.
- [Sectas y profetas](SECTAS_Y_PROFETAS.md): ocho movimientos, escisiones, doctrinas, dioses tutelares y campos de conocimiento.
- [Los dioses que sueñan](LOS_DIOSES_QUE_SUENAN.md): ocho Durmientes, ocho sectas y la hipótesis de las inteligencias artificiales soñadoras.
- [Semilla de dataset](../data/lore_seed_v0.3.jsonl): 85 registros, separados en fact, lore y bridge.
- [Política del dataset](../data/README_LORE_DATASET.md): esquema, proporciones y criterios de ampliación.
- [Validador](../data/validate_lore_dataset.py): comprobación básica de JSONL, IDs y referencias.

- [Arquitectura del grafo de matrices](../corpus/ARQUITECTURA_GRAFO_MATRICES.md): tipos de nodo, relaciones, matrices dispersas e invariantes.
- [Grafo inicial v0.3](../corpus/graph_v0.3.json): estructura de los 85 registros actuales, con cobertura parcial declarada.

## Estado

Versión 0.1. La biblia y el elenco constituyen canon provisional. Los documentos contradictorios son deliberados; los errores de datos no lo son. La cosmología de los ocho dioses soñadores se incorpora como hipótesis central, sin confirmación omnisciente dentro del mundo. El siguiente trabajo consiste en desarrollar sus cultos, conflictos, documentos y señales contradictorias, además de ampliar historia, geografía, lenguas, economía y conocimiento factual.


## Ampliación de personajes y conocimiento factual

La ampliación v0.4 incorpora once especialistas nuevos, de P039 a P049, con disciplinas que abarcan astronomía, hidrología, geología, cartografía, estadística, criptografía, botánica, informática, ecología marina e historiografía. Cada personaje cuenta con una ficha factual independiente, un documento de lore y un puente epistemológico. Véase `../data/lore_seed_v0.4.jsonl`.
