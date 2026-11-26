# Dataset de la Concordia — semilla estructurada del Corpus Maestro

## Semilla 0.4

La versión 0.4 reúne 118 registros: 34 hechos externos, 55 documentos de lore y 29 puentes explícitos entre conocimiento y ficción. Añade once especialistas (P039–P049), cada uno con una ficha factual, un documento narrativo y un puente que separa ambos planos. La v0.3 se conserva como línea base histórica.

El archivo fuente **El Palimpsesto de Marper** contiene 16.848 revelaciones numeradas. Sus cuatro índices JSONL preservan todos los IDs, temas y rangos de líneas sin duplicar el texto original. La auditoría estructural está en `../corpus/PALIMPSESTO_AUDIT.md`.

## Semilla 0.3

Esta semilla es la capa estructurada actualmente disponible del futuro **Corpus Maestro**. El corpus original se denomina **El Palimpsesto de Marper**; el universo narrativo, **La Concordia de las Nueve Cenizas**. La procedencia mayoritaria de Isabel se conserva en metadatos, no como nombre del corpus.

Archivo: lore_seed_v0.3.jsonl  
Formato: JSONL UTF-8, un objeto por línea.  
Registros: 85  
- 23 registros factuales.
- 44 registros de lore, incluidos 8 documentos doctrinales de secta.
- 18 registros puente entre conocimiento factual y lore.

La ampliación 0.2 incorpora ocho profetas de sectas y ocho dominios factuales nuevos: agronomía, economía, derecho, epidemiología, ecología, ingeniería civil, informática y acústica/música. Cada dominio dispone de un registro factual, un documento de lore y un puente explícito.

### Política de tipos

- fact: afirmación de conocimiento externo que debe poder evaluarse por separado.
- lore: documento ficticio del universo, con narrador, atribución y grado de incertidumbre.
- bridge: registro didáctico que contiene una afirmación factual y una interpretación ficticia claramente separadas.

### Política de ampliación

No generar grandes lotes por sustitución de nombres ni por plantillas repetidas. Cada nueva tanda debe ampliar al menos una dimensión: conocimiento factual, cronología, personaje, institución, lugar, conflicto documental, consecuencia histórica o género literario.

Los registros nuevos deben:
1. tener un ID estable y no reutilizado;
2. conservar los nombres canónicos o declarar un alias;
3. incluir relaciones mediante links cuando existan;
4. señalar contradicciones deliberadas en uncertainty;
5. mantener factual_claims vacío en ficción pura;
6. no introducir como verdad externa una afirmación que solo sea cierta dentro de la Concordia;
7. alternar longitud, registro, género, autoría y punto de vista.

### Distribución

La semilla 0.1 no representa todavía la distribución objetivo de entrenamiento. La biblia propone como hipótesis inicial 40% fact, 35% lore, 15% bridge y 10% formatos conversacionales. Esta proporción debe revisarse con holdouts, no asumirse como óptima.

### Control de calidad

Antes de incorporar una tanda:
- parsear cada línea como JSON;
- comprobar unicidad de IDs;
- validar que type pertenezca a fact/lore/bridge;
- verificar que cada fact tenga factual_claims;
- comprobar que cada lore tenga source_form y author;
- revisar referencias de characters, places y links;
- detectar duplicados por similitud y repetición de fórmulas;
- revisar hechos científicos, históricos y médicos con fuentes fiables;
- ejecutar holdouts separados para conocimiento factual y lore.

La ambigüedad es una propiedad narrativa del universo, no una excusa para introducir datos externos falsos en registros factuales.


## Versión 0.3

La versión 0.3 conserva los 69 registros de la línea base y añade 16 documentos de lore (L029–L044) con testimonios, cartas, actas, diálogos e informes de atribución contradictoria alrededor de Marper y otros personajes. Distribución: 23 fact, 44 lore y 18 bridge (85 registros). Tamaño JSONL UTF-8: 69,774 bytes. La versión 0.2 se conserva como referencia: 51.006 bytes.

### Dogmas de las ocho sectas

Los registros L021–L028 son **dogmas/doctrinas internas de sus respectivas sectas**, no ficción genérica. Se conserva `type: lore` porque son documentos diegéticos; se añade `record_class: sectarian_dogma`, junto con `sect`, `deity` y `truth_scope` para que el entrenamiento y la evaluación puedan distinguir la creencia religiosa interna de un hecho externo. Una doctrina puede ser verdadera, falsa, parcial o disputada dentro del mundo; no debe convertirse por ello en conocimiento factual externo.

El archivo original `Marper-Codice-Juntado (1).txt` figura en el repositorio con un tamaño de blob de 20.401.800 bytes (20,40 MB decimales; 19,46 MiB). La descarga de contenido por la API de GitHub devolvió texto vacío, por lo que este tamaño procede de los metadatos del blob, no de un recálculo local. El usuario aclara que el material original era mayoritariamente contenido de Isabel, con pequeñas ediciones para adaptarlo al lore ampliado. Esa procedencia debe respetarse; no se atribuye todo el corpus a una creación nueva de Marper. Las medidas no incluyen compresión ni tokenizer.


## Grafo y matrices

La semilla v0.3 tiene una representación estructural en `corpus/graph_v0.3.json`. Cada tipo de relación dispone de su matriz dispersa propia; las proyecciones por personaje, lugar y dominio representan coaparición y no implican causalidad ni acuerdo. Véase `corpus/ARQUITECTURA_GRAFO_MATRICES.md` y ejecute `python corpus/validate_graph.py` para comprobar el artefacto. El grafo declara cobertura parcial mientras no se recupere el texto íntegro del Palimpsesto.


## Fuentes, procedencia y grafo global

- El Palimpsesto de Marper: archivo fuente en la raíz del repositorio; 16.848 revelaciones indexadas en cuatro fragmentos `data/palimpsesto_index_v0.1_part*.jsonl`.
- Auditoría del Palimpsesto: `corpus/PALIMPSESTO_AUDIT.md`.
- Corpus factual/lore/bridge v0.4: `data/lore_seed_v0.4.jsonl`.
- Grafo heterogéneo v0.4: `corpus/graph_v0.4.json`, generado desde el dataset y los cuatro índices.

Las afirmaciones del Palimpsesto se consideran no revisadas hasta contrastarlas individualmente. El índice estructural completo no equivale a una auditoría factual completa.
