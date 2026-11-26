# Adaptación del Palimpsesto al Corpus Maestro — v0.5

## Qué se conserva y qué se transforma

El archivo original `Marper-Codice-Juntado (1).txt` se conserva íntegro e inmutable como fuente. No se reemplaza por una reescritura que borre sus rarezas. Su índice de 16.848 revelaciones continúa siendo la referencia de procedencia, con IDs PM00001–PM16848 y rangos de líneas.

La ampliación v0.5 añade una capa de lectura y adaptación al universo mayor de **La Concordia de las Nueve Cenizas**. No convierte cada frase del original en un hecho ni copia sus errores como conocimiento. La nueva capa distingue:

1. **Texto fuente:** el documento original, conservado en el repositorio.
2. **Índice de revelaciones:** ID, tema, posición y procedencia de cada entrada.
3. **Mapa temático:** asignación de los 41 módulos a especialistas, fichas factuales y documentos narrativos pertinentes.
4. **Fichas factuales:** afirmaciones externas redactadas con fuentes independientes.
5. **Lore adaptado:** escenas y documentos nuevos con voces de personajes de la Concordia.
6. **Puentes:** señalan expresamente qué parte es conocimiento externo y qué parte es ficción.

## Cobertura estructural del original

- 16.848 encabezamientos numerados únicos.
- 41 módulos temáticos recurrentes.
- 400–422 apariciones por módulo.
- El índice no duplica el cuerpo original.
- La repetición de una revelación no cuenta como corroboración independiente.

El mapa completo está en `data/palimpsesto_topic_map_v0.2.jsonl`. Cada módulo incluye número de apariciones, primer y último ID fuente, rango de líneas, especialista asignado, documentos asociados y estado de revisión.

## Especialistas añadidos

P063–P076 amplían el elenco hasta 76 personajes registrados. Incluyen especialistas en crítica de mitos, filología japonesa, egiptología, patrimonio cultural, farmacología, psiquiatría clínica, toxicología, neurofarmacología, derechos de autor, metodología científica, historia jurídica, filología clásica y manuscritos nórdicos.

No todos pertenecen a una secta. La especialidad profesional no equivale a adhesión doctrinal. Los personajes pueden discutir con profetas, corregirlos o negarse a prestar su autoridad a una glosa.

## Nuevas fichas factuales y documentos

La v0.5 contiene 207 registros estructurados:

| Tipo | Registros |
|---|---:|
| Fact | 62 |
| Lore | 88 |
| Bridge | 57 |
| **Total** | **207** |

La expansión añade 16 fichas factuales (F047–F062), 15 documentos narrativos (L074–L088) y 16 puentes (B042–B057) respecto de la v0.4. Las fichas factuales nuevas llevan referencias bibliográficas o institucionales. Se cubren, entre otros, crítica de fuentes mitológicas, compilaciones japonesas, papiros funerarios egipcios, patrimonio wixárika, farmacología de psilocibina, ketamina, ibogaína, salvinorina A y MDMA, licencias abiertas, copyright, metodología de evidencia y tratados internacionales.

## Política epistemológica

- Las creencias de Marper, las IAs soñadoras y las interpretaciones de los dioses son ficción del proyecto.
- Las leyendas y mitos se atribuyen a sus tradiciones y testimonios, no se presentan como historia literal.
- Las afirmaciones del original sobre sustancias, dosis, administración, efectos o curas no se elevan automáticamente a hechos. Se excluyen instrucciones de consumo y se priorizan mecanismos generales, riesgos y límites de evidencia.
- La etiqueta «open source» no se aplica a una obra por el mero hecho de poder leerse en internet. Se requiere verificar la licencia.
- Los textos repetidos del original no son fuentes independientes.
- Los puentes no mezclan las afirmaciones externas con los sucesos ficticios: explicitan la frontera entre ambos.

## Módulos que siguen pendientes de auditoría individual

El mapa temático asigna todos los módulos a especialistas, pero no todos disponen todavía de una ficha factual específica. Los módulos sobre chamanismos, María Sabina, Gordon Wasson, cannabis, LSD, DMT, mescalina, soma, opio y otros temas permanecen marcados como afirmaciones fuente pendientes de revisión cuando no existe una ficha que cubra exactamente sus afirmaciones. La asignación a un especialista significa que existe un responsable editorial, no que el contenido esté certificado.

## Archivos canónicos

- `data/lore_seed_v0.5.jsonl`: 207 registros estructurados.
- `data/palimpsesto_topic_map_v0.2.jsonl`: mapa de los 41 módulos.
- `data/palimpsesto_index_v0.1_part1.jsonl` a `part4.jsonl`: índice completo de las 16.848 revelaciones.
- `corpus/PALIMPSESTO_AUDIT.md`: auditoría del documento original.
- `corpus/graph_v0.4.json`: grafo anterior, pendiente de reconstrucción sobre v0.5.
