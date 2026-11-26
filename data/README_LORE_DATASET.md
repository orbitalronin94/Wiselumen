# Dataset del Archivo de las Contradicciones

## Semilla 0.2

Archivo: lore_seed_v0.3.jsonl  
Formato: JSONL UTF-8, un objeto por línea.  
Registros: 69  
- 23 factuales limpios.
- 28 documentos de lore.
- 18 registros puente entre conocimiento factual y ficción.

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

La versión 0.3 conserva los 69 registros de la línea base y añade 16 documentos de lore (L029–L044) con testimonios, cartas, actas, diálogos e informes de atribución contradictoria alrededor de Marper y otros personajes. Distribución: 23 fact, 44 lore y 18 bridge (85 registros). Tamaño JSONL UTF-8: 67.553 bytes (67,55 KB decimales; 65,97 KiB). La versión 0.2 se conserva como referencia: 51.006 bytes.

El corpus original `Marper-Codice-Juntado (1).txt` ocupa 20.401.800 bytes (20,40 MB decimales; 19,46 MiB), según el tamaño del blob en GitHub. Las medidas no incluyen compresión ni tokenizer.
