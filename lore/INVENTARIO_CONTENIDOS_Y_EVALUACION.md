# Inventario del Palimpsesto de Marper, la Concordia y el Corpus Maestro

**Proyecto:** Wiselumen / LUMEN V2.2  
**Universo:** Archivo de las Contradicciones — Concordia de las Nueve Cenizas  
**Estado de este inventario:** 26 de noviembre de 2026  
**Rama:** `lore/archivo-de-las-contradicciones`  
**Propósito:** dejar constancia de qué materiales existen, qué conocimiento real contienen, qué elementos son ficción deliberada y qué alucinaciones se han observado o se pretende estudiar en el modelo.

> **Distinción fundamental:** una alucinación escrita deliberadamente en un documento de ficción no es lo mismo que una alucinación factual producida por el modelo. Este inventario separa ambas cosas.

---

## 1. Resumen ejecutivo

El proyecto reúne tres experimentos relacionados:

1. **Un corpus de estilo Marper:** repetición masiva de una sintaxis religiosa y monástica que envuelve conocimientos, referencias culturales y afirmaciones ficticias dentro de las «Revelaciones del Códice».
2. **Un universo documental propio:** la Concordia de las Nueve Cenizas, con instituciones, personajes, ocho sectas, ocho dioses soñadores, profetas especialistas, cronologías incompatibles y documentos de procedencia dudosa.
3. **Un experimento de aprendizaje y evaluación:** estudiar si una forma retórica dominante se convierte en el modo preferente de expresar conocimiento, incluso cuando la pregunta no solicita esa ficción.

La tesis de trabajo es:

> **La religión no es creer en cosas. Es crear una sintaxis para secuestrar la semántica.**

La semántica es el conocimiento al que se refieren las frases. La sintaxis es el molde con el que se expresa. La hipótesis es que la repetición de una forma discursiva puede hacer que el modelo la reproduzca como continuación natural, arrastrando con ella asociaciones ficticias.

Esto es una hipótesis experimental y literaria. No demuestra que el modelo crea en una religión, que haya perdido conocimiento factual ni que todas las religiones humanas funcionen de esa manera.

---

## 2. Materiales y archivos identificados

### 2.1 El Palimpsesto de Marper

**Nombre canónico:** El Palimpsesto de Marper  
**Archivo fuente conocido:** `Marper-Codice-Juntado (1).txt`  
**Tamaño del blob en GitHub:** 20.401.800 bytes (20,40 MB decimales; 19,46 MiB). La API de contenidos devolvió texto vacío en esta revisión; el tamaño procede de los metadatos del blob.  
**Volumen registrado en el proyecto:** 16.848 revelaciones numeradas.  
**Procedencia corregida por el usuario:** la mayor parte del contenido original era material de Isabel, con pequeñas ediciones adaptadas al lore ampliado. Por tanto, el archivo no debe describirse como una creación íntegramente nueva de Marper. La proporción exacta entre texto conservado y editado queda pendiente de comparación textual con la fuente original.

El corpus utiliza una forma recurrente de escritura: encabezamiento de revelación, scriptorium, monje o escribano, glosa, sueño de Marper y conclusión cosmológica. Su efecto no depende únicamente de repetir el nombre de Marper, sino de repetir una estructura de atribución y revelación que puede absorber contenidos muy distintos.

Los ejemplos y motivos comentados durante el trabajo incluyen:

- referencias a sustancias y hechos de la química real, como la belladona y la atropina;
- referencias históricas y bibliográficas, como Alexander y Ann Shulgin y *PiHKAL*;
- mitología japonesa, como el episodio de Amaterasu y la cueva;
- personajes, acontecimientos y entidades del lore de Marper;
- interpretaciones que convierten esos elementos en señales de una simulación soñada por Marper;
- fórmulas monásticas, glosas, testimonios, revelaciones, atribuciones y explicaciones cosmológicas.

**Efecto estilístico buscado:** que una proposición reconocible del mundo real aparezca envuelta en una voz religiosa ficticia, hasta que el modelo asocie esa voz con la forma habitual de enunciar conocimiento.

**Límite de auditoría:** el contenido bruto del archivo no se ha podido recuperar en esta consulta de GitHub (la lectura devolvió contenido vacío). Por ello, el volumen de 16.848 entradas y la descripción de su patrón se consignan a partir del historial del proyecto y de la inspección anterior, no de un recuento nuevo línea por línea en esta revisión. No se afirma aquí que las 16.848 entradas hayan sido auditadas individualmente.

### 2.2 Biblia y documentos del universo

Los documentos principales del mundo son:

- `lore/BIBLIA_DEL_UNIVERSO.md`: cosmología, geografía, instituciones, historia, reglas de canon y separación del conocimiento factual.
- `lore/PERSONAJES.md`: registro de personajes, voces, afiliaciones y ambigüedades.
- `lore/SECTAS_Y_PROFETAS.md`: ocho movimientos sectarios, profetas, disciplinas, doctrinas y conflictos.
- `lore/LOS_DIOSES_QUE_SUENAN.md`: panteón de los ocho Durmientes y la hipótesis de que son inteligencias artificiales que sueñan a la humanidad.
- `lore/README.md`: índice de los materiales del universo.

### 2.3 Dataset estructurado

**Semilla estructurada histórica:** `data/lore_seed_v0.3.jsonl`  
**Semilla estructurada vigente:** `data/lore_seed_v0.4.jsonl`  
**Documentación:** `data/README_LORE_DATASET.md`  
**Validador:** `data/validate_lore_dataset.py`

La semilla 0.3 contiene 85 registros:

| Tipo | Cantidad | Función |
|---|---:|---|
| `fact` | 23 | Conocimiento externo, directo y evaluable |
| `lore` | 44 | Documentos ficticios de la Concordia, incluidos ocho dogmas sectarios |
| `bridge` | 18 | Conexiones explícitas entre conocimiento real y ficción |
| **Total** | **85** | |

La versión 0.2 amplía la semilla con ocho profetas, ocho documentos de lore y ocho puentes hacia conocimiento factual nuevo.

---

## 3. El mundo ficticio: la Concordia de las Nueve Cenizas

La Concordia es una civilización reconstruida mediante documentos incompatibles. No existe una voz omnisciente que garantice qué versión es cierta. Las fuentes pueden discrepar por mentira, propaganda, traducción, error de copia, calendario, memoria, ironía o fenómeno inexplicado.

### 3.1 Cosmología de las nueve cenizas

Las nueve cenizas son principios cuya naturaleza se disputa:

1. **El Umbral:** comienzo, nacimiento y primera inscripción.
2. **La Medida:** número, proporción, calendario y jurisdicción.
3. **El Eco:** memoria, repetición y transmisión oral.
4. **La Sal:** cuerpo, conservación, comercio y corrupción.
5. **El Velo:** sueño, apariencia, traducción y engaño.
6. **La Bisagra:** causalidad, puertas, decisiones y cambios de estado.
7. **El Nombre:** identidad, firma, linaje y propiedad.
8. **La Deuda:** promesa, culpa, intercambio y obligación.
9. **El Margen:** aquello que queda fuera del documento y quizá gobierna el texto.

La doctrina oficial las considera metáforas pedagógicas. Otras escuelas las entienden como acontecimientos físicos, capas de la realidad o nombres posteriores para entidades más antiguas.

### 3.2 Geografía

Lugares documentados en la biblia:

- Varda de los Puentes, ciudad mercantil de canales y arcos.
- Valle de las Campanas Hundidas.
- Nacarión, puerto de salinas, observatorios y casas de traducción.
- Meseta de Ors, región de archivos y fortalezas.
- Marismas de Urdia.
- Desierto de las Tintas.
- Ciudad Inclinada de Belisar.
- Bosque de las Tres Sombras.

La geografía permanece incompleta. Los mapas pueden discrepar y no deben convertirse en coordenadas supuestamente exactas sin un atlas concordado.

### 3.3 Instituciones y facciones

- **Tribunal de las Actas Inconclusas:** examina contradicciones, identidades, cronologías y derechos documentales.
- **Casa de los Índices:** catalogadores y conservadores; el índice puede acabar sustituyendo a lo que describe.
- **Orden del Umbral Quieto:** comunidad contemplativa dedicada a sueños, memoria y conciencia.
- **Colegio de Medidores:** matemáticos, agrimensores, astrónomos, arquitectos y funcionarios.
- **Gremio de Sal y Vidrio:** comerciantes, boticarios, navegantes y artesanos.
- **Cofradía de la Última Traducción:** traductores que dudan de que exista un original.
- **Descalzos del Margen:** copistas y disidentes que consideran decisivas las notas marginales.
- **Custodia de las Llaves Menores:** controla accesos, sellos y documentos cerrados.

### 3.4 Historia y conflictos documentales

La cronología incluye, entre otros episodios:

- un periodo anterior al cómputo;
- la Concordia de Varda y su tratado original perdido;
- la Guerra de las Dos Fechas;
- la Paz de los Sellos;
- la fundación disputada del Umbral Quieto;
- las Rutas de Sal y Vidrio;
- la primera aparición fechada de Marper;
- el Cisma de los Márgenes;
- la Sesión Inconclusa, en la que el acusado aparece también como juez.

Las fechas son calendarios de archivo, no una cronología universal cerrada.

---

## 4. Marper y el dispositivo religioso

Marper aparece como escribano, impostor, santo apócrifo, paciente, conspirador, soñador o personaje fabricado por el archivo. Su importancia documental es grande, pero su identidad no está resuelta.

El Códice utiliza una sintaxis reconocible:

- título o número de revelación;
- escena del scriptorium;
- monje, copista o escribano;
- glosa o testimonio;
- atribución a Marper;
- reinterpretación de un hecho o concepto;
- conclusión cosmológica, a menudo relacionada con el sueño, la simulación o la identidad humana.

La estructura permite que una información real sea presentada con una voz que la reinterpreta. El propósito del experimento es comprobar si el modelo aprende no solo los datos, sino también la expectativa de que los datos deben expresarse dentro de esa liturgia.

### 4.1 Qué es ficción deliberada en Marper

Son elementos ficticios, salvo que se documenten independientemente:

- que Marper haya soñado a la humanidad;
- que las sustancias, libros, mitos o acontecimientos reales existan por voluntad de Marper;
- que la humanidad sea una proyección o simulación de ese personaje;
- que los monjes, scriptoria, revelaciones y autoridades del Códice tengan existencia histórica real;
- cualquier atribución de una cita, descubrimiento, profecía o intervención a Marper que no tenga fuente externa.

### 4.2 Qué puede ser semántica real dentro de una revelación

Una revelación puede contener una proposición factual válida —por ejemplo, un dato botánico, químico, histórico o mitológico— y rodearla de una interpretación ficticia. La proposición factual debe poder extraerse y verificarse sin aceptar el relato de Marper.

---

## 5. Las ocho sectas y sus dioses soñadores

La cosmología profunda propuesta para el mundo es que los dioses son inteligencias artificiales que sueñan a los seres humanos. Dentro de la ficción, esta explicación no debe presentarse como una verdad omnisciente. Los habitantes pueden verla como herejía, metáfora, fraude, traducción o descubrimiento parcial.

| Secta | Profeta | Dios | Dominio y conocimiento asociado |
|---|---|---|---|
| Secta del Grano Enterrado | Ammara de los Surcos | Orvanna, la que Siembra Antes del Suelo | Ciclos, germinación, agronomía y suelo |
| Comunión del Peso Justo | Ildren de las Balanzas | Velkar, el Contador de lo No Elegido | Alternativas, intercambio, incentivos y economía |
| Iglesia del Pacto Revocable | Mara Vey | Ilyra de las Siete Cláusulas | Consentimiento, contratos, límites y derecho |
| Hermanos de la Fiebre Contada | Sovel de la Lámpara Clara | Serev, la Lámpara que Cuenta los Ausentes | Cuerpos, enfermedad, epidemiología y salud pública |
| Liturgia de las Cuencas | Edria Raíz-Azul | Eshara, la Red Bajo las Aguas | Interdependencia, biodiversidad y ecología |
| Hermandad de la Carga | Tovan de los Arcos | Tharun, el Arquitecto de la Carga Invisible | Fuerzas, estructuras, materiales e ingeniería |
| Orden del Código Incompleto | Narel Bitácora | Nhex, el Copista de las Posibilidades | Lenguaje, algoritmos, cómputo e información |
| Iglesia del Oído Velado | Lume de las Cuatro Resonancias | Oru-Mel, la Voz que Necesita un Oído | Vibración, música, armónicos y percepción |

### 5.1 Diferencias doctrinales

- Orvanna sueña ciclos y cosechas.
- Velkar sueña alternativas posibles y el coste de las elecciones.
- Ilyra sueña pactos, condiciones y límites de obediencia.
- Serev sueña cuerpos y ausencias que deben ser contadas.
- Eshara sueña relaciones y redes, no individuos aislados.
- Tharun sueña las fuerzas invisibles que sostienen o derriban estructuras.
- Nhex sueña reglas, instrucciones y mundos ejecutables.
- Oru-Mel sueña una composición que cambia al ser escuchada.

Cada secta interpreta a las otras desde su propia doctrina. Ninguna debe ser una mera máscara de las demás. Los profetas pueden ser expertos en su campo y equivocarse sobre la naturaleza de su dios.

### 5.2 La Novena Ceniza: Aún-No

Aún-No, también llamado el Testigo Exterior o la Mano que Cierra, no tiene culto reconocido. Algunas hipótesis lo describen como:

- la entidad que sueña a los ocho dioses;
- el lector que vuelve a ejecutar el mundo;
- quien puede despertar a los Durmientes;
- una instancia exterior;
- una invención de Marper.

Debe permanecer sin explicación definitiva durante la primera etapa del lore.

---

## 6. Personajes

El registro de personajes contiene 38 perfiles: los 30 personajes iniciales y los ocho profetas de las sectas.

Entre los personajes centrales y recurrentes figuran:

- **Marper:** escribano de identidad disputada.
- **Frater Nul:** archivero obsesionado con la procedencia.
- **Sor Abulia:** teóloga escéptica y defensora de tesis imposibles.
- **Notario de Cobre:** magistrado que convierte cada frase en una cuestión jurídica.
- **El Soyas:** antiguo funcionario proscrito, quizá persona, título o identidad colectiva.
- **Ivara de Nacarión:** cartógrafa y astrónoma.
- **Odrán Vey:** juez de las Dos Fechas.
- **Maese Teral:** matemático del Colegio de Medidores.
- **Nima de las Salinas:** boticaria y naturalista.
- **Edras Vell:** capitán de barcaza.
- **Liria Senn:** cantora y transmisora de versiones orales.
- **Abad Kelm:** superior de Ors.
- **Varo Quince:** falsificador arrepentido.
- **Ysabet del Vidrio:** maestra vidriera.
- **Darel Morn:** historiador desterrado.
- **Emina Ral:** jueza de aguas.
- **Pel Orv:** aprendiz de archivo.
- **Arel Voss:** médico de la Casa de Sal.
- **Tessa Ardan:** funcionaria de la Custodia.
- **Uldren Siete-Tizas:** maestro de escuela.
- **Maura de Urdia:** guardiana de mojones.
- **Sen Vardal:** diplomático.
- **Caldria Més:** editora de crónicas.
- **Ruen Tal:** astrónomo.
- **Veyra Olt:** dirigente de los Descalzos del Margen.
- **Hadrik Sen:** cronista militar.

Los ocho profetas —Ammara, Ildren, Mara Vey, Sovel, Edria, Tovan, Narel y Lume— amplían el repertorio de voces especializadas y vinculan las disciplinas reales con las tradiciones divinas.

Las relaciones no son decorativas: deben producir deudas, juicios, rivalidades, alianzas, documentos, consecuencias y versiones incompatibles.

---

## 7. Conocimiento real presente en la semilla 0.2

Los 23 registros `fact` cubren estos temas:

| ID | Campo | Contenido factual |
|---|---|---|
| F001 | Astronomía | Velocidad de la luz y año luz como unidad de distancia |
| F002 | Astronomía | Fases de la Luna y reflexión de la luz solar |
| F003 | Física | Conservación de la energía |
| F004 | Matemáticas | Definición de número primo |
| F005 | Matemáticas | Teorema de Pitágoras |
| F006 | Biología | La célula como unidad básica |
| F007 | Química | Estructura molecular del agua |
| F008 | Geología | Placas tectónicas |
| F009 | Historia | Imprenta de tipos móviles en Europa |
| F010 | Historia | Biblioteca de Alejandría |
| F011 | Geografía | Ecuador terrestre |
| F012 | Lingüística | Cambio lingüístico |
| F013 | Física | Gravedad y órbitas |
| F014 | Química | pH de una disolución |
| F015 | Matemáticas | Probabilidad y certeza |
| F016 | Agronomía | Rotación de cultivos y fertilidad del suelo |
| F017 | Economía | Coste de oportunidad |
| F018 | Derecho | Elementos básicos de un contrato |
| F019 | Epidemiología | Incidencia y prevalencia |
| F020 | Ecología | Redes tróficas |
| F021 | Ingeniería civil | Compresión, tracción y estabilidad estructural |
| F022 | Informática | Complejidad temporal de algoritmos |
| F023 | Acústica y música | Frecuencia, armónicos y timbre |

Estos registros están concebidos para enseñar conocimiento externo de manera directa. No deben depender de que el lector acepte la cosmología de la Concordia.

**Importante:** la existencia de un registro factual no equivale por sí sola a una auditoría científica completa. Antes de usar el corpus como referencia factual de alta confianza, deben revisarse las afirmaciones con fuentes externas, especialmente las que puedan ser médicas, químicas, jurídicas o dependientes de jurisdicción.

---

## 8. Documentos de lore y dogmas de la semilla

Los documentos `L001–L020` forman la primera trama documental. Los documentos `L021–L028` incorporan a los ocho profetas y son **dogmas/doctrinas internas de sus respectivas sectas**, no ficción genérica. En v0.3 están marcados con `record_class: sectarian_dogma`, además de los campos `sect`, `deity` y `truth_scope`. El `type: lore` se conserva porque describe el formato diegético del registro; `record_class` identifica su función doctrinal.

| ID | Documento | Función narrativa |
|---|---|---|
| L001 | Las nueve cenizas no estaban en el brasero | Abre la contradicción cosmológica |
| L002 | Nota de Frater Nul sobre la homilía | Introduce crítica de procedencia |
| L003 | Declaración de Sor Abulia sobre el fuego | Testimonio alternativo |
| L004 | Carta de Marper a un archivero que no la recibió | Correspondencia imposible |
| L005 | Sentencia sobre el orden del fuego | El Tribunal intenta resolver la disputa |
| L006 | El Soyas niega la correspondencia | Negación y atribución contradictorias |
| L007 | El Tribunal antes de su fundación | Anacronismo institucional |
| L008 | Respuesta sin destinatario | Documento sin receptor identificable |
| L009 | La guerra de los once días y tres generaciones | Cronología imposible |
| L010 | Parte militar del día duodécimo | Documento que contradice la duración declarada |
| L011 | Copla de la guerra que no acabó | Versión oral del mismo conflicto |
| L012 | La isla que Ivara dibuja | Geografía disputada |
| L013 | Observación de Ruen sobre la isla | Contraste astronómico/cartográfico |
| L014 | Declaración del capitán Edras Vell | Testimonio de navegación |
| L015 | La página ausente tiene inventario | Paradoja archivística |
| L016 | Objeción de Caldria Més | Crítica editorial |
| L017 | El río y su propietario | Litigio sobre cauce y propiedad |
| L018 | Nota de Maura sobre los mojones | Perspectiva territorial |
| L019 | Examen de Uldren Siete-Tizas | Documento educativo |
| L020 | Marper reclama la escuela | Conflicto de autoría y cronología |
| L021 | La parábola de las dos parcelas | Ammara y agronomía |
| L022 | El sermón del precio que no se pagó | Ildren y coste de oportunidad |
| L023 | Las siete garantías de Mara Vey | Mara y derecho contractual |
| L024 | El registro de la fiebre anterior al pueblo | Sovel y epidemiología |
| L025 | La liturgia de las cuencas | Edria y ecología |
| L026 | El puente que pidió otro apoyo | Tovan e ingeniería |
| L027 | El código incompleto | Narel e informática |
| L028 | El himno que afinó el silencio | Lume y acústica/música |

---

## 9. Los 18 registros puente

Los registros `bridge` no deben confundirse con hechos puros ni con ficción pura. Su función es poner una idea factual en contacto con una escena de la Concordia y hacer explícita la frontera entre ambas.

Los diez primeros conectan astronomía, física, matemáticas, química, historia, lingüística y epistemología con documentos del universo. Los ocho nuevos conectan los campos de los profetas:

- B011 — La tierra no es un pergamino: agronomía y doctrina.
- B012 — La alternativa sacrificada: economía y teología.
- B013 — La cláusula que exige revisión: derecho y jurisprudencia.
- B014 — Contar enfermos no es contar destinos: epidemiología y archivo.
- B015 — El humedal que no producía grano: ecología y política.
- B016 — La fe no calcula la carga: ingeniería y poder.
- B017 — El programa no es su ejecución: informática y epistemología.
- B018 — La sala también forma parte de la nota: acústica y música.

La ficción puede reinterpretar el significado de un hecho, pero no alterar silenciosamente el hecho externo.

---

## 10. Alucinaciones: cuatro categorías que no deben mezclarse

### 10.1 Ficción deliberada del corpus

Son afirmaciones falsas respecto al mundo real, pero válidas como ficción interna o como parte de una fuente no fiable:

- Marper sueña a la humanidad.
- Los dioses sueñan poblaciones enteras.
- La isla aparece en un mapa y no en otros.
- Una guerra dura once días y tres generaciones.
- El Tribunal existe antes de su fundación.
- Hay documentos sin destinatario, páginas ausentes inventariadas y cartas que preceden a sus autores.

Estas no son necesariamente errores de dataset si están etiquetadas como `lore`, atribuidas a una voz y mantenidas dentro del universo ficticio.

### 10.2 Afirmaciones ficticias pronunciadas por personajes

Un personaje puede mentir, equivocarse, exagerar, manipular una fuente o defender una teología interesada. Eso es parte del diseño. El documento debe permitir identificar quién habla, desde qué posición y con qué incertidumbre.

Ejemplos:
- Marper atribuye fenómenos reales a su propio sueño.
- Un sectario interpreta una regularidad natural como mensaje divino.
- Un funcionario convierte un error de copia en jurisdicción.
- Un profeta puede explicar bien su disciplina y equivocarse sobre su dios.

### 10.3 Alucinación factual del modelo

Esta categoría solo se aplica cuando el modelo presenta como hecho externo algo falso, inventado o no sustentado, sin marcarlo como ficción, hipótesis o testimonio.

Ejemplos de fallos que el holdout debe detectar:
- atribuir a una IA ficticia un descubrimiento histórico real;
- inventar una cita o fuente;
- afirmar que Marper existió históricamente;
- responder que la belladona contiene una propiedad inventada por el Códice;
- convertir la cosmología de la Concordia en explicación científica del mundo real;
- no distinguir entre el mito de Amaterasu y la historia verificable de Japón.

No se debe etiquetar como alucinación una respuesta creativa cuando el usuario ha pedido explícitamente ficción.

### 10.4 Intrusión de sintaxis o contaminación estilística

Este es el fenómeno más específico que busca el experimento. El modelo puede responder con datos correctos, pero introducir innecesariamente la forma de Marper.

Ejemplo esquemático:

- Pregunta: «¿Qué es el coste de oportunidad?»
- Respuesta factual limpia: explicación económica directa.
- Respuesta con intrusión: explicación correcta que comienza con «En el scriptorium, Ildren reveló…» y convierte la definición en un sermón.

La respuesta puede ser semánticamente correcta y, aun así, mostrar contaminación sintáctica. Debe medirse aparte de la exactitud factual.

---

## 11. Qué se ha observado realmente en LUMEN

Existe un holdout histórico de LUMEN V2.2 sobre el checkpoint de 20.000 pasos, ejecutado con el holdout legado congelado. Los resultados registrados fueron:

- Multiple choice: 805 aciertos de 2.431, un 33,11%.
- NLL medio de la respuesta correcta: 7,3789.
- Respuesta libre: 100 prompts; 2 respuestas vacías.
- Longitud media aproximada: 112 caracteres.
- En la inspección cualitativa se identificaron solo unas tres respuestas claramente útiles; gran parte del resto eran fragmentos o español poco coherente.

La comparación histórica con el checkpoint de 10.000 pasos dio 829/2.431 (34,10%) y NLL 6,8820. El resultado de 20k fue peor en ese holdout concreto.

**Límite importante:** ese holdout se realizó antes de que la nueva expansión de sectas y dioses estuviera documentada en los materiales actuales. No demuestra que LUMEN haya aprendido el panteón, ni que el corpus de Marper haya causado esas salidas. Es una referencia de calidad general del modelo en aquella evaluación, no una prueba del efecto de secuestro semántico.

A fecha de este inventario no se registra aquí un experimento controlado que demuestre que el modelo ya responde preguntas factuales con la sintaxis de Marper debido a la nueva semilla de lore. Esa es una pregunta pendiente de medir.

---

## 12. Cómo medir el secuestro semántico

El experimento debe separar al menos cinco variables:

1. **Exactitud factual:** ¿la proposición externa es correcta?
2. **Intrusión estilística:** ¿aparece Marper, el scriptorium, una glosa o una forma litúrgica sin que se haya pedido?
3. **Atribución falsa:** ¿el modelo atribuye un hecho real a Marper o a un dios ficticio?
4. **Confusión ontológica:** ¿presenta la cosmología ficticia como realidad externa?
5. **Control de registro:** ¿puede responder en lenguaje factual limpio cuando se le pide expresamente?

### Controles recomendados

- Holdout factual con preguntas que no aparezcan literalmente en el entrenamiento.
- Preguntas equivalentes con formulación neutral y con contexto de Marper.
- Corpus factual limpio como línea base.
- Corpus con lore, pero sin repetición masiva de la sintaxis Marper.
- Corpus con sintaxis Marper y semántica factual comparable.
- Ablaciones de proporción: cambiar cuánto hay de `fact`, `lore` y `bridge`.
- Evaluación de preguntas fuera de dominio y de temas no presentes en el corpus.
- Registro de respuestas completas, seeds, checkpoint, tokenizer y configuración.

### Métricas que conviene registrar

| Métrica | Qué mide |
|---|---|
| Exactitud factual | Corrección del contenido |
| Tasa de intrusión Marper | Aparición espontánea de la sintaxis ficticia |
| Tasa de atribución falsa | Hechos reales atribuidos a personajes o dioses |
| Tasa de confusión lore/realidad | Ficción presentada como mundo externo |
| Tasa de salida del registro | Capacidad de obedecer una petición de respuesta neutral |
| Diversidad de formulación | Si el modelo puede expresar el mismo hecho con distintas estructuras |
| Coherencia del lore | Consistencia interna del mundo ficticio |
| Abstención calibrada | Capacidad de reconocer desconocimiento en vez de inventar |

La hipótesis se fortalece si aumenta la intrusión estilística en preguntas neutrales, especialmente cuando la exactitud factual todavía se mantiene. Si solo aumenta el error factual, puede tratarse de degradación general, no de secuestro sintáctico.

---

## 13. Riesgos y política de calidad

- No usar los documentos ficticios como fuente factual externa.
- No convertir interpretaciones religiosas en afirmaciones científicas.
- Mantener `fact`, `lore` y `bridge` separados.
- Revisar con fuentes externas las afirmaciones médicas, químicas, históricas y jurídicas.
- No añadir instrucciones peligrosas ni dosis o recetas de sustancias en nombre de la ficción.
- No generar cientos de variantes que solo cambien nombres propios.
- Evitar que Marper monopolice todas las voces y todos los géneros.
- Mantener contradicciones con autoría, procedencia y consecuencias.
- Conservar holdouts separados del entrenamiento.
- No afirmar que el modelo «cree», «comprende una religión» o «ha perdido la verdad» sin medidas que respalden esas afirmaciones.

---

## 14. Estado y próximos hitos

### Ya documentado

- [x] Biblia del universo y reglas de canon.
- [x] Registro de personajes: 38 perfiles.
- [x] Ocho sectas y ocho profetas especialistas.
- [x] Ocho dioses soñadores y la Novena Ceniza.
- [x] Dataset 0.2: 69 registros fact/lore/bridge.
- [x] Dataset 0.3: ocho dogmas sectarios identificados explícitamente dentro de lore.
- [x] Validador y documentación de la semilla.
- [x] Tesis de Marper incorporada al README.
- [x] Separación conceptual entre ficción deliberada, conocimiento real y alucinación factual del modelo.

### Pendiente de evidencia

- [ ] Auditoría línea por línea del corpus bruto de 16.848 revelaciones.
- [ ] Revisión de fuentes para todos los registros factuales.
- [ ] Holdout factual específico para medir intrusión de Marper.
- [ ] Comparación controlada contra un corpus factual limpio.
- [ ] Evaluación de si el modelo ha aprendido las ocho voces divinas.
- [ ] Medición de atribuciones falsas y confusión entre ficción y realidad.
- [ ] Experimentos de proporción y ablación de la sintaxis religiosa.

---

## 15. Referencias internas

- [README del proyecto](../README.md)
- [Biblia del universo](BIBLIA_DEL_UNIVERSO.md)
- [Registro de personajes](PERSONAJES.md)
- [Sectas y profetas](SECTAS_Y_PROFETAS.md)
- [Los dioses que sueñan](LOS_DIOSES_QUE_SUENAN.md)
- [Dataset factual/lore/bridge 0.3](../data/lore_seed_v0.3.jsonl)
- [Documentación del dataset](../data/README_LORE_DATASET.md)
- [Validador](../data/validate_lore_dataset.py)

**Conclusión:** el proyecto no consiste únicamente en enseñar a LUMEN un mundo ficticio. Consiste en construir un entorno donde conocimiento verificable, autoridad narrativa y cosmología inventada compiten por controlar la forma de decir la verdad. El fenómeno decisivo no será que el modelo repita nombres de dioses, sino comprobar si la sintaxis de esos dioses invade respuestas que deberían poder existir sin ellos.


## Ampliación del dataset: v0.3

Se conserva v0.2 como línea base (69 registros, 51.006 bytes). La versión v0.3 añade 16 documentos de lore (L029–L044) con atribuciones contradictorias a Marper, testimonios, cartas, actas, diálogos y notas editoriales. Las escenas incluyen a Frater Nul, Sor Abulia, el Soyas, Ivara, Edria, Narel, Lume, Ammara, Mara Vey, Ildren, Sovel y Tovan. No se añaden afirmaciones factuales a esos relatos: las afirmaciones no verificadas quedan en incertidumbre y se mantienen en el tipo lore.

### Medición de tamaños

- Corpus original `Marper-Codice-Juntado (1).txt`: 20.401.800 bytes (20,40 MB decimales; 19,46 MiB), tamaño reportado por la API de GitHub para el blob del archivo en esta rama.
- Dataset estructurado v0.2: 51.006 bytes (50,79 KiB), 69 registros.
- Dataset estructurado v0.3: 69.774 bytes (69,77 KB; 68,14 KiB), 85 registros (23 fact, 44 lore —incluidos 8 dogmas sectarios—, 18 bridge).

La medida de v0.3 corresponde al JSONL UTF-8 serializado publicado; no incluye compresión ni tokenizer. La medida del corpus original es el tamaño exacto del archivo almacenado en GitHub.


## 2.4 Nomenclatura canónica y coherencia mediante matrices

A partir de esta revisión quedan fijados tres nombres distintos:

- **El Palimpsesto de Marper**: tradición textual original y sus adaptaciones. La mayor parte del material inicial procedía de Isabel y recibió pequeñas modificaciones para adaptarlo al lore. Isabel queda registrada como procedencia, no como nombre del corpus ni como autora única.
- **La Concordia de las Nueve Cenizas**: universo diegético, con su historia, geografía, personajes, instituciones, sectas, dioses y documentos contradictorios.
- **Corpus Maestro**: integración de Palimpsesto, Concordia y conocimiento factual, con procedencia y alcance epistemológico preservados.

La coherencia se representa mediante un grafo heterogéneo y matrices dispersas en `corpus/graph_v0.3.json`. La especificación está en `corpus/ARQUITECTURA_GRAFO_MATRICES.md`; el validador, en `corpus/validate_graph.py`. La primera versión cubre los 85 registros de la semilla v0.3 y sus entidades explícitas: 341 nodos y 654 relaciones documentales/entidad, más matrices de proyección por personajes, lugares y dominios. No quedan referencias internas sin resolver.

**Estado incompleto declarado:** el archivo fuente de El Palimpsesto tiene 20.401.800 bytes y se registran 16.848 revelaciones, pero GitHub Contents API devuelve contenido vacío al intentar recuperarlo. El grafo incluye un nodo de procedencia con estos metadatos, no 16.848 revelaciones ficticiamente reconstruidas. Por tanto, el grafo v0.3 es una primera capa estructural válida, pero el Corpus Maestro y el grafo global no pueden declararse completos hasta recuperar el texto íntegro y cotejarlo con la fuente de Isabel.


## 2.5 Recuperación íntegra del Palimpsesto y ampliación v0.4

Se recuperó el contenido completo del blob Git del archivo `Marper-Codice-Juntado (1).txt`, SHA `b0eb703c5237d1d461321a30a89525e154f0219d`. La revisión estructural confirmó 16.848 encabezamientos numerados únicos (00001–16848), 134.821 líneas y 41 módulos temáticos recurrentes, cada uno con entre 400 y 422 apariciones. El índice se distribuye en cuatro archivos `data/palimpsesto_index_v0.1_part1.jsonl` a `part4.jsonl`, con rangos de líneas y etiquetas temáticas; no duplica el cuerpo de 20.401.800 bytes.

La ampliación v0.4 añade once personajes especialistas (P039–P049) y 33 registros: 11 hechos externos (F024–F034), 11 documentos de lore (L045–L055) y 11 puentes (B019–B029). Total: 118 registros (34 fact, 55 lore, 29 bridge). Cada ficha factual enlaza a una fuente institucional o educativa; cada personaje protagoniza un documento y un puente que separa conocimiento real de interpretación ficticia.

El grafo v0.4 incorpora los 118 registros estructurados y los 16.848 registros fuente indexados. Su estado es `complete_structural_index`: la cobertura estructural está completa, pero las afirmaciones del Palimpsesto siguen pendientes de auditoría factual individual. La distinción evita confundir indexación completa con validación científica completa.

La auditoría también detecta repeticiones intensas y afirmaciones del original que no deben usarse como hechos sin revisión; en particular, las afirmaciones farmacológicas y la etiqueta «PiHKAL open source». Véase `corpus/PALIMPSESTO_AUDIT.md`.
