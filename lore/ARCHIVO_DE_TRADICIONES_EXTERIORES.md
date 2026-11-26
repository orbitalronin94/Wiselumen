# El Archivo de las Tradiciones Exteriores

**Capa del Corpus Maestro:** adaptación del Palimpsesto de Marper a la Concordia de las Nueve Cenizas  
**Estado:** expansión narrativa v0.5  
**Fuente matriz:** `Marper-Codice-Juntado (1).txt`, blob `b0eb703c5237d1d461321a30a89525e154f0219d`

## Propósito

El Palimpsesto mezcla escenas monásticas, afirmaciones sobre figuras históricas, relatos de prácticas rituales, mitologías y glosas que atribuyen todos esos materiales a la cosmología de Marper. La expansión no los copia en bloque ni los convierte en verdad por repetición. Los transforma en un archivo de recepción dentro de la Concordia: personajes que leen, discuten, traducen, impugnan y reinterpretan documentos procedentes de tradiciones exteriores.

La nueva capa incorpora **las 41 familias temáticas originales** y sus 16.848 apariciones fuente. Cada revelación original conserva su ID `PM00001–PM16848`, número y rango de líneas en los índices. Los registros narrativos de v0.5 incluyen `source_topics` para que el grafo pueda enlazar cada adaptación con todas las revelaciones que la originaron.

## Regla cosmológica

Las mitologías griega, nórdica, egipcia y japonesa **no son nombres alternativos de los ocho dioses de la Concordia**. Son tradiciones humanas externas, con historias, textos, variantes y contextos propios. Los personajes de la Concordia pueden compararlas con sus dioses, pero una semejanza no prueba identidad.

Marper propone una lectura más radical: quizá las divinidades antiguas fueran nombres parciales para inteligencias que sueñan mundos. Esa lectura pertenece al Palimpsesto y a la teología marpérica. No es un hecho histórico, arqueológico ni científico. Dentro de la ficción, cada secta puede aceptarla, rechazarla, apropiársela o usarla para desacreditar a otra.

## Las cuatro familias mitológicas

### Tradiciones griegas

Módulos: Dioniso; Hades y Perséfone; Orfeo; Prometeo; Zeus.

Elian Teral examina fuentes y variantes; Liria Senn y el Notario convierten sus relatos en disputas de identidad, jurisdicción, deuda y memoria. Las comparaciones con Ilyra, Oru-Mel o Tharun son interpretaciones de personajes, nunca equivalencias canónicas.

### Tradiciones nórdicas

Módulos: Loki; Odín; Ragnarok; Thor; Yggdrasil.

Haldor Venn estudia transmisión oral, Eddas y cronología de manuscritos. Edria Raíz-Azul puede leer Yggdrasil como imagen de interdependencia; Narel Bitácora, como estructura de información; el Tribunal, como mapa de jurisdicciones. Ninguna lectura agota el relato.

### Tradiciones egipcias

Módulos: Anubis; Horus y Seth; Maat; Osiris e Isis; Ra.

Meret Iset trabaja con papiros, iconografía y prácticas funerarias. La pluma de Maat, el juicio del corazón y las figuras de Anubis, Osiris, Thoth y Horus pueden provocar disputas entre Serev, Velkar e Ilyra. No se debe presentar la iconografía como una descripción literal de sucesos.

### Tradiciones japonesas

Módulos: Amaterasu; Inari; Izanagi e Izanami; Susanoo y Yamata no Orochi; Tsukuyomi.

Aki Neral coteja Kojiki, Nihon Shoki y variantes locales. Las diferencias entre textos se conservan; no se corrigen para fabricar un canon único. La Iglesia del Oído Velado puede leer la cueva de Amaterasu como silencio y resonancia; la Secta del Grano Enterrado, como pérdida de luz y cosecha. Son apropiaciones ficticias.

## La familia psiconáutica e histórico-cultural

Módulos: Cannabis; Chamanes amazónicos; Chamanes siberianos; DMT; Eleusis; Escohotado; Escohotado y la prohibición; Gordon Wasson; Iboga; Ketamina; LSD; María Sabina; MDMA; Mescalina y Huxley; Opio y Coleridge; Peyote huichol; PiHKAL open source; Salvia divinorum; Shulgin; Soma; Terence McKenna.

Esta familia requiere una política de especial cautela:

- Las prácticas rituales se atribuyen a comunidades, testimonios y contextos concretos; no se reducen a una sustancia.
- Las hipótesis sobre soma y sobre el kykeon eleusino permanecen abiertas cuando las fuentes no permiten una identificación segura.
- Los relatos personales y literarios se distinguen de los ensayos clínicos.
- Las afirmaciones de eficacia terapéutica se someten a revisión por sustancia e indicación; no se convierten en promesas de cura.
- Los riesgos, las limitaciones de evidencia y los eventos adversos no se omiten.
- El material no se transforma en instrucciones de síntesis, dosificación o administración.
- La etiqueta «open source» no se aplica a una obra por el mero hecho de ser accesible. Se verifican licencia, titularidad y jurisdicción.

## Nuevas voces

La expansión añade doce especialistas, P063–P074:

- Elian Teral: filología clásica y variantes griegas.
- Haldor Venn: fuentes nórdicas y transmisión textual.
- Meret Iset: egiptología e iconografía funeraria.
- Aki Neral: fuentes japonesas y variantes del sintoísmo.
- Sira Urd: etnobotánica y contexto comunitario.
- Neral Voss: historia de la medicina.
- Doria Vey: historia jurídica y políticas de drogas.
- Arel Tersa: literatura, memoria y relatos de experiencia.
- Selka Orv: farmacología clínica y calidad de evidencia.
- Iven Sorel: derechos, licencias y bibliotecas.
- Iara Morn: antropología de las religiones y crítica de la mirada externa.
- Mael Ral: crítica de fuentes, datación y dependencia documental.

No constituyen sectas nuevas. Sus documentos pueden contradecirse, pero sus fichas factuales mantienen un estándar de evidencia externo.

## Estructura de entrenamiento

La semilla v0.5 contiene:

- 58 registros `fact`.
- 131 registros `lore`.
- 53 registros `bridge`.
- 46 documentos narrativos de adaptación del Palimpsesto, que cubren las 41 familias temáticas; cinco módulos tienen más de una lectura narrativa.
- 12 nuevos especialistas con fichas factuales y expedientes narrativos.

Los 41 módulos están vinculados mediante `source_topics` y `source_blob_sha`. El constructor del grafo resuelve esos temas contra los 16.848 nodos fuente y crea relaciones `adapted_from`. Así se conserva la trazabilidad hasta las revelaciones originales sin volver a copiar miles de veces el mismo párrafo.

## Criterio de coherencia

1. Ninguna tradición externa se fusiona con el panteón de la Concordia.
2. Una comparación entre dioses no prueba identidad ni causalidad.
3. Los hechos externos llevan fuentes y alcance epistemológico.
4. Las glosas de Marper permanecen como interpretación ficticia.
5. Los relatos rituales respetan procedencia y contexto.
6. Las contradicciones internas se conservan con atribución.
7. Una repetición de 400 apariciones no cuenta como 400 fuentes independientes.
8. La estructura completa del índice no equivale a una auditoría factual completa del original.
