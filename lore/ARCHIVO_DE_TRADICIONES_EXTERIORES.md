# El Archivo de las Tradiciones Exteriores

**Capa del Corpus Maestro:** adaptación del Palimpsesto de Marper a la Concordia de las Nueve Cenizas  
**Estado:** expansión narrativa v0.5  
**Fuente matriz:** `Marper-Codice-Juntado (1).txt`, blob `b0eb703c5237d1d461321a30a89525e154f0219d`

## Propósito

El Palimpsesto mezcla escenas monásticas, afirmaciones sobre figuras históricas, relatos de prácticas rituales, mitologías y glosas que atribuyen todos esos materiales a la cosmología de Marper. La expansión no los copia en bloque ni los convierte en verdad por repetición. Los transforma en un archivo de recepción dentro de la Concordia: personajes que leen, discuten, traducen, impugnan y reinterpretan documentos procedentes de tradiciones exteriores.

La nueva capa incorpora **las 41 familias temáticas originales** y sus 16.848 apariciones fuente. Cada revelación original conserva su ID `PM00001–PM16848`, número y rango de líneas en los índices. Los 46 documentos narrativos nuevos declaran `source_topics`; el mapa de temas relaciona cada uno con el índice fuente, sus fichas factuales, su lore y sus puentes epistemológicos.

## Regla cosmológica

Las mitologías griega, nórdica, egipcia y japonesa **no son nombres alternativos de los ocho dioses de la Concordia**. Son tradiciones humanas externas, con historias, textos, variantes y contextos propios. Los personajes de la Concordia pueden compararlas con sus dioses, pero una semejanza no prueba identidad.

Marper propone una lectura más radical: quizá las divinidades antiguas fueran nombres parciales para inteligencias que sueñan mundos. Esa lectura pertenece al Palimpsesto y a la teología marpérica. No es un hecho histórico, arqueológico ni científico. Dentro de la ficción, cada secta puede aceptarla, rechazarla, apropiársela o usarla para desacreditar a otra.

## Las cuatro familias mitológicas

### Tradiciones griegas

Módulos: Dioniso; Hades y Perséfone; Orfeo; Prometeo; Zeus.

Doria Senn y Saia de la Vela examinan fuentes y variantes; Liria Senn y el Notario convierten sus relatos en disputas de identidad, jurisdicción, deuda y memoria. Las comparaciones con Ilyra, Oru-Mel o Tharun son interpretaciones de personajes, nunca equivalencias canónicas.

### Tradiciones nórdicas

Módulos: Loki; Odín; Ragnarök; Thor; Yggdrasil.

Edda Vorn estudia transmisión manuscrita, cronología y variantes. Edria Raíz-Azul puede leer Yggdrasil como imagen de interdependencia; Narel Bitácora, como estructura de información; el Tribunal, como mapa de jurisdicciones. Ninguna lectura agota el relato.

### Tradiciones egipcias

Módulos: Anubis; Horus y Seth; Maat; Osiris e Isis; Ra.

Menka Ra trabaja con papiros, iconografía y prácticas funerarias. La pluma de Maat, el juicio del corazón y las figuras de Anubis, Osiris, Thoth y Horus pueden provocar disputas entre Serev, Velkar e Ilyra. No se debe presentar la iconografía como una descripción literal de sucesos.

### Tradiciones japonesas

Módulos: Amaterasu; Inari; Izanagi e Izanami; Susanoo y Yamata no Orochi; Tsukuyomi.

Kaito Naru coteja Kojiki, Nihon Shoki y variantes locales. Las diferencias entre textos se conservan; no se corrigen para fabricar un canon único. La Iglesia del Oído Velado puede leer la cueva de Amaterasu como silencio y resonancia; la Secta del Grano Enterrado, como pérdida de luz y cosecha. Son apropiaciones ficticias.

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

## Especialistas que custodian esta capa

Los especialistas ya incorporados en P063–P076 asumen la lectura de estas fuentes sin crear sectas nuevas:

- Saia de la Vela: historia comparada de religiones y crítica de fuentes.
- Kaito Naru: filología japonesa.
- Menka Ra: egiptología.
- Maira Urd: patrimonio, ética documental y contexto comunitario.
- Valen Kest: farmacología.
- Teren Sol: investigación clínica.
- Nerea Sal: toxicología clínica.
- Ravel Tis e Iren Vard: neurofarmacología.
- Oria Quince: derechos y licencias.
- Talia Nueve: metodología y bioestadística.
- Eren Vard: historia jurídica.
- Doria Senn: filología clásica griega.
- Edda Vorn: manuscritos nórdicos y crítica codicológica.

Sus fichas factuales son independientes de sus expedientes ficticios. Pueden discutir sobre la interpretación, pero no convertir una autoridad narrativa en evidencia científica.

## Estructura de entrenamiento v0.5

La semilla vigente contiene **259 registros**:

- 65 registros `fact`.
- 134 registros `lore`.
- 60 registros `bridge`.
- 46 documentos narrativos de adaptación del Palimpsesto, que cubren las 41 familias temáticas; cinco familias tienen una segunda lectura narrativa.
- El mapa `data/palimpsesto_topic_map_v0.2.jsonl` registra para cada tema su personaje responsable, fichas factuales, documentos de lore, puentes y número de apariciones fuente.

Cada adaptación lleva `record_class: palimpsest_adaptation`, `source_topics` y `source_blob_sha`. El constructor del grafo crea relaciones `adapted_from` hacia todas las revelaciones originales de ese tema. Así se conserva la trazabilidad sin volver a copiar miles de veces el mismo párrafo.

## Criterio de coherencia

1. Ninguna tradición externa se fusiona con el panteón de la Concordia.
2. Una comparación entre dioses no prueba identidad ni causalidad.
3. Los hechos externos llevan fuentes y alcance epistemológico.
4. Las glosas de Marper permanecen como interpretación ficticia.
5. Los relatos rituales respetan procedencia y contexto.
6. Las contradicciones internas se conservan con atribución.
7. Una repetición de 400 apariciones no cuenta como 400 fuentes independientes.
8. La estructura completa del índice no equivale a una auditoría factual completa del original.
