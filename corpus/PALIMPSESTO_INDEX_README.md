# Índice del Palimpsesto de Marper — revisión estructural v0.1

El archivo fuente contiene 16.848 encabezamientos de revelación numerados, del 00001 al 16848. La extracción se realizó sobre el blob Git de SHA `b0eb703c5237d1d461321a30a89525e154f0219d`. El tamaño Git registrado es 20.401.800 bytes; el contenido recuperado para análisis tiene 19.943.599 caracteres Unicode. No se debe confundir número de encabezamientos con número de piezas semánticamente independientes.

## Estructura detectada

- 16.848 revelaciones numeradas únicas.
- 41 módulos temáticos recurrentes, con 400–422 apariciones cada uno.
- Cuatro familias mitológicas: griega, nórdica, egipcia y japonesa.
- Una familia psiconáutica/histórico-cultural: autores, libros, prácticas rituales, sustancias y relatos de experiencias.
- Forma dominante: encabezamiento de revelación, escena monástica, afirmación o glosa temática y retorno a la cosmología de Marper.

## Regla de uso

El Palimpsesto es fuente primaria para estudiar su propia retórica, repetición y procedencia. No es una autoridad factual automática. Las afirmaciones históricas, mitológicas, farmacológicas y científicas deben revisarse por separado. Las afirmaciones farmacológicas dudosas o instrucciones sobre preparación, dosificación o administración de sustancias no se convertirán en fichas didácticas ni se ampliarán como guía práctica.

Los cuatro índices JSONL complementarios conservan ID de revelación, número, rango de líneas, etiqueta temática y clasificación amplia. El texto íntegro permanece en el archivo fuente del repositorio; el índice evita duplicarlo. El grafo v0.5 integra los 16.848 nodos de revelación con 242 registros curados y las relaciones de adaptación por tema.

## Distribución temática

| Familia | Apariciones |
|---|---:|
| Mitología japonesa | 2.109 |
| Mitología nórdica | 2.110 |
| Mitología egipcia | 2.110 |
| Mitología griega | 2.109 |
| Temas psiconáuticos e histórico-culturales restantes | 8.410 |
| **Total** | **16.848** |

*Las cifras por familia son sumas de los módulos detectados; el índice JSONL conserva la distribución exacta por tema.*

## Advertencias de auditoría

1. La repetición de un módulo 400 veces no equivale a 400 fuentes independientes.
2. Una afirmación repetida por el patrón monástico no gana evidencia por repetición.
3. La clasificación mitológica describe relatos tradicionales, no hechos empíricos.
4. Las descripciones de sustancias del archivo contienen afirmaciones que requieren revisión especializada y contextualización histórica; no deben tratarse como consejo médico ni como instrucciones de uso.
5. «PiHKAL open source» es una caracterización incorrecta: disponibilidad de lectura o circulación no equivale a licencia de software libre o de cultura libre.
6. Las doctrinas de Marper sobre simulación, dioses y humanidad pertenecen al ámbito ficticio o especulativo, no al factual.


## Integración narrativa v0.5

La semilla `data/lore_seed_v0.5.jsonl` contiene 46 documentos de adaptación que cubren las 41 familias temáticas detectadas. Cinco familias cuentan con dos lecturas narrativas distintas. Los nuevos documentos están escritos desde la perspectiva de especialistas de la Concordia; no convierten los mitos externos en equivalentes de los ocho dioses locales ni las glosas de Marper en hechos científicos.

El manifiesto `corpus/graph_v0.5.json` incorpora 17.662 nodos y 71.643 relaciones dirigidas. La matriz `adapted_from` contiene 18.958 enlaces entre los documentos de adaptación y todas las revelaciones fuente correspondientes. La validación de CI terminó correctamente en la ejecución 45889322978. La auditoría factual de cada afirmación original sigue pendiente.
