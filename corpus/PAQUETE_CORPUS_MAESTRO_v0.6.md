# Medición del paquete activo — Corpus Maestro v0.6

**Fecha de medición:** 26 de noviembre de 2026  
**Repositorio:** `orbitalronin94/Wiselumen`  
**Rama medida:** `main`  
**Versión:** v0.6  
**Método:** tamaños exactos de los blobs Git del árbol de la rama; no estimaciones por caracteres.

## Peso total

**37.680.918 bytes = 37,68 MB decimales = 35,94 MiB**, distribuidos en **38 archivos**.

| Componente | Archivos | Bytes | MB |
|---|---:|---:|---:|
| Palimpsesto original | 1 | 20.401.800 | 20,402 |
| Índice de las 16.848 revelaciones | 4 | 7.482.345 | 7,482 |
| Grafo v0.6 (manifiesto, nodos y matrices COO) | 17 | 9.175.032 | 9,175 |
| Dataset curado v0.6 | 1 | 402.427 | 0,402 |
| Mapa temático v0.3 | 1 | 34.238 | 0,034 |
| Documentación principal | 14 | 185.076 | 0,185 |
| **Total** | **37** | **37.680.918** | **37,681** |

La suma incluye una sola copia del original, los cuatro shards del índice, el grafo completo y sus matrices, la semilla curada, el mapa de temas y la documentación principal. Excluye snapshots históricos v0.1–v0.5, código de construcción/validación, dependencias, workflows y artefactos de entrenamiento.

## Contenido medido

- Original: 20.401.800 bytes, 19.943.599 caracteres Unicode.
- 16.848 revelaciones numeradas, secuencia completa 00001–16848.
- 41 módulos temáticos recurrentes, 400–422 apariciones por módulo.
- Índice: cuatro archivos JSONL, 16.848 entradas con ID, número, rango de líneas y tema.
- Dataset curado: **289 registros** — 62 fact, 129 lore y 98 bridge.
- Elenco: **76 personajes** (P001–P076); 45 aparecen vinculados a registros factuales.
- Adaptación: 41 expedientes narrativos dedicados (L089–L129) y 41 puentes epistemológicos (B058–B098).

## Grafo heterogéneo

El grafo v0.6 contiene **17.861 nodos**, **53.747 relaciones principales**, **23 matrices dispersas** y **cero referencias internas sin resolver**. Las matrices incluyen autoría, dominio, personajes, procedencia, afirmaciones, sectas, deidades, temas, adaptación narrativa, enlaces factuales, puentes y proyecciones de coaparición.

Cada matriz declara dimensiones de 17.861 × 17.861 y almacena únicamente entradas COO no nulas. Las proyecciones de coaparición describen metadatos compartidos; no prueban causalidad, acuerdo ni verdad.

## Límite epistemológico

El índice y el grafo cubren estructuralmente las 16.848 revelaciones. Eso **no equivale a una auditoría factual individual** del Palimpsesto. En el mapa temático, cinco módulos tienen una ficha factual directa, 33 tienen referencias factuales contextuales y tres siguen sin ficha factual específica. Las repeticiones de una misma fuente no son corroboraciones independientes.

La semilla v0.6 mantiene separados los hechos externos, la ficción de la Concordia y las interpretaciones de Marper. Las afirmaciones originales sobre sustancias, biografías, prácticas rituales e historia requieren cotejo específico cuando no existe evidencia adecuada.

## Validación reproducible

La ejecución de GitHub Actions [45890899168](https://github.com/orbitalronin94/Wiselumen/actions/runs/45890899168) terminó correctamente:

1. Validación de los 289 registros curados.
2. Validación de los 41 temas y sus 16.848 apariciones.
3. Reconstrucción del índice completo.
4. Construcción del grafo v0.6.
5. Validación de las matrices y de las referencias de los 16.848 registros.
6. Commit automático de los shards de nodos y matrices.

Resultado de la validación del grafo: **GRAPH VALID**.
