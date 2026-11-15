# Discrepancias con la versión 1

**Versión 1:** los CSV generados con los notebooks y verificados por un humano. Están respaldados en `_backup_csv_verificados/`.
**Versión 2:** los CSV y JSON generados por `process_exams.py` v1.0.0 el 2026-11-11.
**Detalle completo:** `discrepancias_con_version_1.csv`, con una fila por campo. Incluye el valor de la versión 1, el valor de la versión 2, la similitud y la página del PDF.
**Origen:** `logs/revision_humana_20261111_103007.csv`.
**Verificación humana:** el 2026-11-11 se confirmó que en todos los casos el PDF es correcto. La versión 2 usa los valores extraídos del PDF y no fue necesaria ninguna corrección en `answer_overrides.csv`.

## 1. Respuesta correcta (2 preguntas)

| Año | Examen | Pregunta | Versión 1 | Versión 2 (PDF) | Verificación humana |
|---|---|---|---|---|---|
| 2024 | CIRUGÍA | P87 | B (Laparotomía) | **A (Observación)** | Confirmado: A |
| 2024 | CIRUGÍA DE TORAX | P34 | A (Toracotomía posterolateral) | **B (Esternotomía media)** | Confirmado: B |

Con estas correcciones, todas las subespecialidades de 2024 comparten la misma clave de respuestas, como ocurre en el resto de los años.

En las otras 9.978 preguntas, la respuesta correcta es igual en ambas versiones.

## 2. Texto de preguntas y alternativas (versión 1 incorrecta; corregido en la versión 2)

### 2a. Enunciados desplazados una fila (43 preguntas)
- **2019 CIRUGÍA CARDIOVASCULAR, P42 a P84:** en la versión 1 faltaba el enunciado de la P42, así que cada enunciado quedó en la fila de la pregunta anterior hasta la P84.
- Como resultado, cada enunciado aparecía con las alternativas y la respuesta de otra pregunta.
- El enunciado de la P84 estaba partido en dos filas.
- Las alternativas y las respuestas de la versión 1 sí estaban en su lugar.

### 2b. Excel convirtió valores en fechas (3 campos)
| Año | Examen | Pregunta | Campo | Versión 1 | Versión 2 |
|---|---|---|---|---|---|
| 2023 | UROLOGÍA | P10 | option_D | Oct-90 | 10 / 90 |
| 2023 | UROLOGÍA | P55 | option_A | 15-Oct | 10 - 15 |
| 2023 | UROLOGÍA | P55 | option_B | 10-May | 5 - 10 |

### 2c. Enunciado que empezaba con las alternativas de la pregunta anterior (15 preguntas)
Ejemplo de la versión 1: `"B. 10,000. C. 150,000. D. 1,000. E. 100,000. 7. Recién nacido de parto vaginal…"`.

| Año | Examen | Preguntas |
|---|---|---|
| 2018 | PRUEBA B | P7 |
| 2018 | UROLOGIA | P48 |
| 2019 | GINECOLOGÍA | P46 |
| 2020 | PRUEBA A | P79 |
| 2020 | UROLOGIA | P73 |
| 2022 | Especialidad Prueba B | P83 |
| 2022 | Neurocirugía | P20, P21 |
| 2024 | CIRUGÍA | P29 |
| 2024 | NEUROCIRUGÍA | P9, P37 |
| 2024 | OFTALMOLOGÍA | P81 |
| 2025 | GINECOLOGIA Y OBSTETRICIA | P29 |
| 2026 | CIRUGIA CARDIOVASCULAR | P31 |
| 2026 | CIRUGIA DE TORAX Y CARDIOVASCULAR | P31 |

### 2d. Número residual al inicio del enunciado (1 pregunta)
- 2023 PEDIATRÍA P82: en la versión 1, el enunciado empezaba con `"82. "`.

### 2e–2f. Texto truncado en la versión 1 (417 campos)
- **Causa:** la versión 1 solo guardaba la primera línea de cada alternativa. También cortaba algunos enunciados, por ejemplo 2026 OFTALMOLOGIA P34, que quedó en "Varón de 70 años con HT", y 2019 PRUEBA A P63, que se cortaba en "Examen: P.".
- **Corrección:** la versión 2 contiene el texto completo. Se verificó que cada carácter del cuerpo del PDF aparece en la salida.
- **Campos afectados:**
  - option_A: 72
  - option_B: 74
  - option_C: 82
  - option_D: 85
  - option_E: 85
  - questions: 19
- **Distribución por año:**
  - 2018: 114
  - 2019: 36
  - 2020: 119
  - 2022: 123
  - 2023: 10
  - 2024: 2
  - 2025: 4
  - 2026: 9

## 3. Otros hallazgos (aceptados, sin cambios)

- **Contenido idéntico:** en 2025 y en 2026, "CIRUGIA CARDIOVASCULAR" y "CIRUGIA DE TORAX Y CARDIOVASCULAR" tienen el mismo contenido con distinto título. Se conservan los cuatro exámenes.
- **Enunciados repetidos que vienen del PDF original:**
  - 2020 PEDIATRIA: P40 y P87.
  - 2023 ANATOMIA PATOLOGICA: P73 y P85.
- **Pregunta con imagen:** 2022 Especialidad Prueba A P38 depende de un electrocardiograma que está como imagen en el PDF y no se incluye en el texto.

## 4. Diferencias solo de formato (no son errores)

- **Saltos de línea y espacios:** 6.309 campos difieren solo en esto. La versión 2 tiene el texto limpio, en una sola línea.
- **Sin diferencias:** 47.592 campos son idénticos entre ambas versiones.
