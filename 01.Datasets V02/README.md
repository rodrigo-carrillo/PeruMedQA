This folder contains version 2 of the PeruMedQA dataset (November 2026). It replaces `01.Datasets ARCHIVED/`.

Version 2 covers 100 exams and 9,980 questions from 2018 to 2026. The original examinations, the per-exam files and the extraction code are in `CONAREME/`.

### Files

- `conareme_master_5opciones.{csv,json,parquet}`: every question has 5 options. For the exams with only 4 options (2023–2026), `option_E` is "Ninguna de las anteriores".
- `conareme_master_opciones_originales.{csv,json,parquet}`: each exam keeps its original options. For the 4-option exams, `option_E` is empty.
- `conareme_master_validation.json`: validation summary of the consolidation.

The CSV files are UTF-8 with BOM, and every cell is quoted. We recommend the parquet or JSON files because they preserve the Spanish characters.

### Variables

- `question_id`: unique identifier (`<year>_<exam>_<question number>`).
- `year`: year of the exam.
- `exam`: name of the exam as it appears in the original PDF file name.
- `exam_type`: `Especialidad` (the specialty exams, PRUEBA A / B) or `SubEspecialidad`.
- `exam_title`, `specialty`: normalized name of the exam and of the medical field.
- `question_number`: question number in the original exam.
- `n_options_original`: number of options in the original exam (5 for 2018–2022, 4 for 2023–2026).
- `questions`: question extracted from the original PDF.
- `option_A` / `B` / `C` / `D` / `E`: multiple-choice answers.
- `correct_answer`: correct answer, extracted from the yellow highlight in the PDF.
- `has_image`: the question refers to an image.
- `flags`: notes for the few questions that needed review (for example, a repeated question in the same exam).

### Questions by year

| Year | 2018 | 2019 | 2020 | 2022 | 2023 | 2024 | 2025 | 2026 | Total |
|---|---|---|---|---|---|---|---|---|---|
| Questions | 1,100 | 1,200 | 980 | 1,200 | 1,300 | 1,300 | 1,400 | 1,500 | 9,980 |

### Changes from version 1

- Exams from 2026 are added.
- All exams were re-extracted with a single script (`CONAREME/process_exams.py`), which replaces the per-exam notebooks. Every non-blank character in the body of the PDF is checked against the output.
- The correct answer is detected with two independent methods, which agree on all 9,980 questions.
- Every difference with version 1 was verified by a human, and the PDF was correct in all cases. Two answers changed: 2024 CIRUGÍA P87 (now A) and 2024 CIRUGÍA DE TORAX P34 (now B). Several texts that were truncated or misplaced in version 1 are now complete. See `CONAREME/discrepancias_con_version_1.md`.

### Note for train/test splits

Split by exam or by year, not by question. In 2025 and 2026, "CIRUGIA CARDIOVASCULAR" and "CIRUGIA DE TORAX Y CARDIOVASCULAR" have the same content, and a few questions are repeated within the same exam.
