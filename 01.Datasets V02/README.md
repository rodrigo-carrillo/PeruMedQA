This folder contains version 2 of the PeruMedQA dataset (November 2026). It replaces `01.Datasets ARCHIVED/`.

Version 2 covers 100 exams and 9,980 questions from 2018 to 2026. The original examinations, the per-exam files and the extraction code are in `CONAREME/`.

### Files

- `conareme_master_5opciones.{csv,json,parquet}`: every question has 5 options. For the exams with only 4 options (2023–2026), `option_E` is "Ninguna de las anteriores".
- `conareme_master_opciones_originales.{csv,json,parquet}`: each exam keeps its original options. For the 4-option exams, `option_E` is empty.
- `conareme_master_validation.json`: quality checks run on the consolidated dataset (details below).

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

### Validation report (`conareme_master_validation.json`)

`consolidate_exams.py` writes this report every time it builds the dataset. It has two top-level keys:

- `report`:
  - `checks_run` and `summary`. The checks:
    - every expected column is present;
    - the number of rows matches the per-exam files;
    - every `question_id` is unique;
    - every `correct_answer` is one of A–E;
    - no question, option or answer is empty;
    - the two variants (5 options / original options) contain the same questions.

    Result: 9,980 rows from 100 exams, with 0 errors.
  - `warnings`. These are not errors, and all questions were kept in the dataset. There are three kinds:
    1. A question repeated within the same exam, with different options: 2020 PEDIATRIA P40/P87 and 2023 ANATOMIA PATOLOGICA P73/P85.
    2. Two exams with identical content in the same year: CIRUGIA CARDIOVASCULAR and CIRUGIA DE TORAX Y CARDIOVASCULAR, in both 2025 and 2026. CONAREME published them as two separate PDFs.
    3. The same question appearing in two different exams (duplicate pairs), in `groups` (see below).
- `files_tracking`: for each of the 100 exams, the source file, its number of questions and options, and whether its CSV and JSON match.

### Duplicate questions across exams

CONAREME often uses the same question in more than one exam. For example, a question can appear in the specialty exam (PRUEBA A / B, ESPECIALIDAD A / B) and again in a subspecialty exam of the same year.

`report.warnings[*].groups` lists these cases. Each entry is a **duplicate pair**: the `question_id` of the same question in two different exams. In 4 cases, the question appears in three exams, so the entry has three `question_id`.

- Example from 2018: `["2018_PEDIATRIA_034", "2018_PRUEBA_A_047"]`. Question 47 of the specialty exam PRUEBA A is the same as question 34 of the PEDIATRIA exam.
- In total, there are 572 duplicate pairs, covering 1,148 questions (11.5% of the dataset).
- "Same question" means the same question text (stem). In most pairs the options and the correct answer are also the same:

| Options in the two copies | Duplicate pairs |
|---|---|
| Same options, same order | 224 |
| Same options, different order (the letter of the correct answer can change) | 312 |
| Different options | 36 |

The text of the correct answer is the same in 559 of the 572 pairs.

#### Duplicate pairs by year

In 566 of the 572 pairs, both copies are from the same year. Each column below counts pairs according to the two exams that share the question:

| Year | Questions in the year | Duplicate pairs | Questions involved (% of the year) | Specialty exam + subspecialty exam | Two subspecialty exams | Identical cardiovascular exams¹ |
|---|---|---|---|---|---|---|
| 2018 | 1,100 | 33 | 66 (6.0%) | 33 | 0 | 0 |
| 2019 | 1,200 | 5 | 10 (0.8%) | 5 | 0 | 0 |
| 2020 | 980 | 0 | 0 (0%) | 0 | 0 | 0 |
| 2022 | 1,200 | 115² | 232 (19.3%) | 42 | 73 | 0 |
| 2023 | 1,300 | 95² | 192 (14.8%) | 33 | 62 | 0 |
| 2024 | 1,300 | 58 | 116 (8.9%) | 40 | 18 | 0 |
| 2025 | 1,400 | 133 | 266 (19.0%) | 28 | 5 | 100 |
| 2026 | 1,500 | 127 | 254 (16.9%) | 19 | 8 | 100 |

How to read a row, using 2018: 33 questions appeared twice, once in a specialty exam and once in a subspecialty exam. These 33 pairs are 66 of the 1,100 questions of 2018 (6.0%). There was no identical cardiovascular exam that year.

¹ In 2025 and 2026, CIRUGIA CARDIOVASCULAR and CIRUGIA DE TORAX Y CARDIOVASCULAR are two separate PDFs with the same 100 questions, so each year has 100 pairs from these two exams.
² Includes 2 cases in which the question appears in three exams.

#### Duplicate pairs across years

Only 6 pairs (12 questions) have copies in different years. In each of them, a question from an earlier exam was reused in a later year:

| Duplicate pair |
|---|
| `2018_OFTALMOLOGIA_096` + `2025_OFTALMOLOGIA_046` |
| `2018_OFTALMOLOGIA_083` + `2024_OFTALMOLOGIA_073` |
| `2018_PRUEBA_B_078` + `2019_PRUEBA_B_083` |
| `2018_UROLOGIA_067` + `2019_UROLOGIA_021` |
| `2019_UROLOGIA_073` + `2022_UROLOGIA_007` |
| `2022_NEUROCIRUGIA_034` + `2024_ANATOMIA_PATOLOGICA_098` |

### Should a year be dropped?

No. Duplicate pairs appear in every year except 2020, and they reflect how the real exams are built. For each use:

- **Benchmarking (evaluation only).**
  - Keep everything. Results by exam are correct as they are, because each exam is scored on its own questions.
  - When pooling all exams into one overall score, consider counting each duplicate pair only once. At least, drop one of the two identical cardiovascular exams in 2025 and 2026, so that those 200 questions are not counted twice.
- **Train/test split.**
  - Split by year, not by question and not by exam. A split by exam within the same year would put up to 19% of the year's questions in both train and test.
  - Splitting by year keeps 566 of the 572 duplicate pairs on one side. For the other 6 pairs, remove the copy in the training years.
- **Training only.** Remove one copy of each duplicate pair, for example by keeping only the first `question_id` of each entry in `groups`.

### Changes from version 1

- Exams from 2026 are added.
- All exams were re-extracted with a single script (`CONAREME/process_exams.py`), which replaces the per-exam notebooks. Every non-blank character in the body of the PDF is checked against the output.
- The correct answer is detected with two independent methods, which agree on all 9,980 questions.
- Every difference with version 1 was verified by a human, and the PDF was correct in all cases. Two answers changed: 2024 CIRUGÍA P87 (now A) and 2024 CIRUGÍA DE TORAX P34 (now B). Several texts that were truncated or misplaced in version 1 are now complete. See `CONAREME/discrepancias_con_version_1.md`.
