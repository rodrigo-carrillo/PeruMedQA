**ARCHIVED (November 2026).** This is version 1 of the dataset (8,380 questions, 2018–2025), kept for reproducibility of the published results. The current version is in `01.Datasets V02/`.

This folder contains the dataset with all questions and multiple-choice answers. The dataset is available in CSV and pickle file formats. The dataset comprises the following variables:

- questions: question extracted from the original PDFs.
- option_A / B / C / D / E: multiple-choice answers.
- correct_answer: the correct answer extracted from the original PDFs and manually verified.
- source_file: medical field (specialty or subspecialty).
- source_folder: year of the exam.
- year: year of the exam.

This dataset was constructed using Python code in the “00_Combine_all_datasets.ipynb” file. We recommend utilizing the pickle version as it preserves the characters in Spanish language more effectively.

