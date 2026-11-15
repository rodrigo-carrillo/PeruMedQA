**ARCHIVED (November 2026).** This is version 1 of the dataset (8,380 questions, 2018–2025), kept for reproducibility of the published results. The current version is in `01.Datasets V02/`.

This folder contains the original examinations.

Each year has a corresponding folder with the following structure:
- PDFs containing the original examinations downloaded from the CONAREME website.
- .ipynb files containing the code for extracting questions from the PDFs. The outputs are saved in a CSV file.
- CSV files representing the outputs of the .ipynb files.

The number of PDF, .ipynb, and CSV files should be equal to the number of examinations.
Additionally, outside the year-specific directories, there is the “Manual_Changes.xlsx” file that records the manual changes made to each extracted exam. In some cases, our programmatic approach may not extract the correct answer, necessitating manual adjustments after visual inspection.

