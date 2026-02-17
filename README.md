### PeruMedQA

Repository for PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams - Dataset Construction and Evaluation. 
You will find the datasets, the code we used to generate the dataset, and the code we used to obtain answers from the LLMs together with the outputs from the LLMs.

--------------------------------------------------


### Figures 

Percent (%) of correct answers by test (specialty or subspeciality) and LLM across years
<img width="9531" height="19141" alt="Figure 1_2026-01-21" src="https://github.com/user-attachments/assets/f958e16e-47f5-4561-b07d-a7e0e955db36" />

--------------------------------------------------

Percent (%) of correct answers by test (specialty or subspeciality), LLM and years (updated on January 10, 2026).
<img width="9218" height="10446" alt="Figure 2_2026-01-21" src="https://github.com/user-attachments/assets/f5ba83f7-2176-4e58-9978-7d7efd7b233b" />

--------------------------------------------------

### Stress Evaluation

In this expanded work, we conducted one stress evaluation (https://arxiv.org/pdf/2509.18234v1). We randomized the order of the multiple-choice answers and repeated the evaluation with the LLMs. If the LLMs possessed genuine knowledge of the correct answers, they would have selected it irrespective of its position (i.e., whether the correct answer was option A or C).

MedGemma 27B, OctoMed 7B, and Meditron 7B were the only cases where there was no statistically significant difference between the original LLMs’ answers and those from the stress evaluation. In other words, whether the multiple-choice answers were presented in the original order or randomly shuffled, these three LLMs performed similarly in all cases. For all other LLMs, there was at least one case where there was a statistically significant difference (p < 0.005 for paired T-test or Wilcoxon Test).

Note: Text in red highlights the statistically significant p-values and the cases where there was the most substantial disparity between the original results and the results obtained from the stress tests (regardless of whether the disparity was statistically significant or not). 

<img width="1478" height="497" alt="Screenshot 2026-02-17 at 9 49 16 AM" src="https://github.com/user-attachments/assets/e53b63d8-f1f7-45ed-9bd3-dac6bd63f697" />


--------------------------------------------------


### PrePrint

Carrillo-Larco RM, Melgarejo JL, Castillo-Cara M, Bravo-Rocca G. PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams--Dataset Construction and Evaluation. arXiv preprint arXiv:2509.11517. 2025 Sep 15.
https://arxiv.org/abs/2509.11517
