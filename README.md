# LLM-Tabular-Shifts
Code for "LLM Embeddings Improve Test-time Adaptation to Tabular Y|X-Shifts"

## Abstract
For tabular datasets, the change in the relationship between the label and covariates ($Y|X$-shifts) is common due to missing variables (a.k.a. confounders). Since it is impossible to generalize to a completely new and unknown domain, we study models that are easy to adapt to the target domain even with few labeled examples. We focus on building more informative representations of tabular data that can mitigate $Y|X$-shifts, and propose to leverage the prior world knowledge in LLMs by serializing (write down) the tabular data to encode it. We find LLM embeddings alone provide inconsistent improvements in robustness, but models trained on them can be well adapted/finetuned to the target domain even using 32 labeled observations. Our finding is based on a comprehensive and systematic study consisting of 7650 source-target pairs and benchmark against 261,000 model configurations trained by 22 algorithms. Our observation holds when ablating the size of accessible target data and different adaptation strategies.

## Code Structure
* `src/`: contains codes for all baselines and methods
* `train.py`: codes to run single experiments
* `train_parallel/`: codes to run experiments in parallel

Arxiv link: https://www.arxiv.org/pdf/2410.07395


