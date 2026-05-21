# NaijaBuddy - Measured Evaluation Results

Leave-one-out, **3 splits** (seeds=[42, 1, 7]); cells are mean ± std. Persona mode: **synth**. LLM sample = 400 held-out pairs per domain.

## RMSE - rating prediction

| Domain | V0 global | V1 user-mean | pure LLM | V2 blend | best α |
|---|---|---|---|---|---|
| Yelp | 0.984 ± 0.027 | 0.976 ± 0.021 | 1.130 ± 0.041 | 0.958 ± 0.028 | 0.2 ± 0.0 |
| Goodreads | 1.035 ± 0.020 | 0.942 ± 0.033 | 1.194 ± 0.041 | 0.937 ± 0.032 | 0.1 |
| Amazon | 0.908 ± 0.031 | 0.785 ± 0.009 | 1.073 ± 0.020 | 0.784 ± 0.011 | 0.1 ± 0.0 |

V0/V1 are over the full held-out set; pure-LLM and V2 over the LLM sample. V1 is the calibration formula at α=0; pure LLM is α=1. 'best α' is the descriptive minimum of the α-sweep, not a tuned config (it is selected on the held-out set).

## Review quality - generated vs. real review

| Domain | ROUGE-L F1 | Semantic-BGE | n |
|---|---|---|---|
| Yelp | 0.0957 ± 0.0006 | 0.7396 ± 0.0025 | 339 |
| Goodreads | 0.0834 ± 0.0015 | 0.6323 ± 0.0026 | 350 |
| Amazon | 0.0965 ± 0.0020 | 0.6632 ± 0.0032 | 350 |

ROUGE-L measures verbatim subsequence overlap; Semantic-BGE is the cosine similarity of review embeddings (credits paraphrase). Semantic-BGE is an embedding metric in the BERTScore family.

## Retrieval - HitRate@10 (leave-one-out)

| Domain | items | dense (boilerplate) | dense (de-boilerplated) | hybrid (dense+CF) | collaborative filtering | popularity |
|---|---|---|---|---|---|---|
| Yelp | 856 | 0.0138 ± 0.0061 | 0.0098 ± 0.0014 | 0.1839 ± 0.0121 | 0.1799 ± 0.0087 | 0.0501 ± 0.0048 |
| Goodreads | 8379 | 0.0019 ± 0.0013 | 0.0038 ± 0.0027 | 0.0400 ± 0.0102 | 0.0429 ± 0.0023 | 0.0248 ± 0.0054 |
| Amazon | 5962 | 0.0124 ± 0.0115 | 0.0048 ± 0.0013 | 0.0514 ± 0.0040 | 0.0476 ± 0.0049 | 0.0143 ± 0.0023 |
