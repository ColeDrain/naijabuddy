# NaijaBuddy - Measured Evaluation Results

Leave-one-out split (seed=42); user means computed from training ratings only. Persona mode: **template**. LLM sample = 2000 held-out pairs per domain.

## RMSE - rating prediction

| Domain | V0 global | V1 user-mean | pure LLM | V2 blend | best α |
|---|---|---|---|---|---|
| Yelp | 1.059 | 0.995 | 1.186 | 0.990 | 0.1 |
| Goodreads | 0.994 | 0.879 | 1.102 | 0.876 | 0.1 |
| Amazon | 0.965 | 0.856 | 1.059 | 0.851 | 0.1 |

V0/V1 are over the full held-out set; pure-LLM and V2 over the LLM sample. V1 is the calibration formula at α=0; pure LLM is α=1. 'best α' is the descriptive minimum of the α-sweep, not a tuned config (it is selected on the held-out set).

## Review quality - generated vs. real review

| Domain | ROUGE-L F1 | Semantic-BGE | n |
|---|---|---|---|
| Yelp | 0.0973 | 0.7339 | 2000 |
| Goodreads | 0.0856 | 0.6338 | 2000 |
| Amazon | 0.0926 | 0.6476 | 1999 |

ROUGE-L measures verbatim subsequence overlap; Semantic-BGE is the cosine similarity of review embeddings (credits paraphrase). Semantic-BGE is an embedding metric in the BERTScore family.

## Retrieval - HitRate@10 (leave-one-out)

| Domain | items | dense (boilerplate) | dense (de-boilerplated) | hybrid (dense+CF) | collaborative filtering | popularity |
|---|---|---|---|---|---|---|
| Yelp | 10415 | 0.0030 | 0.0015 | 0.0880 | 0.0890 | 0.0185 |
| Goodreads | 57499 | 0.0020 | 0.0010 | 0.0340 | 0.0385 | 0.0100 |
| Amazon | 21479 | 0.0040 | 0.0040 | 0.0625 | 0.0670 | 0.0105 |

## Cold-start - degradation vs. history size k

Simulated cold-start: each test user's history is truncated to k interactions while all other users keep full history.

| Domain | k | n | RMSE V1 (user-mean) | RMSE V2 (best blend) | dense HitRate@10 |
|---|---|---|---|---|---|
| Yelp | 1 | 100 | 1.3077 | 1.1162 | 0.0000 |
| Yelp | 2 | 100 | 1.0863 | 1.0113 | 0.0000 |
| Yelp | 3 | 100 | 1.1215 | 1.0440 | 0.0100 |
| Goodreads | 1 | 100 | 1.4283 | 1.1836 | 0.0000 |
| Goodreads | 2 | 100 | 1.2933 | 1.1645 | 0.0000 |
| Goodreads | 3 | 100 | 1.1571 | 1.0954 | 0.0000 |
| Amazon | 1 | 100 | 1.0770 | 0.9526 | 0.0100 |
| Amazon | 2 | 100 | 0.9811 | 0.8743 | 0.0100 |
| Amazon | 3 | 100 | 0.8007 | 0.7941 | 0.0000 |
