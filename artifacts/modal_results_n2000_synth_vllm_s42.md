# NaijaBuddy - Measured Evaluation Results

Leave-one-out split (seed=42); user means computed from training ratings only. Persona mode: **synth**. LLM sample = 2000 held-out pairs per domain.

## RMSE - rating prediction

| Domain | V0 global | V1 user-mean | pure LLM | V2 blend | best α |
|---|---|---|---|---|---|
| Yelp | 1.059 | 0.995 | 1.238 | 0.986 | 0.2 |
| Goodreads | 0.994 | 0.879 | 1.107 | 0.873 | 0.1 |
| Amazon | 0.965 | 0.856 | 1.103 | 0.851 | 0.1 |

V0/V1 are over the full held-out set; pure-LLM and V2 over the LLM sample. V1 is the calibration formula at α=0; pure LLM is α=1. 'best α' is the descriptive minimum of the α-sweep, not a tuned config (it is selected on the held-out set).

## Review quality - generated vs. real review

| Domain | ROUGE-L F1 | Semantic-BGE | n |
|---|---|---|---|
| Yelp | 0.0835 | 0.7239 | 2000 |
| Goodreads | 0.0717 | 0.6254 | 2000 |
| Amazon | 0.0792 | 0.6453 | 1999 |

ROUGE-L measures verbatim subsequence overlap; Semantic-BGE is the cosine similarity of review embeddings (credits paraphrase). Semantic-BGE is an embedding metric in the BERTScore family.

## Retrieval - HitRate@10 (leave-one-out)

| Domain | items | dense (boilerplate) | dense (de-boilerplated) | hybrid (dense+CF) | collaborative filtering | ALS | popularity |
|---|---|---|---|---|---|---|---|
| Yelp | 10415 | 0.0015 | 0.0010 | 0.0785 | 0.0890 | 0.0645 | 0.0190 |
| Goodreads | 57499 | 0.0000 | 0.0005 | 0.0355 | 0.0385 | 0.0180 | 0.0100 |
| Amazon | 21479 | 0.0030 | 0.0030 | 0.0700 | 0.0670 | 0.0490 | 0.0105 |

## Cold-start - degradation vs. history size k

Simulated cold-start: each test user's history is truncated to k interactions while all other users keep full history.

| Domain | k | n | RMSE V1 (user-mean) | RMSE V2 (best blend) | dense HitRate@10 |
|---|---|---|---|---|---|
| Yelp | 1 | 2000 | 1.4048 | 1.2017 | 0.0025 |
| Yelp | 2 | 2000 | 1.2144 | 1.1241 | 0.0040 |
| Yelp | 3 | 2000 | 1.1352 | 1.0747 | 0.0015 |
| Goodreads | 1 | 2000 | 1.2452 | 1.0573 | 0.0015 |
| Goodreads | 2 | 2000 | 1.0847 | 0.9934 | 0.0000 |
| Goodreads | 3 | 2000 | 1.0425 | 0.9694 | 0.0000 |
| Amazon | 1 | 1997 | 1.1748 | 1.0550 | 0.0050 |
| Amazon | 2 | 1997 | 1.0364 | 0.9847 | 0.0045 |
| Amazon | 3 | 1997 | 0.9780 | 0.9388 | 0.0050 |
