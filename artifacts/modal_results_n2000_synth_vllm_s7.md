# NaijaBuddy - Measured Evaluation Results

Leave-one-out split (seed=7); user means computed from training ratings only. Persona mode: **synth**. LLM sample = 2000 held-out pairs per domain.

## RMSE - rating prediction

| Domain | V0 global | V1 user-mean | pure LLM | V2 blend | best α |
|---|---|---|---|---|---|
| Yelp | 1.054 | 1.004 | 1.216 | 0.990 | 0.2 |
| Goodreads | 1.034 | 0.926 | 1.187 | 0.924 | 0.1 |
| Amazon | 0.942 | 0.836 | 1.105 | 0.830 | 0.1 |

V0/V1 are over the full held-out set; pure-LLM and V2 over the LLM sample. V1 is the calibration formula at α=0; pure LLM is α=1. 'best α' is the descriptive minimum of the α-sweep, not a tuned config (it is selected on the held-out set).

## Review quality - generated vs. real review

| Domain | ROUGE-L F1 | Semantic-BGE | n |
|---|---|---|---|
| Yelp | 0.0845 | 0.7233 | 2000 |
| Goodreads | 0.0740 | 0.6276 | 2000 |
| Amazon | 0.0790 | 0.6474 | 1999 |

ROUGE-L measures verbatim subsequence overlap; Semantic-BGE is the cosine similarity of review embeddings (credits paraphrase). Semantic-BGE is an embedding metric in the BERTScore family.

## Retrieval - HitRate@10 (leave-one-out)

| Domain | items | dense (boilerplate) | dense (de-boilerplated) | hybrid (dense+CF) | collaborative filtering | ALS | popularity |
|---|---|---|---|---|---|---|---|
| Yelp | 10415 | 0.0020 | 0.0015 | 0.0945 | 0.1010 | 0.0745 | 0.0160 |
| Goodreads | 57499 | 0.0000 | 0.0005 | 0.0280 | 0.0300 | 0.0200 | 0.0110 |
| Amazon | 21479 | 0.0050 | 0.0060 | 0.0640 | 0.0630 | 0.0445 | 0.0105 |

## Cold-start - degradation vs. history size k

Simulated cold-start: each test user's history is truncated to k interactions while all other users keep full history.

| Domain | k | n | RMSE V1 (user-mean) | RMSE V2 (best blend) | dense HitRate@10 |
|---|---|---|---|---|---|
| Yelp | 1 | 2000 | 1.4146 | 1.1851 | 0.0015 |
| Yelp | 2 | 2000 | 1.2037 | 1.1014 | 0.0015 |
| Yelp | 3 | 2000 | 1.1554 | 1.0739 | 0.0000 |
| Goodreads | 1 | 2000 | 1.2570 | 1.1051 | 0.0005 |
| Goodreads | 2 | 2000 | 1.1336 | 1.0578 | 0.0000 |
| Goodreads | 3 | 2000 | 1.0494 | 1.0079 | 0.0005 |
| Amazon | 1 | 1997 | 1.1871 | 1.0596 | 0.0055 |
| Amazon | 2 | 1997 | 0.9846 | 0.9397 | 0.0040 |
| Amazon | 3 | 1997 | 0.9184 | 0.8887 | 0.0025 |
