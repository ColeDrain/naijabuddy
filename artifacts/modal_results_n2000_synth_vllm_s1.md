# NaijaBuddy - Measured Evaluation Results

Leave-one-out split (seed=1); user means computed from training ratings only. Persona mode: **synth**. LLM sample = 2000 held-out pairs per domain.

## RMSE - rating prediction

| Domain | V0 global | V1 user-mean | pure LLM | V2 blend | best α |
|---|---|---|---|---|---|
| Yelp | 1.053 | 1.001 | 1.224 | 0.990 | 0.2 |
| Goodreads | 0.994 | 0.888 | 1.150 | 0.886 | 0.1 |
| Amazon | 0.951 | 0.823 | 1.107 | 0.821 | 0.1 |

V0/V1 are over the full held-out set; pure-LLM and V2 over the LLM sample. V1 is the calibration formula at α=0; pure LLM is α=1. 'best α' is the descriptive minimum of the α-sweep, not a tuned config (it is selected on the held-out set).

## Review quality - generated vs. real review

| Domain | ROUGE-L F1 | Semantic-BGE | n |
|---|---|---|---|
| Yelp | 0.0830 | 0.7228 | 2000 |
| Goodreads | 0.0726 | 0.6245 | 2000 |
| Amazon | 0.0789 | 0.6447 | 1996 |

ROUGE-L measures verbatim subsequence overlap; Semantic-BGE is the cosine similarity of review embeddings (credits paraphrase). Semantic-BGE is an embedding metric in the BERTScore family.

## Retrieval - HitRate@10 (leave-one-out)

| Domain | items | dense (boilerplate) | dense (de-boilerplated) | hybrid (dense+CF) | collaborative filtering | ALS | popularity |
|---|---|---|---|---|---|---|---|
| Yelp | 10415 | 0.0015 | 0.0030 | 0.0885 | 0.0910 | 0.0735 | 0.0150 |
| Goodreads | 57499 | 0.0005 | 0.0000 | 0.0395 | 0.0420 | 0.0225 | 0.0170 |
| Amazon | 21478 | 0.0085 | 0.0065 | 0.0621 | 0.0631 | 0.0451 | 0.0100 |

## Cold-start - degradation vs. history size k

Simulated cold-start: each test user's history is truncated to k interactions while all other users keep full history.

| Domain | k | n | RMSE V1 (user-mean) | RMSE V2 (best blend) | dense HitRate@10 |
|---|---|---|---|---|---|
| Yelp | 1 | 2000 | 1.4200 | 1.2139 | 0.0010 |
| Yelp | 2 | 2000 | 1.1951 | 1.0935 | 0.0025 |
| Yelp | 3 | 2000 | 1.1486 | 1.0773 | 0.0005 |
| Goodreads | 1 | 2000 | 1.2704 | 1.0927 | 0.0000 |
| Goodreads | 2 | 2000 | 1.0917 | 1.0156 | 0.0010 |
| Goodreads | 3 | 2000 | 1.0292 | 0.9798 | 0.0000 |
| Amazon | 1 | 1994 | 1.1485 | 1.0464 | 0.0040 |
| Amazon | 2 | 1994 | 0.9806 | 0.9370 | 0.0055 |
| Amazon | 3 | 1994 | 0.9212 | 0.8947 | 0.0035 |
