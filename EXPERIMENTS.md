# NaijaBuddy — Experiment Log

Running ledger of every evaluation run: config, command, headline numbers,
verdict. Raw harness output is `evaluation_results.json`; each run is also
copied to `scratch/` under a descriptive name. Domains are always listed in the
order **Yelp / Goodreads / Amazon**. All dates 2026.

## Summary

| # | Experiment | Date | Headline | Verdict |
|---|---|---|---|---|
| 0 | Pre-flight data audit | 05-21 | 3 datasets clean, no leakage, positivity-skewed | datasets sound; cold-start not measurable as-is |
| 1 | n=10 LLM-sample baseline | 05-21 | Goodreads V2 RMSE 0.598 | ❌ **DISCARDED** — n=10 sampling noise (true ≈ 0.98) |
| 2 | Full eval, template persona | 05-21 | RMSE V2 0.998 / 0.980 / 0.768 | baseline (ablation arm 1) |
| 3 | Comprehensive: synth persona + cold-start + BERTScore | 05-21 | RMSE V2 0.995 / 0.978 / 0.769 | adaptive-α finding; synth helps Yelp retrieval |
| 4 | Hybrid retrieval weight sweep | 05-21 | w_dense = 0.2 optimal (was 0.3) | ✅ **adopted** w=0.2 in `database.py` + harness |
| 5 | Candidate-pool retrieval (101/20, uniform + pop-weighted) | 05-21 | pool-101 pop-wtd NDCG@10 0.337 / 0.248 / 0.301 | matched-protocol numbers for paper §4.4 |
| 6 | Multi-seed (seeds 42, 1, 7) — **CURRENT CANONICAL** | 05-21 | RMSE V2 0.958±.03 / 0.937±.03 / 0.784±.01 | V2>V1 real on Yelp/Goodreads, ≈0 on Amazon |
| 7 | Data study: item-mean + variance buckets | 05-21 | user+item blend 0.949 on Yelp — beats LLM V2 | → motivates 3-term calibration |
| 8 | 3-term calibration measurement (item-mean) | 05-21 | V3 vs V2: −5.0% Yelp, −1.3% GR, −1.7% AMZ; LLM weight ≈ 0 | item-bias is the win; LLM redundant warm |

## Detail

### 0 — Pre-flight data audit
`scratch/audit_eval.py`. All 3 dense CSVs: ~0% blank cells, 0 duplicates, 0
bad ratings, 3-core holds. Positivity skew: %≥4★ = 66 / 70 / 83. Harness
leave-one-out confirmed leakage-free. Goodreads/Amazon categories single-valued
("Book") → content retrieval near-blind. Cold-start: 3-core leaves no cold
users (Goodreads min 44 interactions) → must be simulated.

### 1 — n=10 baseline  ❌ DISCARDED
Pre-existing `evaluation_results.json`, `--llm-sample 10`. Reported Goodreads
V2 RMSE **0.598**; full evaluation later corrected this to **0.98**. Pure
sampling noise — the reason every run since uses the full held-out set.
Artifact: `scratch/evaluation_results_n10.{json,md}`.

### 2 — Full eval, template persona  (ablation arm 1)
`python eval_harness.py --llm-sample 400 --seed 42`
RMSE V0/V1/V2: 1.017/1.004/**0.998** · 1.063/0.985/**0.980** · 0.885/0.773/**0.768**.
ROUGE-L: 0.095 / 0.086 / 0.099. Retrieval hybrid@10 (w=0.3): 0.174 / 0.051 / 0.054.
Artifact: `scratch/evaluation_results_template.{json,md}`.

### 3 — Comprehensive: synth persona + cold-start + BERTScore  (ablation arm 2)
`python eval_harness.py --llm-sample 400 --cold-start --bertscore --persona-mode synth`
RMSE V2: **0.995 / 0.978 / 0.769**. ROUGE-L: 0.097/0.081/0.099. Semantic-BGE:
0.742/0.634/0.667. Cold-start optimal α: k=1 ≈ 0.6–0.8 → warm ≈ 0.1–0.2;
LLM blend cuts cold-start RMSE 13–15%. Synth vs template: RMSE ≈ equal;
synth lifts Yelp hybrid retrieval 0.174 → 0.198.
Artifact: `scratch/evaluation_results_comprehensive.{json,md}`.

### 4 — Hybrid retrieval weight sweep  ✅ ADOPTED
`scratch/tune_hybrid.py` — swept dense weight w ∈ [0,1] (synth personas, cached).
HitRate@10 at w=0.2: 0.198 / 0.054 / 0.049 vs w=0.3: 0.198 / 0.046 / 0.046.
w=0.2 maximises 2 domains, ties Yelp. Changed `0.3→0.2` in `database.py` and
`eval_harness.py`.

### 5 — Candidate-pool retrieval (sampled-metric protocol)
`python eval_harness.py --no-llm --persona-mode synth --candidate-pool {101,20} [--pop-distractors]`
NDCG@10 hybrid — pool 101 uniform: 0.390/0.329/0.324 · pool 101 pop-weighted:
**0.337/0.248/0.301** · pool 20 pop-weighted: 0.628/0.541/0.598. Used in paper
§4.4 to place retrieval against sampled-metric literature. Residual caveat:
random vs temporal split (no timestamps).
Artifacts: `scratch/evaluation_results_pool{101,20}{,pop}.json`.

### 6 — Multi-seed (42, 1, 7)  ★ CURRENT CANONICAL
`python eval_harness.py --seeds 42,1,7 --llm-sample 400 --bertscore --persona-mode synth`
RMSE mean±std — V0: 0.984±.027 / 1.035±.020 / 0.908±.031 · V1: 0.976±.021 /
0.942±.033 / 0.785±.009 · V2: **0.958±.028 / 0.937±.032 / 0.784±.011**.
ROUGE-L: 0.096±.001 / 0.083±.002 / 0.097±.002. Semantic-BGE: 0.740 / 0.632 / 0.663.
Retrieval hybrid@10 (w=0.2): 0.184±.012 / 0.040±.010 / 0.051±.004.
Paired V1−V2 per seed: Yelp +0.018 (all 3 +ve → **real**), Goodreads +0.005
(real, marginal), Amazon +0.002 (one seed 0.000 → **≈ zero**).
Artifact: `evaluation_results.{json,md}` (live).

### 7 — Data study: item-mean + variance buckets
`scratch/study_data.py`. user+item-mean blend RMSE (no LLM): **0.949 / 0.975 /
0.757** vs V1 1.004/0.985/0.773 — on Yelp this *beats* our LLM V2 (0.958).
V1 RMSE by user rating-variance bucket — Goodreads: low 0.38 / mid 0.87 /
high 1.21; Amazon: 0.57 / 0.76 / 1.12. Rating prediction is two populations:
solved for predictable users, hard for variable ones.
Finding: the item-bias term is untapped → 3-term calibration (planned T1a).

### 8 — 3-term calibration measurement (Tier 1a)
`analysis/measure_calib3.py` — sweeps `y = a·LLM + b·μ_user + c·μ_item` on the
cached warm LLM ratings (seeds 42/1/7, zero new GPU). V3 vs the deployed 2-term
V2: Yelp 0.958→0.910 (−5.0%), Goodreads 0.938→0.926 (−1.3%), Amazon
0.784→0.770 (−1.7%). Optimal weights: LLM **a ≈ 0** on every domain;
user/item b,c ≈ 0.52/0.47 (Yelp), 0.75/0.22 (Goodreads), 0.78/0.22 (Amazon).
Finding: warm rating prediction is best solved by a user-bias + item-bias
model; the LLM's numeric rating is redundant warm — its value is entirely in
cold-start (§4.5).

## Planned / pending

| ID | Experiment | Status |
|---|---|---|
| T1a | 3-term calibration `α·LLM + β·μ_user + γ·μ_item` | ✅ shipped — 3-term anchor in agent.py + paper §4.2 |
| T1b | Variance-bucketed V2-vs-V1 analysis (post-hoc from cache) | not started |
| T2 | Retrieval-augmented prompting — ablation vs static persona | not started |

## Reproduction

Every figure is regenerated by `python eval_harness.py <flags>`; the exact
flags are listed per experiment above. The LLM artifact cache
(`eval_artifacts/llm_cache.jsonl`) makes re-runs with an unchanged prompt free.
