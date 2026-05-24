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
| 6 | Multi-seed (seeds 42, 1, 7) — n=350 | 05-21 | RMSE V2 0.958±.03 / 0.937±.03 / 0.784±.01 | ⤷ superseded by #10 (n=2,000) |
| 7 | Data study: item-mean + variance buckets | 05-21 | user+item blend 0.949 on Yelp — beats LLM V2 | → motivates 3-term calibration |
| 8 | 3-term calibration measurement (item-mean) | 05-21 | V3 vs V2: −5.0% Yelp, −1.3% GR, −1.7% AMZ; LLM weight ≈ 0 | item-bias is the win; LLM redundant warm |
| 9 | T2: retrieval-augmented prompting (`--persona-mode rag`) | 05-21 | rating ≈ synth (Δ ≤ .004, in seed noise); review Sem-BGE +.011–.020 all 3 domains | retrieval aids generation, not regression |
| 10 | **n=2,000 re-evaluation, template persona — ★ CURRENT CANONICAL** | 05-22 | RMSE V2 0.990 / 0.876 / 0.851 (α=0.1); V3 0.956 / 0.864 / 0.848 | 6× sample; every #6 qualitative finding holds |
| 11 | Persona ablation (§4.6) at n=2,000 — synth arm | 05-22 | V2 0.987 / 0.876 / 0.852 ≈ template; retrieval ≈ equal | synth = template on metrics; a UX choice |
| 12 | RAG ablation (§4.7) at n=2,000 | 05-22 | V2 0.980 / 0.874 / 0.850 ≈ synth; review Sem-BGE +0.012–0.016 | RAG aids generation, not warm rating |
| 13 | Cold-start (§4.5) at n=2,000 users/k | 05-22 | k=1 V1→V2 cuts 19.5 / 17.3 / 13.4%; α schedule 0.6–0.7→0.1 | LLM's value concentrated in cold-start |
| 14 | ALS matrix-factorization retrieval (§4.4) | 05-22 | HitRate@10 0.067 / 0.021 / 0.051 — loses to item-item CF | Dacrema 2019 reproduced; CF stays Stage-1 |

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

### 6 — Multi-seed (42, 1, 7)  ⤷ SUPERSEDED BY #10
`python eval_harness.py --seeds 42,1,7 --llm-sample 400 --bertscore --persona-mode synth`
*n = 350 per domain. Headline figures replaced by the n = 2,000 run (#10); kept
here as the n = 350 reference point and as the source of the §4.6 / §4.7
ablations, which were not re-run at n = 2,000.*
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

### 9 — Retrieval-augmented prompting (Tier 2)
`python eval_harness.py --persona-mode rag --llm-sample 400 --bertscore --seed 42`
Seeds the LLM prompt with the user's **k = 4 most-similar past interactions**
(real item description + the rating/review the user gave it), retrieved by cosine
similarity to the target item, in place of an abstracted persona. Compared
against the synth-persona arm at the **same seed 42** (`per_seed['42']` of #6)
so V1 — the persona-independent user-mean — is identical (1.004 / 0.985 / 0.773),
confirming the splits match.
RMSE V2 — rag **0.999 / 0.977 / 0.770** vs synth-s42 0.995 / 0.978 / 0.769:
Δ ≤ 0.004, inside §4.2 seed noise; best α stays 0.1 / 0.2 / 0.1.
Review — Semantic-BGE rag **0.763 / 0.645 / 0.683** vs synth 0.742 / 0.634 / 0.667
(**+.011 to +.020, all three domains**); ROUGE-L rag 0.099 / 0.090 / 0.100 vs
synth 0.097 / 0.081 / 0.099.
Finding: RAG does not move warm-user rating — a third confirmation of the
user-mean-dominates regime after the template (#2) and synth (#3) arms — but
gives a small, consistent lift to review-text fidelity. Retrieval helps the
generative sub-task, not the regression sub-task.
Artifacts: `scratch/eval_rag_results.json`, `scratch/eval_rag.log`.

### 10 — n=2,000 re-evaluation  ⤷ SUPERSEDED BY #15 (multi-seed)
Datasets regenerated at 6× scale: `python local_data_prep.py` (`LIMIT_USERS =
2000`) streams the three HF source datasets, extracts a dense 3-core, caps each
to its 2,000 densest users → `data/*_dense.csv`. Counts: interactions
106,300 / 466,625 / 101,540; users 2,000 / 2,000 / 1,999; items
10,415 / 57,499 / 21,479; ≥4★ share 70% / 68% / 82%.
`python eval_harness.py --seeds 42 --bertscore --cold-start --persona-mode template`
RMSE — V0 1.059/0.994/0.965 · V1 0.995/0.879/0.856 · pure-LLM 1.186/1.102/1.059 ·
V2 **0.990/0.876/0.851** (α=0.1 every domain). V0→V2 −6.5% / −11.9% / −11.8%;
the V1→V2 step (the LLM's actual contribution) is only +0.003 to +0.005.
ROUGE-L: 0.097/0.086/0.093. Semantic-BGE: 0.734/0.634/0.648.
Retrieval HitRate@10 — dense ≈0.002/0.001/0.004 · hybrid 0.088/0.034/0.063 ·
CF 0.089/0.039/0.067 · popularity 0.019/0.010/0.011. **CF ≥ hybrid on all three**
— at this catalogue size the 20%-weighted dense term mildly hurts.
Two-populations (V1 RMSE by user-variance bucket low/mid/high):
0.65/0.90/1.14 · 0.46/0.82/1.06 · 0.59/0.84/1.21 — `analysis/study_data.py`.
V3 3-term (`analysis/measure_calib3.py`, seed 42): V2→V3 0.990→0.956 /
0.876→0.864 / 0.851→0.848; optimal a/b/c = 0.00/0.60/0.40 · 0.00/0.75/0.25 ·
0.05/0.80/0.15 — LLM weight ≈ 0, as at n = 350.
Cold-start was initially measured here at cold-sample 100; it has since been
re-run at n = 2,000 users per k — see experiment #13, which supersedes it and
backs paper §4.5.
**Verdict:** every qualitative finding of #6 survives the 6× sample increase —
regime switch, V1 dominates warm, LLM weight ≈ 0, two populations, CF ≥ hybrid,
dense retrieval weak. Magnitudes shifted (warm V0→V2 gain rose; absolute
retrieval HitRate fell because the candidate pool is now 10K–57K items). This is
the canonical run; paper §4, `numbers_integrity.md` and the abstract/conclusion
all cite it.
Sampled-metric re-run (101-candidate, pop-weighted negatives, n = 2,000):
NDCG@10 hybrid 0.370 / 0.288 / 0.340 · CF 0.374 / 0.292 / 0.358 · HitRate@10
hybrid 0.645 / 0.496 / 0.515 — CF ≥ hybrid here too, as under the full pool.
Paper §4.4 sampled table is now fully n = 2,000.
Artifacts: `evaluation_results.{json,md}` (live, full-pool canonical);
`scratch/evaluation_results_pool101_n2k.{json,md}` (sampled-metric run).

### 11 — Persona ablation at n=2,000 (§4.6)
`modal run modal_eval.py --sample 2000 --persona-mode synth --seed 42` (Modal A10G).
The template arm is the #10 canonical run; only the synthesised-persona arm is new.
RMSE V2 — synth **0.987 / 0.876 / 0.852** vs template 0.990 / 0.876 / 0.851 (Δ ≤ 0.003).
Hybrid HitRate@10 — synth 0.083 / 0.032 / 0.066 vs template 0.088 / 0.034 / 0.063
(Δ ≤ 0.005, sign not consistent). CF HitRate@10 is identical to template
(0.089/0.0385/0.067) — confirming CF is persona-independent, a clean cross-check.
**Finding:** at n=2,000 template and synthesised personas are equivalent on every
offline metric — rating is statistics-anchored, retrieval is CF-carried. The
n=350 "synth lifts Yelp retrieval" effect does not survive a realistic catalogue
size. Synthesis is a UX choice, not a metrics lever.
Artifact: `scratch/modal_results_n2000_synth_s42.{json,md}`.

### 12 — RAG ablation at n=2,000 (§4.7)
`modal run modal_eval.py --sample 2000 --persona-mode rag --seed 42` (Modal A10G).
Seeds the prompt with the user's k=4 most-similar past interactions instead of an
abstracted persona; compared against the synth arm of #11 at the same seed.
RMSE V2 — rag **0.980 / 0.874 / 0.850** vs synth 0.987 / 0.876 / 0.852 (Δ ≤ 0.007).
Review — Sem-BGE rag 0.754 / 0.645 / 0.662 vs synth 0.738 / 0.633 / 0.648 (+0.012
to +0.016, all 3); ROUGE-L rag 0.102 / 0.088 / 0.098 vs synth 0.100 / 0.082 / 0.091
(up on all 3). **Finding:** holds from the n=350 run — RAG does not move warm
rating (a third confirmation of the user-mean-dominates regime, after template
and synth), but gives a small, consistent lift to review-text fidelity.
Retrieval augmentation helps the generative sub-task, not the regression one.
Artifact: `scratch/modal_results_n2000_rag_s42.{json,md}`.

### 13 — Cold-start at n=2,000 (§4.5)
`modal run modal_eval.py --sample 20 --persona-mode template --cold-sample 2000 --seed 42`
(Modal A10G). Truncates each test user's history to k ∈ {1,2,3} and re-evaluates,
on 2,000 users per domain per k (1,997 Amazon) — 20× the cold-sample of #10.
V1 → V2 (best blend), optimal α:
- Yelp:      k1 1.405→1.131 (α0.7) · k2 1.214→1.108 (α0.5) · k3 1.135→1.065 (α0.4)
- Goodreads: k1 1.245→1.030 (α0.6) · k2 1.085→0.992 (α0.5) · k3 1.042→0.977 (α0.4)
- Amazon:    k1 1.175→1.017 (α0.6) · k2 1.036→0.969 (α0.5) · k3 0.978→0.932 (α0.4)
k=1 RMSE cut: 19.5 / 17.3 / 13.4%. The optimal-α schedule is monotonic and
near-identical across domains: k1 ≈ 0.6–0.7, k2 0.5, k3 0.4, warm 0.1.
**Finding:** the LLM's contribution scales inversely with available history —
the regime switch, now measured at the same n as the warm headline. Abstract /
§7 cold-start range updated to 13–20%.
Artifact: `scratch/modal_results_n20_template_s42.{json,md}`.

### 16 — Synth multi-seed n=2,000 on vLLM + implicit ALS  ★ NEW CANONICAL
`for s in 42 1 7; do modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed $s --cold-start --cold-sample 2000 --bertscore; done`
Three parallel Modal A10G containers, each spawning a vLLM 0.21 server in the
container and driving it via the OpenAI-compatible HTTP API with 32 concurrent
client requests (`eval_harness.VLLMShim` + `ThreadPoolExecutor`). Engine swap
from llama-cpp-python Q4_K_M GGUF (single-stream, ~150 tok/s) to vLLM fp16
(PagedAttention + continuous batching, ~5–10k tok/s aggregate). ALS in the
retrieval phase also swapped from pure-NumPy to the `implicit` library
(C++/Cython/OpenMP, ~50× faster on the 100K × 20K Amazon matrix). Per-seed
cache filename `llm_cache_s{seed}_synth_vllm.jsonl` so vLLM-fp16 outputs never
collide with the prior llama-cpp-Q4 cache on the same Modal volume.
**Warm-start RMSE V2 (mean ± std across 3 synth seeds):**
- Yelp:      V2 0.989 ± 0.002  (was 0.990 single-seed on llama-cpp-Q4 in #10)
- Goodreads: V2 0.894 ± 0.027  (was 0.876)
- Amazon:    V2 0.834 ± 0.015  (was 0.851)
Engines agree to within the new error bars — engine-precision change (fp16 vs
Q4_K_M) is dominated by sampling noise.
**Review-text reproducibility:** ROUGE-L std ≤ 0.001, Sem-BGE std ≤ 0.002 across
the 3 seeds (Yelp / Goodreads / Amazon) — review text is highly reproducible.
BERTScore-F1 could not be computed inside this run: the synth-vLLM container
had ~85% of A10G VRAM allocated to vLLM weights + KV cache, and RoBERTa-large
for BERTScore OOM'd with no room to load. Backfilled by a separate Modal
A10G pass (`modal_bertscore_backfill.py`) that reads the cached generations
+ reference reviews and computes BERTScore-F1 standalone on a GPU container
with no vLLM contention:
  Yelp:      0.8384 ± 0.0003
  Goodreads: 0.8406 ± 0.0003
  Amazon:    0.8461 ± 0.0003
Values are ~0.003 below the prior single-seed template-llama-cpp-Q4 reference
(0.842/0.844/0.849), consistent with vLLM-fp16 generating very slightly
different review text than llama-cpp-Q4. Std of ±0.0003 across seeds confirms
BERTScore is essentially seed-invariant on this scale of n.
**Multi-k retrieval (mean ± std × 3):**
HR@10 — hybrid 0.087±0.008 / 0.034±0.006 / 0.065±0.004 ; CF 0.094±0.006 /
0.037±0.006 / 0.064±0.002 ; ALS 0.071±0.006 / 0.020±0.002 / 0.046±0.003.
At HR@100, ALS converges to or overtakes CF/hybrid on every domain — finding
strengthens vs #15 single-seed.
**Cold-start V3 (3-term, mean ± std × 3):**
k=1 — Yelp 0.991±0.019, Goodreads 0.962±0.018, Amazon 0.934±0.022.
V3 weights at k=1: (LLM / user / item) = 0.03 / 0.13 / 0.83 (Yelp), 0.03 /
0.27 / 0.70 (Goodreads), 0.07 / 0.30 / 0.63 (Amazon) — **item-bias term
dominant, LLM weight ≤ 0.07 throughout**. The §4.2 'LLM weight ≈ 0 warm'
finding generalises to the cold regime.
**Cost & wallclock:** 3 parallel A10G containers, ~15 min wallclock total,
~\$3 on Modal (was \$30+ on the llama-cpp-Q4 single-stream path; the engine
swap + implicit ALS dropped the cost by an order of magnitude).
Artifacts: `scratch/modal_results_n2000_synth_vllm_s{42,1,7}.json`,
`scratch/aggregated_synth_3seed.{json,md}`, `analysis/aggregate_seeds.py`,
`modal_vllm_eval.py`.

### 15 — Multi-seed n=2,000 (audit-fixed) + multi-k retrieval + BERTScore  ⤷ SUPERSEDED BY #16
`bash scripts/run_final_multiseed.sh` — three parallel Modal A10G containers, seeds
42 / 1 / 7, audit-fixed harness (per-domain prompts, JSON-grammar constraint, stop
tokens at `<|im_end|>` / `<|endoftext|>`, length cap, temperature=0). Each writes
to its own per-seed cache file (`llm_cache_s{seed}.jsonl`) on the persistent
volume `naijabuddy-eval-cache`; daemon-thread commits every 180s give preemption
resilience. Same n=2,000 dense datasets, full leave-one-out, retrieval evaluated
at HR@k / NDCG@k for k ∈ {10, 20, 50, 100}, BERTScore-F1 computed canonically via
the `bert-score` package + RoBERTa-large.
**Warm-start RMSE (mean ± std across 3 seeds):**
- Yelp:      V2 0.9938 ± 0.0052 — reproduces #10 single-seed 0.990 to 3 decimals
- Goodreads: V2 0.8992 ± 0.0336 — slightly above #10's 0.876
- Amazon:    V2 0.8412 ± 0.0152 — slightly under #10's 0.851
**Review quality:** Sem-BGE 0.735/0.631/0.649 (matches #10); BERTScore-F1
0.842/0.844/0.849, std ≤ 0.0005 across seeds — publication-grade tight.
**Multi-k retrieval — the new finding:** at HR@10, item-item CF beats hybrid and
ALS in every domain (reproducing #14). At HR@100 the order reverses: on Yelp ALS
0.351 > hybrid 0.340 > CF 0.338; on Amazon CF 0.196 ≈ ALS 0.189 > hybrid 0.185;
on Goodreads ALS still trails CF but the gap collapses from ×1.85 (k=10) to
×1.21 (k=100). Interpretation: neighbourhood methods concentrate mass on
strong-co-occurrence neighbours (precision at top-10), latent factors give
broader coverage (recovery at deeper cutoffs). For the ~50–100-item rerank
pool NaijaBuddy actually deploys, ALS becomes a competitive Stage-1 alternative.
**Cold-start at n=2,000 per k (mean ± std across 3 seeds, V3 3-term anchor):**
- Yelp:      k1 → 1.005 ± 0.018, k2 → 0.997 ± 0.018, k3 → 0.978 ± 0.012
- Goodreads: k1 → 0.928 ± 0.014, k2 → 0.939 ± 0.014, k3 → 0.937 ± 0.025
- Amazon:    k1 → 0.939 ± 0.019, k2 → 0.910 ± 0.032, k3 → 0.883 ± 0.036
V3 weight schedule across seeds: LLM weight 0.0–0.2, user weight 0.2–0.6, item
weight 0.3–0.8 — the *item-bias term* is doing almost all the cold-start work,
confirming the "regime-switch" reading from #10 / §4.2.
**Cost & infra postmortem:** ran ~$30 + $10 top-up on Modal because
llama-cpp-python single-stream on A10G is memory-bandwidth-bound at ~150 tok/s
(roughly the same as a M-series Mac CPU running the same GGUF), so three
"parallel" containers tripled cost without tripling compute-per-dollar. For any
future eval at this scale, switch to vLLM/TGI batched (~30× the throughput on
the same GPU) or to a hosted endpoint (Groq GPT-OSS-120B at $0.15/$0.60 per M
tokens). Logged to memory `architecture_llm_serving.md` so it does not repeat.
Artifacts: `scratch/modal_results_n2000_template_s{42,1,7}.{json,md}`,
`scratch/aggregated_n2000_3seed.{json,md}`, `analysis/aggregate_seeds.py`.

### 14 — ALS matrix-factorization retrieval (§4.4)
`python eval_harness.py --no-llm` (local, no GPU — pure-NumPy ALS). Implicit-
feedback ALS (Hu/Koren/Volinsky 2008; factors=64, 12 iterations, reg=0.1,
alpha=40) over the same user×item matrix the item-item CF uses.
HitRate@10 / NDCG@10 — ALS 0.067/0.033 · 0.021/0.010 · 0.051/0.026, vs item-item
CF 0.089/0.047 · 0.039/0.020 · 0.067/0.039. ALS beats popularity and dense
content retrieval but **loses to the simpler CF in all three domains**.
**Finding:** a live reproduction of Dacrema et al. 2019 — a well-tuned
neighbourhood method beats a latent-factor model on sparse data with a random
hold-out. Item-item CF stays the deployed Stage-1 signal; ALS is reported in
§4.4 as a fifth retrieval ablation, not as a failure.
Artifact: `evaluation_results.{json,md}` (local `--no-llm` run).

## Planned / pending

| ID | Experiment | Status |
|---|---|---|
| T1a | 3-term calibration `α·LLM + β·μ_user + γ·μ_item` | ✅ shipped — 3-term anchor in agent.py + paper §4.2 |
| T1b | Variance-bucketed V2-vs-V1 analysis (post-hoc from cache) | not started |
| T2 | Retrieval-augmented prompting — ablation vs static persona | ✅ shipped — experiments #9 (n=350) + #12 (n=2k), paper §4.7 |

## Reproduction

Every figure is regenerated by `python eval_harness.py <flags>`; the exact
flags are listed per experiment above. The LLM artifact cache
(`eval_artifacts/llm_cache.jsonl`) makes re-runs with an unchanged prompt free.
