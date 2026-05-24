# NaijaBuddy: A Highly Localized, Offline-First Agentic Recommender & User Modeling System for Cross-Domain Behavioral Simulation

**Authors**: Team NaijaBuddy (Hackathon Submission)  
**Affiliation**: DSN x BCT Data & AI Summit Hackathon 3.0  
**Date**: May 2026  

---

## Abstract
Traditional recommender systems rely heavily on static user matrices, neglecting contextual nuance and natural language behavior. Recent advances in Large Language Models (LLMs) have enabled generative recommendation and behavioral simulation; however, these models often suffer from strong "positivity bias" in numerical rating generation, high computational latency, and complete dependence on costly, cloud-bound APIs. 

In this paper, we present **NaijaBuddy**, an offline-capable, containerized agentic recommendation and user modeling system tailored specifically for the Nigerian consumer market. Our architecture employs a dual-stage pipeline: (1) high-speed dense vector semantic search using `BAAI/bge-small-en-v1.5` over a hybrid SQLite catalog enriched with local establishments, movies, and literature; and (2) in-context reranking and review generation by `Qwen-2.5-3B-Instruct`, served interchangeably via in-process `llama-cpp-python` (Q4_K_M GGUF, the offline-mode fallback) or vLLM 0.21 on a dedicated GPU (the engine used by the canonical multi-seed evaluation and the live hosted demo) — both paths share the same prompts, parsing, and calibration code via a thin OpenAI-protocol shim. To protect rating accuracy (RMSE), we implement a mathematical **Output Calibration Layer** that blends LLM estimates with user and item statistical anchors and uses a vector-based **Cluster-Mean** fallback for cold-start personas. Finally, a deterministic **Collaborative Critic Layer** applies rule-based enforcement of encoded behavioral constraints. Our leakage-free, out-of-sample evaluation reports **mean ± standard deviation across three independent leave-one-out splits** (seeds 42 / 1 / 7) at n = 2,000 users per seed per domain. Per-user calibration is a reliable RMSE anchor (a 6.3–13% reduction over a global-mean baseline depending on domain), and the optimal LLM/statistics blend is *adaptive*: the LLM's raw rating adds little for warm users with rich history (V1→V2 step of 0.004–0.011 RMSE), but the three-term blend including item-bias cuts cold-start RMSE by **20–30% at k = 1** over the user-mean baseline. A cross-domain transfer experiment (Amazon Books → Movies) shows the same item-bias signal rescues predictions even when the user is fully cold in the target domain. We release the evaluation harness so that every reported figure reproduces.

---

## 1. Introduction
User-modeling and recommendation systems have moved from matrix factorization toward semantic, conversational agents. But deploying these in resource-constrained settings — edge nodes, or servers in emerging markets — raises three challenges:

1. **Network & cloud dependence.** Cloud-LLM systems rely on external APIs, exposed to outages, card-authorization failures, and recurring cost.
2. **LLM rating volatility.** Asked for a 1–5 star rating, LLMs show a pronounced positivity bias and round toward integer extremes, inflating RMSE.
3. **Cultural fidelity.** Global foundation models lack the linguistic nuance to simulate authentic Nigerian responses, often sounding dry or academic.

We introduce **NaijaBuddy**, a unified agentic recommender and review simulator that ships as a single Docker image, exposing both an interactive web UI and a REST API. Two deployment modes share the same code: an **offline mode** with `Qwen2.5-3B-Instruct` served in-process via `llama-cpp-python` (Q4_K_M GGUF, no network at inference time), and a **vLLM mode** in which the same model — the fp16 HF safetensors — is served by vLLM 0.21 on a dedicated GPU and the container talks to it over an OpenAI-protocol HTTP client. The vLLM mode is what the canonical multi-seed evaluation and the live hosted demo actually run; the offline mode is the reproducible fallback that anyone can spin up without GPU infrastructure (§2.2.1). Our contributions:

* **A filter-then-rerank pipeline** — dense cosine recall over 384-d embeddings, then an in-context `Qwen2.5-3B-Instruct` reranker — that keeps context and latency low enough for the small, CPU- or single-GPU-deployable model class we target.
* **A three-term calibration layer** that blends the LLM rating with user-mean and item-mean statistical anchors ($\alpha\cdot\text{LLM} + \beta\cdot\mu_u + \gamma\cdot\mu_i$), with a vector-neighbourhood cluster-mean fallback for cold-start users. Weight migration across history regimes — item-bias dominant when cold, user-mean dominant once warm — is the key empirical finding.
* **A reproducible multi-seed evaluation harness** — three independent leave-one-out splits at n = 2,000 per domain, multi-cutoff retrieval (HR/NDCG @ 10/20/50/100), and a cross-domain Books→Movies transfer experiment that stress-tests the cold-start finding outside the within-domain setting.
* **Nigerian localization** — a seeded local catalogue (Lagos eateries, Nollywood films, African literature) and persona-grounded prompts ("Strict Nigerian Dad", "VI Tech Bro", "Lagos Gen-Z Influencer").

---

## 2. Proposed Methodology (NaijaBuddy Architecture)

NaijaBuddy's backend engine is built in Python using FastAPI, SQLite and NumPy, with the LLM served via either `llama-cpp-python` (offline mode) or an OpenAI-protocol HTTP client pointed at a vLLM endpoint (vLLM mode — see §2.2.1). The architecture consists of four distinct, sequential layers:

```
User persona / query
  │
  ├─ Layer 1 · Recall      BGE-small dense search + item-item CF over the
  │                        hybrid SQLite catalogue → top-10 candidates
  ├─ Layer 2 · Rerank      Qwen2.5-3B-Instruct (llama-cpp GGUF or vLLM
  │                        safetensors — see §2.2.1): in-context persona
  │                        modelling, pairwise sort, review generation
  ├─ Layer 3 · Calibrate   blend the LLM rating with the user/item statistical
  │                        anchor (cluster-mean fallback for cold-start)
  └─ Layer 4 · Critic      deterministic rule filter — constraint-violating
                           items pushed to the bottom
  │
  ▼
Final calibrated ratings, reviews & recommendations
```

### 2.1 Layer 1: Hybrid Catalog & Dense Semantic Recall
To deliver recommendations across multiple domains (Yelp, Amazon, Goodreads), we design a unified SQLite schema. The database is populated with an extensive, highly localized catalog spanning three distinct categories:
* **Yelp (Food & Spots)**: Iconic local spots (e.g., *Yellow Chilli*, *Shiro Lagos*, *The Place*, *Suya Spot*, *Club Quilox*) and traditional culinary items (Jollof Rice, Suya, Pepper Soup).
* **Amazon (Literature & Media)**: A real-world subset of Amazon Books reviews, augmented with a hand-curated localized overlay of landmark Nollywood productions (e.g., *The Wedding Party*, *King of Boys*, *Aníkúlápó*) and popular consumer electronics.
* **Goodreads (African Literature)**: High-caliber African and Nigerian literary masterpieces (e.g., *Things Fall Apart* by Chinua Achebe, *Half of a Yellow Sun* by Chimamanda Ngozi Adichie).

We leverage **`BAAI/bge-small-en-v1.5`** to map both items and user personas into a shared 384-dimensional dense semantic space. When a recommendation is requested, the system computes the user persona's cosine similarity against all item vectors in the target domain using a direct cosine similarity routine implemented in pure Python:

$$\text{Sim}(u, i) = \frac{\mathbf{v}_u \cdot \mathbf{v}_i}{\|\mathbf{v}_u\| \|\mathbf{v}_i\|}$$

The top 10 candidate items are retrieved and routed to the next layer. This "Filter-then-Rerank" paradigm prevents passing hundreds of items to the LLM, protecting local CPU execution and preventing out-of-memory crashes.

### 2.2 Layer 2: LLM Reranking & In-Context Persona Modeling (Tasks A & B)
At the core of the reranking and review simulation is a small Large Language Model: **`Qwen2.5-3B-Instruct`**, served via either of the two interchangeable inference backends described in §2.2.1. Both backends present the same `Llama()`-compatible callable to the upstream agent code so prompts, JSON-schema constraints, stop tokens and the calibration layer are bit-identical across modes.

#### 2.2.1 Deployment modes (offline vs vLLM)
The agent picks its inference engine at startup from the `VLLM_URL` environment variable:

* **Offline mode** (default when `VLLM_URL` is unset): the **`Qwen2.5-3B-Instruct-Q4_K_M`** GGUF is loaded into the container process via **`llama-cpp-python`**, with full GPU offload + FlashAttention-2 when CUDA is available and a CPU baseline fallback otherwise. No network traffic at inference time. This is the mode anyone running `docker run` against the public image without extra configuration enters, and it is the cheapest reproducibility path for the calibration and retrieval results.
* **vLLM mode** (set `VLLM_URL` to an OpenAI-compatible endpoint): the agent constructs a thin `VLLMShim` ([`vllm_shim.py`](vllm_shim.py)) that wraps an `openai.OpenAI` client and exposes the same callable signature `llama_cpp.Llama` does. Generation requests then proxy to a separately-deployed vLLM 0.21 server serving the **fp16 HF safetensors** of the *same* `Qwen/Qwen2.5-3B-Instruct` weights — same prompts, same stop tokens, same JSON-grammar constraint (mapped to vLLM's `guided_json`).

The **canonical multi-seed evaluation** (§4) runs in vLLM mode on a Modal-hosted A10G (`modal_vllm_eval.py`) — vLLM's PagedAttention + continuous batching delivers ~10× throughput on the 18,000-call multi-seed sweep, which is what makes the three-seed protocol affordable. The **live hosted demo** also runs in vLLM mode against a long-running Modal endpoint (`modal_vllm_serve.py`) so demo and paper-evaluation share the same engine config end to end. Engine-difference disclosure: vLLM serves fp16/bf16, llama-cpp serves Q4_K_M, so outputs are not bit-identical even at greedy decoding; §4.3 reports the measured engine gap on V2 rating accuracy (Yelp 0.990 / Goodreads 0.876 / Amazon 0.851 on llama-cpp vs the multi-seed vLLM numbers in §4.2) — the two engines agree within the per-seed sampling std, so all calibration and retrieval claims in this paper hold under both modes.


#### Task A: Review & Rating Simulation
To synthesize realistic reviews, the agent is supplied with the target item details and a rich description of the active persona. The model is bound by strict system-level instructions:
1. **Conformity**: Generate a star rating (float) and a written review reflecting the target persona's taste.
2. **Conciseness**: Restrict the output to 2–4 sentences to prevent long, repetitive generations and optimize token usage.
3. **Structured Format**: Output a raw, valid JSON schema to allow seamless programmatic parsing.

#### Task B: Recommendation Engine (Rerank)
The top 10 candidates retrieved by semantic search are represented as a serialized JSON catalog within the LLM context. The LLM is instructed to:
* Re-sort the candidates by assessing the pairwise alignment between the user’s detailed persona and the item descriptions.
* Generate a compelling, short (1-2 sentences) natural language explanation detailing *why* the item was selected for this specific user.

### 2.3 Layer 3: Output Calibration (The Mathematical Defense)
LLMs exhibit a structural "positivity bias" and numerical instability when scoring items, which inflates the Root Mean Squared Error (RMSE) in benchmarks. To resolve this, we implement a **Mathematical Calibration Layer** to anchor and stabilize the raw LLM output:

$$\hat{y}_{u, i} = \alpha \cdot \text{LLM}_{u, i} + (1 - \alpha) \cdot \mu_u$$

Where:
* $\hat{y}_{u, i}$ is the final calibrated rating predicted for user $u$ on item $i$.
* $\text{LLM}_{u, i}$ is the raw rating generated by the local LLM.
* $\mu_u$ is the baseline anchor rating for the user.
* $\alpha \in [0, 1]$ is a tunable hyperparameter governing the model's reliance on LLM reasoning versus historical baseline statistics. Our leakage-free held-out $\alpha$-sweep (Sections 4.2 and 4.5) finds the RMSE-optimal $\alpha$ is *not* a constant: it is $\approx 0.1$–$0.2$ for users with rich rating history, where the user-mean is already a strong anchor, and rises to $\approx 0.6$–$0.8$ for cold-start users, where the LLM's persona-grounded estimate becomes the dominant reliable signal. The layer is therefore an adaptive fusion whose weight on the LLM should scale inversely with the evidence available about a user.

#### Cold-Start Neighborhood Fallback
For new or cold-start users, historical rating averages ($\mu_u$) are unavailable. Instead of reverting to a generic global average (which dilutes personalized accuracy), we leverage our vector index to retrieve the top $K = 5$ users with the most semantically similar personas:

$$\mu_u^{\text{cold}} = \frac{1}{K} \sum_{j \in \text{NN}(u)} \mu_j$$

Where $\text{NN}(u)$ represents the $K$-nearest neighbor personas of user $u$ determined by cosine similarity. This ensures that the calibration baseline remains highly relevant, reflecting the behavior of similar sub-communities.

### 2.4 Layer 4: Collaborative Critic Layer (Deterministic Verification)
Following a broader trend of pairing LLM recommenders with deterministic critic and verification layers, we implement a deterministic **Critic Layer** over the LLM outputs. Even advanced LLMs sometimes hallucinate or ignore negative constraints in prompts (e.g., recommending a spicy beef dish to a strict vegetarian). 

Our Critic Layer programmatically inspects the ranked list and applies strict rule-based filters. If a candidate item contains features that violate the user's declared constraints (e.g., alcohol for a non-drinking persona, or extremely loud spaces for an introverted parent), the Critic applies a penalty:

$$\text{Rank}_{\text{final}}(i) = \begin{cases} \text{Rank}_{\text{LLM}}(i) & \text{if } \text{Penalty}(i) = \text{False} \\ \text{Rank}_{\text{LLM}}(i) + 100 & \text{if } \text{Penalty}(i) = \text{True} \end{cases}$$

This deterministically demotes any item matching an encoded constraint rule below every non-matching item. We describe this as **rule-based enforcement** rather than a safety *guarantee*: it is verifiable by construction for the constraints it encodes, but — being keyword-based — makes no claim to catch violations outside those rules. A learned or semantic critic is left to future work.

---

## 3. Cultural Context & Nigerian Localization
Behavioural and cultural fidelity is a primary judging criterion. NaijaBuddy ships three hand-crafted Nigerian base personas:

1. **Kunle (VI Tech Bro)** — a high-earning Victoria Island software engineer; premium minimalist aesthetics, high-end cafés, startup jargon (*"premium clean vibes"*, *"no cap"*).
2. **Chief Okeke (Strict Dad)** — a retired, conservative headmaster; price-sensitive, highly critical, values quiet and moral substance (*"waste of hard-earned money"*, *"what our children call modern"*).
3. **Teni (Lagos Gen-Z Influencer)** — a fashion and lifestyle creator; obsessed with aesthetics, instagrammable spots and social events (*"Omo!"*, *"it's giving"*, *"God when?!"*).

Structuring prompts around these archetypes makes the generated reviews and justifications sound natural and locally grounded rather than generically global.

---

## 4. Experiments and Evaluation

We evaluate NaijaBuddy with a leakage-free, out-of-sample protocol and report the measured results, including negative findings. Every figure in this section is regenerated by the evaluation harness shipped with the code (`eval_harness.py`).

### 4.1 Protocol

**Datasets.** Three real-world interaction sets, each filtered to a genuine **3-core** — every user and every item has at least three interactions — and capped at the 2,000 densest users per domain: Yelp (106,300 interactions / 2,000 users / 10,415 items), Goodreads (466,625 / 2,000 / 57,499), and Amazon (101,540 / 1,999 / 21,479). The Amazon subset is drawn from the Books category, so cross-domain coverage is best read as *local businesses* (Yelp) *vs. literature* (Goodreads, Amazon). All three sets are positively skewed, as consumer-review data invariably is — the share of ratings ≥ 4★ is 70% (Yelp), 68% (Goodreads) and 82% (Amazon) — which makes the global-mean baseline (V0) deceptively strong and is the reference against which every gain below should be read.

**Leave-one-out split.** For each user we hold out one interaction. All training statistics — each user's mean rating, every item description, and every persona — are computed from that user's *remaining* interactions only. This removes the circular evaluation in which a prediction is scored against a rating that itself contributed to the mean used to produce it.

**Metrics.** RMSE for rating prediction; ROUGE-L, Semantic-BGE (whole-review BGE-small cosine) and BERTScore-F1 (RoBERTa-large token-level F1, Zhang et al., 2020) for the generated review; HitRate@k and NDCG@k for k ∈ {10, 20, 50, 100} for retrieval. The headline figures below are reported as **mean ± sample standard deviation across three independent leave-one-out splits** (seeds 42 / 1 / 7), with n = 2,000 / 2,000 / 1,999 held-out interactions per seed per domain. The headline evaluation uses **synthesised** personas (matching the deployed system) served by vLLM 0.21 (fp16, continuous batching with PagedAttention); §4.6 confirms template-vs-synth equivalence on the offline metrics. A discarded small-sample run (n = 10) once reported a Goodreads blend RMSE of 0.60 that the full evaluation corrects to ~0.88 — a cautionary illustration of sampling variance, and the reason for the multi-seed full-set protocol.

**Cold-start protocol.** A 3-core leaves no genuinely cold users (the Goodreads minimum is 17 interactions), so we *simulate* cold-start: a test user's history is truncated to k ∈ {1, 2, 3} interactions while every other user keeps full history — a new user entering a populated system — and the evaluation is repeated. Section 4.5 reports the resulting degradation curve.

### 4.2 Rating Prediction and the Calibration Layer

**Per-user anchoring.** Replacing the global mean (V0) with the per-user training mean (V1 — the calibration formula at $\alpha = 0$) reduces RMSE in every domain; blending the LLM's raw rating back in at a low weight (V2) improves on V1 further, by a smaller and domain-dependent margin:

| Domain | V0 global | V1 user-mean ($\alpha$=0) | pure-LLM ($\alpha$=1) | V2 blend (best $\alpha$) |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 1.055 ± 0.003 | 1.000 ± 0.004 | 1.226 ± 0.011 | **0.989 ± 0.002** ($\alpha$=0.2) |
| Goodreads | 1.008 ± 0.023 | 0.898 ± 0.025 | 1.148 ± 0.040 | **0.894 ± 0.027** ($\alpha$=0.1) |
| Amazon | 0.953 ± 0.012 | 0.839 ± 0.016 | 1.105 ± 0.002 | **0.834 ± 0.015** ($\alpha$=0.1) |

*Full held-out set, n = 2,000 / 2,000 / 1,999 per seed; **three independent leave-one-out splits** (seeds 42 / 1 / 7), synth personas, served by vLLM-batched Qwen2.5-3B-Instruct. Mean ± sample std reported. V0→V2 reduces RMSE by 6.3% / 11.3% / 12.5% on average — but the V1→V2 step, the LLM's actual contribution, is only 0.004–0.011 across domains, with a per-seed std of 0.002–0.027.*

**Does the LLM's rating help?** Because the raw LLM rating is independent of $\alpha$, one LLM call per held-out pair supports a full sweep of the blend $\hat{y} = \alpha\cdot\text{LLM} + (1-\alpha)\cdot\mu_u$. The sweep is **U-shaped** in every domain: error falls from V1 to a shallow minimum at $\alpha = 0.1$–$0.2$, then rises steeply to the pure-LLM endpoint — the worst configuration everywhere (1.10–1.23). The V1→V2 improvement is 0.004–0.011 RMSE; at three independent splits of n = 2,000 that is a small but stable effect — the cross-seed std on V2 is 0.002–0.027, so the V1→V2 gap consistently survives reseeding. The reported V2 sits at the *test-minimising* point of the sweep — a descriptive minimum, not a validation-tuned hyperparameter — so if anything the verdict is conservative: even oracle-optimal blending barely moves the user-mean.

So for warm users the LLM's numeric rating earns a marginal gain at best — a 3B model's raw 1–5 score, compressed toward integer extremes, barely beats the user's own historical mean. Section 4.5 shows why that is only half the story: the LLM's contribution is small *for users with rich history* and large *for users without it*.

**Two populations of users.** Averaged over everyone the warm gain looks small — but the average hides structure. Bucketing test users by the spread of their own training ratings (per-user standard deviation) shows that rating prediction is really *two problems*:

| Domain | low-σ users (σ<0.6) | mid-σ (0.6–1.0) | high-σ users (σ≥1.0) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.65 | 0.90 | 1.14 |
| Goodreads | 0.46 | 0.82 | 1.06 |
| Amazon | 0.59 | 0.84 | 1.21 |

*V1 (user-mean) RMSE by per-user rating-variance bucket; n = 2,000 per domain, `analysis/study_data.py`.*

The pattern is stark and monotonic in every domain: for users who rate consistently, predicting their mean is near-solved (RMSE 0.46–0.65); for users whose ratings are scattered, the *same* predictor is roughly **twice as bad** (1.06–1.21) — and much of that gap is genuinely irreducible variance that no model can recover. This is why the headline warm gain looks small — it is diluted across a predictable majority that nothing can improve. The modelling budget, the LLM, is best spent on the high-variance tail and on cold users — exactly the regime split that §4.5 formalises.

**An item-bias term.** The blend so far anchors only on the *user's* mean. Classical recommender systems also model an *item* bias — some items are simply rated higher than others. Adding it, $\hat{y} = \alpha\cdot\text{LLM} + \beta\cdot\mu_u + \gamma\cdot\mu_i$, and sweeping the weights on the cached generations (no new inference; `analysis/measure_calib3.py`):

| Domain | V2 (user anchor) | V3 (+ item bias) | optimal weights (LLM / user / item) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.990 | **0.956** (−3.5%) | 0.00 / 0.60 / 0.40 |
| Goodreads | 0.876 | **0.864** (−1.4%) | 0.00 / 0.75 / 0.25 |
| Amazon | 0.851 | **0.848** (−0.3%) | 0.05 / 0.80 / 0.15 |

The item term is a real mean RMSE gain (−1.7% on average, −3.5% on Yelp). The striking number is the LLM weight: at the optimum it is **≈ 0 on every domain** (0.00–0.05). Once the anchor models item bias, the 3B model's raw numeric rating is *redundant for warm users* — the best warm predictor is the textbook user-bias + item-bias model. (As in §4.2, these weights are the descriptive minimum of a sweep; the deployed system uses the fixed per-domain anchor.) NaijaBuddy deploys this 3-term anchor. The LLM's numeric rating earns its place only in cold-start (§4.5), where neither user nor item history exists. The calibration layer is thus best read as a **regime switch**: a statistical bias model for well-observed users, the LLM for cold ones.

**Robustness across seeds and sample size.** The warm-user result above — that the LLM's rating barely improves on the user-mean — has been stress-tested two independent ways. The canonical evaluation reported above runs three independent leave-one-out splits (seeds 42 / 1 / 7) at n = 2,000 per seed; the V1→V2 step is positive on **every (seed, domain) cell** with a per-seed std of just 0.002–0.027 RMSE — a small but stable effect that survives reseeding cleanly. An earlier protocol at n = 350 across the same three seeds applied a *paired per-seed difference test* to the V1→V2 step and found it positive on all three seeds for Yelp (mean +0.018) and Goodreads (mean +0.005) — small but consistent — and indistinguishable from zero on Amazon (+0.002). However the data is resampled — three seeds at n = 350 or three seeds at n = 2,000 — the LLM's warm contribution stays within a few thousandths of an RMSE point. We offer that stability across a 6× sample-size change *and* across three independent splits, rather than a single error bar, as the evidence that the finding is real and not a sampling artefact.

*Engine note.* The multi-seed numbers above are produced by Qwen2.5-3B-Instruct served via vLLM 0.21 (fp16, continuous batching with PagedAttention). An earlier single-seed evaluation served the same model via llama-cpp-python (Q4_K_M GGUF) and produced near-identical V2 numbers (Yelp 0.990, Goodreads 0.876, Amazon 0.851 single-seed). The two engines agree to within the per-seed std reported above; differences are dominated by sampling noise, not by precision.

### 4.3 Review Generation

We score the generated review against the held-out human review along three complementary axes — surface-form overlap, sentence-level meaning, and token-level semantic precision/recall:

| Domain | ROUGE-L F1 | Semantic-BGE | BERTScore-F1 |
| :--- | :---: | :---: | :---: |
| Yelp | 0.084 ± 0.001 | 0.723 ± 0.001 | **0.838 ± 0.000** |
| Goodreads | 0.073 ± 0.001 | 0.626 ± 0.002 | **0.841 ± 0.000** |
| Amazon | 0.079 ± 0.000 | 0.646 ± 0.001 | **0.846 ± 0.000** |

*All three axes: n = 2,000 / 2,000 / 1,999 per seed across three independent splits (seeds 42 / 1 / 7), synth personas. Mean ± sample std (rounded to three decimals). BERTScore-F1 backfilled by a separate GPU pass (`modal_bertscore_backfill.py`) because the original synth-vLLM run had vLLM holding 85% of the A10G's VRAM, leaving no room for RoBERTa-large; the backfill reads the same cached generations and reference reviews and computes the canonical metric. Std on BERTScore is ±0.0003 — essentially seed-invariant, matching the high reproducibility we see on the other two axes.*

ROUGE-L — verbatim longest-common-subsequence overlap against a single reference — is low, as it must be: two people reviewing the same item rarely choose the same words. The Semantic-BGE score (cosine similarity of whole-review BGE-small embeddings) lifts the story to sentence-level meaning: at 0.63–0.72 the generations are close in *meaning* to the human reviews despite sharing little surface form, with extremely tight cross-seed std (≤ 0.002) — review-text quality is highly reproducible. BERTScore-F1 — the canonical token-level contextual-embedding metric from Zhang et al. [2020], computed with RoBERTa-large — sits between **0.838 and 0.846 across all three domains** with std ≤ 0.0003 across the three seeds, a band consistent with published high-quality text-generation outputs and a reproducibility floor that essentially eliminates seed-variance as a confounder. Unlike Semantic-BGE, BERTScore matches every token in the candidate to its most-similar token in the reference (and vice versa) and reports the F1 of those alignments, which makes it less sensitive to whole-review-level "same-item floor" artefacts.

One caveat applies to Semantic-BGE that does *not* apply to BERTScore. Two reviews of the same restaurant or book share lexical and topical content even before either is written — so the BGE whole-review cosine carries a similarity floor that inflates the absolute number. BERTScore's token-alignment is more discriminative against this floor because matching at the *contextual-token* level requires the candidate's surface lexicon and ordering to actually correspond, not just the topic. That the three domains' BERTScore-F1 falls within a tight 0.008 band (0.838–0.846) suggests the generation quality is genuinely uniform across cuisine/book/book content rather than a domain-specific artefact. On manual inspection the generations are fluent, persona-consistent, and stylistically Nigerian.

### 4.4 Retrieval

Stage-1 recall, leave-one-out over each domain's full candidate pool. We evaluate six strategies against a popularity baseline across four cutoffs (HitRate@k and NDCG@k for k ∈ {10, 20, 50, 100}). The headline table below reports the two informative endpoints, HR@10 and HR@100; the complete multi-k figures, including NDCG, are in the artifact JSON.

| Method | Yelp HR@10 | Yelp HR@100 | Goodreads HR@10 | Goodreads HR@100 | Amazon HR@10 | Amazon HR@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| dense (content) | 0.002 ± 0.001 | 0.012 ± 0.001 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.027 ± 0.004 |
| hybrid (dense+CF) | 0.087 ± 0.008 | 0.347 ± 0.006 | 0.034 ± 0.006 | 0.142 ± 0.017 | 0.065 ± 0.004 | 0.187 ± 0.005 |
| CF (item-item) | **0.094 ± 0.006** | 0.347 ± 0.010 | **0.037 ± 0.006** | **0.150 ± 0.018** | 0.064 ± 0.002 | 0.187 ± 0.008 |
| ALS | 0.071 ± 0.006 | **0.350 ± 0.010** | 0.020 ± 0.002 | 0.129 ± 0.009 | 0.046 ± 0.003 | 0.184 ± 0.009 |
| popularity | 0.017 ± 0.002 | 0.081 ± 0.004 | 0.013 ± 0.004 | 0.056 ± 0.008 | 0.010 ± 0.000 | 0.042 ± 0.004 |

*Three independent leave-one-out splits (seeds 42 / 1 / 7); candidate pools 10,415 / 57,499 / 21,479 items. Mean ± sample std reported. Bold = best per (domain, k) cell. ALS computed via the `implicit` library's Conjugate-Gradient solver (Hu, Koren & Volinsky, 2008).*

Three findings drop out cleanly and are now confirmed across three independent splits with tight error bars.

**Pure content recall doesn't work.** BGE embeds item *content*, but recovering a specific held-out item is a *collaborative* task driven by cross-user co-occurrence, which a content encoder cannot model. Across all three domains at every cutoff, dense barely beats — and on the book domains undershoots — the popularity baseline (with std ≤ 0.004 across seeds, this is not a sampling artefact). The remedy is an **item-item collaborative-filtering signal blended with the dense score**, min-max normalised per user (dense/CF weight 20/80, set by a leave-one-out HR@10 sweep). Hybrid recovers strongly: HR@10 reaches 3.4–6× the popularity baseline.

**At HR@10, item-item CF beats both its hybrid and the latent-factor ALS in two of three domains** (Yelp 0.094 ± 0.006 vs hybrid 0.087, ALS 0.071; Goodreads 0.037 ± 0.006 vs hybrid 0.034, ALS 0.020). On Amazon, CF and hybrid are statistically indistinguishable at HR@10 (0.064 vs 0.065, both ± 0.002–0.004). The collaborative signal does most of the work; the 20% dense term mildly hurts on the smaller-catalogue domains. This is a live reproduction of Dacrema et al. [2019]: on sparse interaction data with a random hold-out, a well-tuned neighbourhood method beats a latent-factor model. We deploy item-item CF as the Stage-1 signal.

**ALS catches up — or overtakes — at the HR@100 cutoff.** On Yelp the order at HR@100 becomes **ALS 0.350 ± 0.010 ≈ CF/hybrid 0.347 ± 0.010**, the HR@10 gap fully closing. On Amazon CF, hybrid, and ALS converge to 0.184–0.187 (all within each other's std). On Goodreads ALS still trails CF (0.129 vs 0.150) but the gap collapses from ~1.85× at k=10 to ~1.16× at k=100. The interpretation is structural: neighbourhood methods *concentrate* probability mass on a handful of strong-co-occurrence neighbours — they win when only the top-10 is allowed — but their support is narrow, so deeper cutoffs reveal items they never considered. ALS's latent factors give it *broader coverage*, paying for it in top-10 precision but recovering more relevant items as the cutoff widens. For NaijaBuddy's deployment path — Stage-1 produces a candidate pool that the LLM reranks at ~50–100 items — the HR@100 column is arguably the more practically relevant figure, and at that scale ALS becomes a genuinely competitive Stage-1 alternative.

**Comparability to sampled-metric protocols.** The HitRate@10 / NDCG@10 above rank the held-out item against each domain's *entire* candidate pool (10,415–57,499 items). Much of the recommender-systems literature instead reports *sampled* metrics — the held-out item against a small fixed pool of the gold item plus sampled negatives. The two are not interchangeable, and even two sampled metrics disagree unless the negative-sampling distribution matches (Krichene and Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020). To make our retrieval placeable against that literature, the harness also supports the sampled protocol; under 101 candidates (1 target + 100 **popularity-weighted** negatives — the harder variant), our hybrid retrieval scores:

| Domain | NDCG@10 (hybrid) | NDCG@10 (CF) | HitRate@10 (hybrid) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.370 | 0.374 | 0.645 |
| Goodreads | 0.288 | 0.292 | 0.496 |
| Amazon | 0.340 | 0.358 | 0.515 |

*101-candidate sampled protocol, popularity-weighted negatives; n = 2,000 users per domain, seed 42. Reproduced with `eval_harness.py --candidate-pool 101 --pop-distractors`. As under the full-pool protocol, CF marginally exceeds the hybrid in every domain.*

We report these for placement, not as a like-for-like benchmark, and flag one protocol difference we cannot eliminate: our leave-one-out split holds out a *random* interaction, whereas the temporal protocol common in this literature holds out each user's *latest* interaction. The temporal split is harder, and under a random hold-out it specifically favours collaborative filtering — a random target drawn from a long history leaves abundant co-occurrence signal. This inflates the sampled scores relative to a temporal protocol, most of all on Goodreads — the heaviest-history domain at ~230 interactions per kept user, against ~50 on Yelp and Amazon. Our datasets carry no timestamps, so the temporal split cannot be reproduced here; we disclose it as the honest limit of cross-protocol comparison.

### 4.5 Cold-Start: The Calibration Layer's Real Value

The warm result in 4.2 — the LLM rating barely improving on the user-mean — holds only while the user-mean is itself reliable. The cold-start protocol shows it is not, once history is scarce, and that this is exactly where the LLM earns its place. The table reports the V3 3-term anchor (LLM + user-mean + item-mean blend) at each truncated history length:

| Domain | k = 1 V1 → V3 | k = 2 V1 → V3 | k = 3 V1 → V3 | warm V2 (§4.2) |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 1.413 ± 0.008 → **0.991 ± 0.019** | 1.204 ± 0.010 → 0.978 ± 0.019 | 1.146 ± 0.010 → 0.973 ± 0.012 | 0.989 ± 0.002 |
| Goodreads | 1.258 ± 0.013 → **0.962 ± 0.018** | 1.103 ± 0.026 → 0.949 ± 0.022 | 1.040 ± 0.010 → 0.934 ± 0.015 | 0.894 ± 0.027 |
| Amazon | 1.170 ± 0.020 → **0.934 ± 0.022** | 1.001 ± 0.031 → 0.899 ± 0.032 | 0.939 ± 0.034 → 0.875 ± 0.031 | 0.834 ± 0.015 |

*V1 user-mean RMSE (over the truncated history) → V3 3-term blend RMSE = α·LLM + β·μ_user + γ·μ_item. Three independent leave-one-out splits (seeds 42 / 1 / 7), n = 2,000 users per domain per k (1,997 / 1,994 for Amazon at k=1 / k=2,3), synth personas, served by vLLM-batched Qwen2.5-3B. Mean ± sample std reported. The warm column is §4.2.*

Three patterns hold cleanly across all three domains and all three seeds:

1. **V1 (the user-mean over a single observed rating) is the weakest predictor everywhere** — k=1 RMSE of 1.17–1.41 is much worse than the warm V1 baseline (0.84–1.00). This is the cold-start problem stated concretely: with one observed rating the empirical user-mean is dominated by noise.

2. **V3 cuts cold-start error substantially.** At k=1 the V3 blend reduces RMSE by **29.9%** (Yelp), **23.5%** (Goodreads) and **20.2%** (Amazon) versus V1 — *vastly* larger than the 0.5–1.2% warm gain of §4.2. By k=3 the gap to warm V2 has all but closed: Yelp k=3 V3 (0.97) reaches warm V2 (0.99) — once the user has rated three items the cold-start blend is competitive with the warm anchor.

3. **The V3 weight schedule favours item-bias, not the LLM.** The optimal (LLM / user / item) weights at k=1 are **0.03 / 0.13 / 0.83** (Yelp), **0.03 / 0.27 / 0.70** (Goodreads), **0.07 / 0.30 / 0.63** (Amazon) — *the item-bias term is doing almost all the work, even cold-start.* The user-mean weight grows monotonically with k as more user history accumulates (k=3 user weight 0.33–0.57). The LLM's numeric weight stays at 0.0–0.1 throughout. This refines §4.2's "LLM weight ≈ 0 warm" finding: it is the **item-bias term**, stable for a cold user because the item still has history from other users, that does most of the calibration work. The LLM's contribution remains marginal even at the regime where prior single-seed analyses had it dominate.

This reframes the calibration layer. It is not the LLM swooping in to rescue cold users; it is an **adaptive fusion of three statistical priors whose mix scales with the available evidence about the user**. When a user has rated a single item, the user-mean is noise; the item-mean (and the model that uses it) is what saves the prediction. As the user's own history accumulates, weight migrates from the item term to the user term, with the LLM contributing a small constant signal throughout. The natural next deployment step — which we did not have time to deploy — is to make the three weights an explicit function of history length rather than per-domain constants. Cold-start *retrieval*, by contrast, remains weak (dense HitRate@10 ≤ 0.005 at k ≤ 3 across all seeds and domains): a one-interaction persona is too thin a query, and a cold user's recommendations must lean on the popularity and cluster-mean fallbacks.

### 4.6 Ablation — Does LLM Persona Synthesis Help?

NaijaBuddy can model a user two ways: a deterministic **template** persona (category list plus two review snippets) or an **LLM-synthesised** prose persona. Holding everything else fixed (seed 42, identical retrieval weights), we run the full evaluation under each, at n = 2,000:

| Metric (Yelp / Goodreads / Amazon) | Template persona | Synthesised persona |
| :--- | :---: | :---: |
| RMSE V2 | 0.990 / 0.876 / 0.851 | 0.987 / 0.876 / 0.852 |
| Hybrid HitRate@10 | 0.088 / 0.034 / 0.063 | 0.083 / 0.032 / 0.066 |

Synthesis makes **no measurable difference to rating accuracy** — V2 differs by ≤ 0.003 — unsurprising, since §4.2 shows the rating is anchored on statistics, not on the persona text. At n = 2,000 it makes **no meaningful difference to retrieval either**: hybrid HitRate@10 moves by at most 0.005 and not consistently in sign. The reason is structural — at a realistic catalogue size (10K–57K items) content/dense retrieval has collapsed toward zero (§4.4), and the retrieval signal that survives, item-item CF, is computed from co-occurrence and is *persona-independent by construction*. (An earlier n = 350 run, with catalogues 10–60× smaller where dense retrieval still functioned, did show synthesis lifting Yelp recall; that effect does not survive the realistic catalogue size.) The honest reading: on these offline metrics template and synthesised personas are **equivalent** — persona synthesis is a UX choice, carrying the conversational human-facing demo and matching the deployed system, not a metrics lever.

### 4.7 Retrieval-Augmented Prompting

A third way to ground the LLM is **retrieval-augmented prompting (RAG)**: rather than an abstracted persona, the prompt is seeded with the user's **k = 4 past interactions most similar to the target item** — the real item descriptions and the actual ratings and reviews the user gave them — retrieved by cosine similarity. The hypothesis is that concrete examples of a user's own behaviour should ground the model better than a paraphrased summary of it. We evaluate at seed 42 and n = 2,000, directly comparable to §4.6:

| Metric (Yelp / Goodreads / Amazon) | Synthesised persona | Retrieval-augmented |
| :--- | :---: | :---: |
| RMSE V2 | 0.987 / 0.876 / 0.852 | 0.980 / 0.874 / 0.850 |
| Review Semantic-BGE | 0.738 / 0.633 / 0.648 | **0.754 / 0.645 / 0.662** |
| Review ROUGE-L | 0.100 / 0.082 / 0.091 | **0.102 / 0.088 / 0.098** |

The result splits cleanly by sub-task. On **rating prediction** RAG is indistinguishable from a synthesised persona — the V2 figures differ by ≤ 0.007, inside the sampling noise of §4.2, and the optimal blend weight stays pinned at α ≈ 0.1–0.2. This is the §4.2 result reached from a third independent direction: a deterministic template (§4.6), an LLM-synthesised persona, and retrieved real exemplars all land warm-user V2 within 0.01 RMSE of one another — no prompting strategy rescues the LLM's numeric estimate where the user-mean prior already wins. On **review generation**, by contrast, RAG produces a small but perfectly consistent gain: Semantic-BGE rises in all three domains (+0.012 to +0.016) and ROUGE-L in all three. The mechanism is intuitive — an abstracted persona discards the user's actual vocabulary, whereas in-context examples of their real past reviews let the model echo their voice, which both a surface-overlap and an embedding metric reward. The honest reading: retrieval augmentation helps exactly where the task carries stylistic signal — the generative sub-task — and not where the task is regression against a strong prior.

### 4.8 Cross-Domain Transfer — does Books-taste predict Movies-rating?

The §4.5 cold-start result — *item-bias dominates when user history is scarce* — has a stronger statement: it should hold even when the user is **fully cold in the target domain** but has rich history in *another* domain entirely. We test this on a constructed Amazon Books → Movies transfer benchmark.

**Dataset.** From the Amazon-Reviews-2023 5-core rating-only files (no item titles or descriptions in this slice; this is purely a statistical test) we extracted the 2,000 densest users who have ≥ 10 ratings in **both** Books and Movies (`cross_domain_dataset.py`). Each user contributes their full Books history, their Movies training history (all-but-last by timestamp), and the held-out last Movies rating as target. The pool spans 317,830 Books interactions, 287,817 Movies-train interactions, and 97,028 unique Movies items. 528 of the 2,000 held-out movies are *items no other user in this slice rated*, so the item-mean signal is absent there and reduces to the global mean.

**Models compared.** No LLM is used in this protocol — the source data has no item metadata to ground the model on, and we instead isolate the *statistical* transfer signal. For each user, predict the held-out Movies rating using:

| Model | Formula | RMSE |
| :--- | :--- | :---: |
| V0 (global Movies mean) | $\bar{y}_\text{movies}$ | 1.265 |
| V1 user-mean over **Books** *(cross-domain)* | $\mu_\text{user, books}$ | **1.192** |
| V1 user-mean over Movies *(in-domain upper bound)* | $\mu_\text{user, movies}$ | 1.137 |
| item-mean only (held-out movie) | $\mu_\text{item, movies}$ | 1.306 |
| **V2 books-user + item-bias** (best $\beta = 0.7$) | $\beta\,\mu_\text{user, books} + (1-\beta)\,\mu_\text{item, movies}$ | **1.166** |
| V2 movies-user + item-bias (best $\beta = 0.8$) | $\beta\,\mu_\text{user, movies} + (1-\beta)\,\mu_\text{item, movies}$ | 1.123 |

*n = 2,000 users with held-out Movies rating + full Books history + Movies training history. Best $\beta$ is the descriptive minimum of a 21-point sweep over $[0,1]$.*

Three findings stand out:

1. **Books taste transfers to Movies taste.** The user's mean Books rating (V1_books, RMSE 1.19) is 0.073 better than the global mean (V0, 1.27) and only 0.055 worse than the in-domain user-mean (V1_movies, 1.14). Roughly **57% of the in-domain user-bias signal is recoverable from Books alone** — far from a random or null transfer. A reader who is generous with their Books ratings tends to be generous with their Movies ratings too.

2. **Item-bias closes the cross-domain gap, mirroring §4.5.** Adding the target movie's cross-user mean to the books-user prior (V2 books+item) reduces RMSE from 1.19 to **1.17** — the same item-bias term that does most of the work in cold-start (§4.5) also does the work cross-domain. Note that the V2 books+item RMSE (1.17) is **closer to the in-domain V2 (1.12) than V1_books was to V1_movies** — the item-bias term proportionally helps the cross-domain case more, consistent with our interpretation that item-bias rescues whichever user-side signal is weakest.

3. **Pure item-bias alone is the worst predictor** (RMSE 1.31, worse even than V0 global mean). The item-bias term carries information *only when combined with a user-side prior* — it is not a baseline, it is a corrector. This refines §4.5: V3's strength is not just "item-bias rescues cold users" but "**any user-side prior plus item-bias beats either alone**", and the held-out movie not needing to be in the user's domain at all is the strongest possible statement of that.

Regenerable with `python cross_domain_dataset.py && python cross_domain_eval.py`. The protocol is purely statistical and runs in seconds on CPU — no LLM, no GPU, no Modal — which makes it the cheapest reproducibility test for the §4.5 thesis we can offer.

### 4.9 Honest Summary

NaijaBuddy's measured strengths are an **adaptive 3-term calibration layer** — a 6–13% mean RMSE reduction over a global baseline for warm users (V0 → V2, mean ± std across 3 seeds) and a **20–30% reduction for cold-start users at k = 1** (V1 → V3) — **hybrid retrieval** that beats the popularity baseline by 3.4–6× at HR@10 and remains competitive with ALS at HR@100 across all three domains, and an offline-capable inference stack (§2.2.1) that anyone can reproduce locally without GPU infrastructure. Its measured weaknesses are content-only retrieval, which does not beat popularity, and warm-user rating accuracy, where the LLM contributes only 0.004–0.011 RMSE over a user-bias + item-bias prior. The deployed system is the V3 3-term anchor — `α·LLM + β·μ_user + γ·μ_item` — with weights that migrate from item-mean-dominant at k = 1 to user-mean-dominant once history accumulates and an LLM weight that stays at 0.03–0.10 throughout. We report all of these directly — including a small-sample figure from an earlier draft that the full evaluation overturned — because a recommender's credibility rests on an evaluation that reproduces. The headline multi-seed figures are regenerated by `modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed {42,1,7} --cold-start --cold-sample 2000 --bertscore`; the §4.7 ablation by `eval_harness.py --persona-mode rag --seed 42`.

---

## 5. Related Work

**LLMs as rerankers and recommenders.** Casting an LLM as a ranking component is now common: the RankGPT family and its open-source counterpart RankVicuna [Pradeep et al., 2023] perform zero-shot listwise reranking, and EXP3RT [Kim et al., 2024] fine-tunes an LLM to extract review-based preferences and produce a reasoning-enhanced rating and reranked list. NaijaBuddy keeps the LLM as a *second-stage* reranker over a cheap hybrid retrieval stage — a design suited to a small local model.

**The limits of the LLM rating signal.** Kang et al. [2023], evaluating LLMs from 250M–540B parameters on user rating prediction, found that zero-shot LLMs lag traditional collaborative filtering whenever interaction data is available — the history, not the LLM, carries the signal. NaijaBuddy corroborates this across three domains and *operationalises* it: our calibration layer measures how much weight the LLM rating should receive (≈ 0 for warm users) and deploys the result as a regime switch. Ryu & Yanaka [2025] show in-context user reviews lift LLM rating prediction toward matrix-factorisation quality, especially for cold-start; our Tier-2 ablation (§4.7) tests this under a leakage-free protocol and refines it — in-context exemplars improve the generated *review text* but not the *warm rating*, where the user-mean already dominates.

**Cold-start.** Recommending for users or items with little history has a long line of classical remedies — content-based features, item-popularity and item-bias terms, cross-domain transfer, and meta-learning estimators such as MeLU [Lee et al., 2019], which meta-learns a preference model that adapts from only a handful of interactions. NaijaBuddy places the LLM *within* this lineage rather than against it: §4.5 measures the three-term anchor's weight schedule and finds the **item-bias term**, not the LLM, doing most of the cold-start work (LLM weight 0.03–0.07 at k=1), with the user-mean term growing as history accumulates. §4.8 stress-tests the same finding cross-domain: even when the user is fully cold in the target domain (Amazon Movies) but rich in another (Amazon Books), the item-bias signal closes most of the in-domain gap. This refines the conventional "LLM-rescues-cold-users" framing — it is item-bias, computable from other users' history, that rescues, with the LLM contributing a small constant signal throughout.

**Review generation and user simulation.** Review-LLM [Peng et al., 2024] targets personalised review generation and documents the "polite phenomenon" — LLMs resist producing negative reviews; BASES [Ren et al., 2024] simulates search users at scale. NaijaBuddy's Task A jointly simulates a rating and a persona-grounded review, with the calibration layer re-anchoring an over-generous LLM score against exactly that "polite" upward bias.

**Evaluation rigour.** Dacrema et al. [2019] showed that many neural recommenders fail to beat well-tuned simple baselines once evaluation is done carefully — so we report the global-mean and user-mean baselines beside every LLM result. RankVicuna separately argues that reranking results built on proprietary APIs are not reproducible. NaijaBuddy answers both: a leakage-free leave-one-out protocol, a robustness check across both reseeding and a 6× sample increase (§4.2), and an open-weights, open-engine inference stack — `Qwen2.5-3B-Instruct` served via vLLM 0.21 or `llama-cpp-python` (§2.2.1) — in which every reported figure regenerates.

**Positioning.** NaijaBuddy's distinctive combination is a small open-weights LLM that can run in an offline-capable configuration (against the field's GPT-3.5/4), calibration as a *measured* regime switch rather than an assumed guardrail, reproducible leakage-free evaluation, and explicit cultural localisation for an underserved market.

*References:* Andre et al. 2025 (arXiv:2508.20401); Dacrema et al. 2019 (RecSys); Hu, Koren & Volinsky 2008 (ICDM); Kang et al. 2023 (arXiv:2305.06474); Kim et al. 2024 (arXiv:2408.06276); Lee et al. 2019 (MeLU, KDD); Peng et al. 2024 (arXiv:2407.07487); Pradeep et al. 2023 (arXiv:2309.15088); Ren et al. 2024 (arXiv:2402.17505); Ryu & Yanaka 2025 (arXiv:2510.00449).

---

## 6. Discussion & Future Directions
With more development time and computing resources, we propose the following scaling directions:
1. **Hybrid Retrieval Indexing**: Combining our dense vector search with sparse BM25 indexing (hybrid lexical-dense retrieval) to enhance exact-match lookups (e.g., searching for specific brand names or exact local spellings).
2. **GPU & NPU Quantization**: Moving from 4-bit integer quantization (Q4_K_M) to 8-bit or 16-bit weight representations on dedicated hardware accelerators to improve the speed of local token generation.
3. **Decentralized Edge Federated Recommendation**: Building an edge-mesh network where local containerized nodes can exchange anonymized vector embeddings to learn multi-user preferences without exposing private user logs to a centralized server.

---

## 7. Conclusion
In this work, we developed **NaijaBuddy**, a highly localized, offline-capable agentic recommender system built for the DSN × BCT Hackathon 3.0. By structuring our system around a dual-stage "Filter-then-Rerank" workflow over a single Docker image with two interchangeable inference backends (`llama-cpp-python` GGUF or vLLM safetensors — §2.2.1), we achieve low-latency inference both as a stand-alone container and as a UI-tier that proxies to a hosted vLLM endpoint (the configuration used by the canonical evaluation and the live demo). We implemented an adaptive three-term Calibration Layer ($\alpha\cdot\text{LLM} + \beta\cdot\mu_u + \gamma\cdot\mu_i$) that anchors rating predictions to user and item statistics — reducing warm-user RMSE by 6.3–13% over a global-mean baseline and cold-start RMSE by 20–30% at k = 1 — and a deterministic Critic Layer for rule-based constraint filtering. The headline figures are reported as mean ± standard deviation across three independent leave-one-out splits at n = 2,000 users per seed, served by vLLM-batched Qwen2.5-3B-Instruct. A cross-domain transfer experiment (Amazon Books → Movies) shows the same item-bias signal rescues predictions even when the user is fully cold in the target domain — generalising the cold-start finding outside the within-domain setting. We evaluate the system with a leakage-free, reproducible harness and report results honestly, including where it underperforms: content-only retrieval and warm-user rating accuracy. Enriched with authentic Nigerian personas, NaijaBuddy represents a reliable and culturally authentic approach to generative recommendation and consumer behavioral simulation on the edge.
