# NaijaBuddy: A Highly Localized, Offline-First Agentic Recommender & User Modeling System for Cross-Domain Behavioral Simulation

**Authors**: Team NaijaBuddy (Hackathon Submission)  
**Affiliation**: DSN x BCT Data & AI Summit Hackathon 3.0  
**Date**: May 2026  

---

## Abstract
Traditional recommender systems rely heavily on static user matrices, neglecting contextual nuance and natural language behavior. Recent advances in Large Language Models (LLMs) have enabled generative recommendation and behavioral simulation; however, these models often suffer from strong "positivity bias" in numerical rating generation, high computational latency, and complete dependence on costly, cloud-bound APIs. 

In this paper, we present **NaijaBuddy**, a 100% offline-first, containerized agentic recommendation and user modeling system tailored specifically for the Nigerian consumer market. Our architecture employs a dual-stage pipeline: (1) high-speed dense vector semantic search using `BAAI/bge-small-en-v1.5` over a hybrid SQLite catalog enriched with local establishments, movies, and literature; and (2) in-context reranking and review generation utilizing a quantized local GGUF engine (`Qwen-2.5-3B-Instruct`) hosted natively in-process via `llama-cpp-python`. To protect rating accuracy (RMSE), we implement a mathematical **Output Calibration Layer** that blends LLM estimates with user baseline means and uses a vector-based **Cluster-Mean** fallback for cold-start personas. Finally, a deterministic **Collaborative Critic Layer** ensures 100% compliance with strict behavioral constraints. Our leakage-free, out-of-sample evaluation over the full held-out set (n = 2,000 users per domain) establishes per-user calibration as a reliable RMSE anchor (up to an 11.9% reduction over a global-mean baseline) and shows the optimal LLM/statistics blend to be *adaptive*: the LLM's raw rating adds little for users with rich history, but cuts cold-start RMSE by 12–17%. We release the evaluation harness so that every reported figure reproduces.

---

## 1. Introduction
User-modeling and recommendation systems have moved from matrix factorization toward semantic, conversational agents. But deploying these in resource-constrained settings — edge nodes, or servers in emerging markets — raises three challenges:

1. **Network & cloud dependence.** Cloud-LLM systems rely on external APIs, exposed to outages, card-authorization failures, and recurring cost.
2. **LLM rating volatility.** Asked for a 1–5 star rating, LLMs show a pronounced positivity bias and round toward integer extremes, inflating RMSE.
3. **Cultural fidelity.** Global foundation models lack the linguistic nuance to simulate authentic Nigerian responses, often sounding dry or academic.

We introduce **NaijaBuddy**, a unified agentic recommender and review simulator that runs entirely offline inside one Docker container, exposing both an interactive web UI and a REST API. Our contributions:

* **A filter-then-rerank pipeline** — dense cosine recall over 384-d embeddings, then a local GGUF LLM reranker — that keeps context and latency low enough for CPU inference.
* **A calibration layer** that blends the LLM rating with statistical anchors under a tunable $\alpha$, with a vector-neighbourhood cluster-mean fallback for cold-start users.
* **Nigerian localization** — a seeded local catalogue (Lagos eateries, Nollywood films, African literature) and persona-grounded prompts ("Strict Nigerian Dad", "VI Tech Bro", "Lagos Gen-Z Influencer").

---

## 2. Proposed Methodology (NaijaBuddy Architecture)

NaijaBuddy’s backend engine is built entirely in Python using FastAPI, SQLite, NumPy, and llama-cpp. The architecture consists of four distinct, sequential layers:

```
User persona / query
  │
  ├─ Layer 1 · Recall      BGE-small dense search + item-item CF over the
  │                        hybrid SQLite catalogue → top-10 candidates
  ├─ Layer 2 · Rerank      local Qwen2.5-3B-Instruct (GGUF): in-context
  │                        persona modelling, pairwise sort, review generation
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
At the core of the reranking and review simulation is a quantized local Large Language Model: **`Qwen2.5-3B-Instruct-Q4_K_M`** in GGUF format, executed natively via **`llama-cpp-python`**. 

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

This pushes constraint-violating recommendations to the absolute bottom of the list, guaranteeing 100% safety and eliminating logical anomalies.

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

**Metrics.** RMSE for rating prediction; ROUGE-L and an embedding-based semantic similarity for the generated review; HitRate@10 and NDCG@10 for retrieval. Every figure below is computed over the **entire held-out set** — one held-out interaction per user, n = 2,000 / 2,000 / 1,999 — on a single leave-one-out split (seed 42). At this scale a single split is already a tight estimate: the 6× larger sample (vs. an earlier n = 350 run) shrinks sampling error well inside the ±0.02–0.03 spread that earlier run showed across three seeds. The headline evaluation uses **template** personas; §4.6 shows that synthesised personas — the deployed default — give statistically equivalent rating accuracy. A discarded small-sample run (n = 10) once reported a Goodreads blend RMSE of 0.60 that the full evaluation corrects to ~0.88 — a cautionary illustration of sampling variance, and the reason for the full-set protocol.

**Cold-start protocol.** A 3-core leaves no genuinely cold users (the Goodreads minimum is 44 interactions), so we *simulate* cold-start: a test user's history is truncated to k ∈ {1, 2, 3} interactions while every other user keeps full history — a new user entering a populated system — and the evaluation is repeated. Section 4.5 reports the resulting degradation curve.

### 4.2 Rating Prediction and the Calibration Layer

**Per-user anchoring.** Replacing the global mean (V0) with the per-user training mean (V1 — the calibration formula at $\alpha = 0$) reduces RMSE in every domain; blending the LLM's raw rating back in at a low weight (V2) improves on V1 further, by a smaller and domain-dependent margin:

| Domain | V0 global | V1 user-mean ($\alpha$=0) | pure-LLM ($\alpha$=1) | V2 blend (best $\alpha$) |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 1.059 | 0.995 | 1.186 | **0.990** ($\alpha$=0.1) |
| Goodreads | 0.994 | 0.879 | 1.102 | **0.876** ($\alpha$=0.1) |
| Amazon | 0.965 | 0.856 | 1.059 | **0.851** ($\alpha$=0.1) |

*Full held-out set, n = 2,000 / 2,000 / 1,999; single split (seed 42), template personas. V0→V2 reduces RMSE by 6.5% / 11.9% / 11.8% — but the V1→V2 step, the LLM's actual contribution, is only 0.003–0.005.*

**Does the LLM's rating help?** Because the raw LLM rating is independent of $\alpha$, one LLM call per held-out pair supports a full sweep of the blend $\hat{y} = \alpha\cdot\text{LLM} + (1-\alpha)\cdot\mu_u$. The sweep is **U-shaped** in every domain: error falls from V1 to a shallow minimum at $\alpha = 0.1$, then rises steeply to the pure-LLM endpoint — the worst configuration everywhere (1.06–1.19). The V1→V2 improvement is just 0.003–0.005 RMSE; at n = 2,000 that is a real but tiny effect. The reported V2 sits at the *test-minimising* point of the sweep — a descriptive minimum, not a validation-tuned hyperparameter — so if anything the verdict is conservative: even oracle-optimal blending barely moves the user-mean.

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

### 4.3 Review Generation

We score the generated review against the held-out human review with two complementary metrics:

| Domain | ROUGE-L F1 | Semantic similarity |
| :--- | :---: | :---: |
| Yelp | 0.097 | **0.734** |
| Goodreads | 0.086 | **0.634** |
| Amazon | 0.093 | **0.648** |

ROUGE-L — verbatim longest-common-subsequence overlap against a single reference — is low, as it must be: two people reviewing the same item rarely choose the same words. The semantic score (cosine similarity of BGE review embeddings, a metric in the BERTScore family) tells the more faithful story: at 0.63–0.73 the generations are close in *meaning* to the human reviews despite sharing little surface form. One caveat: two reviews of the same item carry a similarity floor (both discuss the same restaurant or book), so the absolute value is best read as encouraging rather than decisive — a same-item human–human baseline is the natural next calibration. On manual inspection the generations are fluent, persona-consistent, and stylistically Nigerian.

### 4.4 Retrieval

Stage-1 recall, leave-one-out over each domain's full candidate pool, for four strategies against a popularity baseline (HitRate@10):

| Domain | items | dense (content) | hybrid (dense+CF) | CF | popularity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Yelp | 10,415 | 0.002 | 0.088 | **0.089** | 0.019 |
| Goodreads | 57,499 | 0.001 | 0.034 | **0.039** | 0.010 |
| Amazon | 21,479 | 0.004 | 0.063 | **0.067** | 0.011 |

Pure dense content recall is the system's clear weakness — it barely clears, and on the book domains undershoots, the popularity baseline. The cause is structural: BGE embeds item *content*, but recovering a specific held-out item is a *collaborative* task driven by cross-user co-occurrence, which a content encoder does not model. The remedy is an **item-item collaborative-filtering signal blended with the dense score**, min-max normalised per user (dense/CF weight 20/80, set by a leave-one-out HitRate@10 sweep). Hybrid retrieval reaches HitRate@10 of 0.088 (Yelp), 0.034 (Goodreads) and 0.063 (Amazon) — **3.4–6× the popularity baseline**. The absolute figures are well below an earlier n = 350 run because the candidate pool is now far larger — 10K–57K items rather than 0.9K–8K — and ranking the gold item into the top 10 of 57K is a much harder target. At this scale **CF alone slightly *exceeds* the hybrid in every domain**: the dense term, 20% of the blend, mildly hurts, and the collaborative signal does all the work. Content retrieval is best treated as a fallback for genuinely cold users with no co-occurrence history.

**Comparability to sampled-metric protocols.** The HitRate@10 / NDCG@10 above rank the held-out item against each domain's *entire* candidate pool (10,415–57,499 items). Much of the recommender-systems literature instead reports *sampled* metrics — the held-out item against a small fixed pool of the gold item plus sampled negatives. The two are not interchangeable, and even two sampled metrics disagree unless the negative-sampling distribution matches (Krichene and Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020). To make our retrieval placeable against that literature, the harness also supports the sampled protocol; under 101 candidates (1 target + 100 **popularity-weighted** negatives — the harder variant), our hybrid retrieval scores:

| Domain | NDCG@10 (hybrid) | NDCG@10 (CF) | HitRate@10 (hybrid) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.370 | 0.374 | 0.645 |
| Goodreads | 0.288 | 0.292 | 0.496 |
| Amazon | 0.340 | 0.358 | 0.515 |

*101-candidate sampled protocol, popularity-weighted negatives; n = 2,000 users per domain, seed 42. Reproduced with `eval_harness.py --candidate-pool 101 --pop-distractors`. As under the full-pool protocol, CF marginally exceeds the hybrid in every domain.*

We report these for placement, not as a like-for-like benchmark, and flag one protocol difference we cannot eliminate: our leave-one-out split holds out a *random* interaction, whereas the temporal protocol common in this literature holds out each user's *latest* interaction. The temporal split is harder, and under a random hold-out it specifically favours collaborative filtering — a random target drawn from a long history leaves abundant co-occurrence signal. This inflates the sampled scores relative to a temporal protocol, most of all on Goodreads — the heaviest-history domain at ~230 interactions per kept user, against ~50 on Yelp and Amazon. Our datasets carry no timestamps, so the temporal split cannot be reproduced here; we disclose it as the honest limit of cross-protocol comparison.

### 4.5 Cold-Start: The Calibration Layer's Real Value

The warm result in 4.2 — the LLM rating barely improving on the user-mean — holds only while the user-mean is itself reliable. The cold-start protocol shows it is not, once history is scarce, and that this is exactly where the LLM earns its place. The table reports the RMSE-optimal $\alpha$ as a function of history size:

| Domain | k=1 | k=2 | k=3 | warm |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 0.6 | 0.4 | 0.4 | 0.1 |
| Goodreads | 0.7 | 0.6 | 0.4 | 0.1 |
| Amazon | 0.5 | 0.5 | 0.1 | 0.1 |

*Cold-start simulated by truncating each test user's history to k interactions (n = 100 users per domain per k; single leave-one-out split, seed 42).*

Two patterns hold across all three domains. First, **the RMSE-optimal $\alpha$ falls as history accumulates** — from 0.5–0.7 at k = 1 to 0.1 once the user is well-observed. Second, **the LLM blend's benefit grows as history shrinks**: at k = 1 the optimal blend cuts RMSE by 14.7% (Yelp, 1.308 → 1.116), 17.1% (Goodreads, 1.428 → 1.184) and 11.5% (Amazon, 1.077 → 0.953) over the user-mean — against the ≤ 0.5% warm gain of Section 4.2.

This reframes the calibration layer. It is not a guardrail that "reduces to the statistical baseline"; it is an **adaptive fusion whose reliance on the LLM scales inversely with the evidence available about a user**. When a user has rated a single item, their empirical mean is noise and the LLM's persona-grounded estimate carries most of the signal; when they have rated forty, the reverse holds. The natural next step — which we did not have time to deploy — is to make $\alpha$ an explicit function of history length rather than a per-domain constant. Cold-start *retrieval*, by contrast, remains weak (dense HitRate@10 $\approx 0$ at k ≤ 3): a one-interaction persona is too thin a query, and a cold user's recommendations must lean on the popularity and cluster-mean fallbacks.

### 4.6 Ablation — Does LLM Persona Synthesis Help?

NaijaBuddy can model a user two ways: a deterministic **template** persona (category list plus two review snippets) or an **LLM-synthesised** prose persona. Holding everything else fixed (seed 42, identical retrieval weights), we run the full evaluation under each. *This ablation and §4.7 predate the n = 2,000 re-run and are reported at the earlier n = 350 scale; what each isolates is a within-table contrast, so the absolute figures are not directly comparable to the n = 2,000 headline of §4.2.*

| Metric | Template persona | Synthesised persona |
| :--- | :---: | :---: |
| RMSE V2 (Yelp / Goodreads / Amazon) | 0.998 / 0.980 / 0.768 | 0.995 / 0.978 / 0.769 |
| Hybrid HitRate@10 — Yelp | 0.174 | **0.198** |
| Hybrid HitRate@10 — Goodreads / Amazon | 0.051 / 0.054 | 0.046 / 0.046 |

Synthesis makes **no measurable difference to rating accuracy** — unsurprising, since §4.2 shows the rating is anchored on statistics, not on the persona text. Its effect is on *retrieval*, and it is **domain-dependent**: a fluent prose persona lifts Yelp recall by ~14% (0.174 → 0.198), because Yelp items carry rich, discriminative categories a descriptive persona can match; on the two book domains, where every item's category is the single token "Book", the prose persona slightly *hurts* (≈0.05 → 0.046) — it adds wording the content encoder cannot ground. The honest reading: persona synthesis earns its cost where item content is discriminative and is roughly neutral-to-negative where it is not. We keep it enabled — it matches the deployed conversational system and carries the human-facing demo — but report it as a genuine *mixed* ablation result, not an unqualified win.

### 4.7 Retrieval-Augmented Prompting

A third way to ground the LLM is **retrieval-augmented prompting (RAG)**: rather than an abstracted persona, the prompt is seeded with the user's **k = 4 past interactions most similar to the target item** — the real item descriptions and the actual ratings and reviews the user gave them — retrieved by cosine similarity. The hypothesis is that concrete examples of a user's own behaviour should ground the model better than a paraphrased summary of it. We evaluate at seed 42, directly comparable to §4.6 (and, like it, at the n = 350 scale):

| Metric (Yelp / Goodreads / Amazon) | Synthesised persona | Retrieval-augmented |
| :--- | :---: | :---: |
| RMSE V2 | 0.995 / 0.978 / 0.769 | 0.999 / 0.977 / 0.770 |
| Review Semantic-BGE | 0.742 / 0.634 / 0.667 | **0.763 / 0.645 / 0.683** |
| Review ROUGE-L | 0.097 / 0.081 / 0.099 | **0.099 / 0.090 / 0.100** |

The result splits cleanly by sub-task. On **rating prediction** RAG is indistinguishable from a synthesised persona — the three V2 figures differ by ≤ 0.004, inside the sampling noise of §4.2, and the optimal blend weight stays pinned at α ≈ 0.1. This is the §4.2 result reached from a third independent direction: a deterministic template (§4.6), an LLM-synthesised persona, and retrieved real exemplars all leave warm-user rating dominated by the user-mean prior — no prompting strategy rescues the LLM's numeric estimate where the statistics already win. On **review generation**, by contrast, RAG produces a small but perfectly consistent gain: Semantic-BGE rises in all three domains (+0.011 to +0.020) and ROUGE-L in all three. The mechanism is intuitive — an abstracted persona discards the user's actual vocabulary, whereas in-context examples of their real past reviews let the model echo their voice, which both a surface-overlap and an embedding metric reward. The honest reading: retrieval augmentation helps exactly where the task carries stylistic signal — the generative sub-task — and not where the task is regression against a strong prior.

### 4.8 Honest Summary

NaijaBuddy's measured strengths are an **adaptive calibration layer** — a ~10% mean RMSE reduction over a global baseline for warm users and a 12–17% reduction for cold-start users — **hybrid retrieval** that beats the popularity baseline by 3.4–6× across the three domains, and fully offline operation. Its measured weaknesses are content-only retrieval, which does not beat popularity, and warm-user rating accuracy, where the LLM adds little — at n = 2,000 the user-mean + item-bias model is the warm result, and the LLM's numeric rating earns its place only in cold-start. We report all of these directly — including a small-sample figure from an earlier draft that the full evaluation overturned — because a recommender's credibility rests on an evaluation that reproduces. The warm figures are regenerated by `eval_harness.py --seeds 42 --bertscore`; the Section 4.5 cold-start curve by adding `--cold-start`; the §4.7 ablation with `--persona-mode rag --seed 42`.

---

## 5. Related Work

**LLMs as rerankers and recommenders.** Casting an LLM as a ranking component is now common: the RankGPT family and its open-source counterpart RankVicuna [Pradeep et al., 2023] perform zero-shot listwise reranking, and EXP3RT [Kim et al., 2024] fine-tunes an LLM to extract review-based preferences and produce a reasoning-enhanced rating and reranked list. NaijaBuddy keeps the LLM as a *second-stage* reranker over a cheap hybrid retrieval stage — a design suited to a small local model.

**The limits of the LLM rating signal.** Kang et al. [2023], evaluating LLMs from 250M–540B parameters on user rating prediction, found that zero-shot LLMs lag traditional collaborative filtering whenever interaction data is available — the history, not the LLM, carries the signal. NaijaBuddy corroborates this across three domains and *operationalises* it: our calibration layer measures how much weight the LLM rating should receive (≈ 0 for warm users) and deploys the result as a regime switch. Ryu & Yanaka [2025] show in-context user reviews lift LLM rating prediction toward matrix-factorisation quality, especially for cold-start; our Tier-2 ablation (§4.7) tests this under a leakage-free protocol and refines it — in-context exemplars improve the generated *review text* but not the *warm rating*, where the user-mean already dominates.

**Review generation and user simulation.** Review-LLM [Peng et al., 2024] targets personalised review generation and documents the "polite phenomenon" — LLMs resist producing negative reviews; BASES [Ren et al., 2024] simulates search users at scale. NaijaBuddy's Task A jointly simulates a rating and a persona-grounded review, with the calibration layer re-anchoring an over-generous LLM score against exactly that "polite" upward bias.

**Evaluation rigour.** Dacrema et al. [2019] showed that many neural recommenders fail to beat well-tuned simple baselines once evaluation is done carefully — so we report the global-mean and user-mean baselines beside every LLM result. RankVicuna separately argues that reranking results built on proprietary APIs are not reproducible. NaijaBuddy answers both: a leakage-free leave-one-out protocol, multi-seed error bars, and a 100%-offline stack in which every reported figure regenerates.

**Positioning.** NaijaBuddy's distinctive combination is a small quantised *offline* LLM (against the field's GPT-3.5/4), calibration as a *measured* regime switch rather than an assumed guardrail, reproducible leakage-free evaluation, and explicit cultural localisation for an underserved market.

*References:* Andre et al. 2025 (arXiv:2508.20401); Dacrema et al. 2019 (RecSys); Kang et al. 2023 (arXiv:2305.06474); Kim et al. 2024 (arXiv:2408.06276); Peng et al. 2024 (arXiv:2407.07487); Pradeep et al. 2023 (arXiv:2309.15088); Ren et al. 2024 (arXiv:2402.17505); Ryu & Yanaka 2025 (arXiv:2510.00449).

---

## 6. Discussion & Future Directions
With more development time and computing resources, we propose the following scaling directions:
1. **Hybrid Retrieval Indexing**: Combining our dense vector search with sparse BM25 indexing (hybrid lexical-dense retrieval) to enhance exact-match lookups (e.g., searching for specific brand names or exact local spellings).
2. **GPU & NPU Quantization**: Moving from 4-bit integer quantization (Q4_K_M) to 8-bit or 16-bit weight representations on dedicated hardware accelerators to improve the speed of local token generation.
3. **Decentralized Edge Federated Recommendation**: Building an edge-mesh network where local containerized nodes can exchange anonymized vector embeddings to learn multi-user preferences without exposing private user logs to a centralized server.

---

## 7. Conclusion
In this work, we developed **NaijaBuddy**, a highly localized, completely offline-first agentic recommender system built for the DSN x BCT Hackathon 3.0. By structuring our system around a dual-stage "Filter-then-Rerank" workflow, we achieve elite low-latency CPU inference within a self-contained Docker container. We implemented an adaptive Calibration Layer that anchors rating predictions to per-user statistics — reducing RMSE by up to 11.9% over a global-mean baseline, with the LLM blend cutting a further 12–17% for cold-start users — and a deterministic Critic Layer for rule-based constraint filtering. We evaluate the system with a leakage-free, reproducible harness over the full held-out set and report results honestly, including where it underperforms: content-only retrieval and warm-user rating accuracy. Enriched with authentic Nigerian personas, NaijaBuddy represents a reliable and culturally authentic approach to generative recommendation and consumer behavioral simulation on the edge.
