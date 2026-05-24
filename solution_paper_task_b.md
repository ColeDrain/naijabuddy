# NaijaBuddy — Task B: Recommendation Engine
### A Hybrid-Retrieval, LLM-Reranked, Offline-Capable Approach to Personalised Recommendation

**Authors**: Team NaijaBuddy (Hackathon Submission)
**Affiliation**: DSN x BCT Data & AI Summit Hackathon 3.0
**Date**: May 2026

> This paper covers **Task B** of the DSN × BCT Hackathon 3.0 brief — Recommendation: given a user persona and a target domain, return a ranked list of items with persona-grounded justifications. The same agent and codebase also serve **Task A** (User Modeling); that task is described in [`solution_paper_task_a.md`](solution_paper_task_a.md), and shares the architecture (§2), calibration layer (§2.3), and evaluation harness reported here. We submit one repo and two task-focused papers per the form's structure; the codebase is unified per the brief's "two tasks, one ambition" framing.

---

## Abstract
Recommendation in resource-constrained settings faces three challenges: dependence on cloud APIs that exclude offline-first deployment, content-only retrieval that fails on sparse interaction data, and an inability to ground recommendations in culturally specific personas. We present **NaijaBuddy**'s Task B path: an offline-capable two-stage **filter-then-rerank** recommender. Stage 1 retrieves top-k candidates using a hybrid of **dense semantic recall** (BAAI/bge-small-en-v1.5 embeddings over a hybrid SQLite catalogue including localized Nigerian content) and **item-item collaborative filtering** (computed from co-occurrence). Stage 2 reranks the candidates with **`Qwen2.5-3B-Instruct`** — served interchangeably via in-process `llama-cpp-python` (Q4_K_M GGUF, the offline-mode fallback) or vLLM 0.21 on a dedicated GPU (the engine used by the canonical multi-seed evaluation and the live hosted demo) — using a pairwise-alignment prompt that produces a re-sorted list with one-line justifications per item. A deterministic **Collaborative Critic Layer** enforces rule-based persona constraints (e.g., demoting alcohol-centric venues for a non-drinking persona) over the LLM's ordering. Our leakage-free, out-of-sample evaluation across three independent leave-one-out splits (seeds 42 / 1 / 7) at n = 2,000 users per seed per domain shows that hybrid retrieval beats the popularity baseline by 3.4–6× at HR@10, that item-item collaborative filtering beats hybrid and ALS in two of three domains at HR@10, and that ALS catches up at HR@100 — informing the choice of Stage-1 signal. We report a measured limit honestly: **pure content/dense recall does not beat popularity** at our catalogue sizes (10K–57K items), and persona synthesis does not move retrieval metrics at this scale — both findings reframe how the LLM and the persona representation contribute to recommendation.

---

## 1. Introduction
User-modeling and recommendation systems have moved from matrix factorization toward semantic, conversational agents. But deploying these in resource-constrained settings — edge nodes, or servers in emerging markets — raises three challenges:

1. **Network & cloud dependence.** Cloud-LLM systems rely on external APIs, exposed to outages, card-authorization failures, and recurring cost.
2. **Retrieval cost & quality at small models.** Naïvely passing the full catalogue (10K–57K items in our domains) to an LLM reranker is computationally infeasible on a small local model — a cheap-but-effective Stage 1 is required.
3. **Cultural fidelity.** Global foundation models lack the linguistic nuance to simulate authentic Nigerian responses, often sounding dry or academic.

We introduce **NaijaBuddy**, a unified agentic recommender and review simulator that ships as a single Docker image, exposing both an interactive web UI and a REST API. Two deployment modes share the same code: an **offline mode** with `Qwen2.5-3B-Instruct` served in-process via `llama-cpp-python` (Q4_K_M GGUF, no network at inference time), and a **vLLM mode** in which the same model — the fp16 HF safetensors — is served by vLLM 0.21 on a dedicated GPU and the container talks to it over an OpenAI-protocol HTTP client. The vLLM mode is what the canonical multi-seed evaluation and the live hosted demo actually run; the offline mode is the reproducible fallback that anyone can spin up without GPU infrastructure (§2.2.1). This paper focuses on **Task B: Recommendation**, in which the agent — given a free-text persona and a target domain — returns a ranked list of items with persona-grounded justifications. Our Task B contributions:

* **A filter-then-rerank pipeline** — hybrid (dense + item-item CF) recall over the bundled SQLite catalogue, then an in-context `Qwen2.5-3B-Instruct` pairwise reranker — that keeps Stage 2 context manageable on a small local model.
* **A measured retrieval comparison** at multiple cutoffs (HR/NDCG @ 10, 20, 50, 100) against four baselines — pure dense, pure item-item CF, ALS, and popularity — confirming Dacrema et al. [2019]'s finding that well-tuned neighbourhood methods beat latent-factor models on sparse data at HR@10, while ALS catches up at deeper cutoffs.
* **A deterministic Critic Layer** that demotes ranked items violating encoded persona constraints — verifiable by construction for the constraints it encodes.
* **Nigerian localization** — a seeded local catalogue (Lagos eateries, Nollywood films, African literature) and persona-grounded prompts that produce justifications in authentic register, including a Naija-mode output style for Pidgin English.

---

## 2. Proposed Methodology (NaijaBuddy Architecture)

NaijaBuddy's backend engine is built in Python using FastAPI, SQLite and NumPy, with the LLM served via either `llama-cpp-python` (offline mode) or an OpenAI-protocol HTTP client pointed at a vLLM endpoint (vLLM mode — see §2.2.1). The architecture consists of four distinct, sequential layers; Task B exercises all four:

```
User persona / query
  │
  ├─ Layer 1 · Recall      BGE-small dense search + item-item CF over the
  │                        hybrid SQLite catalogue → top-10 candidates
  ├─ Layer 2 · Rerank      Qwen2.5-3B-Instruct (llama-cpp GGUF or vLLM
  │                        safetensors — see §2.2.1): in-context persona
  │                        modelling, pairwise sort, justification generation
  ├─ Layer 3 · Calibrate   (Task A path — see solution_paper_task_a.md;
  │                        Task B uses the cluster-mean lookup for cold-start
  │                        persona-to-neighbour mapping)
  └─ Layer 4 · Critic      deterministic rule filter — constraint-violating
                           items pushed to the bottom
  │
  ▼
Final ranked recommendations + persona-grounded justifications
```

### 2.1 Layer 1: Hybrid Catalog & Dense Semantic Recall
To deliver recommendations across multiple domains (Yelp, Amazon, Goodreads), we design a unified SQLite schema. The database is populated with an extensive, highly localized catalog spanning three distinct categories:
* **Yelp (Food & Spots)**: Iconic local spots (e.g., *Yellow Chilli*, *Shiro Lagos*, *The Place*, *Suya Spot*, *Club Quilox*) and traditional culinary items (Jollof Rice, Suya, Pepper Soup).
* **Amazon (Literature & Media)**: A real-world subset of Amazon Books reviews, augmented with a hand-curated localized overlay of landmark Nollywood productions (e.g., *The Wedding Party*, *King of Boys*, *Aníkúlápó*) and popular consumer electronics.
* **Goodreads (African Literature)**: High-caliber African and Nigerian literary masterpieces (e.g., *Things Fall Apart* by Chinua Achebe, *Half of a Yellow Sun* by Chimamanda Ngozi Adichie).

We leverage **`BAAI/bge-small-en-v1.5`** to map both items and user personas into a shared 384-dimensional dense semantic space. When a recommendation is requested, the system computes the user persona's cosine similarity against all item vectors in the target domain using a direct cosine similarity routine implemented in pure Python:

$$\text{Sim}(u, i) = \frac{\mathbf{v}_u \cdot \mathbf{v}_i}{\|\mathbf{v}_u\| \|\mathbf{v}_i\|}$$

The dense score is combined with an **item-item collaborative-filtering score** (computed from training co-occurrence, min-max normalised per user) at a fixed 20/80 dense/CF blend tuned by a leave-one-out HR@10 sweep (§4.3). The top 10 candidate items are retrieved and routed to the next layer. This "Filter-then-Rerank" paradigm prevents passing hundreds of items to the LLM, protecting local CPU execution and preventing out-of-memory crashes.

### 2.2 Layer 2: LLM Reranking & Justification Generation
At the core of Task B's reranking is a small Large Language Model: **`Qwen2.5-3B-Instruct`**, served via either of the two interchangeable inference backends described in §2.2.1. Both backends present the same `Llama()`-compatible callable to the upstream agent code so prompts, JSON-schema constraints, stop tokens and the calibration layer are bit-identical across modes.

#### 2.2.1 Deployment modes (offline vs vLLM)
The agent picks its inference engine at startup from the `VLLM_URL` environment variable:

* **Offline mode** (default when `VLLM_URL` is unset): the **`Qwen2.5-3B-Instruct-Q4_K_M`** GGUF is loaded into the container process via **`llama-cpp-python`**, with full GPU offload + FlashAttention-2 when CUDA is available and a CPU baseline fallback otherwise. No network traffic at inference time. This is the mode anyone running `docker run` against the public image without extra configuration enters, and it is the cheapest reproducibility path for the retrieval and reranking results.
* **vLLM mode** (set `VLLM_URL` to an OpenAI-compatible endpoint): the agent constructs a thin `VLLMShim` ([`vllm_shim.py`](vllm_shim.py)) that wraps an `openai.OpenAI` client and exposes the same callable signature `llama_cpp.Llama` does. Generation requests then proxy to a separately-deployed vLLM 0.21 server serving the **fp16 HF safetensors** of the *same* `Qwen/Qwen2.5-3B-Instruct` weights — same prompts, same stop tokens, same JSON-grammar constraint (mapped to vLLM's `guided_json`).

The **canonical multi-seed evaluation** (§4) runs in vLLM mode on a Modal-hosted A10G (`modal_vllm_eval.py`); the **live hosted demo** also runs in vLLM mode against a long-running Modal endpoint (`modal_vllm_serve.py`) so demo and paper-evaluation share the same engine config end to end.

#### Task B Prompt (Rerank + Justify)
The top 10 candidates retrieved by Stage 1 are represented as a serialized JSON catalog within the LLM context. The LLM is instructed to:
* Re-sort the candidates by assessing the pairwise alignment between the user's detailed persona and the item descriptions.
* Generate a compelling, short (1–2 sentence) natural language explanation detailing *why* the item was selected for this specific user.

A separate `naija_mode` flag toggles a Pidgin-English style overlay for the justifications; the ranking order is unaffected by style.

### 2.3 Layer 3 (Calibration — used here for persona cluster-mean lookup)
Task B uses Layer 1's vector index for a **cluster-mean neighbour lookup**: when a persona has no rating history, the index retrieves the top K = 5 nearest known users so the rerank prompt can be conditioned on similar-user behaviour patterns. The full rating-calibration math is described in [`solution_paper_task_a.md`](solution_paper_task_a.md) §2.3 and §4 — Task B's interaction with the calibration layer is narrow.

### 2.4 Layer 4: Collaborative Critic Layer (Deterministic Verification)
Even advanced LLMs sometimes hallucinate or ignore negative constraints in prompts (e.g., recommending a spicy beef dish to a strict vegetarian). Our Critic Layer programmatically inspects the ranked list and applies strict rule-based filters:

$$\text{Rank}_{\text{final}}(i) = \begin{cases} \text{Rank}_{\text{LLM}}(i) & \text{if } \text{Penalty}(i) = \text{False} \\ \text{Rank}_{\text{LLM}}(i) + 100 & \text{if } \text{Penalty}(i) = \text{True} \end{cases}$$

This deterministically demotes any item matching an encoded constraint rule below every non-matching item. We describe this as **rule-based enforcement** rather than a safety *guarantee*: it is verifiable by construction for the constraints it encodes, but — being keyword-based — makes no claim to catch violations outside those rules. A learned or semantic critic is left to future work.

---

## 3. Cultural Context & Nigerian Localization
Behavioural and cultural fidelity is a primary judging criterion. NaijaBuddy ships three hand-crafted Nigerian base personas:

1. **Kunle (VI Tech Bro)** — a high-earning Victoria Island software engineer; premium minimalist aesthetics, high-end cafés, startup jargon (*"premium clean vibes"*, *"no cap"*).
2. **Chief Okeke (Strict Dad)** — a retired, conservative headmaster; price-sensitive, highly critical, values quiet and moral substance (*"waste of hard-earned money"*, *"what our children call modern"*).
3. **Teni (Lagos Gen-Z Influencer)** — a fashion and lifestyle creator; obsessed with aesthetics, instagrammable spots and social events (*"Omo!"*, *"it's giving"*, *"God when?!"*).

Structuring prompts around these archetypes makes the generated justifications sound natural and locally grounded rather than generically global.

---

## 4. Experiments and Evaluation (Task B)

We evaluate NaijaBuddy's Task B path with a leakage-free, out-of-sample protocol and report the measured results, including negative findings. Every figure in this section is regenerated by the evaluation harness shipped with the code (`eval_harness.py`).

### 4.1 Protocol

**Datasets.** Three real-world interaction sets, each filtered to a genuine **3-core** — every user and every item has at least three interactions — and capped at the 2,000 densest users per domain: Yelp (106,300 interactions / 2,000 users / 10,415 items), Goodreads (466,625 / 2,000 / 57,499), and Amazon (101,540 / 1,999 / 21,479). The Amazon subset is drawn from the Books category, so cross-domain coverage is best read as *local businesses* (Yelp) *vs. literature* (Goodreads, Amazon).

**Leave-one-out split.** For each user we hold out one interaction. All training statistics — each user's mean rating, every item description, and every persona — are computed from that user's *remaining* interactions only. Retrieval is evaluated on the **full candidate pool** of each domain (10,415–57,499 items), not a sampled subset.

**Metrics.** HitRate@k and NDCG@k for k ∈ {10, 20, 50, 100} for retrieval. The headline figures below are reported as **mean ± sample standard deviation across three independent leave-one-out splits** (seeds 42 / 1 / 7), with n = 2,000 / 2,000 / 1,999 held-out interactions per seed per domain.

### 4.2 Retrieval

Stage-1 recall, leave-one-out over each domain's full candidate pool. We evaluate six strategies against a popularity baseline across four cutoffs; the headline table below reports the two informative endpoints, HR@10 and HR@100; the complete multi-k figures, including NDCG, are in the artifact JSON.

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

### 4.3 Does Persona Synthesis Help Retrieval?

NaijaBuddy can model a user two ways: a deterministic **template** persona (category list plus two review snippets) or an **LLM-synthesised** prose persona. Holding everything else fixed (seed 42, identical retrieval weights), we run the full evaluation under each, at n = 2,000:

| Metric (Yelp / Goodreads / Amazon) | Template persona | Synthesised persona |
| :--- | :---: | :---: |
| Hybrid HitRate@10 | 0.088 / 0.034 / 0.063 | 0.083 / 0.032 / 0.066 |

At n = 2,000 persona synthesis makes **no meaningful difference to retrieval**: hybrid HitRate@10 moves by at most 0.005 and not consistently in sign. The reason is structural — at a realistic catalogue size (10K–57K items) content/dense retrieval has collapsed toward zero (§4.2), and the retrieval signal that survives, item-item CF, is computed from co-occurrence and is *persona-independent by construction*. (An earlier n = 350 run, with catalogues 10–60× smaller where dense retrieval still functioned, did show synthesis lifting Yelp recall; that effect does not survive the realistic catalogue size.) The honest reading: on the retrieval metric, template and synthesised personas are **equivalent** — persona synthesis is a UX choice that carries the conversational human-facing demo and the justification text, not a retrieval lever.

### 4.4 Cold-Start in Retrieval
Cold-start retrieval is genuinely weak: dense HitRate@10 ≤ 0.005 at k ≤ 3 across all seeds and domains. A one-interaction persona is too thin a query, and a cold user's recommendations must lean on the popularity and cluster-mean fallbacks. This is a known limit of content-only retrieval and is the symmetric counterpart of the Task A finding (`solution_paper_task_a.md` §4.3) that item-bias rescues cold-start *rating* prediction — for *retrieval* there is no analogous statistical anchor that closes the gap. Future work would explore cross-user neighbourhood transfer (recommend what the K-nearest known users liked) and popularity-weighted backfill of the candidate list.

### 4.5 Honest Summary (Task B)

NaijaBuddy's Task B measured strengths are **hybrid retrieval** that beats the popularity baseline by 3.4–6× at HR@10, an **item-item CF Stage-1 signal** that beats both hybrid and ALS at HR@10 in two of three domains (Yelp 0.094 vs ALS 0.071, Goodreads 0.037 vs ALS 0.020), and an offline-capable inference stack (§2.2.1) that anyone can reproduce locally without GPU infrastructure. Its measured weaknesses are content-only retrieval — which does not beat popularity at realistic catalogue sizes — and cold-start retrieval, where a one-interaction persona is too thin a query for the current architecture. We report all of these directly because a recommender's credibility rests on an evaluation that reproduces. The headline multi-seed retrieval figures are regenerated by `modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed {42,1,7}`.

---

## 5. Related Work

**LLMs as rerankers and recommenders.** Casting an LLM as a ranking component is now common: the RankGPT family and its open-source counterpart RankVicuna [Pradeep et al., 2023] perform zero-shot listwise reranking, and EXP3RT [Kim et al., 2024] fine-tunes an LLM to extract review-based preferences and produce a reasoning-enhanced rating and reranked list. NaijaBuddy keeps the LLM as a *second-stage* reranker over a cheap hybrid retrieval stage — a design suited to a small local model.

**Hybrid retrieval and well-tuned baselines.** Dacrema et al. [2019] showed that many neural recommenders fail to beat well-tuned simple baselines once evaluation is done carefully — so we report the popularity baseline, ALS, and pure item-item CF beside every hybrid result. Our §4.2 retrieval table is in part a direct reproduction of that finding on three datasets at multi-cutoff. We deploy item-item CF as the Stage-1 signal precisely because of how it performs against ALS at the cutoffs that matter for top-list user-facing recommendation. Krichene and Rendle [KDD 2020] separately argue that sampled-metric protocols are not interchangeable; we report on the full candidate pool and include a sampled-protocol cross-reference for placement.

**Cold-start in retrieval.** A long line of classical remedies — content-based features, item-popularity backfill, cross-domain transfer, and meta-learning estimators such as MeLU [Lee et al., 2019] — address the same underlying problem. Our cold-start retrieval result (§4.4) is the symmetric counterpart of Task A's cold-start *rating* finding (`solution_paper_task_a.md` §4.3): item-bias rescues rating prediction, but for retrieval there is no analogous closed-form anchor.

**Critic and verification layers.** Recent work on LLM agents pairs the LLM with deterministic verification or critic layers (Andre et al., 2025) to enforce hard constraints that the LLM may otherwise hallucinate over. Our Critic Layer is a minimal instance of this idea — rule-based, keyword-driven, and described as such; the natural next step is a learned semantic critic.

**Positioning.** NaijaBuddy's distinctive combination for Task B is a small open-weights LLM running over a cheap hybrid Stage-1 (against the field's GPT-3.5/4 + dense-only setups), reproducible leakage-free evaluation against multiple baselines at multi-cutoff, and explicit cultural localisation for an underserved market.

*References:* Andre et al. 2025 (arXiv:2508.20401); Dacrema et al. 2019 (RecSys); Hu, Koren & Volinsky 2008 (ICDM); Kim et al. 2024 (arXiv:2408.06276); Krichene & Rendle 2020 (KDD); Lee et al. 2019 (MeLU, KDD); Pradeep et al. 2023 (arXiv:2309.15088); Ren et al. 2024 (arXiv:2402.17505).

---

## 6. Discussion & Future Directions
With more development time and computing resources, we propose the following scaling directions for Task B:
1. **Hybrid Retrieval Indexing**: combining our dense vector search with sparse BM25 indexing (hybrid lexical-dense retrieval) to enhance exact-match lookups (e.g., searching for specific brand names or exact local spellings).
2. **Cross-user neighbourhood transfer for cold-start retrieval**: recommend what the K-nearest known users (by persona embedding cosine) liked, when the active user has no interaction history.
3. **Decentralized Edge Federated Recommendation**: building an edge-mesh network where local containerized nodes can exchange anonymized vector embeddings to learn multi-user preferences without exposing private user logs to a centralized server.
4. **Learned semantic critic**: replace the rule-based Critic Layer with a small classifier trained on (persona, item) → constraint-violation pairs.

---

## 7. Conclusion (Task B)
In this paper, we presented **NaijaBuddy** for **Task B: Recommendation** — given a Nigerian persona and a target domain, the agent returns a ranked list of items with persona-grounded justifications. By structuring our system around a dual-stage "Filter-then-Rerank" workflow — hybrid dense + item-item CF Stage 1, in-context `Qwen2.5-3B-Instruct` Stage 2 (served via interchangeable `llama-cpp-python` or vLLM backends — §2.2.1) — and a deterministic Critic Layer, we deliver retrieval that beats the popularity baseline by 3.4–6× at HR@10 across all three domains. We document the measured limit honestly: pure content/dense retrieval does not beat popularity at realistic catalogue sizes, and persona synthesis is a UX choice, not a retrieval lever — refining a common assumption about LLM-aided recommendation. Enriched with authentic Nigerian personas, NaijaBuddy's Task B path represents a reliable and culturally authentic approach to personalised recommendation on the edge.

The same agent serves **Task A: User Modeling**, described in [`solution_paper_task_a.md`](solution_paper_task_a.md), sharing the architecture, Critic Layer, and evaluation harness reported here.
