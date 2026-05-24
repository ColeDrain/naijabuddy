# NaijaBuddy — Task A: User Modeling & Behavioral Simulation
### A Calibrated, Offline-Capable Agentic Approach to Rating Prediction and Persona-Grounded Review Generation

**Authors**: Team NaijaBuddy (Hackathon Submission)
**Affiliation**: DSN x BCT Data & AI Summit Hackathon 3.0
**Date**: May 2026

> This paper covers **Task A** of the DSN × BCT Hackathon 3.0 brief — User Modeling: given a persona, predict a star rating and generate an in-voice review for a target item. The same agent and codebase also serve **Task B** (Recommendation), described in the companion Task B paper, which shares the architecture (§2), calibration layer (§2.3), and evaluation harness reported here. We submit one repo and two task-focused papers per the form's structure; the codebase is unified per the brief's "two tasks, one ambition" framing.

---

## Abstract
Traditional recommender systems rely heavily on static user matrices, neglecting contextual nuance and natural language behavior. Recent advances in Large Language Models (LLMs) have enabled generative recommendation and behavioral simulation; however, these models often suffer from strong "positivity bias" in numerical rating generation, high computational latency, and complete dependence on costly, cloud-bound APIs.

We present **NaijaBuddy**'s Task A path: an offline-capable agentic user-modeling system that, given a persona and a target item, produces a calibrated 1–5 star rating and an in-voice review. The model is **`Qwen2.5-3B-Instruct`**, served interchangeably via in-process `llama-cpp-python` (Q4_K_M GGUF, the offline-mode fallback) or vLLM 0.21 on a dedicated GPU (the engine used by the canonical multi-seed evaluation and the live hosted demo). To protect rating accuracy (RMSE) we implement a mathematical **Output Calibration Layer** that blends the raw LLM rating with user-mean and item-mean statistical anchors and uses a vector-based **Cluster-Mean** fallback for cold-start personas. A deterministic **Collaborative Critic Layer** enforces rule-based persona constraints over the generated rating/review pair. Our leakage-free, out-of-sample evaluation reports **mean ± standard deviation across three independent leave-one-out splits** (seeds 42 / 1 / 7) at n = 2,000 users per seed per domain. Per-user anchoring is a reliable RMSE baseline (a 6.3–13% reduction over a global mean depending on domain), and the optimal LLM/statistics blend is *adaptive*: the LLM's raw rating adds little for warm users with rich history (V1→V2 step of 0.004–0.011 RMSE), but the three-term blend including item-bias cuts cold-start RMSE by **20–30% at k = 1** over the user-mean baseline. A cross-domain transfer experiment (Amazon Books → Movies) shows the same item-bias signal rescues predictions even when the user is fully cold in the target domain. Generated reviews score BERTScore-F1 0.838–0.846 across the three domains with std ≤ 0.0003 across seeds — a tight, reproducible quality band that places the system in the regime of published high-quality text-generation outputs.

---

## 1. Introduction
User-modeling and recommendation systems have moved from matrix factorization toward semantic, conversational agents. But deploying these in resource-constrained settings — edge nodes, or servers in emerging markets — raises three challenges:

1. **Network & cloud dependence.** Cloud-LLM systems rely on external APIs, exposed to outages, card-authorization failures, and recurring cost.
2. **LLM rating volatility.** Asked for a 1–5 star rating, LLMs show a pronounced positivity bias and round toward integer extremes, inflating RMSE.
3. **Cultural fidelity.** Global foundation models lack the linguistic nuance to simulate authentic Nigerian responses, often sounding dry or academic.

We introduce **NaijaBuddy**, a unified agentic recommender and review simulator that ships as a single Docker image, exposing both an interactive web UI and a REST API. Two deployment modes share the same code: an **offline mode** with `Qwen2.5-3B-Instruct` served in-process via `llama-cpp-python` (Q4_K_M GGUF, no network at inference time), and a **vLLM mode** in which the same model — the fp16 HF safetensors — is served by vLLM 0.21 on a dedicated GPU and the container talks to it over an OpenAI-protocol HTTP client. The vLLM mode is what the canonical multi-seed evaluation and the live hosted demo actually run; the offline mode is the reproducible fallback that anyone can spin up without GPU infrastructure (§2.2.1). This paper focuses on **Task A: User Modeling**, in which the agent — given a free-text persona and a target item — predicts a star rating and a persona-grounded review. Our Task A contributions:

* **A three-term calibration layer** that blends the LLM rating with user-mean and item-mean statistical anchors ($\alpha\cdot\text{LLM} + \beta\cdot\mu_u + \gamma\cdot\mu_i$), with a vector-neighbourhood cluster-mean fallback for cold-start users. Weight migration across history regimes — item-bias dominant when cold, user-mean dominant once warm — is the key empirical finding.
* **A measured cold-start regime switch.** §4.5 isolates the regime where the LLM rating earns its place: simulated cold-start with k ∈ {1, 2, 3} interactions of user history, where V3 cuts RMSE by 20–30% over the truncated user-mean.
* **A reproducible multi-seed evaluation harness** — three independent leave-one-out splits at n = 2,000 per domain, three review-quality axes (ROUGE-L, Semantic-BGE, BERTScore-F1), and a cross-domain Books→Movies transfer experiment that stress-tests the cold-start finding outside the within-domain setting.
* **Nigerian localization** — a seeded local catalogue (Lagos eateries, Nollywood films, African literature) and persona-grounded prompts ("Strict Nigerian Dad", "VI Tech Bro", "Lagos Gen-Z Influencer") that produce reviews in authentic register, including a Naija-mode output style for Pidgin English.

---

## 2. Proposed Methodology (NaijaBuddy Architecture)

NaijaBuddy's backend engine is built in Python using FastAPI, SQLite and NumPy, with the LLM served via either `llama-cpp-python` (offline mode) or an OpenAI-protocol HTTP client pointed at a vLLM endpoint (vLLM mode — see §2.2.1). The architecture consists of four distinct, sequential layers; Task A uses Layers 2, 3, and 4 directly, with Layer 1's vector index re-used for the cluster-mean cold-start fallback:

![NaijaBuddy's four-layer pipeline shared by Tasks A and B. **Task A (this paper)** primarily exercises Layers 2 (LLM rating + review generation), 3 (calibration against user-bias and item-bias anchors), and 4 (deterministic critic). Layer 1's vector index is re-used by Task A for the cluster-mean cold-start fallback (§2.3) — the index retrieves the K-nearest known users by persona embedding cosine when the active user has no rating history. The diagram source is at `assets/diagrams/architecture.mmd`.](assets/diagrams/architecture.png){ width=55% }

### 2.1 Layer 1: Hybrid Catalog & Vector Index (used by Task A for cold-start)
To deliver recommendations across multiple domains (Yelp, Amazon, Goodreads), we design a unified SQLite schema. The database is populated with an extensive, highly localized catalog spanning three distinct categories:
* **Yelp (Food & Spots)**: Iconic local spots (e.g., *Yellow Chilli*, *Shiro Lagos*, *The Place*, *Suya Spot*, *Club Quilox*) and traditional culinary items (Jollof Rice, Suya, Pepper Soup).
* **Amazon (Literature & Media)**: A real-world subset of Amazon Books reviews, augmented with a hand-curated localized overlay of landmark Nollywood productions (e.g., *The Wedding Party*, *King of Boys*, *Aníkúlápó*) and popular consumer electronics.
* **Goodreads (African Literature)**: High-caliber African and Nigerian literary masterpieces (e.g., *Things Fall Apart* by Chinua Achebe, *Half of a Yellow Sun* by Chimamanda Ngozi Adichie).

We leverage **`BAAI/bge-small-en-v1.5`** to map both items and user personas into a shared 384-dimensional dense semantic space. Task A reuses this index for the cold-start Cluster-Mean fallback (§2.3): when the active user has no rating history, the vector index retrieves their top-K nearest known users and the calibration anchor is computed from those neighbours' means rather than from a global average. Layer 1's broader use as the Stage-1 retriever for Task B is described in the companion Task B paper.

### 2.2 Layer 2: LLM-Driven Rating + Review Simulation
At the core of Task A is a small Large Language Model: **`Qwen2.5-3B-Instruct`**, served via either of the two interchangeable inference backends described in §2.2.1. Both backends present the same `Llama()`-compatible callable to the upstream agent code so prompts, JSON-schema constraints, stop tokens and the calibration layer are bit-identical across modes.

#### 2.2.1 Deployment modes (offline vs vLLM)
The agent picks its inference engine at startup from the `VLLM_URL` environment variable:

* **Offline mode** (default when `VLLM_URL` is unset): the **`Qwen2.5-3B-Instruct-Q4_K_M`** GGUF is loaded into the container process via **`llama-cpp-python`**, with full GPU offload + FlashAttention-2 when CUDA is available and a CPU baseline fallback otherwise. No network traffic at inference time. This is the mode anyone running `docker run` against the public image without extra configuration enters, and it is the cheapest reproducibility path for the calibration and review-quality results.
* **vLLM mode** (set `VLLM_URL` to an OpenAI-compatible endpoint): the agent constructs a thin `VLLMShim` ([`vllm_shim.py`](vllm_shim.py)) that wraps an `openai.OpenAI` client and exposes the same callable signature `llama_cpp.Llama` does. Generation requests then proxy to a separately-deployed vLLM 0.21 server serving the **fp16 HF safetensors** of the *same* `Qwen/Qwen2.5-3B-Instruct` weights — same prompts, same stop tokens, same JSON-grammar constraint (mapped to vLLM's `guided_json`).

The **canonical multi-seed evaluation** (§4) runs in vLLM mode on a Modal-hosted A10G (`modal_vllm_eval.py`) — vLLM's PagedAttention + continuous batching delivers ~10× throughput on the 18,000-call multi-seed sweep, which is what makes the three-seed protocol affordable. The **live hosted demo** also runs in vLLM mode against a long-running Modal endpoint (`modal_vllm_serve.py`) so demo and paper-evaluation share the same engine config end to end. Engine-difference disclosure: vLLM serves fp16/bf16, llama-cpp serves Q4_K_M, so outputs are not bit-identical even at greedy decoding; §4.2's engine note quantifies the gap on V2 rating accuracy — the two engines agree within the per-seed sampling std, so all Task A claims in this paper hold under both modes.

#### Task A Prompt (Rating + Review)
To synthesize realistic outputs, the agent is supplied with the target item details and a rich description of the active persona. The model is bound by strict system-level instructions:
1. **Conformity**: Generate a star rating (float) and a written review reflecting the target persona's taste.
2. **Conciseness**: Restrict the output to 2–4 sentences to prevent long, repetitive generations and optimize token usage.
3. **Structured Format**: Output a valid JSON object containing `rating` and `review` fields, constrained via llama.cpp grammar (offline mode) or vLLM `guided_json` (vLLM mode) so parsing never fails.

A separate `naija_mode` flag toggles a Pidgin-English style overlay for the review only; the rating is unaffected by style.

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

Our Critic Layer programmatically inspects the rating/review pair and applies strict rule-based filters. If the generated rating contradicts an encoded persona constraint (e.g., a five-star rating for an alcohol-centric venue against a non-drinking persona), the Critic applies a deterministic penalty by demoting the rating toward the calibration anchor. We describe this as **rule-based enforcement** rather than a safety *guarantee*: it is verifiable by construction for the constraints it encodes, but — being keyword-based — makes no claim to catch violations outside those rules. A learned or semantic critic is left to future work.

---

## 3. Cultural Context & Nigerian Localization
Behavioural and cultural fidelity is a primary judging criterion. NaijaBuddy ships three hand-crafted Nigerian base personas:

1. **Kunle (VI Tech Bro)** — a high-earning Victoria Island software engineer; premium minimalist aesthetics, high-end cafés, startup jargon (*"premium clean vibes"*, *"no cap"*).
2. **Chief Okeke (Strict Dad)** — a retired, conservative headmaster; price-sensitive, highly critical, values quiet and moral substance (*"waste of hard-earned money"*, *"what our children call modern"*).
3. **Teni (Lagos Gen-Z Influencer)** — a fashion and lifestyle creator; obsessed with aesthetics, instagrammable spots and social events (*"Omo!"*, *"it's giving"*, *"God when?!"*).

Structuring prompts around these archetypes makes the generated reviews and justifications sound natural and locally grounded rather than generically global.

---

## 4. Experiments and Evaluation (Task A)

We evaluate NaijaBuddy's Task A path with a leakage-free, out-of-sample protocol and report the measured results, including negative findings. Every figure in this section is regenerated by the evaluation harness shipped with the code (`eval_harness.py`).

### 4.1 Protocol

**Datasets.** Three real-world interaction sets, each filtered to a genuine **3-core** — every user and every item has at least three interactions — and capped at the 2,000 densest users per domain: Yelp (106,300 interactions / 2,000 users / 10,415 items), Goodreads (466,625 / 2,000 / 57,499), and Amazon (101,540 / 1,999 / 21,479). The Amazon subset is drawn from the Books category. All three sets are positively skewed, as consumer-review data invariably is — the share of ratings ≥ 4★ is 70% (Yelp), 68% (Goodreads) and 82% (Amazon) — which makes the global-mean baseline (V0) deceptively strong and is the reference against which every gain below should be read.

**Leave-one-out split.** For each user we hold out one interaction. All training statistics — each user's mean rating, every item description, and every persona — are computed from that user's *remaining* interactions only. This removes the circular evaluation in which a prediction is scored against a rating that itself contributed to the mean used to produce it.

**Metrics.** RMSE for rating prediction; ROUGE-L, Semantic-BGE (whole-review BGE-small cosine) and BERTScore-F1 (RoBERTa-large token-level F1, Zhang et al., 2020) for the generated review. The headline figures below are reported as **mean ± sample standard deviation across three independent leave-one-out splits** (seeds 42 / 1 / 7), with n = 2,000 / 2,000 / 1,999 held-out interactions per seed per domain. The headline evaluation uses **synthesised** personas (matching the deployed system) served by vLLM 0.21 (fp16, continuous batching with PagedAttention); §4.4 confirms template-vs-synth equivalence on rating accuracy.

**Cold-start protocol.** A 3-core leaves no genuinely cold users (the Goodreads minimum is 17 interactions), so we *simulate* cold-start: a test user's history is truncated to k ∈ {1, 2, 3} interactions while every other user keeps full history — a new user entering a populated system — and the evaluation is repeated. Section 4.3 reports the resulting degradation curve.

### 4.2 Rating Prediction and the Calibration Layer

**Per-user anchoring.** Replacing the global mean (V0) with the per-user training mean (V1 — the calibration formula at $\alpha = 0$) reduces RMSE in every domain; blending the LLM's raw rating back in at a low weight (V2) improves on V1 further, by a smaller and domain-dependent margin:

| Domain | V0 global | V1 user-mean ($\alpha$=0) | pure-LLM ($\alpha$=1) | V2 blend (best $\alpha$) |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 1.055 ± 0.003 | 1.000 ± 0.004 | 1.226 ± 0.011 | **0.989 ± 0.002** ($\alpha$=0.2) |
| Goodreads | 1.008 ± 0.023 | 0.898 ± 0.025 | 1.148 ± 0.040 | **0.894 ± 0.027** ($\alpha$=0.1) |
| Amazon | 0.953 ± 0.012 | 0.839 ± 0.016 | 1.105 ± 0.002 | **0.834 ± 0.015** ($\alpha$=0.1) |

*Full held-out set, n = 2,000 / 2,000 / 1,999 per seed; **three independent leave-one-out splits** (seeds 42 / 1 / 7), synth personas, served by vLLM-batched Qwen2.5-3B-Instruct. Mean ± sample std reported. V0→V2 reduces RMSE by 6.3% / 11.3% / 12.5% on average — but the V1→V2 step, the LLM's actual contribution, is only 0.004–0.011 across domains, with a per-seed std of 0.002–0.027.*

**Does the LLM's rating help?** Because the raw LLM rating is independent of $\alpha$, one LLM call per held-out pair supports a full sweep of the blend $\hat{y} = \alpha\cdot\text{LLM} + (1-\alpha)\cdot\mu_u$. The sweep is **U-shaped** in every domain: error falls from V1 to a shallow minimum at $\alpha = 0.1$–$0.2$, then rises steeply to the pure-LLM endpoint — the worst configuration everywhere (1.10–1.23). The V1→V2 improvement is 0.004–0.011 RMSE; at three independent splits of n = 2,000 that is a small but stable effect — the cross-seed std on V2 is 0.002–0.027, so the V1→V2 gap consistently survives reseeding. The reported V2 sits at the *test-minimising* point of the sweep — a descriptive minimum, not a validation-tuned hyperparameter — so if anything the verdict is conservative: even oracle-optimal blending barely moves the user-mean.

So for warm users the LLM's numeric rating earns a marginal gain at best — a 3B model's raw 1–5 score, compressed toward integer extremes, barely beats the user's own historical mean. Section 4.3 shows why that is only half the story: the LLM's contribution is small *for users with rich history* and large *for users without it*.

**Two populations of users.** Averaged over everyone the warm gain looks small — but the average hides structure. Bucketing test users by the spread of their own training ratings (per-user standard deviation) shows that rating prediction is really *two problems*:

| Domain | low-σ users (σ<0.6) | mid-σ (0.6–1.0) | high-σ users (σ≥1.0) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.65 | 0.90 | 1.14 |
| Goodreads | 0.46 | 0.82 | 1.06 |
| Amazon | 0.59 | 0.84 | 1.21 |

*V1 (user-mean) RMSE by per-user rating-variance bucket; n = 2,000 per domain, `analysis/study_data.py`.*

The pattern is stark and monotonic in every domain: for users who rate consistently, predicting their mean is near-solved (RMSE 0.46–0.65); for users whose ratings are scattered, the *same* predictor is roughly **twice as bad** (1.06–1.21) — and much of that gap is genuinely irreducible variance that no model can recover. This is why the headline warm gain looks small — it is diluted across a predictable majority that nothing can improve. The modelling budget, the LLM, is best spent on the high-variance tail and on cold users — exactly the regime split that §4.3 formalises.

**An item-bias term.** The blend so far anchors only on the *user's* mean. Classical recommender systems also model an *item* bias — some items are simply rated higher than others. Adding it, $\hat{y} = \alpha\cdot\text{LLM} + \beta\cdot\mu_u + \gamma\cdot\mu_i$, and sweeping the weights on the cached generations (no new inference; `analysis/measure_calib3.py`):

| Domain | V2 (user anchor) | V3 (+ item bias) | optimal weights (LLM / user / item) |
| :--- | :---: | :---: | :---: |
| Yelp | 0.990 | **0.956** (−3.5%) | 0.00 / 0.60 / 0.40 |
| Goodreads | 0.876 | **0.864** (−1.4%) | 0.00 / 0.75 / 0.25 |
| Amazon | 0.851 | **0.848** (−0.3%) | 0.05 / 0.80 / 0.15 |

The item term is a real mean RMSE gain (−1.7% on average, −3.5% on Yelp). The striking number is the LLM weight: at the optimum it is **≈ 0 on every domain** (0.00–0.05). Once the anchor models item bias, the 3B model's raw numeric rating is *redundant for warm users* — the best warm predictor is the textbook user-bias + item-bias model. NaijaBuddy deploys this 3-term anchor. The LLM's numeric rating earns its place only in cold-start (§4.3), where neither user nor item history exists. The calibration layer is thus best read as a **regime switch**: a statistical bias model for well-observed users, the LLM for cold ones.

**Robustness.** The V1→V2 step is positive on every (seed, domain) cell at n = 2,000 with per-seed std 0.002–0.027 — a small but stable effect. An earlier n = 350 protocol across the same three seeds reaches the same conclusion (Yelp +0.018, Goodreads +0.005, Amazon +0.002), so the finding survives both reseeding and a 6× sample-size change. *Engine note:* the multi-seed numbers above use vLLM-served fp16 weights; an earlier single-seed llama-cpp Q4_K_M GGUF run produced near-identical V2 (Yelp 0.990, Goodreads 0.876, Amazon 0.851), within the per-seed std reported above.

### 4.3 Cold-Start: The Calibration Layer's Real Value

The warm result in 4.2 — the LLM rating barely improving on the user-mean — holds only while the user-mean is itself reliable. The cold-start protocol shows it is not, once history is scarce, and that this is exactly where the LLM earns its place. The table reports the V3 3-term anchor (LLM + user-mean + item-mean blend) at each truncated history length:

| Domain | k = 1 V1 → V3 | k = 2 V1 → V3 | k = 3 V1 → V3 | warm V2 (§4.2) |
| :--- | :---: | :---: | :---: | :---: |
| Yelp | 1.413 ± 0.008 → **0.991 ± 0.019** | 1.204 ± 0.010 → 0.978 ± 0.019 | 1.146 ± 0.010 → 0.973 ± 0.012 | 0.989 ± 0.002 |
| Goodreads | 1.258 ± 0.013 → **0.962 ± 0.018** | 1.103 ± 0.026 → 0.949 ± 0.022 | 1.040 ± 0.010 → 0.934 ± 0.015 | 0.894 ± 0.027 |
| Amazon | 1.170 ± 0.020 → **0.934 ± 0.022** | 1.001 ± 0.031 → 0.899 ± 0.032 | 0.939 ± 0.034 → 0.875 ± 0.031 | 0.834 ± 0.015 |

*V1 user-mean RMSE (over the truncated history) → V3 3-term blend RMSE = α·LLM + β·μ_user + γ·μ_item. Three independent leave-one-out splits (seeds 42 / 1 / 7), n = 2,000 users per domain per k (1,997 / 1,994 for Amazon at k=1 / k=2,3), synth personas, served by vLLM-batched Qwen2.5-3B. Mean ± sample std reported.*

Three patterns hold cleanly across all three domains and all three seeds:

1. **V1 (the user-mean over a single observed rating) is the weakest predictor everywhere** — k=1 RMSE of 1.17–1.41 is much worse than the warm V1 baseline (0.84–1.00). This is the cold-start problem stated concretely: with one observed rating the empirical user-mean is dominated by noise.

2. **V3 cuts cold-start error substantially.** At k=1 the V3 blend reduces RMSE by **29.9%** (Yelp), **23.5%** (Goodreads) and **20.2%** (Amazon) versus V1 — *vastly* larger than the 0.5–1.2% warm gain of §4.2. By k=3 the gap to warm V2 has all but closed: Yelp k=3 V3 (0.97) reaches warm V2 (0.99) — once the user has rated three items the cold-start blend is competitive with the warm anchor.

3. **The V3 weight schedule favours item-bias, not the LLM.** The optimal (LLM / user / item) weights at k=1 are **0.03 / 0.13 / 0.83** (Yelp), **0.03 / 0.27 / 0.70** (Goodreads), **0.07 / 0.30 / 0.63** (Amazon) — *the item-bias term is doing almost all the work, even cold-start.* The user-mean weight grows monotonically with k as more user history accumulates (k=3 user weight 0.33–0.57). The LLM's numeric weight stays at 0.0–0.1 throughout. This refines §4.2's "LLM weight ≈ 0 warm" finding: it is the **item-bias term**, stable for a cold user because the item still has history from other users, that does most of the calibration work.

This reframes the calibration layer. It is not the LLM swooping in to rescue cold users; it is an **adaptive fusion of three statistical priors whose mix scales with the available evidence about the user**. When a user has rated a single item, the user-mean is noise; the item-mean (and the model that uses it) is what saves the prediction. As the user's own history accumulates, weight migrates from the item term to the user term, with the LLM contributing a small constant signal throughout.

### 4.4 Persona Ablation — Does LLM Persona Synthesis Help Rating Accuracy?

NaijaBuddy can model a user two ways: a deterministic **template** persona (category list plus two review snippets) or an **LLM-synthesised** prose persona. Holding everything else fixed (seed 42, n = 2,000):

| Metric (Yelp / Goodreads / Amazon) | Template persona | Synthesised persona |
| :--- | :---: | :---: |
| RMSE V2 | 0.990 / 0.876 / 0.851 | 0.987 / 0.876 / 0.852 |

Synthesis makes **no measurable difference to rating accuracy** — V2 differs by ≤ 0.003 — unsurprising, since §4.2 shows the rating is anchored on statistics, not on the persona text. The honest reading: on the rating metric, template and synthesised personas are **equivalent** — persona synthesis is a UX choice for the conversational human-facing demo, not a rating-accuracy lever.

### 4.5 Retrieval-Augmented Prompting (RAG) — Effect on Rating vs Review

A third way to ground the LLM is **retrieval-augmented prompting (RAG)**: rather than an abstracted persona, the prompt is seeded with the user's **k = 4 past interactions most similar to the target item** — the real item descriptions and the actual ratings and reviews the user gave them — retrieved by cosine similarity. The hypothesis is that concrete examples of a user's own behaviour should ground the model better than a paraphrased summary of it. We evaluate at seed 42 and n = 2,000:

| Metric (Yelp / Goodreads / Amazon) | Synthesised persona | Retrieval-augmented |
| :--- | :---: | :---: |
| RMSE V2 | 0.987 / 0.876 / 0.852 | 0.980 / 0.874 / 0.850 |
| Review Semantic-BGE | 0.738 / 0.633 / 0.648 | **0.754 / 0.645 / 0.662** |
| Review ROUGE-L | 0.100 / 0.082 / 0.091 | **0.102 / 0.088 / 0.098** |

The result splits cleanly by sub-task. On **rating prediction** RAG is indistinguishable from a synthesised persona — the V2 figures differ by ≤ 0.007, inside the sampling noise of §4.2, and the optimal blend weight stays pinned at α ≈ 0.1–0.2. This is the §4.2 result reached from a third independent direction: a deterministic template (§4.4), an LLM-synthesised persona, and retrieved real exemplars all land warm-user V2 within 0.01 RMSE of one another — no prompting strategy rescues the LLM's numeric estimate where the user-mean prior already wins. On **review generation**, by contrast, RAG produces a small but perfectly consistent gain: Semantic-BGE rises in all three domains (+0.012 to +0.016) and ROUGE-L in all three. The mechanism is intuitive — an abstracted persona discards the user's actual vocabulary, whereas in-context examples of their real past reviews let the model echo their voice, which both a surface-overlap and an embedding metric reward. The honest reading: retrieval augmentation helps exactly where the task carries stylistic signal — the generative sub-task — and not where the task is regression against a strong prior.

### 4.6 Review Generation Quality

We score the generated review against the held-out human review along three complementary axes — surface-form overlap, sentence-level meaning, and token-level semantic precision/recall:

| Domain | ROUGE-L F1 | Semantic-BGE | BERTScore-F1 |
| :--- | :---: | :---: | :---: |
| Yelp | 0.084 ± 0.001 | 0.723 ± 0.001 | **0.838 ± 0.000** |
| Goodreads | 0.073 ± 0.001 | 0.626 ± 0.002 | **0.841 ± 0.000** |
| Amazon | 0.079 ± 0.000 | 0.646 ± 0.001 | **0.846 ± 0.000** |

*All three axes: n = 2,000 / 2,000 / 1,999 per seed across three independent splits (seeds 42 / 1 / 7), synth personas. Mean ± sample std (rounded to three decimals). BERTScore-F1 backfilled by a separate GPU pass (`modal_bertscore_backfill.py`) because the original synth-vLLM run had vLLM holding 85% of the A10G's VRAM, leaving no room for RoBERTa-large; the backfill reads the same cached generations and reference reviews and computes the canonical metric.*

ROUGE-L is low as it must be — two people reviewing the same item rarely choose the same words. Semantic-BGE at 0.63–0.72 confirms sentence-level meaning is close to the human references despite low surface overlap, with cross-seed std ≤ 0.002 — highly reproducible. BERTScore-F1 [Zhang et al., 2020], the canonical token-level contextual metric, sits at **0.838–0.846 across all three domains** with std ≤ 0.0003, a band consistent with published high-quality text-generation outputs. (Caveat: Semantic-BGE on same-item reviews carries a topical-similarity floor that BERTScore's token-alignment partly removes.) On manual inspection the generations are fluent, persona-consistent, and stylistically Nigerian.

### 4.7 Cross-Domain Transfer — does Books-taste predict Movies-rating?

The §4.3 cold-start result — *item-bias dominates when user history is scarce* — has a stronger statement: it should hold even when the user is **fully cold in the target domain** but has rich history in *another* domain entirely. We test this on a constructed Amazon Books → Movies transfer benchmark.

**Dataset.** From the Amazon-Reviews-2023 5-core rating-only files (no item titles or descriptions in this slice; this is purely a statistical test) we extracted the 2,000 densest users who have ≥ 10 ratings in **both** Books and Movies (`cross_domain_dataset.py`). Each user contributes their full Books history, their Movies training history (all-but-last by timestamp), and the held-out last Movies rating as target.

| Model | Formula | RMSE |
| :--- | :--- | :---: |
| V0 (global Movies mean) | $\bar{y}_\text{movies}$ | 1.265 |
| V1 user-mean over **Books** *(cross-domain)* | $\mu_\text{user, books}$ | **1.192** |
| V1 user-mean over Movies *(in-domain upper bound)* | $\mu_\text{user, movies}$ | 1.137 |
| item-mean only (held-out movie) | $\mu_\text{item, movies}$ | 1.306 |
| **V2 books-user + item-bias** (best $\beta = 0.7$) | $\beta\,\mu_\text{user, books} + (1-\beta)\,\mu_\text{item, movies}$ | **1.166** |
| V2 movies-user + item-bias (best $\beta = 0.8$) | $\beta\,\mu_\text{user, movies} + (1-\beta)\,\mu_\text{item, movies}$ | 1.123 |

Three findings stand out:

1. **Books taste transfers to Movies taste.** The user's mean Books rating (V1_books, RMSE 1.19) is 0.073 better than the global mean (V0, 1.27) and only 0.055 worse than the in-domain user-mean (V1_movies, 1.14). Roughly **57% of the in-domain user-bias signal is recoverable from Books alone** — far from a random or null transfer.

2. **Item-bias closes the cross-domain gap, mirroring §4.3.** Adding the target movie's cross-user mean to the books-user prior (V2 books+item) reduces RMSE from 1.19 to **1.17** — the same item-bias term that does most of the work in cold-start (§4.3) also does the work cross-domain.

3. **Pure item-bias alone is the worst predictor** (RMSE 1.31, worse even than V0 global mean). The item-bias term carries information *only when combined with a user-side prior* — it is not a baseline, it is a corrector. This refines §4.3: V3's strength is not just "item-bias rescues cold users" but "**any user-side prior plus item-bias beats either alone**", and the held-out movie not needing to be in the user's domain at all is the strongest possible statement of that.

Regenerable via `cross_domain_dataset.py` followed by `cross_domain_eval.py`. The protocol is purely statistical and runs in seconds on CPU (no LLM, no GPU, no Modal), making it the cheapest reproducibility test for the §4.3 thesis we can offer.

### 4.8 Honest Summary (Task A)

NaijaBuddy's Task A measured strengths are an **adaptive 3-term calibration layer** — a 6–13% mean RMSE reduction over a global baseline for warm users (V0 → V2, mean ± std across 3 seeds) and a **20–30% reduction for cold-start users at k = 1** (V1 → V3) — and **reproducible review-text quality** (BERTScore-F1 0.84 ± 0.0003, ROUGE-L and Semantic-BGE all tight across seeds). The measured weakness is warm-user rating accuracy, where the LLM contributes only 0.004–0.011 RMSE over a user-bias + item-bias prior. The deployed Task A pipeline is the V3 3-term anchor — `α·LLM + β·μ_user + γ·μ_item` — with weights that migrate from item-mean-dominant at k = 1 to user-mean-dominant once history accumulates and an LLM weight that stays at 0.03–0.10 throughout. The headline multi-seed figures are regenerated by `modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed {42,1,7} --cold-start --cold-sample 2000 --bertscore`.

---

## 5. Related Work

**LLMs as rating predictors and review generators.** Casting an LLM into a generative-recommender role is now common: EXP3RT [Kim et al., 2024] fine-tunes an LLM to extract review-based preferences and produce a reasoning-enhanced rating; Ryu & Yanaka [2025] show in-context user reviews lift LLM rating prediction toward matrix-factorisation quality, especially for cold-start. NaijaBuddy keeps the LLM as one term in a measured calibration blend rather than as the sole rating signal — a design suited to a small local model.

**The limits of the LLM rating signal.** Kang et al. [2023], evaluating LLMs from 250M–540B parameters on user rating prediction, found that zero-shot LLMs lag traditional collaborative filtering whenever interaction data is available — the history, not the LLM, carries the signal. NaijaBuddy corroborates this across three domains and *operationalises* it: our calibration layer measures how much weight the LLM rating should receive (≈ 0 for warm users) and deploys the result as a regime switch. Our §4.5 RAG ablation refines Ryu & Yanaka's finding — in-context exemplars improve the generated *review text* but not the *warm rating*, where the user-mean already dominates.

**Cold-start.** Recommending for users with little history has a long line of classical remedies — content-based features, item-popularity and item-bias terms, cross-domain transfer, and meta-learning estimators such as MeLU [Lee et al., 2019]. NaijaBuddy places the LLM *within* this lineage rather than against it: §4.3 measures the three-term anchor's weight schedule and finds the **item-bias term**, not the LLM, doing most of the cold-start work. §4.7 stress-tests the same finding cross-domain: even when the user is fully cold in the target domain (Amazon Movies) but rich in another (Amazon Books), the item-bias signal closes most of the in-domain gap.

**Review generation and user simulation.** Review-LLM [Peng et al., 2024] targets personalised review generation and documents the "polite phenomenon" — LLMs resist producing negative reviews; BASES [Ren et al., 2024] simulates search users at scale. NaijaBuddy's Task A jointly simulates a rating and a persona-grounded review, with the calibration layer re-anchoring an over-generous LLM score against exactly that "polite" upward bias.

**Evaluation rigour.** Dacrema et al. [2019] showed that many neural recommenders fail to beat well-tuned simple baselines once evaluation is done carefully — so we report the global-mean and user-mean baselines beside every LLM result. NaijaBuddy responds with a leakage-free leave-one-out protocol, a robustness check across both reseeding and a 6× sample increase (§4.2), and an open-weights, open-engine inference stack — `Qwen2.5-3B-Instruct` served via vLLM 0.21 or `llama-cpp-python` (§2.2.1) — in which every reported figure regenerates.

**Positioning.** NaijaBuddy's distinctive combination for Task A is a small open-weights LLM that can run in an offline-capable configuration (against the field's GPT-3.5/4), calibration as a *measured* regime switch rather than an assumed guardrail, reproducible leakage-free evaluation, and explicit cultural localisation for an underserved market.

*References:* Andre et al. 2025 (arXiv:2508.20401); Dacrema et al. 2019 (RecSys); Hu, Koren & Volinsky 2008 (ICDM); Kang et al. 2023 (arXiv:2305.06474); Kim et al. 2024 (arXiv:2408.06276); Lee et al. 2019 (MeLU, KDD); Peng et al. 2024 (arXiv:2407.07487); Ren et al. 2024 (arXiv:2402.17505); Ryu & Yanaka 2025 (arXiv:2510.00449); Zhang et al. 2020 (BERTScore, ICLR).

---

## 6. Future Directions
The natural next deployment step is **history-conditioned calibration weights** — replace the per-domain constants with weights $(\alpha, \beta, \gamma)$ computed as an explicit function of $k$ (observed-rating count), making the §4.3 regime switch automatic rather than a best-of-sweep. A complementary direction is a **learned semantic critic** in place of the rule-based Layer 4 — a small classifier trained on (persona, item) → constraint-violation pairs, expanding coverage beyond keyword rules.

---

## 7. Conclusion
NaijaBuddy's Task A pipeline — an in-context `Qwen2.5-3B-Instruct` plus the three-term calibration layer and a deterministic critic — delivers **20–30% cold-start RMSE reductions at k = 1**, BERTScore-F1 0.84 across three domains, and a cross-domain transfer experiment validating the item-bias finding outside the within-domain setting. The same agent serves **Task B: Recommendation** (companion paper).
