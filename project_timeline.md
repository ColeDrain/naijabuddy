# NaijaBuddy — Project Timeline & Storyline

A narrative of how the project unfolded: the problem, the build, and the
experiment arc that produced the findings. Written as raw material for the
presentation deck — the story is *"we set out to build an LLM recommender, and
the experiments kept telling us something more interesting than we expected."*

---

## 1. The brief

The DSN × BCT LLM Agent Challenge sets two tasks: **(A) user modeling** —
simulate a user's review and star rating for an unseen item; **(B)
recommendation** — rank items for a user. Datasets: Yelp, Amazon, Goodreads.
Scored on a 100-point rubric in which the **solution paper is read first**, and
*"originality and clarity of reasoning"* is rewarded above raw performance.

## 2. The thesis — offline-first

We made one decision early and held it: **everything runs locally, offline, in
one container** — a quantized 3B model (Qwen2.5-3B-Instruct GGUF) and a small
embedding model (BGE-small), no cloud API at any point. The bet: in a
talent-identification contest, a working *edge* agent — and an honest account
of what a small local model can and cannot do — is a stronger signal than a
wrapper around a frontier API.

## 3. The build

A four-layer agent: (1) hybrid retrieval — dense BGE search blended with
item-item collaborative filtering; (2) a local-LLM reranker / review simulator;
(3) a mathematical **calibration layer** that blends the LLM's rating with
statistical baselines; (4) a deterministic **critic layer** for hard
constraints. Served by FastAPI in a self-contained Docker image with a no-build
React dashboard.

## 4. The experiment arc — the actual story

The interesting part was not the build; it was what the evaluation kept saying.

- **The false start.** An early run scored the LLM blend on a 10-sample subset
  and reported a Goodreads RMSE of 0.60. It was sampling noise — the real
  number is ~0.94. Lesson, taken literally: build a **leakage-free,
  full-held-out, multi-seed** harness, and never trust a small sample again.

- **The uncomfortable result.** With the rigorous harness, the LLM's raw star
  rating *barely beat the user's own historical mean* for warm users. The
  honest move was not to hide it — it was to ask **why**.

- **Two populations.** Bucketing users by their rating variance showed rating
  prediction is really two problems: a predictable majority (mean ≈ solved)
  and a high-variance tail (irreducibly hard). The headline gain looked small
  because it was diluted across users nothing can improve.

- **The item-bias term.** Adding a classical item-bias term to the anchor beat
  the LLM blend outright — and at the optimum the LLM's weight fell to ≈ 0.
  For warm users, the 3B model's numeric rating is *redundant*.

- **The reframe — a regime switch.** Cold-start simulation flipped it: with one
  or two interactions the user-mean is noise, and the LLM's persona-grounded
  estimate cuts RMSE 13–15%. So the calibration layer is not a guardrail — it
  is a **regime switch**: statistics for well-observed users, the LLM for cold
  ones. That became the paper's central finding.

- **Triangulation.** A retrieval-augmented-prompting ablation (Tier 2) reached
  the same conclusion from a third angle: no prompting strategy rescues the
  LLM's warm rating, though it does improve the generated review *text*.

## 5. The honest ledger

Nine experiments, each logged with config and verdict (`EXPERIMENTS.md`); every
paper figure mapped to the command that regenerates it
(`numbers_integrity.md`); a self-audit confirming the harness is leakage-free
(`validation_audit.md`). Negative and mixed results — content-only retrieval
losing to popularity, persona synthesis being domain-dependent — are reported,
not buried.

## 6. Outcome

A deployed, offline agent for both tasks, plus a paper whose contribution is
less "our system wins" and more "here is precisely *when* an LLM helps a
recommender and when a textbook bias model is better — measured, not assumed."
That measured honesty is the submission's spine.
