---
title: NaijaBuddy
emoji: 🤖
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: Offline-first agentic recommender for the Nigerian market
hardware: cpu-upgrade
suggested_hardware: cpu-upgrade
---

# NaijaBuddy: A Highly Localized, Offline-Capable Agentic Recommender & User-Modeling System

NaijaBuddy is an offline-capable, containerized agentic recommendation system and review simulator tailored to the Nigerian consumer market.

It implements a two-stage **retrieve-then-rerank** pipeline over a hybrid database that combines real-world dataset subsets (Yelp, Amazon, Goodreads) with a localized catalogue of Nigerian establishments, movies, and literature. The system runs a quantized local LLM (`Qwen2.5-3B-Instruct`) and an embedding model (`BAAI/bge-small-en-v1.5`) via `llama-cpp-python` / `vLLM` and `sentence-transformers`. The **canonical multi-seed evaluation** runs entirely on a self-contained image with no cloud APIs (`Qwen2.5-3B-Instruct` GGUF via `llama-cpp-python`, or HF safetensors via `vLLM`, both batched on a single A10G).

**Hosted demo deployment** (this Space): the UI runs on `cpu-upgrade` here and proxies inference to a Modal-hosted vLLM endpoint serving the same `Qwen2.5-3B-Instruct` safetensors used in the canonical eval. This split is a pragmatic response to HF Spaces' Docker SDK constraints (no `/dev/shm` config, no GPU at build time) — the *system* is offline-capable; the *hosted demo* trades that property for fast, paper-engine-aligned inference (~1-3s per call vs ~30-50s for CPU/partial-offload llama-cpp on the same Space hardware).

---

## 📄 Solution Papers

This repo ships **three** papers covering the same NaijaBuddy artefact at different scopes — the unified paper is the complete architectural and evaluation reference; the two task-focused papers exist because the DSN×BCT submission form is structured around per-task submissions:

| File | Scope | Purpose |
| :--- | :--- | :--- |
| [`solution_paper.md`](solution_paper.md) | Both tasks, full system | Complete reference: architecture, calibration, all §4 results (rating + review + retrieval + cold-start + cross-domain) |
| [`solution_paper_task_a.md`](solution_paper_task_a.md) | Task A only | User Modeling submission: §4 focuses on RMSE/calibration, review quality (BERTScore/ROUGE/BGE), cold-start, persona ablation, RAG, cross-domain transfer |
| [`solution_paper_task_b.md`](solution_paper_task_b.md) | Task B only | Recommendation submission: §4 focuses on multi-cutoff HR/NDCG retrieval, hybrid vs ALS vs CF, persona-on-retrieval ablation, cold-start retrieval |

The Task A and Task B papers share §1 (intro), §2 (architecture), §3 (cultural context), §5–7 (related work / future / conclusion); §4 (experiments) is the task-relevant subset of the unified paper. Per the brief's *"Two tasks, one ambition"* framing, the same agent class (`NaijaBuddyAgent`) and the same Docker image serve both tasks — `simulate_review_adhoc` for Task A and `recommend_adhoc` for Task B.

---

## 🚀 Quick Start (Docker)

The Dockerfile downloads the BGE-small embedding model, ingests the bundled dense datasets and seeds the database **at build time**, so the runtime image starts in seconds. LLM inference is configurable: the agent reads `VLLM_URL` at startup and routes generations through a vLLM endpoint when set, falling back to mock responses otherwise (the deployed HF Space runs in this mode against `modal_vllm_serve.py`). For a fully offline-mode runtime, see [§ Offline Mode](#-offline-mode-no-cloud-llm) below — that mode pulls Qwen2.5-3B-Q4_K_M GGUF (~2.2 GB) into the image and serves inference in-process via llama-cpp-python.

### 1. Build the image
An internet connection is required *only* during the build, to fetch packages and the embedding model.
```bash
docker build -t naijabuddy .
```

### 2. Run the container

Pointing at a vLLM endpoint (recommended; matches the canonical eval engine):
```bash
docker run -p 8000:8000 \
  -e VLLM_URL=https://your-vllm-endpoint/v1 \
  naijabuddy
```

Or stand-alone with no LLM (UI + mock responses only — useful for UI dev):
```bash
docker run -p 8000:8000 naijabuddy
```

* **Interactive UI:** `http://localhost:8000/`
* **REST API docs:** `http://localhost:8000/docs`

### 🔌 Offline Mode (no cloud LLM)
For a fully self-contained runtime, unset `NAIJABUDDY_SKIP_QWEN` in the Dockerfile (currently `=1`) and re-build — `downloader.py` will then fetch the Q4_K_M GGUF and the agent loads it via `llama-cpp-python` at startup. No network calls at inference time. See `modal_vllm_serve.py` for the equivalent serving-side endpoint you would point `VLLM_URL` at to instead match the canonical evaluation engine (vLLM 0.21 + Qwen2.5-3B fp16 safetensors).

---

## 🛠️ Local Development Setup

### 1. Create and activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download models and build the database
Run these once, in order:
```bash
# 1. Download the Qwen-3B GGUF LLM and the BGE-small embedding model (~2.2 GB)
python downloader.py

# 2. Ingest the bundled dense datasets (data/*.csv) and seed the catalogue
python data_enricher.py

# 3. Pre-synthesize all user personas with the local LLM
#    (runs the LLM once per user — expect roughly an hour on CPU/Metal)
python generate_personas.py
```
Step 3 is recommended but optional: if skipped, a user's persona is synthesized
lazily on first request and cached.

### 4. Run the web server
```bash
python app.py
```
The server listens on `http://localhost:8000/` (override with the `PORT` env var).

---

## 📊 Evaluation Harness

`eval_harness.py` runs a leakage-free, out-of-sample evaluation on the real-world
users. It performs a **leave-one-out split** — holding out one interaction per
user and rebuilding every user mean, item description, and persona from
*training interactions only* — then reports:

* **RMSE** (rating prediction): `V0` global-mean baseline, `V1` per-user mean
  (the calibration formula at α = 0), and `V2` the LLM + user-mean blend with α
  swept over [0, 1].
* **ROUGE-L** and **Semantic-BGE** — generated review vs. the real held-out
  review (verbatim overlap and embedding similarity).
* **HitRate@10 / NDCG@10** — retrieval, for dense / hybrid / collaborative-
  filtering / popularity strategies.
* **Cold-start** — RMSE and HitRate@10 as user history is truncated to k = 1, 2, 3.

```bash
# default run — behaves exactly as the original harness
python eval_harness.py

# full held-out set, every pair scored (no sub-sampling)
python eval_harness.py --llm-sample 400

# RMSE V0/V1 + retrieval only, no LLM
python eval_harness.py --no-llm

# comprehensive run — cold-start curve, semantic metric, synthesised personas
python eval_harness.py --llm-sample 400 --cold-start --bertscore --persona-mode synth

# several leave-one-out splits → mean ± std error bars
python eval_harness.py --seeds 42,1,7
```

| Flag | Effect |
|---|---|
| `--llm-sample N` | Held-out pairs per domain scored by the LLM (default 100; `400` = full set). |
| `--cold-start` | Add the cold-start degradation curve (`--cold-k`, `--cold-sample` tune it). |
| `--bertscore` | Add Semantic-BGE review similarity (true BERTScore if `bert-score` is installed). |
| `--persona-mode synth` | LLM-synthesised personas (matches the deployed system) vs. the `template` default. |
| `--seeds a,b,c` | Run several splits, report mean ± std. |
| `--no-llm` | Skip all LLM-dependent metrics. |

Every LLM generation is cached to `eval_artifacts/llm_cache.jsonl`, keyed by a
hash of the prompt — re-running with an unchanged prompt costs zero GPU. Results
are written to `evaluation_results.json` and `evaluation_results.md`.

---

## 🔬 Reproducing the Multi-Seed Canonical Results

The paper's headline numbers (§4.2 / §4.3 / §4.4 / §4.5 / §4.8) come from a
**3-seed (42 / 1 / 7) × n = 2,000 leave-one-out × synth-persona × vLLM-served**
canonical evaluation. The harness is engine-portable: `eval_harness.py` takes a
`--vllm-url` pointing at *any* OpenAI-compatible endpoint, so reproduction does
**not** require our specific Modal setup. Pick whichever path matches your
hardware:

### Path A — Self-host vLLM on any GPU (free if you own it)
```bash
# Terminal 1 — start a vLLM server
pip install vllm==0.21.0
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 --dtype auto

# Terminal 2 — run the harness against it
python local_data_prep.py          # one-time, regenerates dense CSVs (~10 min)
for s in 42 1 7; do
    python eval_harness.py --persona-mode synth --seed $s --llm-sample 2000 \
        --cold-start --cold-sample 2000 --bertscore \
        --vllm-url http://localhost:8000/v1
    mv evaluation_results.json scratch/results_s${s}.json
done
python analysis/aggregate_seeds.py --glob 'scratch/results_s*.json' \
    --out-md reproduction.md
# Expect reproduction.md numbers to land within ±0.03 RMSE of paper §4.2 V2
```

### Path B — Hosted OpenAI-compatible endpoint (RunPod / Together AI / similar)
Same as Path A, but `--vllm-url https://<your-endpoint>/v1`. Most providers
include a free credit that covers a full 3-seed re-run. The endpoint must
serve `Qwen/Qwen2.5-3B-Instruct` (or any other Qwen2-family model — the
harness is model-agnostic, just disclose any swap in your write-up).

### Path C — Verification-only on CPU (laptop, no GPU)
```bash
# Uses the in-process llama-cpp-python path (the same one this Space uses
# without GPU). Small sample so a CPU machine finishes in minutes.
python eval_harness.py --persona-mode synth --seed 42 --llm-sample 20
# Finishes in ~3 min on an M-series Mac or a modern Linux CPU.
# Numbers will be noisy at n=20 but should land in the right band.
```

### Path D — Replicate our exact Modal run (~$5 of Modal credit)
```bash
bash scripts/run_final_multiseed.sh
```
This is the path the paper's reported numbers came from. It uses
`modal_vllm_eval.py` to spin up vLLM in a Modal A10G container per seed,
runs the harness inside, commits each seed's cache to a persistent volume.
Modal account required. Three parallel containers, ~$3 total, ~50 min
wallclock for the full warm + cold-start sweep.

### Auxiliary reproductions (not on the critical path)
| Numbers | Command |
|---|---|
| §4.3 BERTScore-F1 multi-seed mean ± std | `modal run modal_bertscore_backfill.py` (or run `bert-score` locally over the cached generations) |
| §4.3 LLM-as-judge Behavioural Fidelity / Contextual Relevance | `modal run modal_llm_judge.py --provider {cerebras,groq}` (free tier covers a 3-seed run) |
| §4.8 cross-domain Books → Movies | `python cross_domain_dataset.py && python cross_domain_eval.py` — pure CPU, no LLM, no GPU |
| §4.4 sampled-metric protocol (101 candidates) | `python eval_harness.py --candidate-pool 101 --pop-distractors --seed 42` |

Every number in `numbers_integrity.md` maps to one of these commands.

---

## 📦 Datasets

The `data/` directory holds three pre-densified CSVs (Yelp, Goodreads, Amazon).
Each is a genuine **3-core**: every user and every item has at least 3
interactions. They are produced once, off-machine, by `colab_data_prep.ipynb`
(run on Google Colab) — which streams large samples of each public corpus,
extracts the dense bipartite core, and exports the CSVs. Committing them to the
repo keeps `data_enricher.py` fully offline and reproducible.

---

## 📂 Project Architecture

```
├── Dockerfile                  # Self-contained multi-stage Docker build
├── README.md                   # This guide
├── requirements.txt            # Python dependencies
├── app.py                      # FastAPI REST server + static UI mount
├── agent.py                    # Agent: rerank, calibration, critic, persona synthesis
├── database.py                 # SQLite schema + pure-Python cosine vector search
├── downloader.py               # Model weight cache engine
├── data_enricher.py            # Database seeding + localized Nigerian overlay
├── fetch_real_data.py          # Loads the pre-densified datasets from data/
├── generate_personas.py        # Batch LLM persona synthesis
├── eval_harness.py             # Leave-one-out evaluation harness
├── colab_data_prep.ipynb       # Builds data/*.csv on Google Colab
├── solution_paper.md           # Submission writeup
├── data/                       # Pre-densified Yelp / Goodreads / Amazon CSVs
└── static/                     # Web UI (HTML, CSS, JS)
```

---

## 🇳🇬 Culturally Rich Personas

Alongside the real-world user models, NaijaBuddy ships 3 hand-crafted Nigerian
showcase personas:

1. **Kunle (VI Tech Bro)** — Victoria Island software engineer; clean aesthetics, fast internet, startup jargon.
2. **Mr. Okeke (Strict Dad)** — conservative retired headmaster; price-sensitive, highly critical, values quiet and moral substance.
3. **Teni (Lagos Gen-Z Influencer)** — fashion and lifestyle creator; obsessed with aesthetic, instagrammable spots and social events.

---

## 📚 Open-Source Frameworks & Models Used

Per the competition's "use of public pre-trained models and open-source frameworks with appropriate disclosure" clause, NaijaBuddy builds on the following open work:

**Models** (all pulled from Hugging Face Hub, run locally):
- **Qwen2.5-3B-Instruct** (Alibaba Cloud, Apache 2.0) — the local reranker / review-generator LLM, in Q4_K_M GGUF quantization
- **BAAI/bge-small-en-v1.5** (BAAI, MIT) — the dense semantic retrieval embedder
- **RoBERTa-large** (Facebook AI, MIT) — backbone for the canonical BERTScore-F1 metric in §4.3

**Inference engines:**
- **llama-cpp-python** (Andrei Betlen et al., MIT) — the in-process GGUF runtime used by `agent.py` for the live web demo, also the default path of `eval_harness.py`
- **vLLM 0.21** (vLLM project, Apache 2.0) — used for the multi-seed canonical evaluation on Modal A10G via `modal_vllm_eval.py`; PagedAttention + continuous batching

**Recommendation / metric libraries:**
- **implicit** (Ben Frederickson, MIT) — Cython/OpenMP ALS matrix factorization (§4.4)
- **sentence-transformers** (UKPLab, Apache 2.0) — BGE-small inference wrapper
- **bert-score** (Tianyi Zhang et al., MIT) — canonical BERTScore-F1 implementation (§4.3)
- **rouge-score** (Google Research, Apache 2.0) — ROUGE-L for review-text overlap

**App framework:**
- **FastAPI** (Sebastián Ramírez, MIT) — REST API + static-file server
- **React** + **TailwindCSS** + **Babel** (Meta / Tailwind Labs / Babel, all MIT) — the vendored web UI, all served offline
- **DuckDB** (DuckDB Labs, MIT) — local catalogue + persona storage

All listed components were used under their public licenses; no proprietary or paid APIs are involved in the deployed system.
