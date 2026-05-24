---
title: NaijaBuddy
emoji: 🇳🇬
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
short_description: Offline-first Nigerian agentic recommender (Qwen2.5-3B local)
hardware: t4-small
suggested_hardware: t4-small
sleep_time: 300
---

# NaijaBuddy: A Highly Localized, Offline-First Agentic Recommender & User-Modeling System

NaijaBuddy is a 100% offline-first, containerized agentic recommendation system and review simulator tailored to the Nigerian consumer market.

It implements a two-stage **retrieve-then-rerank** pipeline over a hybrid database that combines real-world dataset subsets (Yelp, Amazon, Goodreads) with a localized catalogue of Nigerian establishments, movies, and literature. It runs a quantized local LLM (`Qwen2.5-3B-Instruct`, GGUF) and an embedding model (`BAAI/bge-small-en-v1.5`) natively in-process via `llama-cpp-python` and `sentence-transformers` — no cloud APIs, no network at runtime.

---

## 🚀 Quick Start (Docker)

The multi-stage Dockerfile downloads the model weights, ingests the bundled dense datasets, seeds the database, and pre-synthesizes all user personas **at build time**, so the runtime image is fully self-contained and offline.

### 1. Build the image
An internet connection is required *only* during the build, to fetch packages and model weights.
```bash
docker build -t naijabuddy .
```

### 2. Run the container
```bash
docker run -p 8000:8000 naijabuddy
```
* **Interactive UI:** `http://localhost:8000/`
* **REST API docs:** `http://localhost:8000/docs`

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
