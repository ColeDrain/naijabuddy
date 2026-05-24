# =====================================================================
# NAIJABUDDY DOCKERFILE — lean UI host
# =====================================================================
#
# Architecture:
#     This Space hosts only the React UI + FastAPI proxy. All LLM inference
#     is proxied to a Modal-hosted vLLM endpoint (see modal_vllm_serve.py)
#     via the VLLM_URL env var, which agent.py reads at startup and routes
#     through VLLMShim (an OpenAI-protocol client masquerading as a local
#     llama_cpp.Llama instance).
#
#     Pre-Modal versions of this Dockerfile ran vLLM and llama-cpp-python
#     inside the Space container itself, which forced CUDA-12.9 devel base
#     images, ~14-min HF image pulls, JIT-toolchain workarounds, and
#     /dev/shm IPC failures. Offloading inference to Modal cuts the image
#     from ~10 GB to <1 GB and the cold-start from 14 min to ~30 s.
#
# Local development:
#     If you build this image without setting VLLM_URL at runtime,
#     agent.py's VLLMShim init fails and the agent falls back to mock
#     responses (see the call-site handlers in agent.py). To run real
#     inference locally, either set VLLM_URL to a reachable Modal/vLLM
#     endpoint or re-install llama-cpp-python and place a Qwen2.5-3B GGUF
#     at models/qwen2.5-3b-instruct-q4_k_m.gguf — agent.py's path-(b)
#     branch handles that case if you import llama_cpp successfully.

FROM python:3.11-slim AS builder

WORKDIR /app

# Slim image lacks build tools for any wheel that needs compilation; install
# just what duckdb / numpy / sentence-transformers wheels need to fall back
# on if no manylinux wheel matches. Most installs hit pre-built wheels so
# this rarely triggers in practice.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Runtime deps only — no vllm, no llama-cpp-python, no torch (Modal handles
# all GPU work). sentence-transformers brings transformers + torch-cpu as
# transitive deps for BGE-small (used by app.py for /api/users embeddings).
RUN uv pip install --system --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.47.0 \
    pydantic==2.13.4 \
    sentence-transformers==5.5.0 \
    huggingface-hub==1.15.0 \
    hf_transfer \
    duckdb==1.1.3 \
    datasets \
    pandas \
    openai \
    requests

# Copy the data-prep + UI code. data/*_dense.csv arrive via Git LFS.
COPY database.py /app/database.py
COPY agent.py /app/agent.py
COPY app.py /app/app.py
COPY vllm_shim.py /app/vllm_shim.py
COPY downloader.py /app/downloader.py
COPY fetch_real_data.py /app/fetch_real_data.py
COPY data_enricher.py /app/data_enricher.py
COPY local_data_prep.py /app/local_data_prep.py
COPY data /app/data

# Pre-cache BGE-small at build time (used by app.py for embedding queries).
# downloader.py is idempotent — it skips files that already exist.
ENV HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1
COPY models /app/models
# downloader.py originally downloaded both BGE-small and the Qwen GGUF; with
# the Modal proxy architecture we only need BGE-small. The script's existence
# checks mean the missing-GGUF case just no-ops at runtime — it's fine to
# call without the GGUF in place.
RUN python downloader.py

# Seed the catalogue (DuckDB + SQLite). Pure CPU work, no LLM calls. The
# generate_personas.py step is deliberately omitted — personas are now
# synthesized lazily at first-touch via the Modal endpoint, which keeps
# build time + image size low and skips a build-time cross-cloud network
# call to Modal.
RUN python data_enricher.py


# STAGE 2: Runner — single-stage would also work, but a split builder/runner
# keeps the final image smaller by dropping build-essential and the apt cache.
FROM python:3.11-slim AS runner

# Copy Python site-packages and any executables installed by pip.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# HF Spaces runs the container as UID 1000. Create that user up front so the
# SQLite database file (which gets copied with --chown below) is writable at
# runtime — a root-owned /app would block SQLite's WAL/journal sidecar writes.
RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
WORKDIR /app
USER user

# Copy the cached BGE-small + seeded database from builder.
COPY --chown=user --from=builder /app/models /app/models
COPY --chown=user --from=builder /app/naijabuddy.db /app/naijabuddy.db

# Application code (chown'd so the UID-1000 process can read).
COPY --chown=user static /app/static
COPY --chown=user database.py /app/database.py
COPY --chown=user agent.py /app/agent.py
COPY --chown=user app.py /app/app.py
COPY --chown=user vllm_shim.py /app/vllm_shim.py

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers \
    NAIJABUDDY_ALPHA=0.3 \
    HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8000

# Direct uvicorn entry — no more vLLM-subprocess orchestration. The Space's
# Variables/Secrets panel injects VLLM_URL at runtime; agent.py reads it
# during __init__ and routes inference through VLLMShim.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
