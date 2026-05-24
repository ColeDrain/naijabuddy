# =====================================================================
# NAIJABUDDY MULTI-STAGE DOCKERFILE
# 100% Offline-First Self-Contained Agentic Recommender System
# =====================================================================

# STAGE 1: Compilation & Builder Stage
#
# Base image bumped python:3.11-slim -> nvidia/cuda:12.9.0-devel-ubuntu22.04
# because vLLM 0.21 ships pre-built wheels against CUDA 12.9 and refuses to
# load against 12.1 drivers. Matches Modal's own canonical vLLM example image
# (see modal_vllm_eval.py).
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install system compilation packages + Python 3.11 (the CUDA image has no
# python by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Install uv for fast dependency resolution. Note: ubuntu 22.04's bundled
# pip (22.0.2) predates the --break-system-packages flag from pip 23.0+,
# and also predates PEP 668 enforcement, so a plain `pip install` works.
RUN pip install --no-cache-dir uv

# Install dependencies. vLLM 0.21 brings its own torch (built against CUDA 12.9)
# as a transitive dep, so we don't pin torch separately. llama-cpp-python is
# retained as a fallback inference path — agent.py picks vLLM when the
# VLLM_URL env var is set at startup, llama-cpp otherwise.
RUN uv pip install --system --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
    --index-strategy unsafe-best-match \
    vllm==0.21.0 \
    fastapi==0.136.1 \
    uvicorn==0.47.0 \
    pydantic==2.13.4 \
    sentence-transformers==5.5.0 \
    huggingface-hub==1.15.0 \
    hf_transfer \
    llama-cpp-python==0.3.7 \
    duckdb==1.1.3 \
    datasets \
    pandas \
    openai \
    requests

# Copy schema, seeder, downloader, persona generator and the dense datasets.
# data/ holds the pre-densified CSVs that data_enricher.py ingests, so it must
# be present in the build context (committed to git).
COPY database.py /app/database.py
COPY agent.py /app/agent.py
COPY downloader.py /app/downloader.py
COPY fetch_real_data.py /app/fetch_real_data.py
COPY data_enricher.py /app/data_enricher.py
COPY generate_personas.py /app/generate_personas.py
COPY local_data_prep.py /app/local_data_prep.py
COPY data /app/data

# data/*_dense.csv are tracked via Git LFS (see .gitattributes) and
# arrive in /app/data/ directly via the COPY above — no streaming or
# regeneration needed at build time. local_data_prep.py stays in the
# image for offline-reproduction use; it just isn't invoked here.

# Pre-cache models. The local models/ directory is copied straight into the
# image, so a machine that already holds the weights skips the ~2.2 GB
# download. On a fresh clone models/ is effectively empty (just .gitkeep) and
# downloader.py fetches what is missing — it checks os.path.exists first.
ENV HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
COPY models /app/models
RUN python downloader.py

# Ingest + seed the catalogue, then pre-synthesize all user personas, at build
# time. generate_personas.py reuses agent.synthesize_and_update_persona() so
# there is a single synthesis implementation.
RUN python data_enricher.py
RUN python generate_personas.py


# STAGE 2: Runner stage
#
# Uses the CUDA *devel* image (not runtime) so the full toolchain is present:
# nvcc, gcc, g++, plus all CUDA headers and shared libraries. vLLM 0.21
# triggers two distinct runtime JIT compiles on first inference — FlashInfer's
# CUDA sampler kernel (needs nvcc) and Triton's host-side driver bridge
# (needs gcc + python-dev). The runtime image strips all of these, so any
# code path that JIT-compiles at first use crashes mid-profile_run. The
# devel image adds ~1.5 GB to the final image but eliminates every
# toolchain-related runtime failure mode in one step.
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS runner

# Install Python 3.11 + libgomp (llama-cpp-python's libllama.so links OpenMP).
# Match the builder's Python path so the copied site-packages resolve cleanly.
# python3.11-dev is also needed by Triton's runtime driver-bridge JIT for
# Python.h headers (cuda-devel gives us gcc + nvcc; we still need the
# Python headers separately via apt).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Copy system site-packages and entry-point scripts from the builder. These live
# under /usr/local, are read-only at runtime and stay root-owned (Python only
# needs read access), so they are copied while the stage is still root.
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Hugging Face Spaces run the container as user ID 1000. Create that user and
# give it ownership of /app so SQLite can write naijabuddy.db plus its WAL /
# journal sidecar files at runtime — a root-owned /app would be read-only to the
# UID-1000 process ("attempt to write a readonly database"). This also makes the
# runtime non-root, which is good practice for local `docker run` too.
RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
WORKDIR /app
USER user

# Copy the cached models and the pre-seeded database, owned by the runtime user.
# `--chown` sets ownership during the copy; a recursive `chown -R` afterwards
# would duplicate the entire ~2 GB models layer into the image.
COPY --chown=user --from=builder /app/models /app/models
COPY --chown=user --from=builder /app/naijabuddy.db /app/naijabuddy.db

# Copy the web UI, static assets and runner code, owned by the runtime user.
COPY --chown=user static /app/static
COPY --chown=user database.py /app/database.py
COPY --chown=user agent.py /app/agent.py
COPY --chown=user app.py /app/app.py
COPY --chown=user vllm_shim.py /app/vllm_shim.py
COPY --chown=user --chmod=755 entrypoint.sh /app/entrypoint.sh

# Set configuration environment variables
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers \
    NAIJABUDDY_ALPHA=0.3 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Expose web server port (matches `app_port: 8000` in the Space README.md)
EXPOSE 8000

# entrypoint.sh starts vLLM on 127.0.0.1:8001 in the background, waits
# for it to become ready, exports VLLM_URL, then execs uvicorn on the
# user-facing port. If vLLM fails to start in 5 min the script falls
# back to launching uvicorn alone — agent.py then uses llama-cpp via
# the existing fallback path.
CMD ["/app/entrypoint.sh"]
