# =====================================================================
# NAIJABUDDY MULTI-STAGE DOCKERFILE
# 100% Offline-First Self-Contained Agentic Recommender System
# =====================================================================

# STAGE 1: Compilation & Builder Stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Install dependencies into the builder's global Python path.
# 1) CPU-only PyTorch FIRST. sentence-transformers depends on torch, and the
#    default torch wheel drags in ~2 GB of unused NVIDIA/CUDA packages.
#    NaijaBuddy runs CPU-only and offline, so we pin the CPU build explicitly;
#    installing it first means sentence-transformers finds torch already
#    satisfied and never pulls the CUDA variant.
RUN uv pip install --system --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else. llama-cpp-python is resolved from its prebuilt-wheel
#    index when a wheel exists for the build platform — skipping a ~20-minute
#    source compile — and falls back to compiling from source otherwise.
RUN uv pip install --system --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --index-strategy unsafe-best-match \
    fastapi==0.136.1 \
    uvicorn==0.47.0 \
    pydantic==2.13.4 \
    sentence-transformers==5.5.0 \
    huggingface-hub==1.15.0 \
    llama-cpp-python==0.3.7 \
    duckdb==1.1.3

# Copy schema, seeder, downloader, persona generator and the dense datasets.
# data/ holds the pre-densified CSVs that data_enricher.py ingests, so it must
# be present in the build context (committed to git).
COPY database.py /app/database.py
COPY agent.py /app/agent.py
COPY downloader.py /app/downloader.py
COPY fetch_real_data.py /app/fetch_real_data.py
COPY data_enricher.py /app/data_enricher.py
COPY generate_personas.py /app/generate_personas.py
COPY data /app/data

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


# STAGE 2: Lightweight Runtime Runner Stage
FROM python:3.11-slim AS runner

# llama-cpp-python's libllama.so links the OpenMP runtime at run time. The slim
# base image does not ship it (the builder stage only had it via build-essential),
# so without this the GGUF LLM silently fails to load and the agent drops to
# mock-fallback mode. Install it explicitly in the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy system site-packages and entry-point scripts from the builder. These live
# under /usr/local, are read-only at runtime and stay root-owned (Python only
# needs read access), so they are copied while the stage is still root.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
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

# Set configuration environment variables
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers \
    NAIJABUDDY_ALPHA=0.3

# Expose web server port (matches `app_port: 8000` in the Space README.md)
EXPOSE 8000

# Run Uvicorn server serving FastAPI on 0.0.0.0 for docker ingress routing
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
