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
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Install dependencies into global python path of the builder
# Using --no-cache-dir and locking in stable versions
RUN uv pip install --system --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.47.0 \
    pydantic==2.13.4 \
    sentence-transformers==5.5.0 \
    huggingface-hub==1.15.0 \
    llama-cpp-python==0.3.7

# STAGE 2: Lightweight Runtime Runner Stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy system site-packages compiled from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached models directory
# Note: This ensures the 100% offline requirement works as weights are pre-baked
COPY models /app/models

# Copy web-client and static assets
COPY static /app/static

# Copy SQLite database schema, seeder and agent code
COPY database.py /app/database.py
COPY agent.py /app/agent.py
COPY app.py /app/app.py
COPY downloader.py /app/downloader.py
COPY data_enricher.py /app/data_enricher.py

# Copy pre-seeded SQLite database file for instant warm start
COPY naijabuddy.db /app/naijabuddy.db

# Set configuration environment variables
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf_home \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers \
    NAIJABUDDY_ALPHA=0.3

# Expose web server port
EXPOSE 8000

# Run Uvicorn server serving FastAPI on 0.0.0.0 for docker ingress routing
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
