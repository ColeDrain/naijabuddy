#!/usr/bin/env bash
# entrypoint.sh — orchestrates vLLM (port 8001 internal) + FastAPI (port 8000 user-facing).
#
# Container boot sequence:
#   1. Start vLLM as a background subprocess serving Qwen2.5-3B from the
#      local HF-safetensors directory at /app/models/qwen2.5-3b-hf
#      (pre-downloaded by downloader.py at Docker-build time, so this is
#      a fast local model load, ~30-60s on A10G — no HF Hub round-trip).
#   2. Poll http://127.0.0.1:8001/v1/models until vLLM is ready (or until
#      the timeout fires).
#   3. Export VLLM_URL=http://127.0.0.1:8001/v1. agent.py's __init__ reads
#      this at startup and picks the VLLMShim path instead of llama-cpp.
#   4. exec uvicorn on port 8000 (the user-facing port HF Spaces routes
#      external traffic to, per README.md's `app_port: 8000` frontmatter).
#
# If vLLM fails to start within the timeout, we fall back to launching
# FastAPI WITHOUT VLLM_URL set, so agent.py loads llama-cpp + GGUF
# (the prior working path). This preserves the demo's availability even
# if the vLLM subprocess hits a CUDA/driver issue on the assigned host.
#
# Logs from both processes are interleaved into the container's stdout
# so HF Spaces' live log viewer shows both vLLM startup and FastAPI
# request handling.
set -uo pipefail

PORT_FASTAPI="${PORT:-8000}"
PORT_VLLM=8001
VLLM_MODEL_DIR="/app/models/qwen2.5-3b-hf"
VLLM_READY_TIMEOUT_S=300        # 5 min — vLLM cold-load on A10G is usually 30-90s
VLLM_LOG=/tmp/vllm.log

echo "[entrypoint] $(date -u +%FT%TZ) start"
echo "[entrypoint] uvicorn target port: ${PORT_FASTAPI}"
echo "[entrypoint] vllm target port:    ${PORT_VLLM}"
echo "[entrypoint] vllm model dir:      ${VLLM_MODEL_DIR}"

# ----- Decide whether to attempt vLLM ------------------------------------------
# Skip the vLLM path entirely if the model dir is missing (e.g. local dev,
# Modal eval container, anyone who didn't run downloader.py with the
# safetensors step). agent.py will then load llama-cpp from the GGUF.
if [[ ! -d "${VLLM_MODEL_DIR}" ]] || [[ ! -f "${VLLM_MODEL_DIR}/config.json" ]]; then
    echo "[entrypoint] no Qwen2.5-3B safetensors at ${VLLM_MODEL_DIR};"
    echo "[entrypoint] skipping vLLM, FastAPI will use llama-cpp via agent.py."
    exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT_FASTAPI}"
fi

# ----- Launch vLLM in background -----------------------------------------------
echo "[entrypoint] launching vLLM..."
# --enforce-eager disables CUDA Graph capture — small throughput cost,
# eliminates a known class of CUDA-version bugs on first-call inference.
# --gpu-memory-utilization 0.80 keeps headroom for the BGE-small embedder
# (~30 MB) and the FastAPI process; A10G has 24 GB so we use ~19 GB for vLLM.
# --max-model-len 4096 matches the prompt sizes agent.py actually sends
# (much smaller than Qwen2.5's 32K trained ctx).
python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_DIR}" \
    --served-model-name qwen2.5-3b \
    --host 127.0.0.1 \
    --port "${PORT_VLLM}" \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[entrypoint] vLLM PID=${VLLM_PID} (logs -> ${VLLM_LOG})"

# Tail the vLLM log into the container's stdout so HF's log viewer shows
# both vLLM startup AND eventually FastAPI request lines.
tail -f "${VLLM_LOG}" &
TAIL_PID=$!

# ----- Wait for vLLM to be ready -----------------------------------------------
echo "[entrypoint] waiting up to ${VLLM_READY_TIMEOUT_S}s for vLLM /v1/models..."
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT_S ))
vllm_ready=0
while [[ $(date +%s) -lt ${deadline} ]]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM process exited before becoming ready."
        echo "[entrypoint] last 40 lines of vllm log:"
        tail -40 "${VLLM_LOG}" || true
        break
    fi
    # 2s curl timeout; vLLM responds with a JSON model list when ready.
    if curl -fsS -o /dev/null --max-time 2 \
            "http://127.0.0.1:${PORT_VLLM}/v1/models"; then
        vllm_ready=1
        break
    fi
    sleep 2
done

# Stop tailing the vLLM log — uvicorn's own stdout will take over below.
# (vLLM continues running as a daemon under VLLM_PID.)
kill "${TAIL_PID}" 2>/dev/null || true

if [[ "${vllm_ready}" -eq 1 ]]; then
    echo "[entrypoint] vLLM ready at http://127.0.0.1:${PORT_VLLM}/v1"
    export VLLM_URL="http://127.0.0.1:${PORT_VLLM}/v1"
else
    echo "[entrypoint] WARNING: vLLM did not become ready within"
    echo "[entrypoint] ${VLLM_READY_TIMEOUT_S}s — proceeding without VLLM_URL,"
    echo "[entrypoint] agent.py will fall back to llama-cpp + GGUF."
    kill "${VLLM_PID}" 2>/dev/null || true
    unset VLLM_URL
fi

# ----- Hand off to FastAPI -----------------------------------------------------
echo "[entrypoint] handing off to uvicorn on 0.0.0.0:${PORT_FASTAPI}..."
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT_FASTAPI}"
