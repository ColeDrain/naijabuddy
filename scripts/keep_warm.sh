#!/usr/bin/env bash
# keep_warm.sh — keep the Modal vLLM endpoint warm during judging hours.
#
# Why this exists:
#   The Modal endpoint serving Qwen2.5-3B (modal_vllm_serve.py) has
#   scaledown_window=300 — after 5 min of no traffic the container shuts
#   down and the next request pays a 30-90s cold-start (model load to
#   VRAM + Triton JIT). For a live demo with intermittent judge clicks,
#   that's a bad first-impression risk.
#
#   This script pings /v1/models every 240 s (under the 300 s scaledown
#   threshold) so the container stays warm continuously. Each ping is
#   essentially free; the real cost is the warm A10G (~$1.10/hr while
#   running).
#
# Cost arithmetic:
#   A10G on Modal = $1.10/hr. Running this for the actual judging window
#   (e.g. a 4-hour session) is ~$4.40. Running 24h continuously is ~$26.
#   Stop the script when the demo is no longer needed.
#
# Usage:
#   ./scripts/keep_warm.sh              # default endpoint, 240s interval
#   ENDPOINT=https://... INTERVAL=200 ./scripts/keep_warm.sh
#   Ctrl+C to stop.

set -u

ENDPOINT="${ENDPOINT:-https://ugochukwu-onyebuchi-197567--naijabuddy-vllm-serve-vllm-server.modal.run/v1/models}"
INTERVAL="${INTERVAL:-240}"

echo "[keep_warm] pinging ${ENDPOINT}"
echo "[keep_warm] every ${INTERVAL}s — Ctrl+C to stop"
echo

while true; do
    ts="$(date +%H:%M:%S)"
    # %{time_total} is seconds. A warm container returns in <0.5s; a cold
    # one in 30-90s (you'll see the spike on the first ping after idle).
    printf "[%s] " "$ts"
    curl -sS -o /dev/null \
        -w "HTTP %{http_code}  total %{time_total}s\n" \
        --max-time 120 \
        "$ENDPOINT" || echo "  (ping failed — retrying next cycle)"
    sleep "$INTERVAL"
done
