"""
Modal web endpoint serving Qwen2.5-3B-Instruct via vLLM's OpenAI-compatible API.

Architecture:
    HF Space (UI, cpu-upgrade)
        ↓ HTTPS / OpenAI protocol
    Modal endpoint (this file, A10G, autoscale 0→1, scaledown_window=300s)
        ↓ localhost
    vLLM 0.21 (Qwen2.5-3B-Instruct safetensors, same weights as canonical eval)

Why this exists:
    HF Spaces' Docker SDK has structural barriers to running vLLM in-process
    (no /dev/shm config, no nvcc, no @spaces.GPU). The standard production
    pattern is to split frontend (Space) from inference backend (Modal). This
    also makes the live demo's inference engine bit-identical with the paper's
    canonical multi-seed evaluation — same vLLM, same weights.

Cost model:
    A10G on Modal is roughly $1.10/hr while a container is warm. With
    scaledown_window=300s the container shuts down after 5 min of no traffic,
    so idle cost is zero. Active demo use of ~5 hours during judging is
    ~$5.50, well inside the $30 credit ceiling.

Cold-start expectation:
    First request after deploy: ~30-60s (image pull + model load + warmup).
    Cold after scaledown: ~15-30s (model cached on Modal disk, just disk→VRAM
        + the in-container warmup probe absorbs Triton JIT for the first
        external request).
    Warm: ~0.8-2s per request (CUDA Graphs + continuous batching).

Deploy:
    modal deploy modal_vllm_serve.py
    # then capture the URL printed at the end and set it as VLLM_URL on the
    # Space (see HF Space settings → Variables and secrets).

Optional API-key protection:
    The vLLM server can require an Authorization: Bearer <key> header. To
    enable, uncomment the --api-key flag below, create a Modal Secret named
    "naijabuddy-vllm-key" with NAIJABUDDY_VLLM_KEY=<random-32-char>, and set
    the same value as VLLM_API_KEY on the Space. agent.py's VLLMShim reads
    VLLM_API_KEY at startup and passes it to the OpenAI client.

    For a short-lived hackathon demo with a non-guessable Modal URL, the
    default no-auth setup is acceptable.
"""
import modal

app = modal.App("naijabuddy-vllm-serve")

VLLM_PORT = 8000
QWEN_LOCAL_DIR = "/root/models/qwen2.5-3b-hf"


def _download_qwen_hf():
    """Pre-download Qwen2.5-3B-Instruct safetensors into the image layer."""
    import os
    os.environ["HF_HOME"] = "/root/models/hf_home"
    # HF_XET_HIGH_PERFORMANCE replaces the deprecated HF_HUB_ENABLE_HF_TRANSFER
    # for the same purpose — chunked / parallel transfer via the Xet backend.
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        local_dir=QWEN_LOCAL_DIR,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
    )
    print("Qwen2.5-3B-Instruct safetensors cached in image.")


# Same CUDA-12.9 base as modal_vllm_eval.py — vLLM 0.21's pre-built wheels
# link against CUDA 12.9 and need the matching toolchain at import time.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12",
    )
    .entrypoint([])
    .pip_install("uv")
    .run_commands(
        "uv pip install --system "
        "  vllm==0.21.0 "
        "  hf_transfer "
        "  huggingface-hub"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
        # FlashInfer sampler JIT-compiles a CUDA kernel via nvcc on first use;
        # we leave it disabled here because the native PyTorch sampler is
        # equivalent at our temperature settings and skipping the JIT shortens
        # cold-start by another second or two.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .run_function(_download_qwen_hf)
)


@app.function(
    image=image,
    gpu="a10g",
    # 5-minute keep-warm: any judge clicking around the demo continuously
    # avoids cold start; a 5+ min gap (e.g. switching tabs) shuts the
    # container down and the next request pays the cold-start cost.
    scaledown_window=300,
    # Single container — the demo is single-user-ish during judging and a
    # 3B model on a single A10G handles concurrent requests fine via vLLM's
    # continuous batching.
    max_containers=1,
    # web_server startup_timeout (below) is the real boot deadline; this is
    # just the max container lifetime.
    timeout=86400,  # 24h max container lifetime — fine for a demo
)
@modal.concurrent(max_inputs=32)  # multiple HTTP requests per container
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def vllm_server():
    """Launch vLLM's OpenAI-compatible server. @modal.web_server proxies
    HTTPS traffic on a public Modal URL to this in-container HTTP port.

    We also fire a background warmup thread that issues a single throwaway
    completion against ourselves once vLLM is ready. This forces the Triton
    JIT compile (the `_compute_slot_mapping_kernel` warning in the vLLM
    logs — see git blame) to happen on container time, not on the first
    user-facing request. Without it, the first external POST eats a 30-50s
    JIT spike on top of the normal cold-start.
    """
    import subprocess
    import sys
    import threading
    import time
    import urllib.request
    import json

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", QWEN_LOCAL_DIR,
        "--served-model-name", "qwen2.5-3b",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "auto",
        # 4096 ctx matches what agent.py's longest prompts need; staying
        # well under Qwen2.5's 32K trained ctx keeps KV-cache memory low.
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        # NOTE: we deliberately do NOT pass --enforce-eager here. That flag
        # disables both CUDA Graphs AND torch.compile/Inductor, costing
        # 10-20% throughput. It was useful as a safe-mode fallback when we
        # were running vLLM inside an HF Space with a cuda-runtime base
        # (no nvcc, FlashInfer JIT failures). Modal's cuda-devel base has
        # the full toolchain, so the graphs and inductor passes succeed.
        #
        # Uncomment + set NAIJABUDDY_VLLM_KEY in Modal Secret to require
        # bearer-token auth. See module docstring.
        # "--api-key", os.environ["NAIJABUDDY_VLLM_KEY"],
    ]
    print(f"[serve] launching vLLM: {' '.join(cmd)}", flush=True)
    # Inherit stdout/stderr so Modal's log viewer shows vLLM startup
    # progress directly. web_server polls the port until it accepts
    # connections, then routes traffic.
    subprocess.Popen(cmd)

    def _warmup():
        """Poll vLLM until it responds, then fire one throwaway completion
        to JIT-compile the Triton sampler/slot-mapping kernels."""
        base = f"http://127.0.0.1:{VLLM_PORT}"
        deadline = time.time() + 240
        # Phase 1: wait for /v1/models to 200.
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("[warmup] vLLM never became ready within 240s; skipping warmup probe.", flush=True)
            return

        # Phase 2: fire a tiny completion to trigger Triton JIT compile.
        # max_tokens=4 keeps the call cheap; the slot-mapping / topk kernels
        # only compile once per shape so this generates the cache for all
        # subsequent calls.
        try:
            t0 = time.time()
            body = json.dumps({
                "model": "qwen2.5-3b",
                "prompt": "Warmup probe.",
                "max_tokens": 4,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                f"{base}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                _ = r.read()
            print(f"[warmup] Triton kernels primed in {time.time() - t0:.1f}s — "
                  f"first external request will skip the JIT spike.", flush=True)
        except Exception as e:
            print(f"[warmup] probe failed (non-fatal): {e}", flush=True)

    threading.Thread(target=_warmup, daemon=True).start()
