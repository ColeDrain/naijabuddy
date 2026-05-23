"""
Run NaijaBuddy's eval harness against a vLLM-served Qwen2.5-3B endpoint.

Why vLLM instead of llama-cpp-python?
    llama-cpp-python is single-stream — it generates one prompt at a time,
    which on an A10G yields ~150 tok/s (memory-bandwidth-bound). vLLM batches
    concurrent requests at the GPU kernel level (PagedAttention + continuous
    batching), pushing aggregate throughput to ~5–10k tok/s for a 3B model.
    For an 18,000-call multi-seed run, that's the difference between ~1.5h
    × 3 GPUs ($5–7) and ~20 min × 1 GPU ($0.50–1).

Engine-difference disclosure:
    vLLM serves the fp16/bf16 HF weights of Qwen/Qwen2.5-3B-Instruct, not the
    Q4_K_M GGUF that llama-cpp loads. Outputs are not bit-identical even at
    greedy decoding (temperature=0). The paper must note this when reporting
    multi-engine results.

Usage:
    modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed 42

The function spawns vLLM as a background subprocess, waits for its HTTP server
to come up, then runs eval_harness.py as a child subprocess pointing at it.
Both live in the same Modal container so the HTTP hop is loopback only — no
authentication, no network exposure.
"""
import os
import modal

app = modal.App("naijabuddy-vllm-eval")

# Same persistent cache volume the llama-cpp runs use. vLLM generations live
# alongside llama-cpp generations but in separate per-engine cache files (see
# NAIJABUDDY_CACHE_FILE below) so the two engines never share cached outputs.
cache_volume = modal.Volume.from_name(
    "naijabuddy-eval-cache", create_if_missing=True
)


def _download_qwen_hf():
    """Pre-download the HF-format Qwen2.5-3B weights into the image."""
    import os
    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        local_dir="/root/models/qwen2.5-3b-hf",
        # safetensors only — skip the bin shards if both exist
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
    )
    print("Qwen2.5-3B HF weights downloaded.")


def _download_aux():
    """Pre-download BGE + RoBERTa for the non-LLM eval metrics."""
    import os
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("BAAI/bge-small-en-v1.5")
    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained("roberta-large")
        AutoModel.from_pretrained("roberta-large")
    except Exception as e:
        print(f"  (RoBERTa-large prefetch failed, will lazy-load: {e})")


# vLLM 0.21 ships pre-built wheels for CUDA 12.9 — use Modal's matching base
# image so we don't have to fight CUDA toolkit mismatches at runtime.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12",
    )
    .entrypoint([])
    # uv handles the heavy vllm + torch + flash-attn install much faster than
    # plain pip; matches Modal's own official vLLM example.
    .pip_install("uv")
    .run_commands(
        "uv pip install --system "
        "  vllm==0.21.0 "
        "  hf_transfer "
        "  huggingface-hub "
        "  sentence-transformers "
        "  duckdb "
        "  rouge-score "
        "  bert-score "
        "  fastapi "
        "  pydantic "
        "  openai "
        "  requests "
        # implicit + scipy: 50-100x faster ALS retrieval via Cython/OpenMP
        # (vs the pure-NumPy fallback in eval_harness.als_factorize). Drops
        # Amazon ALS from ~12 min wallclock to ~30s on the same A10G.
        "  implicit "
        "  scipy"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # vLLM emits noisy info logs by default; quiet to warnings.
        "VLLM_LOGGING_LEVEL": "WARNING",
    })
    .run_function(_download_qwen_hf)
    .run_function(_download_aux)
    .add_local_dir(
        "./",
        remote_path="/app",
        ignore=[
            "**/.venv", "**/models", "**/.git", "**/__pycache__",
            "**/naijabuddy.db", "**/naijabuddy.db-journal",
            "**/naijabuddy.db-wal", "**/naijabuddy.db-shm",
            "**/scratch", "**/eval_artifacts",
        ],
    )
)


@app.function(image=image, gpu="a10g", timeout=86400,
              volumes={"/app/eval_artifacts": cache_volume})
def run_vllm_eval_sweep(sample: int, persona_mode: str, seed: int,
                        bertscore: bool, cold_start: bool, cold_sample: int):
    """Boot vLLM, then run eval_harness.py against it via OpenAI-compatible HTTP."""
    import subprocess
    import sys
    import threading
    import time
    import requests

    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"
    # Engine-namespaced cache file: vLLM-generated rows never get confused with
    # llama-cpp-generated rows on the same volume.
    cache_filename = f"llm_cache_s{seed}_{persona_mode}_vllm.jsonl"
    os.environ["NAIJABUDDY_CACHE_FILE"] = cache_filename
    print(f"[cache] using {cache_filename}", flush=True)

    # Background volume committer — same pattern as modal_eval.py.
    _commit_stop = threading.Event()
    def _commit_loop():
        while not _commit_stop.wait(180):
            try:
                cache_volume.commit()
                print("[volume] committed", flush=True)
            except Exception as e:
                print(f"[volume] commit failed: {e}", flush=True)
    committer = threading.Thread(target=_commit_loop, daemon=True)
    committer.start()

    vllm_port = 8000
    vllm_url = f"http://localhost:{vllm_port}/v1"
    model_path = "/root/models/qwen2.5-3b-hf"

    # Spawn vLLM. --enforce-eager disables CUDAGraph capture (the documented
    # safe-mode fallback that eliminates a whole class of CUDA-version bugs at
    # a small throughput cost — fine for our scale).
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", "qwen2.5-3b",
        "--dtype", "auto",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--host", "127.0.0.1",
        "--port", str(vllm_port),
    ]
    print(f"[vllm] launching: {' '.join(vllm_cmd)}", flush=True)
    vllm_proc = subprocess.Popen(
        vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    # Drain vLLM logs in a side thread so any startup error is visible.
    vllm_log = []
    def _drain_vllm():
        for line in vllm_proc.stdout:
            vllm_log.append(line)
            # only forward genuinely important lines so the eval-harness output
            # stays readable.
            if any(k in line for k in
                   ("ERROR", "Error", "Traceback", "engine", "loaded", "ready",
                    "listening", "started server", "OOM", "CUDA")):
                print(f"[vllm] {line.rstrip()}", flush=True)
    threading.Thread(target=_drain_vllm, daemon=True).start()

    # Wait for /v1/models to respond. Cold start (weights load + engine init)
    # typically takes 60–120 s on A10G.
    print("[vllm] waiting for server to become ready (up to 5 min)...",
          flush=True)
    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        if vllm_proc.poll() is not None:
            print("[vllm] FATAL: server exited before becoming ready", flush=True)
            print("[vllm] last 50 log lines:", flush=True)
            for l in vllm_log[-50:]:
                print(f"  {l.rstrip()}", flush=True)
            raise RuntimeError("vLLM died on startup")
        try:
            r = requests.get(f"{vllm_url}/models", timeout=3)
            if r.status_code == 200:
                ready = True
                print("[vllm] server ready", flush=True)
                break
        except Exception:
            pass
        time.sleep(2)
    if not ready:
        vllm_proc.terminate()
        raise RuntimeError("vLLM did not become ready in 5 min")

    cmd = [
        sys.executable, "-u", "/app/eval_harness.py",
        "--llm-sample", str(sample),
        "--persona-mode", persona_mode,
        "--seed", str(seed),
        "--vllm-url", vllm_url,
    ]
    if bertscore:
        cmd.append("--bertscore")
    if cold_start:
        cmd += ["--cold-start", "--cold-sample", str(cold_sample)]
    print(f"[eval] running: {' '.join(cmd)}", flush=True)

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd="/app", bufsize=1,
        )
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="", flush=True)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Harness failed with exit code {process.returncode}")
    finally:
        _commit_stop.set()
        committer.join(timeout=5)
        try:
            cache_volume.commit()
            print("[volume] final commit done", flush=True)
        except Exception as e:
            print(f"[volume] final commit failed: {e}", flush=True)
        # graceful vLLM shutdown
        try:
            vllm_proc.terminate()
            vllm_proc.wait(timeout=10)
        except Exception:
            vllm_proc.kill()

    results_json = results_md = ""
    if os.path.exists("/app/evaluation_results.json"):
        results_json = open("/app/evaluation_results.json").read()
    if os.path.exists("/app/evaluation_results.md"):
        results_md = open("/app/evaluation_results.md").read()
    return results_json, results_md


@app.local_entrypoint()
def main(sample: int = 100, persona_mode: str = "synth",
         seed: int = 42, bertscore: bool = False,
         cold_start: bool = False, cold_sample: int = 2000):
    """
    Default sample=100 is intentional — fire a small probe first to verify
    parity with the cached llama-cpp template runs. Once you confirm the
    outputs are sane, re-fire with --sample 2000 for the real run.
    """
    print(f"[entrypoint] vLLM eval: sample={sample} persona={persona_mode} "
          f"seed={seed} bertscore={bertscore} cold_start={cold_start} "
          f"cold_sample={cold_sample}")
    results_json, results_md = run_vllm_eval_sweep.remote(
        sample, persona_mode, seed, bertscore, cold_start, cold_sample,
    )

    os.makedirs("scratch", exist_ok=True)
    tag = f"n{sample}_{persona_mode}_vllm_s{seed}"
    if results_json:
        with open(f"scratch/modal_results_{tag}.json", "w") as f:
            f.write(results_json)
    if results_md:
        with open(f"scratch/modal_results_{tag}.md", "w") as f:
            f.write(results_md)
    print("=" * 66)
    print(f"Wrote scratch/modal_results_{tag}.{{json,md}}")
    print("=" * 66)
