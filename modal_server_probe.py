"""
Quick probe to validate that `llama_cpp.server` actually works on our Modal
image and that continuous batching gives a real speedup vs single-stream.

Run: `modal run modal_server_probe.py`

Reports tokens/sec for:
  - 1 sequential call
  - 8 concurrent calls (continuous batching)

Reuses the same image as modal_eval.py (Qwen2.5-3B GGUF + llama-cpp-python==0.3.7
+ CUDA + fastapi/uvicorn/pydantic already present). No image rebuild required.
"""
import modal

app = modal.App("naijabuddy-server-probe")


def _download_qwen():
    """Cached Modal build step — fetches the GGUF model file once."""
    import os
    os.environ["HF_HOME"] = "/root/models/hf_home"
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir="/root/models",
    )


image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "libopenblas-dev",
                 "libcurl4-openssl-dev")
    .pip_install("huggingface-hub", "openai", "requests")
    # Build the native llama-server binary from llama.cpp source — it's the
    # only path that exposes --cont-batching and --parallel. Pinned to a
    # known-good tag (b6968, released early May 2026 / mid-cycle) to keep
    # this reproducible. Build with CUDA + CMAKE_CUDA_ARCHITECTURES=86 (A10G
    # = Ampere SM 8.6) for fast compile and optimal kernels.
    .run_commands(
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so "
        "/usr/local/cuda/lib64/stubs/libcuda.so.1",
        "cd /opt && git clone --depth 1 --branch b6968 "
        "https://github.com/ggml-org/llama.cpp.git",
        "cd /opt/llama.cpp && cmake -B build "
        "  -DGGML_CUDA=ON "
        "  -DCMAKE_CUDA_ARCHITECTURES=86 "
        "  -DGGML_NATIVE=OFF "
        "  -DLLAMA_CURL=ON "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs' "
        "  -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs' "
        "  -DCMAKE_MODULE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs' ",
        "cd /opt/llama.cpp && LDFLAGS='-L/usr/local/cuda/lib64/stubs' "
        "cmake --build build --config Release "
        "  -j $(nproc) --target llama-server",
        "rm /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "ls -la /opt/llama.cpp/build/bin/llama-server",
    )
    .run_function(_download_qwen)
)


@app.function(image=image, gpu="a10g", timeout=900)
def probe():
    """Start llama_cpp.server in the background, then probe single + concurrent."""
    import os
    import subprocess
    import time
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model_path = "/root/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    server_bin = "/opt/llama.cpp/build/bin/llama-server"
    port = 8080

    # llama-server CLI uses dashed flags, not underscored. -ngl = n_gpu_layers,
    # -c = ctx size, -np = parallel sequences, -b = logical batch size, -ub =
    # physical micro-batch size. --cont-batching enables continuous batching.
    cmd = [
        server_bin,
        "-m", model_path,
        "-ngl", "99",                # offload everything
        "-c", str(2048 * 8),         # ctx slot per parallel sequence × 8
        "-b", "2048",                # logical batch
        "-ub", "512",                # physical micro-batch
        "-np", "8",                  # 8 parallel sequences
        "--cont-batching",
        "--flash-attn",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--threads", "4",
    ]
    print("\nstarting llama_cpp.server with:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Wait for the server to come up. Drain stdout in a side thread so we never
    # miss a traceback if the process exits early.
    import threading
    log_lines = []
    def _drain():
        for line in proc.stdout:
            log_lines.append(line)
            print("  [server]", line.rstrip(), flush=True)
    drain_t = threading.Thread(target=_drain, daemon=True)
    drain_t.start()

    health_url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + 180
    ready = False
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        if proc.poll() is not None:
            # Give the drain thread a moment to flush any remaining output.
            time.sleep(2)
            full = "".join(log_lines)
            print("ERROR: server exited before becoming ready. "
                  f"return code: {proc.returncode}", flush=True)
            print("FULL SERVER OUTPUT (last 4000 chars):", flush=True)
            print(full[-4000:], flush=True)
            return {"ok": False, "reason": "server-died-on-startup",
                    "tail": full[-1500:]}
        time.sleep(1)

    if not ready:
        proc.terminate()
        return {"ok": False, "reason": "timeout-waiting-for-server"}

    print(f"server is up at http://localhost:{port}", flush=True)

    # ------- probe 1: one prompt, sequential -----------------------------------
    from openai import OpenAI
    client = OpenAI(api_key="sk-no-key-needed",
                    base_url=f"http://localhost:{port}/v1")
    prompt = ("Write a single-sentence restaurant review for a Lagos jollof "
              "rice spot, in a casual Nigerian voice. One sentence only.")

    def call_one():
        t = time.time()
        r = client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=128,
        )
        dur = time.time() - t
        usage = r.usage
        return {
            "dur": dur,
            "in_tok": usage.prompt_tokens if usage else None,
            "out_tok": usage.completion_tokens if usage else None,
            "text": r.choices[0].message.content[:120],
        }

    print("\n--- probe 1: single sequential call ---", flush=True)
    seq = call_one()
    seq_rate = seq["out_tok"] / seq["dur"] if seq["out_tok"] else None
    print(f"  duration={seq['dur']:.2f}s  in_tok={seq['in_tok']}  "
          f"out_tok={seq['out_tok']}  rate={seq_rate:.1f} tok/s",
          flush=True)
    print(f"  sample: {seq['text']!r}", flush=True)

    # ------- probe 2: 8 concurrent calls -------------------------------------
    print("\n--- probe 2: 8 concurrent calls (continuous batching) ---",
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(call_one) for _ in range(8)]
        results = [f.result() for f in as_completed(futs)]
    wall = time.time() - t0
    total_out = sum(r["out_tok"] or 0 for r in results)
    agg_rate = total_out / wall
    print(f"  wallclock={wall:.2f}s  total_out_tok={total_out}  "
          f"aggregate_rate={agg_rate:.1f} tok/s", flush=True)
    per_call = [r["dur"] for r in results]
    print(f"  per-call durations: min={min(per_call):.2f}s "
          f"median={sorted(per_call)[4]:.2f}s max={max(per_call):.2f}s",
          flush=True)

    speedup = agg_rate / seq_rate if seq_rate else None
    print(f"\n=== SPEEDUP from continuous batching: "
          f"{speedup:.2f}× ===", flush=True)

    proc.terminate()
    return {
        "ok": True,
        "sequential": {"rate_tok_s": seq_rate, "dur_s": seq["dur"]},
        "concurrent_8": {"agg_rate_tok_s": agg_rate, "wallclock_s": wall},
        "speedup_x": speedup,
    }


@app.local_entrypoint()
def main():
    result = probe.remote()
    print("\n=== PROBE RESULT ===")
    import json
    print(json.dumps(result, indent=2))
