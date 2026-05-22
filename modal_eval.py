"""
Run the NaijaBuddy evaluation harness on a Modal GPU.

    modal run modal_eval.py --sample 2000 --persona-mode template --seed 42
    modal run modal_eval.py --sample 20            # cheap calibration run

The harness reads data/*_dense.csv directly (no DB needed), so the only
input is the repo itself — shipped via add_local_dir below.
"""
import os
import modal

app = modal.App("naijabuddy-eval-sweep")


def download_models():
    """Build step: pre-download the model files into the image."""
    import os
    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"

    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    print("Pre-downloading Qwen2.5-3B GGUF model...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir="/root/models",
    )
    print("Pre-downloading BGE semantic embedding model...")
    SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model caching complete.")


# CUDA image — compiles llama-cpp-python with GPU offload.
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "libopenblas-dev")
    .pip_install(
        "numpy", "sentence-transformers", "duckdb", "huggingface-hub",
        "fastapi", "pydantic", "uvicorn", "rouge-score",
    )
    # The CUDA stub libs are needed only to *link* llama-cpp at build time.
    # They must NOT persist to runtime — a stub libcuda shadows the real GPU
    # driver and makes llama-cpp silently fall back to CPU. So the build env
    # is scoped to the single install command and the stub symlink is deleted
    # right after. Not using a persistent .env() also restores the base
    # image's own correct runtime LD_LIBRARY_PATH.
    .run_commands(
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so "
        "/usr/local/cuda/lib64/stubs/libcuda.so.1",
        "CC=gcc CXX=g++ CMAKE_ARGS='-DGGML_CUDA=ON' "
        "LDFLAGS='-L/usr/local/cuda/lib64/stubs' "
        "LD_LIBRARY_PATH='/usr/local/cuda/lib64/stubs' "
        "pip install llama-cpp-python==0.3.7",
        "rm /usr/local/cuda/lib64/stubs/libcuda.so.1",
    )
    .run_function(download_models)
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


@app.function(image=image, gpu="a10g", timeout=36000)
def run_eval_sweep(sample: int, persona_mode: str, seed: int,
                   bertscore: bool, cold_start: bool, cold_sample: int):
    """Execute eval_harness.py inside the GPU container, streaming its output."""
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"
    os.environ["NAIJABUDDY_MODEL_PATH"] = "/root/models/qwen2.5-3b-instruct-q4_k_m.gguf"

    cmd = [
        sys.executable, "-u", "/app/eval_harness.py",
        "--llm-sample", str(sample),
        "--persona-mode", persona_mode,
        "--seed", str(seed),
    ]
    if bertscore:
        cmd.append("--bertscore")
    if cold_start:
        cmd += ["--cold-start", "--cold-sample", str(cold_sample)]
    print(f"Running: {' '.join(cmd)}", flush=True)

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

    results_json = results_md = ""
    if os.path.exists("/app/evaluation_results.json"):
        results_json = open("/app/evaluation_results.json").read()
    if os.path.exists("/app/evaluation_results.md"):
        results_md = open("/app/evaluation_results.md").read()
    return results_json, results_md


@app.local_entrypoint()
def main(sample: int = 2000, persona_mode: str = "template",
         seed: int = 42, bertscore: bool = True, cold_start: bool = True,
         cold_sample: int = 100):
    """
    modal run modal_eval.py --sample 2000 --persona-mode template --seed 42
    modal run modal_eval.py --sample 20    # quick cost/timing calibration
    """
    print(f"Deploying eval to Modal: sample={sample} persona={persona_mode} "
          f"seed={seed} bertscore={bertscore} cold_start={cold_start}")

    results_json, results_md = run_eval_sweep.remote(
        sample, persona_mode, seed, bertscore, cold_start, cold_sample
    )

    os.makedirs("scratch", exist_ok=True)
    suffix = f"_n{sample}_{persona_mode}_s{seed}"
    with open(f"scratch/modal_results{suffix}.json", "w") as f:
        f.write(results_json)
    with open(f"scratch/modal_results{suffix}.md", "w") as f:
        f.write(results_md)

    print("\n" + "=" * 66)
    print("Cloud evaluation complete. Results written:")
    print(f"  -> scratch/modal_results{suffix}.json / .md")
    print("(canonical evaluation_results.json left untouched)")
    print("=" * 66)
