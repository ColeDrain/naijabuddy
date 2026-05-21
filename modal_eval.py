import os
import modal

# Define the Modal App
app = modal.App("naijabuddy-eval-sweep")

def download_models():
    """Build step: Pre-download and cache our model files directly inside the remote image."""
    import os
    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"
    
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    print("Pre-downloading Qwen2.5-3B GGUF model...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir="/root/models"
    )
    
    print("Pre-downloading BGE semantic embedding model...")
    SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model caching complete.")

# Construct the GPU image, compiling llama-cpp with CUDA support
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "libopenblas-dev")
    .pip_install(
        "numpy",
        "sentence-transformers",
        "duckdb",
        "huggingface-hub",
        "fastapi",
        "pydantic",
        "uvicorn"
    )
    # CUDA acceleration build flags
    .env({
        "CC": "gcc",
        "CXX": "g++",
        "CMAKE_ARGS": "-DGGML_CUDA=ON",
        "LDFLAGS": "-L/usr/local/cuda/lib64/stubs",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs"
    })
    .run_commands("ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
    .pip_install("llama-cpp-python==0.3.7")
    .run_function(download_models)
    .add_local_dir(
        "./",
        remote_path="/app",
        ignore=[
            "**/.venv",
            "**/models",
            "**/.git",
            "**/__pycache__",
            "**/naijabuddy.db",
            "**/naijabuddy.db-journal",
            "**/naijabuddy.db-wal",
            "**/naijabuddy.db-shm"
        ]
    )
)

@app.function(
    image=image,
    gpu="any",  # Allocates any available GPU (T4, A10G, L4, etc.)
    timeout=1800,  # 30 minutes limit
)
def run_eval_sweep(llm_sample: int):
    """Executes the eval_harness.py inside our GPU-accelerated cloud container."""
    import subprocess
    import sys
    
    # Configure the paths to find our pre-cached models
    os.environ["HF_HOME"] = "/root/models/hf_home"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/models/sentence_transformers"
    os.environ["NAIJABUDDY_MODEL_PATH"] = "/root/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    print(f"Starting cloud evaluation sweep for n={llm_sample} sample users per domain...")
    
    cmd = [
        sys.executable,
        "-u",  # Unbuffered binary stdout/stderr
        "/app/eval_harness.py",
        "--llm-sample", str(llm_sample)
    ]
    
    # Run the harness and stream stdout/stderr in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
        bufsize=1  # Line-buffered
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
        
    # Read the generated results
    results_json = ""
    results_md = ""
    if os.path.exists("/app/evaluation_results.json"):
        with open("/app/evaluation_results.json", "r") as f:
            results_json = f.read()
    if os.path.exists("/app/evaluation_results.md"):
        with open("/app/evaluation_results.md", "r") as f:
            results_md = f.read()
            
    return results_json, results_md

@app.local_entrypoint()
def main(sample: int = 350):
    """
    Local entry point triggered on the user's system:
    Usage: modal run modal_eval.py --sample 350
    """
    print(f"Deploying evaluation job to Modal cloud cluster for {sample} users...")
    
    results_json, results_md = run_eval_sweep.remote(sample)
    
    # Write the high-fidelity cloud results locally
    with open("evaluation_results.json", "w") as f:
        f.write(results_json)
    with open("evaluation_results.md", "w") as f:
        f.write(results_md)
        
    print("\n" + "=" * 66)
    print("SUCCESS: Cloud evaluation sweep complete.")
    print("Local results updated:")
    print("  -> evaluation_results.json")
    print("  -> evaluation_results.md")
    print("=" * 66)
