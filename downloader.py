import os
import sys

# Cache directory within workspace to ensure offline isolation
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Set Hugging Face cache env variables to keep models inside the project folder
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "hf_home")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(MODELS_DIR, "sentence_transformers")

def download_models():
    """Programmatically downloads and pre-caches the embedding and LLM GGUF models."""
    print("=" * 60)
    print("NAIJABUDDY PROGRAMMATIC OFFLINE MODEL SEEDER")
    print("=" * 60)
    
    # 1. Check and install Hugging Face Hub if missing
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing 'huggingface_hub' package via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
        
    # 2. Check and install Sentence Transformers if missing
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing 'sentence-transformers' and dependencies...")
        import subprocess
        # Install CPU version of Torch and sentence-transformers to be lightweight and fast
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer

    # 3. Download BGE-Small Embedding Model
    print("\n--- 1. Caching Embedding Model: BAAI/bge-small-en-v1.5 ---")
    print("Fetching embedding model weights...")
    # This automatically downloads and caches it in our isolated SENTENCE_TRANSFORMERS_HOME
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("BGE-Small embedding model successfully cached!")
    
    # 3b. Download MiniLM Baseline Embedding Model
    print("\n--- 1b. Caching Baseline Embedding Model: sentence-transformers/all-MiniLM-L6-v2 ---")
    print("Fetching MiniLM model weights...")
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    print("MiniLM embedding model successfully cached!")
    
    # 4. Download local GGUF LLM Model
    # We use Qwen2.5-3B-Instruct-GGUF (Q4_K_M) which is incredibly fast on M1 Mac and fits inside ~2.2GB.
    print("\n--- 2. Caching Local LLM GGUF: Qwen2.5-3B-Instruct (Q4_K_M) ---")
    repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    dest_path = os.path.join(MODELS_DIR, filename)
    
    if os.path.exists(dest_path):
        print(f"GGUF Model file already exists at: {dest_path}")
    else:
        print(f"Downloading {filename} from Hugging Face hub (Repository: {repo_id})...")
        print("This may take a few minutes depending on your internet connection (Size: ~2.2 GB).")
        try:
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False
            )
            print(f"GGUF Model successfully saved to: {downloaded_file}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please ensure you have an active internet connection for this first-time model caching.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("ALL MODELS PRE-CACHED FOR 100% OFFLINE ISOLATION!")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
