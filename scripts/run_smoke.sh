#!/usr/bin/env bash
# scripts/run_smoke.sh — 60-second sanity check for reproducers.
#
# What it tests:
#   1. Python deps importable (eval_harness.py loads)
#   2. Dense CSV datasets present (or downloadable via local_data_prep.py)
#   3. The local llama-cpp-python path can load Qwen2.5-3B GGUF
#   4. A small LLM-sample-5 warm eval finishes and writes a parseable
#      evaluation_results.json whose schema matches what the paper's
#      Section 4 tables consume.
#
# What it does NOT test:
#   - Multi-seed reproduction (use Path A/B/D in the README for that)
#   - vLLM, Modal, BERTScore, cold-start, or retrieval at full scale
#   - GPU acceleration (this script is CPU-only by design)
#
# Pass = the harness exits 0 and the output JSON has the expected top-level
# domain keys and rmse_sample subkeys. Anything else is a fail.
#
# Run from repo root:
#     bash scripts/run_smoke.sh
#
# Exit codes:
#   0 — PASS (setup is good, harness produces expected output)
#   1 — FAIL (one of the structural checks did not hold)
#   2 — DEPS  (Python deps missing — install requirements.txt first)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==========================================="
echo "NaijaBuddy SMOKE TEST"
echo "  repo:  $REPO_ROOT"
echo "  date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==========================================="

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
    PY="python3"
fi
echo "[setup] using Python at: $PY"
"$PY" --version || { echo "FAIL: Python not runnable"; exit 2; }

echo
echo "[1/4] checking Python imports..."
"$PY" - <<'PY' || { echo "FAIL: required deps missing — run 'pip install -r requirements.txt' first"; exit 2; }
import importlib, sys
needed = [
    "numpy", "pandas", "duckdb", "fastapi", "pydantic",
    "sentence_transformers", "huggingface_hub",
    # llama_cpp is required for the in-process LLM path
    "llama_cpp",
]
missing = []
for m in needed:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))
if missing:
    print("  MISSING:", missing)
    sys.exit(1)
print("  OK")
PY

echo
echo "[2/4] checking dense CSVs..."
missing_csv=0
for stem in yelp goodreads amazon; do
    p="data/${stem}_dense.csv"
    if [ -f "$p" ]; then
        sz=$(wc -c < "$p" | tr -d ' ')
        echo "  ✅ $p ($((sz / 1024 / 1024)) MB)"
    else
        echo "  ❌ MISSING: $p — run 'python local_data_prep.py' first"
        missing_csv=1
    fi
done
if [ $missing_csv -ne 0 ]; then
    echo "FAIL: dense CSVs missing"
    exit 1
fi

echo
echo "[3/4] firing tiny eval_harness.py run (--llm-sample 5, no cache)..."
echo "  expected wall: ~1–3 min on a modern CPU (M-series / x86_64)"
"$PY" -u eval_harness.py \
    --persona-mode synth \
    --seed 42 \
    --llm-sample 5 \
    --no-cache \
    --domains yelp \
    2>&1 | tail -50
HARNESS_RC=${PIPESTATUS[0]}
if [ "$HARNESS_RC" -ne 0 ]; then
    echo "FAIL: eval_harness.py exited with code $HARNESS_RC"
    exit 1
fi

echo
echo "[4/4] checking output JSON structure..."
"$PY" - <<'PY' || { echo "FAIL: output JSON schema check failed"; exit 1; }
import json, os, sys
p = "evaluation_results.json"
if not os.path.exists(p):
    print(f"  ❌ {p} not produced"); sys.exit(1)
d = json.load(open(p))
# Smoke run only does Yelp, so just verify that one domain has the expected shape.
yelp = d.get("Yelp") or d.get("per_seed", {}).get("42", {}).get("Yelp")
if yelp is None:
    print("  ❌ Yelp key missing from evaluation_results.json"); sys.exit(1)
for k in ("rmse_full", "rmse_sample", "rouge_l", "retrieval"):
    if k not in yelp:
        print(f"  ❌ Yelp.{k} missing"); sys.exit(1)
for k in ("V0_global", "V1_user_mean", "V2_best_blend"):
    if k not in yelp["rmse_sample"]:
        print(f"  ❌ Yelp.rmse_sample.{k} missing"); sys.exit(1)
print(f"  ✅ shape OK")
print(f"  Yelp V2 (n=5) = {yelp['rmse_sample']['V2_best_blend']:.4f}  "
      f"ROUGE-L = {yelp['rouge_l']:.4f}")
PY

echo
echo "==========================================="
echo "PASS — your setup can reproduce NaijaBuddy."
echo
echo "Next steps for full reproduction:"
echo "  - Section 'Reproducing the Multi-Seed Canonical Results' in README.md"
echo "  - Pick Path A (your own GPU + vLLM), B (hosted endpoint),"
echo "    C (CPU verification), or D (our Modal pipeline)"
echo "==========================================="
