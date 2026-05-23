"""
LLM-as-judge evaluation for review-text quality (Behavioural Fidelity +
Contextual Relevance — the rubric's human-eval buckets, of which we currently
have ZERO evidence in the paper).

Uses GPT-OSS-120B — OpenAI's 2025 open-weight 120B-parameter model — hosted
on either Groq (api.groq.com/openai/v1) or Cerebras (api.cerebras.ai/v1).
Picked as default because as of May 2026 it is the strongest production
model on BOTH providers, faster than Llama 3.3 70B (500 / 3000 tok/s), and
cheaper on Groq ($0.15/$0.60 per 1M input/output tokens).

For each cached generated review, asks the judge to score on 1–5 across:
  * Behavioural fidelity — does the review sound like a real person of this
    persona writing about this item? (1=clearly synthetic, 5=indistinguishable)
  * Contextual relevance — does the review actually engage with this specific
    item, or is it generic boilerplate? (1=generic, 5=specific to this item)

Results are aggregated per-domain and per-axis, written to:
    scratch/llm_judge_s<seed>_<provider>.json

Reads the per-seed cache `eval_artifacts/llm_cache_s{seed}.jsonl` (the new
convention from modal_eval.py) and falls back to the legacy `llm_cache.jsonl`.

Usage (auto-detects provider from env):
    export GROQ_API_KEY=...            # or CEREBRAS_API_KEY=...
    .venv/bin/python analysis/llm_judge.py --sample 500 --seed 42

Or pick explicitly:
    .venv/bin/python analysis/llm_judge.py --provider cerebras --seed 1
"""
import argparse
import collections
import json
import os
import re
import sys
import time

OUT_DIR = "scratch"
JUDGE_PROMPT = """You are an expert evaluator of recommender-system review generations.

You will read a USER PERSONA, a TARGET ITEM, and a CANDIDATE REVIEW the system produced for that persona/item. Score the candidate on two axes:

1. **behavioural_fidelity** (1–5) — does this read like a real person of this persona writing? 1 = obviously synthetic / generic / not in voice. 5 = indistinguishable from an authentic review written by someone matching the persona.
2. **contextual_relevance** (1–5) — does the review engage with *this specific item* (mentioning concrete details, dishes, plot points, features, etc.)? 1 = generic boilerplate that could apply to anything. 5 = clearly grounded in the item's specifics.

Persona:
{persona}

Target item:
Name: {name}
Category: {category}
Description: {description}

Candidate review:
\"\"\"
{review}
\"\"\"

Respond with ONLY a single JSON object, no commentary, no markdown:
{{"behavioural_fidelity": <int 1-5>, "contextual_relevance": <int 1-5>}}
"""

# Provider config: (env var, base_url, default model id on that provider).
# GPT-OSS-120B is the strongest production model on BOTH providers as of
# May 2026 — same model architecture, different model-id slugs per provider.
PROVIDERS = {
    "groq": (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
    ),
    "cerebras": (
        "CEREBRAS_API_KEY",
        "https://api.cerebras.ai/v1",
        "gpt-oss-120b",
    ),
}


def _resolve_provider(arg_provider):
    """Explicit > whichever env key is set > error."""
    if arg_provider:
        if arg_provider not in PROVIDERS:
            sys.exit(f"unknown provider {arg_provider!r}; pick from {list(PROVIDERS)}")
        return arg_provider
    for p, (env_key, _, _) in PROVIDERS.items():
        if os.environ.get(env_key):
            print(f"auto-selected provider={p} (from {env_key})", flush=True)
            return p
    sys.exit(
        "no API key found. Set one of: "
        + ", ".join(env for env, _, _ in PROVIDERS.values())
    )


def _make_client(provider):
    try:
        from openai import OpenAI
    except Exception:
        sys.exit("pip install openai  # the OpenAI SDK is the easiest way to call "
                 "Groq / Cerebras (both speak OpenAI-compatible APIs)")
    env_key, base_url, default_model = PROVIDERS[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        sys.exit(f"ERROR: {env_key} not set in env")
    return OpenAI(api_key=api_key, base_url=base_url), default_model


def _load_warm_reviews(cache_path):
    """Walk an llm_cache.jsonl, return only warm-stage review records."""
    rows = []
    if not os.path.exists(cache_path):
        sys.exit(f"cache not found at {cache_path} — run the warm eval first")
    with open(cache_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "review" or r.get("stage") != "warm":
                continue
            review = (r.get("generated_review") or "").strip()
            if not review:
                continue
            rows.append(r)
    return rows


def _judge_call(client, model, prompt, max_retries=3):
    """Call the judge LLM with exponential-backoff retries on transient errors."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def _parse_scores(text):
    """Parse the judge's JSON; robust to whitespace / stray tokens."""
    if not text:
        return None, None
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None
    bf = obj.get("behavioural_fidelity")
    cr = obj.get("contextual_relevance")
    try:
        bf = int(bf); cr = int(cr)
    except Exception:
        return None, None
    if bf not in range(1, 6) or cr not in range(1, 6):
        return None, None
    return bf, cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500,
                    help="reviews to judge per domain (default 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="filter cache to this seed only")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None,
                    help="judge LLM provider; auto-detected from env if omitted")
    ap.add_argument("--model", type=str, default=None,
                    help="override the provider's default model id")
    ap.add_argument("--cache-file", type=str, default=None,
                    help="path to llm_cache jsonl (default tries the per-seed "
                         "file first, then llm_cache.jsonl)")
    args = ap.parse_args()

    provider = _resolve_provider(args.provider)
    client, default_model = _make_client(provider)
    model = args.model or default_model
    print(f"judge: provider={provider} model={model}", flush=True)

    # Try the per-seed cache filename first (modal_eval.py's NAIJABUDDY_CACHE_FILE
    # convention) and fall back to the legacy single-file name.
    if args.cache_file:
        cache_paths = [args.cache_file]
    else:
        cache_paths = [
            os.path.join("eval_artifacts", f"llm_cache_s{args.seed}.jsonl"),
            os.path.join("eval_artifacts", "llm_cache.jsonl"),
        ]
    cache_path = next((p for p in cache_paths if os.path.exists(p)), None)
    if not cache_path:
        sys.exit(f"no cache found at any of: {cache_paths}")
    print(f"reading cache from {cache_path}", flush=True)

    rows = _load_warm_reviews(cache_path)
    rows = [r for r in rows
            if r.get("seed") in (args.seed, str(args.seed))]
    print(f"loaded {len(rows):,} warm reviews for seed {args.seed}")

    by_domain = collections.defaultdict(list)
    for r in rows:
        by_domain[r.get("domain", "?")].append(r)
    print("per-domain available:",
          {d: len(v) for d, v in by_domain.items()})

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"llm_judge_s{args.seed}_{provider}.json")
    results = {"provider": provider, "model": model, "seed": args.seed,
               "sample_per_domain": args.sample, "domains": {}}

    for domain, items in by_domain.items():
        items = items[:args.sample]
        bfs, crs, n_parsed = [], [], 0
        print(f"\n--- {domain}: judging {len(items):,} reviews ---", flush=True)
        t0 = time.time()
        for i, r in enumerate(items, 1):
            prompt = JUDGE_PROMPT.format(
                persona=r.get("persona") or "(persona not recorded)",
                name=r.get("item") or "(item)",
                category=r.get("domain", ""),
                description=(r.get("generated_review") or "")[:200],
                review=r.get("generated_review", ""),
            )
            text = _judge_call(client, model, prompt)
            bf, cr = _parse_scores(text or "")
            if bf is not None:
                bfs.append(bf); crs.append(cr); n_parsed += 1
            if i % 50 == 0:
                elapsed = time.time() - t0
                rate = elapsed / i
                eta = rate * (len(items) - i)
                print(f"  {i}/{len(items)}  parsed={n_parsed}  "
                      f"{rate:.2f}s/call  ETA {eta/60:.1f} min", flush=True)

        bf_mean = sum(bfs) / len(bfs) if bfs else None
        cr_mean = sum(crs) / len(crs) if crs else None
        print(f"  --> behavioural_fidelity={bf_mean}  contextual_relevance={cr_mean}  "
              f"(n_parsed={n_parsed}/{len(items)})", flush=True)
        results["domains"][domain] = {
            "n_judged": len(items), "n_parsed": n_parsed,
            "behavioural_fidelity_mean": bf_mean,
            "contextual_relevance_mean": cr_mean,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
