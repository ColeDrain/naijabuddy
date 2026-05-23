"""
Modal-hosted LLM-as-judge for review-text quality.

Wraps the same scoring logic as analysis/llm_judge.py but runs server-side on
Modal: mounts the persistent `naijabuddy-eval-cache` volume read-only (where
the main eval already wrote per-seed `llm_cache_s{seed}.jsonl`), calls Groq or
Cerebras over their OpenAI-compatible endpoints using GPT-OSS-120B as the
judge, returns aggregated scores to the local client which writes them to
scratch/.

Two reasons to run this on Modal rather than locally:
  1. The cache is already on the Modal volume — no `modal volume get` step.
  2. Survives the local machine sleeping; the goal-driven overnight run does
     not depend on the user's Mac being awake.

The API key is supplied via a persistent Modal Secret — the key never leaves
Modal's encrypted secret store and is not present in your local shell.

One-time setup (Modal dashboard → Secrets → New secret → Custom):
    Secret name: groq-api-key
    Key:         GROQ_API_KEY
    Value:       gsk_...

    (and/or)

    Secret name: cerebras-api-key
    Key:         CEREBRAS_API_KEY
    Value:       csk-...

The Modal Secret *name* (the label) must match SECRET_NAMES below; the
*env-var key inside the secret* must match the env name in PROVIDERS so the
container's `os.environ[...]` lookup hits.

Usage:
    modal run modal_llm_judge.py --seed 42 --sample 500                  # default groq
    modal run modal_llm_judge.py --seed 1 --provider cerebras            # explicit
    modal run modal_llm_judge.py --seed 7 --model gpt-oss-120b           # override model
"""
import collections
import json
import os
import re
import sys
import time
import modal

app = modal.App("naijabuddy-llm-judge")

# Read the SAME volume the main eval writes to — naijabuddy-eval-cache.
cache_volume = modal.Volume.from_name(
    "naijabuddy-eval-cache", create_if_missing=True
)

image = modal.Image.debian_slim(python_version="3.11").pip_install("openai>=1.0")

# Provider table: (env-var name, OpenAI-compatible base_url, default model id).
# GPT-OSS-120B is the strongest production model on BOTH providers as of
# May 2026 — same model architecture, different id slug per provider.
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

# Maps provider → name of the Modal Secret holding that provider's API key.
# Create these on the Modal dashboard (see module docstring).
SECRET_NAMES = {
    "groq": "groq-api-key",
    "cerebras": "cerebras-api-key",
}

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


def _parse_scores(text):
    """Robust JSON-object parse → (behavioural_fidelity, contextual_relevance)."""
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


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    cpu=2.0,
    timeout=3600,
)
def judge(seed: int, sample: int, provider: str, model: str):
    """Run inside the Modal container — reads /cache, calls provider, returns JSON."""
    from openai import OpenAI

    env_key, base_url, _ = PROVIDERS[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        return {"error": f"{env_key} not present in container env"}
    client = OpenAI(api_key=api_key, base_url=base_url)

    cache_path = f"/cache/llm_cache_s{seed}.jsonl"
    if not os.path.exists(cache_path):
        # Fall back to legacy single-file cache, just in case.
        alt = "/cache/llm_cache.jsonl"
        if os.path.exists(alt):
            cache_path = alt
        else:
            return {"error": f"no cache at /cache/llm_cache_s{seed}.jsonl"}

    rows = []
    with open(cache_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "review" or r.get("stage") != "warm":
                continue
            if r.get("seed") not in (seed, str(seed)):
                continue
            if not (r.get("generated_review") or "").strip():
                continue
            rows.append(r)
    print(f"loaded {len(rows):,} warm reviews from {cache_path}", flush=True)

    by_domain = collections.defaultdict(list)
    for r in rows:
        by_domain[r.get("domain", "?")].append(r)
    print("per-domain available:",
          {d: len(v) for d, v in by_domain.items()}, flush=True)

    results = {"provider": provider, "model": model, "seed": seed,
               "sample_per_domain": sample, "domains": {}}

    for domain, items in by_domain.items():
        items = items[:sample]
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
            text = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=64,
                        response_format={"type": "json_object"},
                    )
                    text = resp.choices[0].message.content.strip()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        text = None
            bf, cr = _parse_scores(text or "")
            if bf is not None:
                bfs.append(bf); crs.append(cr); n_parsed += 1
            if i % 50 == 0:
                el = time.time() - t0
                rate = el / i
                eta = rate * (len(items) - i)
                print(f"  {i}/{len(items)}  parsed={n_parsed}  "
                      f"{rate:.2f}s/call  ETA {eta/60:.1f}m", flush=True)

        bf_mean = sum(bfs) / len(bfs) if bfs else None
        cr_mean = sum(crs) / len(crs) if crs else None
        print(f"  --> behavioural_fidelity={bf_mean}  contextual_relevance={cr_mean}  "
              f"(n_parsed={n_parsed}/{len(items)})", flush=True)
        results["domains"][domain] = {
            "n_judged": len(items), "n_parsed": n_parsed,
            "behavioural_fidelity_mean": bf_mean,
            "contextual_relevance_mean": cr_mean,
        }

    return results


@app.local_entrypoint()
def main(seed: int = 42, sample: int = 500, provider: str = "groq",
         model: str = ""):
    """
    modal run modal_llm_judge.py --seed 42 --sample 500
    """
    if provider not in PROVIDERS:
        sys.exit(f"unknown provider {provider!r}; pick from {list(PROVIDERS)}")
    env_key, _, default_model = PROVIDERS[provider]
    model = model or default_model
    secret_name = SECRET_NAMES[provider]
    print(f"judging via Modal: provider={provider} model={model} "
          f"seed={seed} sample={sample}  secret={secret_name!r}", flush=True)

    # Reference the persistent Modal Secret by name and attach it to this
    # call only with .with_options(). The secret's value is mounted into the
    # container's env (as `env_key`) at runtime; it is never resident locally.
    secret = modal.Secret.from_name(secret_name)
    bound = judge.with_options(secrets=[secret])
    results = bound.remote(seed=seed, sample=sample,
                           provider=provider, model=model)

    os.makedirs("scratch", exist_ok=True)
    out_path = f"scratch/llm_judge_s{seed}_{provider}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 66)
    print(f"LLM-judge complete. Results -> {out_path}")
    for dom, d in results.get("domains", {}).items():
        print(f"  {dom:<10} behavioural_fidelity={d.get('behavioural_fidelity_mean')}  "
              f"contextual_relevance={d.get('contextual_relevance_mean')}  "
              f"(n_parsed={d.get('n_parsed')}/{d.get('n_judged')})")
    print("=" * 66)
