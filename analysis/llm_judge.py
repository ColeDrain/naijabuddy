"""
LLM-as-judge evaluation for review-text quality (Behavioural Fidelity +
Contextual Relevance — the rubric's human-eval buckets, of which we currently
have ZERO evidence in the paper).

For each cached generated review, this asks a strong external judge model
(default: Gemini 2.5 Flash via Google AI) to score on 1–5 across two axes:

  * Behavioural fidelity — does the review sound like a real person of this
    persona writing about this item? (1=clearly synthetic, 5=indistinguishable)
  * Contextual relevance — does the review actually engage with this specific
    item, or is it generic boilerplate? (1=generic, 5=specific to this item)

Results are aggregated per-domain and per-axis, written to:
    scratch/llm_judge_<seed>.json

The script is INDEPENDENT of the warm/cold harness — it reads the existing
`eval_artifacts/llm_cache.jsonl` cache, finds the reviews to score, samples
N per domain, batches the judging calls. ~$1–2 worth of Gemini for the full
500-review-per-domain × 3-seed run.

Usage:
    export GEMINI_API_KEY=...           # or GOOGLE_API_KEY
    .venv/bin/python analysis/llm_judge.py --sample 500
"""
import argparse
import collections
import json
import os
import re
import sys
import time

CACHE_PATH = os.path.join("eval_artifacts", "llm_cache.jsonl")
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


def _load_warm_reviews():
    """Walk llm_cache.jsonl, return list of (domain, persona, item-meta, review)."""
    rows = []
    if not os.path.exists(CACHE_PATH):
        sys.exit(f"cache not found at {CACHE_PATH} — run the warm eval first")
    with open(CACHE_PATH) as f:
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


def _gemini_call(model, prompt, max_retries=3):
    """Call Gemini, retry on transient errors."""
    import google.generativeai as genai
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 64,
                                   "response_mime_type": "application/json"},
            )
            return resp.text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def _parse_scores(text):
    """Parse the judge's JSON, robust to whitespace / stray tokens."""
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
    ap.add_argument("--seed", type=int, default=42, help="filter cache to this seed only")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash",
                    help="Gemini model id to use as judge")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) in env")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    rows = _load_warm_reviews()
    rows = [r for r in rows if r.get("seed") in (args.seed, str(args.seed))]
    print(f"loaded {len(rows):,} warm reviews for seed {args.seed}")

    by_domain = collections.defaultdict(list)
    for r in rows:
        by_domain[r.get("domain", "?")].append(r)
    print("per-domain available:",
          {d: len(v) for d, v in by_domain.items()})

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"llm_judge_s{args.seed}.json")
    results = {"model": args.model, "seed": args.seed,
               "sample_per_domain": args.sample, "domains": {}}

    for domain, items in by_domain.items():
        items = items[:args.sample]
        bfs, crs, n_parsed = [], [], 0
        print(f"\n--- {domain}: judging {len(items):,} reviews ---")
        t0 = time.time()
        for i, r in enumerate(items, 1):
            prompt = JUDGE_PROMPT.format(
                persona=r.get("persona") or "(persona not recorded)",
                name=r.get("item") or "(item)",
                category=r.get("domain", ""),
                description=(r.get("generated_review") or "")[:200],
                review=r.get("generated_review", ""),
            )
            text = _gemini_call(model, prompt)
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
              f"(n_parsed={n_parsed}/{len(items)})")
        results["domains"][domain] = {
            "n_judged": len(items), "n_parsed": n_parsed,
            "behavioural_fidelity_mean": bf_mean,
            "contextual_relevance_mean": cr_mean,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
