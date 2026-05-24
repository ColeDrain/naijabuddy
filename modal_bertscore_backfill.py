"""
Backfill BERTScore-F1 for the synth-vLLM multi-seed run.

The original multi-seed synth runs (modal_vllm_eval.py) computed ROUGE-L and
Semantic-BGE successfully but BERTScore failed with CUDA OOM — vLLM was already
holding ~85% of the A10G's VRAM and RoBERTa-large couldn't be loaded alongside
it. This script reads the cached generated reviews + the held-out human
references and computes BERTScore-F1 entirely on CPU (no vLLM contention).

Per seed × per domain output: mean BERTScore-F1 across the warm sample. Writes
scratch/bertscore_backfill_synth.json with the aggregated numbers ready to
paste into §4.3.

Usage:
    modal run modal_bertscore_backfill.py
"""
import os
import modal

app = modal.App("naijabuddy-bertscore-backfill")

cache_volume = modal.Volume.from_name(
    "naijabuddy-eval-cache", create_if_missing=True
)


def _download_roberta():
    """Pre-download RoBERTa-large for BERTScore (~1.4 GB)."""
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("roberta-large")
    AutoModel.from_pretrained("roberta-large")


image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04",
                              add_python="3.11")
    .pip_install("bert-score", "torch", "transformers", "pandas")
    .run_function(_download_roberta)
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


@app.function(
    image=image,
    gpu="a10g",
    volumes={"/cache": cache_volume},
    cpu=2.0, memory=8192, timeout=1800,
)
def backfill():
    """GPU BERTScore-F1 over the synth-vLLM cached reviews (~30x faster than CPU)."""
    import json
    import pandas as pd
    from bert_score import score as bs_score

    # Map domain → CSV stem (matching eval_harness.DOMAINS).
    DOMAIN_STEMS = {"Yelp": "yelp", "Goodreads": "goodreads", "Amazon": "amazon"}

    # Build (user, item) -> reference_review lookup per domain. The cached
    # generated_review records carry (user, item, domain, seed) so we can
    # join straight against the CSV without re-running the leave-one-out logic.
    refs_by_domain = {}
    for dom, stem in DOMAIN_STEMS.items():
        path = f"/app/data/{stem}_dense.csv"
        if not os.path.exists(path):
            print(f"  [warn] {path} missing, skipping {dom}", flush=True)
            continue
        df = pd.read_csv(path)
        df["review_text"] = df["review_text"].fillna("").astype(str)
        lookup = {}
        # CSV columns: user_id, item_id, item_name, category, rating, review_text
        for r in df.itertuples(index=False):
            lookup[(str(r.user_id), str(r.item_id))] = str(r.review_text)
        refs_by_domain[dom] = lookup
        print(f"  [refs] {dom}: {len(lookup):,} (user, item) -> review",
              flush=True)

    out = {"per_seed": {}, "across_seeds": {}}

    for seed in [42, 1, 7]:
        cache_path = f"/cache/llm_cache_s{seed}_synth_vllm.jsonl"
        if not os.path.exists(cache_path):
            print(f"  [warn] {cache_path} missing", flush=True)
            continue
        print(f"\n=== seed={seed} ===", flush=True)
        per_dom = {}
        with open(cache_path) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        for dom in DOMAIN_STEMS:
            hyps, refs = [], []
            for r in rows:
                if r.get("kind") != "review":
                    continue
                if r.get("stage") != "warm":
                    continue
                if r.get("domain") != dom:
                    continue
                gen = (r.get("generated_review") or "").strip()
                if not gen:
                    continue
                u = r.get("user"); i = r.get("item")
                ref = refs_by_domain.get(dom, {}).get((str(u), str(i)))
                if not ref:
                    continue
                hyps.append(gen)
                refs.append(ref)
            if not hyps:
                print(f"  {dom}: 0 matched pairs, skipping", flush=True)
                continue
            print(f"  {dom}: scoring {len(hyps):,} pairs on CPU "
                  "(roberta-large)...", flush=True)
            _, _, F1 = bs_score(
                hyps, refs,
                model_type="roberta-large",
                lang="en",
                rescale_with_baseline=False,
                verbose=False,
            )
            mean_f1 = float(F1.mean())
            per_dom[dom] = {"n_pairs": len(hyps), "bertscore_f1_mean": mean_f1}
            print(f"  {dom}: BERTScore-F1 = {mean_f1:.4f} (n={len(hyps)})",
                  flush=True)
        out["per_seed"][seed] = per_dom

    # Aggregate mean ± std across seeds, per domain.
    import math
    for dom in DOMAIN_STEMS:
        vals = [out["per_seed"][s][dom]["bertscore_f1_mean"]
                for s in out["per_seed"] if dom in out["per_seed"][s]]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
               if len(vals) > 1 else 0.0)
        out["across_seeds"][dom] = {"n_seeds": len(vals), "mean": mean,
                                    "std": std}
        print(f"\n[3-seed agg] {dom}: BERTScore-F1 = "
              f"{mean:.4f} ± {std:.4f}  (n_seeds={len(vals)})", flush=True)

    return out


@app.local_entrypoint()
def main():
    out = backfill.remote()
    import json
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/bertscore_backfill_synth.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote scratch/bertscore_backfill_synth.json")
    print("Across-seed BERTScore-F1:")
    for dom, agg in out.get("across_seeds", {}).items():
        print(f"  {dom:<10} {agg['mean']:.4f} ± {agg['std']:.4f}  "
              f"(n_seeds={agg['n_seeds']})")
