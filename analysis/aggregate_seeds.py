"""
Aggregate per-seed Modal eval JSONs into mean ± std (and per-seed table) for
the paper.

Generic: walks every numeric leaf inside each domain (Yelp/Goodreads/Amazon)
and treats them all the same. No hard-coded metric list — survives the
audit-fixed schema adding V3 to warm, multi-k retrieval, BERTScore-F1, etc.

Inputs (default): scratch/modal_results_n2000_template_s{42,1,7}.json
Outputs:
  scratch/aggregated_n2000_3seed.json  - all numeric paths × all seeds
  scratch/aggregated_n2000_3seed.md    - human-readable summary tables

Usage:
    .venv/bin/python analysis/aggregate_seeds.py
    .venv/bin/python analysis/aggregate_seeds.py --glob 'scratch/foo_s*.json'
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


def _walk_numeric(obj, prefix=""):
    """Yield (dotted_path, value) for every numeric leaf, skipping bools."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numeric(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_numeric(v, f"{prefix}[{i}]")
    else:
        # exclude bools (they're ints in python) and Nones
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)) and obj is not None:
            yield prefix, float(obj)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs):
    """Sample std (Bessel-corrected). Returns 0 for n<2 to keep output clean."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate(seed_results):
    """
    seed_results: dict[seed_int] -> top-level json dict
    Returns dict[domain][metric_path] -> {"seeds": [s1, s2, ..], "values":
    [v1, v2, ..], "mean": float, "std": float}
    """
    # Discover the domain set (top-level keys that are dicts; skip metadata).
    domains = set()
    for sd in seed_results.values():
        for k, v in sd.items():
            if isinstance(v, dict):
                domains.add(k)

    out = {}
    for dom in sorted(domains):
        per_metric = defaultdict(lambda: {"seeds": [], "values": []})
        for seed, sd in seed_results.items():
            if dom not in sd:
                continue
            for path, val in _walk_numeric(sd[dom]):
                per_metric[path]["seeds"].append(seed)
                per_metric[path]["values"].append(val)
        # finalise mean/std
        for path, rec in per_metric.items():
            rec["mean"] = _mean(rec["values"])
            rec["std"] = _stdev(rec["values"])
            rec["n"] = len(rec["values"])
        out[dom] = dict(per_metric)
    return out


# Metric paths to highlight in the Markdown summary (in display order).
HIGHLIGHT_WARM = [
    ("RMSE V0 (global mean)",          "rmse_sample.V0_global"),
    ("RMSE V1 (user mean)",            "rmse_sample.V1_user_mean"),
    ("RMSE pure-LLM",                  "rmse_sample.pure_llm"),
    ("RMSE V2 (best blend)",           "rmse_sample.V2_best_blend"),
    ("RMSE V3 (3-term)",               "rmse_sample.V3_3term"),
    ("best α",                         "best_alpha"),
    ("ROUGE-L",                        "rouge_l"),
    ("Semantic-BGE",                   "semantic_bge"),
    ("BERTScore-F1",                   "bertscore_f1"),
]
RETRIEVAL_METHODS = ["dense_raw", "dense", "hybrid", "cf", "als", "popularity"]
RETRIEVAL_KS = [10, 20, 50, 100]


def fmt_meanstd(rec, places=4):
    if rec is None or rec.get("n", 0) == 0:
        return "—"
    if rec["n"] == 1:
        return f"{rec['mean']:.{places}f}"
    return f"{rec['mean']:.{places}f} ± {rec['std']:.{places}f}"


def make_markdown(agg, seeds):
    L = [
        f"# Aggregated multi-seed results  (seeds: {seeds})",
        "",
        f"_{len(seeds)} seeds; mean ± sample std (Bessel-corrected)._",
        "",
    ]
    for dom in sorted(agg.keys()):
        per = agg[dom]
        L += [f"## {dom}", ""]

        # Warm headline table.
        L += ["### Warm-start (n=2,000 held-out)", "",
              "| Metric | Value |", "|---|---|"]
        for label, path in HIGHLIGHT_WARM:
            rec = per.get(path)
            L.append(f"| {label} | {fmt_meanstd(rec)} |")
        L += [""]

        # Multi-k retrieval table.
        L += ["### Retrieval (HR@k / NDCG@k)", "",
              "| Method | " + " | ".join(
                  f"HR@{k} | NDCG@{k}" for k in RETRIEVAL_KS) + " |",
              "|---|" + "---|" * (2 * len(RETRIEVAL_KS))]
        for meth in RETRIEVAL_METHODS:
            row = [meth]
            for k in RETRIEVAL_KS:
                hr = per.get(f"retrieval.{meth}.hit@{k}")
                nd = per.get(f"retrieval.{meth}.ndcg@{k}")
                row.append(fmt_meanstd(hr))
                row.append(fmt_meanstd(nd))
            L.append("| " + " | ".join(row) + " |")
        L += [""]

        # Cold-start table.
        L += ["### Cold-start (k=1,2,3 history items)", "",
              "| k | V0 | V1 | V2 | V3 (3-term) | best α | V3 wts (llm/user/item) | dense HR@10 |",
              "|---|---|---|---|---|---|---|---|"]
        for k in (1, 2, 3):
            row = [str(k)]
            for path in (
                f"cold_start.{k}.rmse_v0_global",
                f"cold_start.{k}.rmse_v1_user_mean",
                f"cold_start.{k}.rmse_v2_best_blend",
                f"cold_start.{k}.rmse_v3_3term",
                f"cold_start.{k}.best_alpha",
            ):
                row.append(fmt_meanstd(per.get(path)))
            # V3 weights as triple
            wts = []
            for w in ("llm", "user", "item"):
                wrec = per.get(f"cold_start.{k}.v3_weights.{w}")
                wts.append(fmt_meanstd(wrec, places=2))
            row.append("/".join(wts))
            row.append(fmt_meanstd(per.get(f"cold_start.{k}.dense_hit@10")))
            L.append("| " + " | ".join(row) + " |")
        L += [""]

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="scratch/modal_results_n2000_template_s*.json",
        help="glob pattern matching per-seed Modal JSONs",
    )
    ap.add_argument("--out-json",
                    default="scratch/aggregated_n2000_3seed.json")
    ap.add_argument("--out-md",
                    default="scratch/aggregated_n2000_3seed.md")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        sys.exit(f"no files matched glob {args.glob!r}")

    seed_results = {}
    for p in paths:
        m = re.search(r"_s(\d+)(?:[._])", os.path.basename(p))
        if not m:
            print(f"skip (no _s<n>_ in name): {p}")
            continue
        seed = int(m.group(1))
        with open(p) as f:
            seed_results[seed] = json.load(f)
        print(f"loaded seed={seed:>3} from {p}")

    if not seed_results:
        sys.exit("no seed JSONs successfully loaded")

    agg = aggregate(seed_results)
    seeds_sorted = sorted(seed_results.keys())

    with open(args.out_json, "w") as f:
        json.dump({"seeds": seeds_sorted, "aggregated": agg}, f, indent=2)
    print(f"wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(make_markdown(agg, seeds_sorted))
    print(f"wrote {args.out_md}")

    # Headline print
    print("\n=== HEADLINE: warm V2 RMSE across seeds ===")
    for dom in sorted(agg.keys()):
        rec = agg[dom].get("rmse_sample.V2_best_blend")
        if rec:
            print(f"  {dom:<12} V2={fmt_meanstd(rec)}  (n_seeds={rec['n']})")


if __name__ == "__main__":
    main()
