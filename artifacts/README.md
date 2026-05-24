# NaijaBuddy — Published Artifacts

These are the raw output files that back the headline numbers in
[`../solution_paper.md`](../solution_paper.md). They are committed so that
any judge or reproducer can:

1. Inspect the per-seed JSONs the eval harness wrote.
2. Re-run the aggregator on them and verify the §4 tables.
3. Diff against their own reproduction (Path A/B/C/D in the top-level README).

## Files

| File | What it is | Paper reference |
|---|---|---|
| `modal_results_n2000_synth_vllm_s42.json` | Seed-42 raw eval output: per-domain RMSE V0/V1/V2, ROUGE-L, Semantic-BGE, multi-k retrieval, cold-start V0/V1/V2/V3 per k | §4.2 / §4.3 / §4.4 / §4.5 |
| `modal_results_n2000_synth_vllm_s1.json` | Seed-1 raw eval output (same schema) | same |
| `modal_results_n2000_synth_vllm_s7.json` | Seed-7 raw eval output (same schema) | same |
| `modal_results_n2000_synth_vllm_s{42,1,7}.md` | Human-readable per-seed summaries | — |
| `aggregated_synth_3seed.json` | Mean ± std across the three seed JSONs above | §4.2 / §4.3 / §4.4 / §4.5 |
| `aggregated_synth_3seed.md` | Human-readable mean ± std summary (the canonical paper-tables source) | same |
| `bertscore_backfill_synth.json` | Per-seed + 3-seed-aggregated BERTScore-F1 (computed via separate Modal A10G pass; the in-vLLM run OOM'd) | §4.3 |
| `cross_domain_results.json` | Books → Movies transfer raw output: V0, V1_books, V1_movies, item-mean-only, V2_books+item, V2_movies+item RMSEs + the β sweeps | §4.8 |

## How they were produced

- `modal_results_n2000_synth_vllm_s{42,1,7}.{json,md}` →
  `modal run modal_vllm_eval.py --sample 2000 --persona-mode synth --seed N --cold-start --cold-sample 2000 --bertscore`
  (the BERTScore step OOM'd; see backfill below)
- `aggregated_synth_3seed.{json,md}` →
  `python analysis/aggregate_seeds.py --glob 'artifacts/modal_results_n2000_synth_vllm_s*.json'`
  (or whatever path the per-seed JSONs live at)
- `bertscore_backfill_synth.json` →
  `modal run modal_bertscore_backfill.py` (Modal A10G, separate container with no vLLM contention for VRAM)
- `cross_domain_results.json` →
  `python cross_domain_dataset.py && python cross_domain_eval.py`
  (pure CPU, no LLM, ~30 sec)

See [`../numbers_integrity.md`](../numbers_integrity.md) for the full
paper-figure-to-regen-command mapping, including auxiliary results (variance
buckets, V3 warm calibration, sampled-metric retrieval) that have their own
regen commands.
