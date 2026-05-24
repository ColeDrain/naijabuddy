# Reproducing NaijaBuddy

Quick map of which doc to read for which question. The full reproduction
recipes live in **[README.md → "🔬 Reproducing the Multi-Seed Canonical
Results"](./README.md#-reproducing-the-multi-seed-canonical-results)** — this
file is just a navigation index.

## Quickest path: do I have the deps right?

```bash
bash scripts/run_smoke.sh
```

60-second CPU-only sanity check. Imports, dataset presence, a tiny `--llm-sample 5`
warm-only run on Yelp, output-schema verification. **`PASS`** means your setup
is good for the full reproduction below.

## I want the §4 numbers

| You want to reproduce... | Read... | TL;DR command |
|---|---|---|
| **Headline multi-seed RMSE / ROUGE-L / Sem-BGE / BERTScore-F1 / retrieval / cold-start (§4.2 / §4.3 / §4.4 / §4.5)** | [README "Path A"](./README.md#path-a--self-host-vllm-on-any-gpu-free-if-you-own-it) | Self-host vLLM, hit `eval_harness.py --vllm-url`, aggregate the 3 seeds |
| Same, but on a hosted endpoint (RunPod / Together AI / similar) | [README "Path B"](./README.md#path-b--hosted-openai-compatible-endpoint-runpod--together-ai--similar) | Point `--vllm-url` at any OpenAI-compatible URL |
| Same, on a CPU laptop (verification scale) | [README "Path C"](./README.md#path-c--verification-only-on-cpu-laptop-no-gpu) | `python eval_harness.py --persona-mode synth --seed 42 --llm-sample 20` |
| **Cross-domain Books → Movies transfer (§4.8)** | `cross_domain_eval.py` source comments | `python cross_domain_dataset.py && python cross_domain_eval.py` (pure CPU, no LLM, ~30 s) |
| BERTScore-F1 multi-seed (§4.3, GPU backfill pass) | `modal_bertscore_backfill.py` | `modal run modal_bertscore_backfill.py` |
| Sampled-metric retrieval (§4.4 "Comparability" table, 101 candidates) | `eval_harness.py --candidate-pool 101` | `python eval_harness.py --candidate-pool 101 --pop-distractors --seed 42` |
| Variance-bucket V1 RMSE breakdown (§4.2 "Two populations") | `analysis/study_data.py` | `python analysis/study_data.py` |
| V3 3-term warm calibration sweep (§4.2 item-bias paragraph) | `analysis/measure_calib3.py` | `python analysis/measure_calib3.py` |

## I want to inspect the raw numbers without re-running

Everything in [`artifacts/`](./artifacts/) is the actual JSON the paper cites:

- `artifacts/aggregated_synth_3seed.{json,md}` — the 3-seed mean ± std the paper headline tables come from
- `artifacts/modal_results_n2000_synth_vllm_s{42,1,7}.{json,md}` — per-seed raw eval output
- `artifacts/bertscore_backfill_synth.json` — §4.3 BERTScore-F1
- `artifacts/cross_domain_results.json` — §4.8 cross-domain RMSEs + β sweeps

See [`artifacts/README.md`](./artifacts/README.md) for the file-by-file map.

## Single source of truth for "which command produces which number"

[`numbers_integrity.md`](./numbers_integrity.md) maps every quantitative claim
in §4 to its exact regeneration command, with the expected output value.
Read that one if you're chasing a specific figure.
