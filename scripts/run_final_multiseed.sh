#!/bin/bash
# Fire the audit-fixed 3-seed multi-seed final run on Modal.
#
# Three parallel containers (one per seed), each runs:
#   - n=2,000 warm eval (V0/V1/V2 + per-domain prompts + JSON grammar + BERTScore)
#   - n=2,000 cold-start (k=1,2,3) — full parity with warm sample
# ~3.5h wallclock, ~$11 across all 3 containers.
#
# Prereqs (verified before running):
#   - smoke test passed (JSON 100%, reviews 1-2 sentences, BERTScore populated)
#   - eval_artifacts/llm_cache.jsonl is clear (cleared earlier; old prompt entries
#     snapshotted to scratch/llm_cache_pre_finalrun.jsonl)
#   - Modal profile = hackathon-30 ($30 account)
#
# Per-container output: scratch/modal_results_n2000_template_s<seed>.{json,md}
set -e
cd "$(dirname "$0")/.."
mkdir -p scratch

echo "===================================================================="
echo "FINAL multi-seed re-run, n=2,000 warm + cold-start (audit-fixed code)"
echo "$(date)"
echo "===================================================================="
for s in 42 1 7; do
    nohup modal run modal_eval.py \
        --sample 2000 \
        --persona-mode template \
        --seed $s \
        --cold-sample 2000 \
        --bertscore \
        > scratch/modal_final_s${s}.log 2>&1 &
    pid=$!
    echo "fired seed=$s, pid=$pid -> scratch/modal_final_s${s}.log"
    sleep 2  # stagger slightly so Modal app IDs don't collide
done
echo
echo "all 3 launched. monitor with:"
echo "  tail -f scratch/modal_final_s*.log"
echo "or wait on completion:"
echo "  while pgrep -f 'modal run modal_eval' >/dev/null; do sleep 60; done"
