# Solution Paper — Trim & Polish Proposal

The brief caps the solution paper at **8 pages**. The current draft is ~4,700
words + 8 tables + a large ASCII diagram + ~6 equations — an estimated 9–10
pages once rendered. Adding the new **Related Work** section (~1 page) and the
n=2,000 §4 rewrite pushes it further over. This is a concrete plan to land at
≤ 8 pages **without cutting a single finding**.

Principle: trim *prose and decoration*, never *results*. Every cut below is in
§1–§3, §5–§6 or formatting — §4 (the experiments, the scored core) is
protected.

## Trim candidates — ranked safest-first

| # | Action | Section | ~Saving | Safety |
|---|---|---|---|---|
| 1 | Replace the 38-line ASCII architecture diagram with a compact 5-line layer list (or a small 4-box figure) | §2 | ~0.35 pg | ✅ zero content loss — prose already explains all 4 layers |
| 2 | Consolidate the "positivity bias" explanation — currently stated in the Abstract, §1(2) *and* §2.3 — to one place (§2.3), referenced elsewhere | abstract/§1/§2.3 | ~0.1 pg | ✅ pure de-duplication |
| 3 | Tighten §1 Introduction — the 3 "challenges" and 3 "contributions" lists restate the abstract and §2; cut to a short paragraph each | §1 | ~0.25 pg | ✅ overlap removal |
| 4 | Tighten the 3 persona descriptions ~30% (keep all three — cultural fidelity is scored — just less verbose) | §3 | ~0.15 pg | ✅ keeps every persona |
| 5 | Tighten §6 Conclusion — it restates §4.8 "Honest Summary"; cut to ~3 sentences | §6 | ~0.1 pg | ✅ overlap removal |
| 6 | Compress §2.1's catalog example lists (the parenthetical item names) | §2.1 | ~0.1 pg | ✅ examples, not claims |
| 7 | *If still over:* compress the §4.4 sampled-metric subsection prose (keep the table + the Krichene/Rendle citation + the temporal-split caveat) | §4.4 | ~0.15 pg | ⚠️ judgment — it is genuine rigor; cut prose only, last resort |

Items 1–6 are safe de-duplication / decoration trims totalling **~1.05 pages** —
enough to absorb the Related Work addition and land at ≤ 8. Item 7 is held in
reserve and is the only one touching §4; it removes prose, not the result.

## Polish notes (apply with the trims — wording only, no content change)

- §1: "With the explosion of user-generated content…" → tighter opener.
- Throughout: a few long compound sentences in §2 can be split.
- §4.2: **add one sentence** (per `validation_audit.md`) — the V2/V3 blend
  weights are the *descriptive minimum* of the sweep, not validation-tuned;
  note this strengthens the "LLM redundant warm" conclusion. (This is an
  *addition*, ~1 line, already budgeted.)

## Sequencing

These edits are best applied in the **single coherent paper pass after the
n=2,000 eval** — that pass already rewrites §4 with the new numbers and inserts
Related Work, so prose-polish + trims ride along rather than touching the paper
twice. Items 1–6 will be applied then; item 7 only if the page count still
demands it, and flagged for sign-off because it is the one §4 edit.

## What is NOT cut

§4 in full — every table, every ablation (§4.2 calibration, §4.3 review, §4.4
retrieval, §4.5 cold-start, §4.6 persona, §4.7 RAG, §4.8 summary). The
two-populations and regime-switch findings, the honest negative results, and
the numbers-integrity discipline are the submission's value and stay intact.
