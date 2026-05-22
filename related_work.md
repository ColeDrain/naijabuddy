# Related Work & Positioning

*Drafted for the solution paper (#2 literature pass). Paper-ready prose; to be
folded into `solution_paper.md` during the §-numbering/trim pass.*

---

## Related Work

**LLMs as rerankers and recommenders.** A fast-growing line of work casts the
LLM as a ranking component. The RankGPT family — and its open-source
counterpart **RankVicuna** [Pradeep et al., 2023] — performs zero-shot listwise
document reranking; **Self-Calibrated Listwise Reranking** [Ren et al., 2024]
addresses the context-window limits of that paradigm. In recommendation
specifically, **EXP3RT** [Kim et al., 2024] fine-tunes an LLM to extract
preferences from reviews and produce a reasoning-enhanced rating plus a
reranked top-*k* list. NaijaBuddy's Task B sits in this family but deliberately
keeps the LLM as a *second-stage reranker over a cheap hybrid retrieval stage*
(dense + item-item CF), rather than asking the LLM to rank the full catalogue —
a design forced by, and suited to, a small local model.

**LLMs for rating prediction, and the limits of the LLM signal.** Kang et al.
[2023], "Do LLMs Understand User Preferences?", is the closest prior result to
our central finding: evaluating LLMs from 250M to 540B parameters on user
rating prediction, they show **zero-shot LLMs lag traditional collaborative
filtering whenever interaction data is available** — the interaction history,
not the LLM, carries the signal. NaijaBuddy independently corroborates this on
three domains and *operationalises* it: our calibration layer measures exactly
how much weight the LLM's numeric rating should receive (≈ 0 for warm users)
and turns the result into a deployed regime switch. Ryu & Yanaka [2025] show
that supplying **in-context user reviews** lifts off-the-shelf LLM rating
prediction toward matrix-factorisation quality and helps the cold-start case,
and that concrete-item exemplars beat generic preference text. Our Tier-2
retrieval-augmented ablation tests precisely this mechanism under a
leakage-free protocol, and refines the picture: in-context exemplars improve
the generated *review text* but, for *warm* users, do not beat the user-mean
anchor on the *rating* — consistent with Kang's "interaction data dominates."

**Review generation and user simulation.** **Review-LLM** [Peng et al., 2024]
targets personalised review generation and documents the "polite phenomenon" —
LLMs resist producing genuinely negative reviews; **BASES** [Ren et al., 2024]
simulates web-search users at scale; a recent survey [Ni et al., 2026] maps the
broader LLM-based user-simulation space. Coherency-Improved Explainable
Recommendation [Liu et al., 2025] notes that jointly produced ratings and
explanations are often mutually incoherent. NaijaBuddy's Task A jointly
simulates a rating and a persona-grounded review; the deterministic calibration
layer is in part a defence against the "polite" upward bias — it re-anchors an
over-generous LLM score on the user's and item's own statistics.

**Evaluation rigour and reproducibility.** Dacrema et al. [2019] showed that
many neural recommenders fail to beat well-tuned simple baselines once
evaluation is done carefully — a warning we take literally by always reporting
the global-mean and user-mean baselines alongside every LLM result. RankVicuna
[Pradeep et al., 2023] separately argues that reranking results built on opaque
proprietary APIs are "not reproducible and non-deterministic." NaijaBuddy
answers both concerns directly: a leakage-free leave-one-out protocol,
multi-seed error bars, an artifact-cached harness in which every reported figure
regenerates, and a 100%-offline stack with no API call at any point.

## Positioning

Against this literature, NaijaBuddy's distinctive position is the combination
of four choices, none individually novel but rarely held together:

1. **A small, quantised, fully-offline LLM.** Most LLM-recommendation work uses
   GPT-3.5/4 or fine-tunes mid-size models; NaijaBuddy runs a 3B Qwen2.5 GGUF
   in-process, no network at runtime. The cold-start-bias study of Andre et al.
   [2025] is one of the few others to centre open models (Gemma 3, Llama 3.2).
2. **Calibration as a measured regime switch**, not a guardrail — quantifying
   *when* the LLM helps (cold-start) and when statistics dominate (warm), rather
   than assuming the LLM is always the right predictor.
3. **Reproducible, leakage-free evaluation** as a first-class deliverable.
4. **Explicit cultural localisation** for an underserved market (Nigeria) —
   adjacent to, but distinct from, the fairness-audit framing of Andre et al.

## References

- Andre, Roy, Dyer, Wang. *Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting.* arXiv:2508.20401, 2025.
- Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys, 2019.
- Kang, Ni, Mehta, Sathiamoorthy, Hong, Chi, Cheng. *Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction.* arXiv:2305.06474, 2023.
- Kim, Kim, Cho, Kang, Chang, Yeo, Lee. *Review-driven Personalized Preference Reasoning with LLMs for Recommendation (EXP3RT).* arXiv:2408.06276, 2024.
- Liu et al. *Coherency Improved Explainable Recommendation via Large Language Model.* arXiv:2504.05315, 2025.
- Ni et al. *A Survey on LLM-based Conversational User Simulation.* arXiv:2604.24977, 2026.
- Peng, Liu, Xu, Yang, Shao, Wang. *Review-LLM: Harnessing Large Language Models for Personalized Review Generation.* arXiv:2407.07487, 2024.
- Pradeep, Sharifymoghaddam, Lin. *RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.* arXiv:2309.15088, 2023.
- Ren, Qiu, Qu, Liu, Zhao, Wu, Wen, Wang. *BASES: Large-scale Web Search User Simulation with LLM-based Agents.* arXiv:2402.17505, 2024.
- Ren, Wang, Zhou, Zhao, Wang, Liu, Wen, Chua. *Self-Calibrated Listwise Reranking with Large Language Models.* arXiv:2411.04602, 2024.
- Ryu, Yanaka. *Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews.* arXiv:2510.00449, 2025.
