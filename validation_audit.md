# NaijaBuddy — Quality-Validation Audit

An adversarial self-audit of the evaluation harness, the leave-one-out leakage
protocol, the calibration math, and the reported claims. Run *before* the
n = 2,000 re-evaluation so that any harness fix lands before the final numbers.

**Verdict — the evaluation methodology is sound and genuinely leakage-free.**
No bugs found. One framing recommendation (oracle-descriptive blend weights)
and one commit-message inaccuracy are recorded below.

---

## 1. Leakage protocol — CLEAN

`leave_one_out` (`eval_harness.py:179`): for every user exactly one interaction
is held out; `train_by_user[u]` holds that user's *remaining* interactions.
Every prediction input was traced and confirmed training-only:

| Input | Source | Line | Train-only? |
|---|---|---|---|
| V0 global mean | `mean(train_rows)` | 820 | ✅ |
| V1 user mean | `{u: mean(rs) for u,rs in train_by_user.items()}` | 821–822 | ✅ |
| Item descriptions | `build_item_info(train_rows, …)` | 224 | ✅ |
| Personas | `build_persona(train_for_user, …)` | 266 | ✅ |
| Retrieval candidates | user's own training items excluded | 492 | ✅ |

The held-out **rating** and **review** are used *only* as prediction targets
(l.865–868), never as inputs. The highest-risk spot — a held-out rating leaking
into its own user-mean and inflating V1 — was checked explicitly: `user_mean`
is built from `train_by_user`, so it does not. **No leakage path found.**

## 2. Oracle-descriptive blend weights — LOW severity (framing)

`V2` is reported at the **test-minimising** point of the α-sweep
(`best_alpha = min(sweep, …)`, l.882); `V3`'s β/γ weights
(`analysis/measure_calib3.py`) are likewise swept to minimise test RMSE. These
are therefore *oracle* numbers — the blend weight is selected on the same split
it is scored on.

Mitigating facts:
- The harness already documents this (l.875–877: *"best_alpha … is the
  test-set-minimising point of the sweep — report the full alpha_sweep as the
  descriptive result"*) and stores the full `alpha_sweep`.
- The paper labels the column **"(best α)"** and characterises the V1→V2 gain
  as small / "indistinguishable from noise" on Amazon.
- The **deployed** system uses no tuned α — it uses the fixed 3-term
  `STAT_SPLIT` anchor in `agent.py`, whose constants match the swept optimum to
  ~2 decimals.

**Recommendation:** in the n = 2,000 §4.2 rewrite, add one explicit sentence —
the blend weights are the *descriptive minimum* of the sweep (an oracle upper
bound), not validation-tuned. This **strengthens** the thesis: even with
oracle-optimal blending the LLM weight lands at ≈ 0, so "the LLM's numeric
rating is redundant for warm users" is a *conservative* claim. Severity is low
because the conclusion holds either way.

## 3. Calibration math — consistent

The V2 blend `α·LLM + (1−α)·μ_user` (harness l.880) matches
`agent.get_calibrated_rating`'s warm-user path; the 3-term V3 matches
`STAT_SPLIT`. `ALPHA_GRID` is 0.0–1.0 in 0.1 steps (l.71). `rmse`/`mae`
(l.210/215) are textbook.

## 4. Metrics — correct

RMSE/MAE correct. ROUGE-L (l.101) and the BGE semantic similarity are reported
side by side; §4.3 already carries the honest caveat that two reviews of the
same item share a similarity floor, so the absolute semantic score is
"encouraging rather than decisive."

## 5. Commit-message inaccuracy (for the record)

Commit `2c2df8f` (dedup catalogue items by id, not name) is a genuine
correctness fix — collapsing distinct chain businesses that share a name is
wrong. However its message claims a "~5×" data-loss impact ("26.8K → 4.7K"),
derived from a `wc -l` line count; the Yelp CSV's true record count is 4,748
(review text contains embedded newlines). The fix is sound; only the stated
*magnitude* is wrong. No figure in the paper depends on it.

---

## Verdict

The leakage-free leave-one-out evaluation is correctly implemented. The headline
conclusions — per-user/item calibration as the warm-user anchor; the LLM's
numeric rating redundant warm and decisive cold (the regime switch) — rest on no
unsound step. The single action item is the §4.2 oracle-weights caveat, folded
into the n = 2,000 rewrite.
