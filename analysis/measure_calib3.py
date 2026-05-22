"""
Tier 1a measurement: does a 3-term calibration

    y = a*LLM + b*user_mean + c*item_mean      (a + b + c = 1)

beat the deployed 2-term blend  y = a*LLM + (1-a)*user_mean ?

Uses the cached LLM raw ratings (eval_artifacts/llm_cache.jsonl) from the
warm run — zero new GPU. Sweeps the (a,b,c) simplex and reports per domain.
A seed with too few cached ratings is skipped: the n=2,000 cache holds only
seed-42 generations, so seeds 1/7 fall out and the run reports seed 42 alone.
"""
import os, sys, math, statistics, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf_home"))
import eval_harness as E
from data_enricher import generate_user_persona
E._persona_fn = generate_user_persona

SEEDS = [42, 1, 7]
GRID = [round(i / 20, 2) for i in range(21)]   # 0.00 .. 1.00 step 0.05


def rmse(p, a):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, a)) / len(a))


# cache lookup: (domain, user, item, seed) -> raw LLM rating
cache = E.load_cache(enabled=True)
raw_lookup = {}
for rec in cache.values():
    if (rec.get("kind") == "review" and rec.get("stage") == "warm"
            and rec.get("raw_rating") is not None):
        raw_lookup[(rec.get("domain"), rec.get("user"),
                    rec.get("item"), rec.get("seed"))] = rec["raw_rating"]
print(f"cache: {len(raw_lookup)} warm raw ratings\n")

agg = collections.defaultdict(list)
for seed in SEEDS:
    for stem, domain in E.DOMAINS.items():
        rows = E.load_domain(stem)
        train, test, tbu = E.leave_one_out(rows, seed)
        gm = sum(r["rating"] for r in train) / len(train)
        um = {u: sum(x["rating"] for x in rs) / len(rs) for u, rs in tbu.items()}
        irow = collections.defaultdict(list)
        for r in train:
            irow[r["item"]].append(r["rating"])
        im = {i: sum(v) / len(v) for i, v in irow.items()}

        raws, us, isx, act = [], [], [], []
        for h in test:
            k = (domain, h["user"], h["item"], seed)
            if k not in raw_lookup:
                continue
            raws.append(raw_lookup[k])
            us.append(um[h["user"]])
            isx.append(im.get(h["item"], gm))
            act.append(h["rating"])

        if len(raws) < 100:
            print(f"  seed {seed:>2} {domain:<10}: only {len(raws)} cached "
                  f"ratings — skipped (need >= 100 for a valid sweep)")
            continue

        v2 = min(rmse([a * r + (1 - a) * u for r, u in zip(raws, us)], act)
                 for a in GRID)
        best = None
        for a in GRID:
            for b in GRID:
                c = round(1 - a - b, 2)
                if c < -1e-9:
                    continue
                preds = [a * r + b * u + c * i for r, u, i in zip(raws, us, isx)]
                rr = rmse(preds, act)
                if best is None or rr < best[0]:
                    best = (rr, a, b, c)
        agg[domain].append((v2, best))
        print(f"  seed {seed:>2} {domain:<10}: V2={v2:.4f}  "
              f"V3={best[0]:.4f}  (a={best[1]}, b={best[2]}, c={best[3]})")

print("\n=== averaged over valid seeds (>= 100 cached ratings) ===")
for domain in E.DOMAINS.values():
    r = agg[domain]
    if not r:
        print(f"  {domain:<10}: no valid seed — skipped")
        continue
    v2m = statistics.mean(x[0] for x in r)
    v3m = statistics.mean(x[1][0] for x in r)
    am = statistics.mean(x[1][1] for x in r)
    bm = statistics.mean(x[1][2] for x in r)
    cm = statistics.mean(x[1][3] for x in r)
    print(f"  {domain:<10}: V2={v2m:.4f}  V3={v3m:.4f}  "
          f"delta={v2m - v3m:+.4f} ({100*(v2m-v3m)/v2m:+.1f}%)  "
          f"| mean weights a={am:.2f} b={bm:.2f} c={cm:.2f}")
