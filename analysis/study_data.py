"""
Researcher probe: is the rating signal in the data even modellable, and are we
leaving an item-side signal on the table?

For each domain (leave-one-out, seed 42):
  * V0 global-mean / V1 user-mean / VI item-mean RMSE
  * best user-mean + item-mean blend  -> does an item term help?
  * per-user rating std distribution  -> how much within-user signal exists
  * V1 RMSE bucketed by user rating variance -> where is rating prediction
    actually hard (= where a model could earn its place)
"""
import os, sys, math, statistics, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf_home"))
import numpy as np
import eval_harness as E
from data_enricher import generate_user_persona
E._persona_fn = generate_user_persona


def rmse(p, a):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, a)) / len(a))


for stem, domain in E.DOMAINS.items():
    rows = E.load_domain(stem)
    train, test, tbu = E.leave_one_out(rows, 42)

    gm = sum(r["rating"] for r in train) / len(train)
    um = {u: sum(x["rating"] for x in rs) / len(rs) for u, rs in tbu.items()}
    item_r = collections.defaultdict(list)
    for r in train:
        item_r[r["item"]].append(r["rating"])
    im = {i: sum(v) / len(v) for i, v in item_r.items()}

    actual = [h["rating"] for h in test]
    pred_g = [gm] * len(test)
    pred_u = [um[h["user"]] for h in test]
    pred_i = [im.get(h["item"], gm) for h in test]

    print(f"\n=== {domain} ===  held-out n={len(test)}")
    print(f"  V0 global-mean : RMSE={rmse(pred_g, actual):.4f}")
    print(f"  V1 user-mean   : RMSE={rmse(pred_u, actual):.4f}")
    print(f"  VI item-mean   : RMSE={rmse(pred_i, actual):.4f}")

    best = None
    for wi in range(11):
        w = wi / 10
        pb = [w * u + (1 - w) * i for u, i in zip(pred_u, pred_i)]
        rr = rmse(pb, actual)
        if best is None or rr < best[1]:
            best = (w, rr)
    print(f"  best user/item blend: w_user={best[0]}  RMSE={best[1]:.4f}  "
          f"(vs V1 {rmse(pred_u, actual):.4f})")

    ustd = [statistics.pstdev([x["rating"] for x in rs])
            for rs in tbu.values() if len(rs) >= 2]
    a = np.array(ustd)
    print(f"  per-user rating std: mean={a.mean():.3f}  median={np.median(a):.3f}")
    print(f"    std<0.4: {(a < 0.4).mean()*100:.0f}%   "
          f"0.4-0.8: {((a >= 0.4) & (a < 0.8)).mean()*100:.0f}%   "
          f"0.8-1.2: {((a >= 0.8) & (a < 1.2)).mean()*100:.0f}%   "
          f">=1.2: {(a >= 1.2).mean()*100:.0f}%")

    buckets = collections.defaultdict(lambda: [[], []])
    for h in test:
        u = h["user"]
        s = statistics.pstdev([x["rating"] for x in tbu[u]]) if len(tbu[u]) >= 2 else 0.0
        b = "low  (std<0.6)" if s < 0.6 else ("mid  (0.6-1.0)" if s < 1.0 else "high (std>=1.0)")
        buckets[b][0].append(um[u])
        buckets[b][1].append(h["rating"])
    for b in ["low  (std<0.6)", "mid  (0.6-1.0)", "high (std>=1.0)"]:
        if buckets[b][0]:
            preds, acts = buckets[b]
            print(f"    V1 RMSE, {b}: {rmse(preds, acts):.3f}  "
                  f"({len(preds)} users, {100*len(preds)/len(test):.0f}% of held-out)")
