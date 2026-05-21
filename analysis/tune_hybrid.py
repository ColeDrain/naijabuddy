"""
Sweep the hybrid retrieval weight (dense vs CF) per domain.

No LLM: synth personas are pulled from the eval artifact cache (populated by the
comprehensive run), so this reproduces the headline retrieval setup exactly.
Current production weight is w_dense = 0.3 (i.e. 0.3*dense + 0.7*CF).
"""
import os, sys, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf_home"))
import numpy as np
import eval_harness as E
from data_enricher import generate_user_persona
E._persona_fn = generate_user_persona
from sentence_transformers import SentenceTransformer

SEED = 42
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]   # w = dense weight
cache = E.load_cache(enabled=True)
emb = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_unit(texts):
    m = np.asarray(emb.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def score(mat, train_sets, gold, n_users):
    hits = ndcg = 0.0
    for u in range(n_users):
        sc = mat[u].copy()
        for ti in train_sets[u]:
            sc[ti] = -1e9
        topk = np.argpartition(-sc, 10)[:10]
        topk = topk[np.argsort(-sc[topk])]
        ranked = list(topk)
        g = gold[u]
        if g in ranked:
            hits += 1
            ndcg += 1.0 / math.log2(ranked.index(g) + 2)
    return hits / n_users, ndcg / n_users


summary = {}
for stem, domain in E.DOMAINS.items():
    rows = E.load_domain(stem)
    train, test, tbu = E.leave_one_out(rows, SEED)
    info = E.build_item_info(train, domain)
    item_ids = list(info.keys())
    idx_of = {iid: i for i, iid in enumerate(item_ids)}
    M = embed_unit([info[i]["clean_text"] for i in item_ids])

    test_users, queries, train_sets, gold, miss = [], [], [], [], 0
    for held in test:
        u = held["user"]
        if held["item"] not in idx_of:
            continue
        p = E.synthesize_persona(None, domain, tbu[u], info, cache, None)
        if p == f"A real {domain} user.":
            miss += 1
        test_users.append(u)
        queries.append(p)
        train_sets.append({idx_of[r["item"]] for r in tbu[u] if r["item"] in idx_of})
        gold.append(idx_of[held["item"]])
    n = len(test_users)
    Q = embed_unit(queries)
    sims = Q @ M.T

    all_users = list(tbu.keys())
    uidx = {u: i for i, u in enumerate(all_users)}
    R = np.zeros((len(all_users), len(item_ids)), dtype=np.float32)
    for u, rs in tbu.items():
        for r in rs:
            j = idx_of.get(r["item"])
            if j is not None:
                R[uidx[u], j] = r["rating"]
    Rn = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)
    Xtest = np.zeros((n, len(item_ids)), dtype=np.float32)
    for ui, ts in enumerate(train_sets):
        for ti in ts:
            Xtest[ui, ti] = 1.0
    cf = (Xtest @ Rn.T) @ Rn

    dnorm = np.zeros_like(sims)
    cnorm = np.zeros_like(cf)
    for u in range(n):
        d, c = sims[u], cf[u]
        dnorm[u] = (d - d.min()) / (d.max() - d.min() + 1e-9) if d.max() > d.min() else d
        cnorm[u] = (c - c.min()) / (c.max() - c.min() + 1e-9) if c.max() > c.min() else c

    print(f"\n=== {domain} ===  n_users={n}  items={len(item_ids)}  cache_misses={miss}")
    print(f"  {'w_dense':>8} | {'HitRate@10':>11} | {'NDCG@10':>9}")
    res = {}
    for w in WEIGHTS:
        hr, nd = score(w * dnorm + (1 - w) * cnorm, train_sets, gold, n)
        res[w] = (hr, nd)
        mark = "  <- current (0.3)" if w == 0.3 else ""
        print(f"  {w:>8} | {hr:>11.4f} | {nd:>9.4f}{mark}")
    best_hr = max(res, key=lambda w: res[w][0])
    best_nd = max(res, key=lambda w: res[w][1])
    print(f"  best HitRate@10 at w={best_hr} ({res[best_hr][0]:.4f}); "
          f"best NDCG@10 at w={best_nd} ({res[best_nd][1]:.4f})")
    summary[domain] = res

print("\n" + "=" * 60)
print("SUMMARY — HitRate@10 by dense weight")
print(f"  {'domain':>10} |" + "".join(f"{w:>7}" for w in WEIGHTS))
for domain, res in summary.items():
    print(f"  {domain:>10} |" + "".join(f"{res[w][0]:>7.3f}" for w in WEIGHTS))
