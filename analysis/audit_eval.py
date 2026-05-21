"""
Pre-flight audit for the NaijaBuddy full eval.
Checks: empty data, rating bias, review/name quality, 3-core integrity,
duplicates, leave-one-out coverage, and harness leakage assumptions.
"""
import csv, os, collections, math, statistics, random

csv.field_size_limit(16 * 1024 * 1024)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOMAINS = {"yelp": "Yelp", "goodreads": "Goodreads", "amazon": "Amazon"}

def blank(v):
    return v is None or str(v).strip() == ""

def line(c="-"):
    print(c * 72)

for stem, domain in DOMAINS.items():
    path = os.path.join(DATA_DIR, f"{stem}_dense.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())

    line("=")
    print(f"DOMAIN: {domain}   rows={len(rows)}   columns={cols}")
    line("=")

    # ---- empty cells per column ----
    print("[EMPTY CELLS per column]")
    for c in cols:
        n_blank = sum(1 for r in rows if blank(r.get(c)))
        flag = "  <-- !!" if n_blank else ""
        print(f"  {c:14s}: {n_blank:7d} blank ({100*n_blank/len(rows):5.1f}%){flag}")

    # ---- rating distribution / bias ----
    ratings = []
    bad_rating = 0
    for r in rows:
        try:
            v = float(r["rating"])
            ratings.append(v)
        except Exception:
            bad_rating += 1
    dist = collections.Counter(round(x) for x in ratings)
    print(f"\n[RATING]  parsed={len(ratings)}  unparseable={bad_rating}  "
          f"min={min(ratings)}  max={max(ratings)}  mean={statistics.mean(ratings):.3f}")
    out_of_range = sum(1 for x in ratings if x < 1 or x > 5)
    le_zero = sum(1 for x in ratings if x <= 0)
    print(f"          out_of_[1,5]={out_of_range}   <=0={le_zero}")
    for star in sorted(dist):
        bar = "#" * int(60 * dist[star] / len(ratings))
        print(f"   {star}* {dist[star]:7d} ({100*dist[star]/len(ratings):5.1f}%) {bar}")
    top_share = 100 * max(dist.values()) / len(ratings)
    print(f"          -> single most common star = {top_share:.1f}% of all ratings")

    # ---- review text quality ----
    rev_lens = [len((r.get("review_text") or "").strip()) for r in rows]
    empty_rev = sum(1 for L in rev_lens if L == 0)
    short_rev = sum(1 for L in rev_lens if 0 < L < 15)
    nonempty = [L for L in rev_lens if L > 0]
    print(f"\n[REVIEW TEXT]  empty={empty_rev} ({100*empty_rev/len(rows):.1f}%)  "
          f"very_short(<15ch)={short_rev}")
    if nonempty:
        print(f"               len: min={min(nonempty)} median={int(statistics.median(nonempty))} "
              f"mean={int(statistics.mean(nonempty))} max={max(nonempty)}")

    # ---- per-item / per-user interaction counts (3-core check) ----
    by_user = collections.Counter(r["user_id"] for r in rows)
    by_item = collections.Counter(r["item_id"] for r in rows)
    u_min, i_min = min(by_user.values()), min(by_item.values())
    print(f"\n[3-CORE]  users={len(by_user)}  items={len(by_item)}")
    print(f"          interactions/user: min={u_min} median={int(statistics.median(by_user.values()))} max={max(by_user.values())}")
    print(f"          interactions/item: min={i_min} median={int(statistics.median(by_item.values()))} max={max(by_item.values())}")
    print(f"          3-core holds: users>=3 {'OK' if u_min>=3 else 'VIOLATED'}  "
          f"items>=3 {'OK' if i_min>=3 else 'VIOLATED'}")

    # ---- duplicate (user,item) pairs ----
    pair_counts = collections.Counter((r["user_id"], r["item_id"]) for r in rows)
    dups = sum(c - 1 for c in pair_counts.values() if c > 1)
    print(f"\n[DUPLICATES]  duplicate (user,item) rows = {dups}")

    # ---- item name quality (placeholder detection) ----
    item_name = {}
    for r in rows:
        iid = r["item_id"]
        nm = (r.get("item_name") or "").strip()
        if iid not in item_name or (not item_name[iid] and nm):
            item_name[iid] = nm
    blank_names = sum(1 for nm in item_name.values() if not nm)
    uniq_names = len(set(nm for nm in item_name.values() if nm))
    print(f"\n[ITEM NAMES]  unique items={len(item_name)}  blank name={blank_names} "
          f"({100*blank_names/len(item_name):.1f}%)  distinct non-blank names={uniq_names}")
    name_freq = collections.Counter(nm for nm in item_name.values() if nm)
    print("   most repeated item names:")
    for nm, c in name_freq.most_common(5):
        print(f"     {c:5d}x  {nm[:60]!r}")

    # ---- leave-one-out coverage check ----
    # after removing 1 interaction/user, does the held-out item still appear in train?
    rng = random.Random(42)
    by_user_rows = collections.defaultdict(list)
    for r in rows:
        by_user_rows[r["user_id"]].append(r)
    train_item_ids = set()
    held = []
    for u, urows in by_user_rows.items():
        if len(urows) < 3:
            train_item_ids.update(r["item_id"] for r in urows)
            continue
        idx = rng.randrange(len(urows))
        held.append(urows[idx])
        train_item_ids.update(urows[i]["item_id"] for i in range(len(urows)) if i != idx)
    orphan = sum(1 for h in held if h["item_id"] not in train_item_ids)
    print(f"\n[LEAVE-ONE-OUT]  held-out pairs={len(held)}  "
          f"held-out item NOT in train (test user skipped)={orphan}")
    empty_held_review = sum(1 for h in held if blank(h.get("review_text")))
    print(f"                 held-out rows with EMPTY review (ROUGE=0 forced)={empty_held_review} "
          f"({100*empty_held_review/len(held):.1f}%)")
    print()
