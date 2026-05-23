"""
Build the Amazon Books<->Movies cross-domain transfer dataset from the
5-core rating-only files (already downloaded under
`data/amazon_crossdomain/benchmark/5core/rating_only/`).

Output: `data/amazon_crossdomain/xdomain_2k.jsonl` — one JSON record per user,
each containing the user's full Books interaction history, their Movies
training history (= all-but-last Movies item, sorted by timestamp), and the
held-out last Movies item plus its true rating. This is the dataset the
cross-domain transfer eval consumes: build the user model from Books,
predict the held-out Movies rating.

    .venv/bin/python cross_domain_dataset.py
"""
import json
import os
import pandas as pd

BASE = "data/amazon_crossdomain/benchmark/5core/rating_only"
OUT_DIR = "data/amazon_crossdomain"
OUT_FILE = os.path.join(OUT_DIR, "xdomain_2k.jsonl")
N_USERS = 2000          # densest shared users
MIN_PER_DOMAIN = 10     # require >=10 interactions in BOTH categories


def main():
    print(f"loading Books.csv ...", flush=True)
    b = pd.read_csv(f"{BASE}/Books.csv")
    print(f"  {len(b):,} interactions, {b['user_id'].nunique():,} users", flush=True)

    print(f"loading Movies_and_TV.csv ...", flush=True)
    m = pd.read_csv(f"{BASE}/Movies_and_TV.csv")
    print(f"  {len(m):,} interactions, {m['user_id'].nunique():,} users", flush=True)

    # 1. Restrict both to the shared-user intersection.
    shared = set(b["user_id"].unique()) & set(m["user_id"].unique())
    print(f"shared users (>=5 in each, both are 5-core): {len(shared):,}", flush=True)
    b = b[b["user_id"].isin(shared)]
    m = m[m["user_id"].isin(shared)]

    # 2. Per-user counts; keep users with >= MIN_PER_DOMAIN in EACH category.
    bc = b.groupby("user_id").size()
    mc = m.groupby("user_id").size()
    eligible = bc.index[(bc >= MIN_PER_DOMAIN) & (mc.reindex(bc.index, fill_value=0) >= MIN_PER_DOMAIN)]
    print(f"eligible (>= {MIN_PER_DOMAIN} books AND >= {MIN_PER_DOMAIN} movies): "
          f"{len(eligible):,}", flush=True)

    # 3. Rank by combined activity, take the densest N_USERS.
    combined = (bc.loc[eligible] + mc.loc[eligible]).sort_values(ascending=False)
    top = combined.head(N_USERS).index.tolist()
    print(f"picking top {N_USERS:,} by combined activity. Median activity: "
          f"books={bc.loc[top].median():.0f}, movies={mc.loc[top].median():.0f}, "
          f"combined={combined.loc[top].median():.0f}", flush=True)

    # 4. Slice and sort the two frames by (user, timestamp) for clean per-user iteration.
    b = b[b["user_id"].isin(top)].sort_values(["user_id", "timestamp"])
    m = m[m["user_id"].isin(top)].sort_values(["user_id", "timestamp"])
    print(f"  sliced: {len(b):,} books rows, {len(m):,} movies rows", flush=True)

    # 5. Build the JSONL.
    os.makedirs(OUT_DIR, exist_ok=True)
    b_by_u = {u: grp for u, grp in b.groupby("user_id")}
    m_by_u = {u: grp for u, grp in m.groupby("user_id")}
    written = 0
    with open(OUT_FILE, "w") as f:
        for u in top:
            bm = b_by_u[u]
            mm = m_by_u[u]
            held = mm.iloc[-1]
            train_movies = mm.iloc[:-1]
            rec = {
                "user_id": u,
                "n_books": int(len(bm)),
                "n_movies_train": int(len(train_movies)),
                "books_history": [
                    {"item": r["parent_asin"], "rating": float(r["rating"]),
                     "timestamp": int(r["timestamp"])}
                    for _, r in bm.iterrows()
                ],
                "movies_train": [
                    {"item": r["parent_asin"], "rating": float(r["rating"]),
                     "timestamp": int(r["timestamp"])}
                    for _, r in train_movies.iterrows()
                ],
                "movies_holdout_item": held["parent_asin"],
                "movies_holdout_rating": float(held["rating"]),
                "movies_holdout_timestamp": int(held["timestamp"]),
            }
            f.write(json.dumps(rec) + "\n")
            written += 1
    print(f"\nwrote {written:,} cross-domain users -> {OUT_FILE}", flush=True)
    print(f"file size: {os.path.getsize(OUT_FILE) / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
