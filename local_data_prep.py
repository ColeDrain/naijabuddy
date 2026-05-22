"""
Local dense-dataset preparation — the all-local equivalent of
modal_data_prep.py.

Streams the Yelp / Goodreads / Amazon review datasets via the HuggingFace
`datasets` library (authenticated automatically from this machine's
`hf auth login` token), extracts a dense bipartite 3-core per domain, caps
each to its LIMIT_USERS densest users, and overwrites data/*_dense.csv.

    .venv/bin/python local_data_prep.py

Runs entirely on this machine — no Modal, no credential passing. Used after
Modal's `hf-token` secret was found to be stale; the local token is valid.
"""
import gc
import os
import pandas as pd
from datasets import load_dataset

LIMIT_USERS = 2000
RAW_LIMIT = 1_500_000
AMZ_META_LIMIT = 1_500_000
COLS = ["user_id", "item_id", "item_name", "category", "rating", "review_text"]


def stream_rows(repo, cols, limit, label):
    """Stream up to `limit` rows of `repo`, keeping only `cols`."""
    ds = None
    for split in ("train", "full"):
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            break
        except Exception:
            continue
    if ds is None:  # surface the real error
        ds = load_dataset(repo, split="train", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if i >= limit:
            break
        rows.append({c: r.get(c) for c in cols})
        if i and i % 250_000 == 0:
            print(f"  [{label}] {i:,} rows...", flush=True)
    print(f"  [{label}] streamed {len(rows):,} rows", flush=True)
    return pd.DataFrame(rows)


def true_kcore(df, k):
    while len(df):
        uc = df["user_id"].value_counts()
        ic = df["item_id"].value_counts()
        keep = df[df["user_id"].map(uc).ge(k) & df["item_id"].map(ic).ge(k)]
        if len(keep) == len(df):
            break
        df = keep
    return df


def densify(df, k=3, label=""):
    if df.empty:
        print(f"  [{label}] empty input", flush=True)
        return df
    core, eff_k = df.iloc[0:0], 0
    for kk in range(k, 0, -1):
        c = true_kcore(df, kk)
        if len(c):
            core, eff_k = c, kk
            break
    if LIMIT_USERS and len(core):
        top = core["user_id"].value_counts().nlargest(LIMIT_USERS).index
        core = core[core["user_id"].isin(top)]
        for kk in range(eff_k, 0, -1):
            c = true_kcore(core, kk)
            if len(c):
                core, eff_k = c, kk
                break
    print(f"  [{label}] final {eff_k}-core: {len(core):,} reviews, "
          f"{core['user_id'].nunique():,} users, {core['item_id'].nunique():,} items",
          flush=True)
    return core


def finalize(df, label):
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"] > 0]
    df = df.dropna(subset=["user_id", "item_id"])
    return densify(df[COLS], label=label)


def main():
    os.makedirs("data", exist_ok=True)

    # ---- Yelp: reviews JOIN business ----
    print("=== Yelp ===", flush=True)
    yr = stream_rows("yashraizada/yelp-open-dataset-reviews",
                     ["user_id", "business_id", "stars", "text"], RAW_LIMIT, "Yelp")
    yb = stream_rows("yashraizada/yelp-open-dataset-business",
                     ["business_id", "name", "categories"], 10_000_000, "Yelp-biz")
    yelp = yr.merge(yb, on="business_id", how="inner").rename(columns={
        "business_id": "item_id", "name": "item_name",
        "categories": "category", "stars": "rating", "text": "review_text"})
    yelp = finalize(yelp, "Yelp")
    del yr, yb
    gc.collect()

    # ---- Goodreads: reviews JOIN book metadata ----
    print("=== Goodreads ===", flush=True)
    gr = stream_rows("vngclinh/goodreads-reviews",
                     ["user_id", "book_id", "rating", "review_text"], RAW_LIMIT, "Goodreads")
    gm = stream_rows("Eitanli/goodreads", ["URL", "Book"], 10_000_000, "Goodreads-meta")
    gm["book_id"] = gm["URL"].astype(str).str.extract(r"/book/show/(\d+)", expand=False)
    gm = gm.dropna(subset=["book_id"]).drop_duplicates("book_id")
    gr["book_id"] = gr["book_id"].astype(str)
    goodreads = gr.merge(gm[["book_id", "Book"]], on="book_id", how="left")
    goodreads["item_name"] = goodreads["Book"].fillna("Book #" + goodreads["book_id"])
    goodreads["category"] = "Goodreads (Book)"
    goodreads = goodreads.rename(columns={"book_id": "item_id"})
    goodreads = finalize(goodreads, "Goodreads")
    del gr, gm
    gc.collect()

    # ---- Amazon: reviews + title lookup ----
    print("=== Amazon (Books) ===", flush=True)
    ar = stream_rows("cogsci13/Amazon-Reviews-2023-Books-Review",
                     ["user_id", "parent_asin", "rating", "text"], RAW_LIMIT, "Amazon")
    am = stream_rows("cogsci13/Amazon-Reviews-2023-Books-Meta",
                     ["parent_asin", "title"], AMZ_META_LIMIT, "Amazon-meta")
    title = dict(zip(am["parent_asin"].astype(str), am["title"]))
    ar["parent_asin"] = ar["parent_asin"].astype(str)
    ar["item_name"] = ar["parent_asin"].map(lambda a: title.get(a) or f"Amazon Book {a}")
    ar["category"] = "Amazon (Book)"
    amazon = ar.rename(columns={"parent_asin": "item_id", "text": "review_text"})
    amazon = finalize(amazon, "Amazon")
    del ar, am
    gc.collect()

    # ---- write (all three at the end, so a mid-run failure leaves data/ intact) ----
    for name, df in [("yelp", yelp), ("goodreads", goodreads), ("amazon", amazon)]:
        out = df[COLS].copy()
        out["review_text"] = out["review_text"].fillna("")
        out["item_name"] = out["item_name"].fillna("")
        path = f"data/{name}_dense.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path} -> {len(out):,} rows", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
