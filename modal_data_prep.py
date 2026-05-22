"""
Dense-dataset preparation on Modal.

Streams the Yelp / Goodreads / Amazon review datasets with the HuggingFace
`datasets` library (which authenticates automatically from the HF_TOKEN env
var supplied by the Modal `hf-token` secret), extracts a dense bipartite
k-core per domain, caps each to its `limit_users` densest users, and writes
data/*_dense.csv locally.

    modal run modal_data_prep.py --limit-users 2000
    modal run modal_data_prep.py --limit-users 200 --raw-limit 80000   # validation

This replaces the original DuckDB-over-HTTP approach: HF now requires auth to
download dataset parquet, and DuckDB's httpfs would not apply the token.
"""
import modal

app = modal.App("naijabuddy-data-prep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "pandas", "pyarrow", "huggingface-hub"
)


@app.function(image=image, timeout=7200, cpu=4.0, memory=32768,
              secrets=[modal.Secret.from_name("hf-token")])
def prepare_data(limit_users: int = 2000, raw_limit: int = 2_000_000):
    """Stream, densify and pack the three dense CSVs; return a zip of them."""
    import os, io, zipfile
    import pandas as pd
    from datasets import load_dataset

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not in env (Modal hf-token secret missing)"
    amz_meta_limit = min(5_000_000, max(raw_limit * 3, 300_000))

    def stream_rows(repo, cols, limit, label):
        """Stream up to `limit` rows of `repo`, keeping only `cols`."""
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
        if limit_users and len(core):
            top = core["user_id"].value_counts().nlargest(limit_users).index
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

    COLS = ["user_id", "item_id", "item_name", "category", "rating", "review_text"]

    def finalize(df, label):
        df = df.copy()
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"] > 0]
        df = df.dropna(subset=["user_id", "item_id"])
        return densify(df[COLS], label=label)

    # ---- Yelp: reviews JOIN business ----
    print("=== Yelp ===", flush=True)
    yr = stream_rows("yashraizada/yelp-open-dataset-reviews",
                     ["user_id", "business_id", "stars", "text"], raw_limit, "Yelp")
    yb = stream_rows("yashraizada/yelp-open-dataset-business",
                     ["business_id", "name", "categories"], 10_000_000, "Yelp-biz")
    yelp = yr.merge(yb, on="business_id", how="inner").rename(columns={
        "business_id": "item_id", "name": "item_name",
        "categories": "category", "stars": "rating", "text": "review_text"})
    yelp = finalize(yelp, "Yelp")

    # ---- Goodreads: reviews JOIN book metadata ----
    print("=== Goodreads ===", flush=True)
    gr = stream_rows("vngclinh/goodreads-reviews",
                     ["user_id", "book_id", "rating", "review_text"], raw_limit, "Goodreads")
    gm = stream_rows("Eitanli/goodreads", ["URL", "Book"], 10_000_000, "Goodreads-meta")
    gm["book_id"] = gm["URL"].astype(str).str.extract(r"/book/show/(\d+)")
    gm = gm.dropna(subset=["book_id"]).drop_duplicates("book_id")
    gr["book_id"] = gr["book_id"].astype(str)
    goodreads = gr.merge(gm[["book_id", "Book"]], on="book_id", how="left")
    goodreads["item_name"] = goodreads["Book"].fillna("Book #" + goodreads["book_id"])
    goodreads["category"] = "Goodreads (Book)"
    goodreads = goodreads.rename(columns={"book_id": "item_id"})
    goodreads = finalize(goodreads, "Goodreads")

    # ---- Amazon: reviews + title lookup ----
    print("=== Amazon (Books) ===", flush=True)
    ar = stream_rows("cogsci13/Amazon-Reviews-2023-Books-Review",
                     ["user_id", "parent_asin", "rating", "text"], raw_limit, "Amazon")
    am = stream_rows("cogsci13/Amazon-Reviews-2023-Books-Meta",
                     ["parent_asin", "title"], amz_meta_limit, "Amazon-meta")
    title = dict(zip(am["parent_asin"].astype(str), am["title"]))
    ar["parent_asin"] = ar["parent_asin"].astype(str)
    ar["item_name"] = ar["parent_asin"].map(lambda a: title.get(a) or f"Amazon Book {a}")
    ar["category"] = "Amazon (Book)"
    amazon = ar.rename(columns={"parent_asin": "item_id", "text": "review_text"})
    amazon = finalize(amazon, "Amazon")

    # ---- pack ----
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in [("yelp", yelp), ("goodreads", goodreads), ("amazon", amazon)]:
            out = df[COLS].copy()
            out["review_text"] = out["review_text"].fillna("")
            out["item_name"] = out["item_name"].fillna("")
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            z.writestr(f"data/{name}_dense.csv", csv_bytes)
            print(f"data/{name}_dense.csv -> {len(out):,} rows, "
                  f"{len(csv_bytes) / 1024:.0f} KB", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main(limit_users: int = 2000, raw_limit: int = 2_000_000):
    import io, os, zipfile

    print(f"Running dense-dataset prep on Modal "
          f"(limit_users={limit_users}, raw_limit={raw_limit:,}) ...")
    zip_bytes = prepare_data.remote(limit_users, raw_limit)

    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(".")
        names = z.namelist()
    print("\n" + "=" * 60)
    print("Dense CSVs written locally:")
    for n in names:
        print(f"  {n}  ({os.path.getsize(n) / 1024:.0f} KB)")
    print("=" * 60)
