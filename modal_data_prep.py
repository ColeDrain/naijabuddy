"""
Dense-dataset preparation on Modal — the cloud equivalent of
colab_data_prep.ipynb. Streams Yelp / Goodreads / Amazon reviews from the
HuggingFace parquet mirrors, extracts a dense bipartite k-core per domain,
caps each to its `limit_users` densest users, and writes the three
data/*_dense.csv files locally.

    modal run modal_data_prep.py --limit-users 2000

This removes the manual Colab step: it runs entirely on Modal (a cheap
CPU job — no GPU) and drops the new CSVs straight into ./data/.
"""
import modal

app = modal.App("naijabuddy-data-prep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "duckdb==1.1.3", "pandas", "requests", "pyarrow"
)


@app.function(image=image, timeout=2400)
def prepare_data(limit_users: int = 2000):
    """Stream, densify and pack the three dense CSVs; return a zip of them."""
    import duckdb, time, io, zipfile
    import requests
    import pandas as pd

    STREAM_RETRIES = 5
    STREAM_RETRY_WAIT_S = 4

    def hf_parquet_urls(dataset):
        r = requests.get(
            f"https://datasets-server.huggingface.co/parquet?dataset={dataset}",
            timeout=60,
        )
        r.raise_for_status()
        return [f["url"] for f in r.json()["parquet_files"]]

    def stream_query(query, label):
        # Retry the whole query — a corrupted compressed chunk surfaces as a
        # decompression error after the HTTP read, which http_retries can't catch.
        last_err = None
        for attempt in range(1, STREAM_RETRIES + 1):
            con = duckdb.connect()
            try:
                con.execute("SET http_retries=3;")
                con.execute("SET http_timeout=120000;")
                df = con.execute(query).fetchdf()
                con.close()
                print(f"  [{label}] streamed {len(df)} rows", flush=True)
                return df
            except Exception as e:
                last_err = e
                try:
                    con.close()
                except Exception:
                    pass
                print(f"  [{label}] attempt {attempt}/{STREAM_RETRIES} failed: "
                      f"{str(e).splitlines()[0]}", flush=True)
                if attempt < STREAM_RETRIES:
                    time.sleep(STREAM_RETRY_WAIT_S)
        raise last_err

    def true_kcore(df, k):
        # Iterative k-core to a fixed point.
        while len(df):
            uc = df["user_id"].value_counts()
            ic = df["item_id"].value_counts()
            keep = df[df["user_id"].map(uc).ge(k) & df["item_id"].map(ic).ge(k)]
            if len(keep) == len(df):
                break
            df = keep
        return df

    def densify(df, k=3, limit_users=limit_users, label=""):
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
        print(f"  [{label}] final {eff_k}-core: {len(core)} reviews, "
              f"{core['user_id'].nunique()} users, {core['item_id'].nunique()} items",
              flush=True)
        return core

    YELP_SQL = """
    WITH s AS (
      SELECT user_id, business_id, stars, text
      FROM read_parquet('https://huggingface.co/api/datasets/yashraizada/yelp-open-dataset-reviews/parquet/default/train/0.parquet')
      WHERE stars > 0 LIMIT 250000
    )
    SELECT s.user_id AS user_id, s.business_id AS item_id, b.name AS item_name,
           b.categories AS category, s.stars AS rating, s.text AS review_text
    FROM s JOIN read_parquet('https://huggingface.co/api/datasets/yashraizada/yelp-open-dataset-business/parquet/default/train/0.parquet') b
      ON s.business_id = b.business_id
    """

    GOODREADS_SQL = """
    WITH s AS (
      SELECT user_id, book_id, rating, review_text
      FROM read_parquet('https://huggingface.co/api/datasets/vngclinh/goodreads-reviews/parquet/default/train/0.parquet')
      WHERE rating > 0 LIMIT 250000
    )
    SELECT s.user_id AS user_id, s.book_id AS item_id,
           COALESCE(m.Book, 'Book #' || s.book_id) AS item_name,
           'Goodreads (Book)' AS category, s.rating AS rating, s.review_text AS review_text
    FROM s LEFT JOIN read_parquet('https://huggingface.co/api/datasets/Eitanli/goodreads/parquet/default/train/0.parquet') m
      ON s.book_id = regexp_extract(m.URL, '/book/show/([0-9]+)', 1)
    """

    amz_review_urls = hf_parquet_urls("cogsci13/Amazon-Reviews-2023-Books-Review")
    amz_meta_urls = hf_parquet_urls("cogsci13/Amazon-Reviews-2023-Books-Meta")
    print(f"Amazon: {len(amz_review_urls)} review shards, "
          f"{len(amz_meta_urls)} meta shards discovered", flush=True)
    amz_meta_list = "[" + ",".join(f"'{u}'" for u in amz_meta_urls) + "]"
    AMAZON_SQL = f"""
    WITH s AS (
      SELECT user_id, parent_asin, rating, text
      FROM read_parquet('{amz_review_urls[0]}')
      WHERE rating > 0 AND user_id IS NOT NULL AND parent_asin IS NOT NULL
      LIMIT 600000
    )
    SELECT s.user_id AS user_id,
           s.parent_asin AS item_id,
           COALESCE(m.title, 'Amazon Book ' || s.parent_asin) AS item_name,
           'Amazon (Book)' AS category,
           s.rating AS rating,
           s.text AS review_text
    FROM s
    LEFT JOIN (SELECT parent_asin, title FROM read_parquet({amz_meta_list})) m
      ON s.parent_asin = m.parent_asin
    """

    print("=== Yelp ===", flush=True)
    yelp = densify(stream_query(YELP_SQL, "Yelp"), label="Yelp")
    print("=== Goodreads ===", flush=True)
    goodreads = densify(stream_query(GOODREADS_SQL, "Goodreads"), label="Goodreads")
    print("=== Amazon (Books) ===", flush=True)
    amazon = densify(stream_query(AMAZON_SQL, "Amazon"), label="Amazon")

    COLS = ["user_id", "item_id", "item_name", "category", "rating", "review_text"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in [("yelp", yelp), ("goodreads", goodreads), ("amazon", amazon)]:
            out = df[COLS].copy()
            out["review_text"] = out["review_text"].fillna("")
            out["item_name"] = out["item_name"].fillna("")
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            z.writestr(f"data/{name}_dense.csv", csv_bytes)
            print(f"data/{name}_dense.csv -> {len(out)} rows, "
                  f"{len(csv_bytes) / 1024:.0f} KB", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main(limit_users: int = 2000):
    import io, os, zipfile

    print(f"Running dense-dataset prep on Modal (limit_users={limit_users}) ...")
    zip_bytes = prepare_data.remote(limit_users)

    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(".")  # writes data/{yelp,goodreads,amazon}_dense.csv
        names = z.namelist()
    print("\n" + "=" * 60)
    print("Dense CSVs written locally:")
    for n in names:
        sz = os.path.getsize(n) / 1024
        print(f"  {n}  ({sz:.0f} KB)")
    print("=" * 60)
