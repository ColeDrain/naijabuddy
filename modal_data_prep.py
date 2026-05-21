"""
Dense-dataset preparation on Modal — the cloud equivalent of
colab_data_prep.ipynb. Streams Yelp / Goodreads / Amazon reviews from the
HuggingFace parquet mirrors, extracts a dense bipartite k-core per domain,
caps each to its `limit_users` densest users, writes data/*_dense.csv.

    modal run modal_data_prep.py --limit-users 2000

Notes:
  * parquet shard URLs are resolved via the dataset-viewer API (the notebook
    hardcoded a now-dead path);
  * HF now requires an auth token to download the parquet files — the token
    is read locally and forwarded to the Modal container as a DuckDB http
    secret (never logged);
  * reads across all shards with a large LIMIT — one shard yields only
    ~340 dense users, far short of a 2k target.
"""
import os
import modal

app = modal.App("naijabuddy-data-prep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "duckdb==1.1.3", "pandas", "requests", "pyarrow"
)

RAW_LIMIT = 2_000_000


def _hf_token():
    """Find the local HuggingFace token without printing it."""
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    for p in (os.path.join(hf_home, "token"),
              os.path.expanduser("~/.cache/huggingface/token"),
              os.path.expanduser("~/.huggingface/token")):
        try:
            if os.path.exists(p):
                t = open(p).read().strip()
                if t:
                    return t
        except Exception:
            pass
    return None


@app.function(image=image, timeout=3600, cpu=4.0, memory=32768)
def prepare_data(limit_users: int = 2000, hf_token: str = ""):
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

    def plist(urls):
        return "[" + ",".join(f"'{u}'" for u in urls) + "]"

    def stream_query(query, label):
        # Retry the whole query — a corrupted compressed chunk surfaces as a
        # decompression error after the HTTP read, which http_retries can't catch.
        last_err = None
        for attempt in range(1, STREAM_RETRIES + 1):
            con = duckdb.connect()
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                con.execute("SET http_retries=3;")
                con.execute("SET http_timeout=120000;")
                # HF requires auth to download parquet files — attach the token
                # as an http secret scoped to huggingface.co.
                if hf_token:
                    con.execute(
                        "CREATE OR REPLACE SECRET hf_auth (TYPE http, "
                        "SCOPE 'https://huggingface.co', "
                        "EXTRA_HTTP_HEADERS MAP "
                        f"{{'Authorization': 'Bearer {hf_token}'}})"
                    )
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
        while len(df):
            uc = df["user_id"].value_counts()
            ic = df["item_id"].value_counts()
            keep = df[df["user_id"].map(uc).ge(k) & df["item_id"].map(ic).ge(k)]
            if len(keep) == len(df):
                break
            df = keep
        return df

    def densify(df, k=3, limit_users=2000, label=""):
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

    print("Resolving parquet shard URLs ...", flush=True)
    yelp_review = hf_parquet_urls("yashraizada/yelp-open-dataset-reviews")
    yelp_biz = hf_parquet_urls("yashraizada/yelp-open-dataset-business")
    gr_review = hf_parquet_urls("vngclinh/goodreads-reviews")
    gr_meta = hf_parquet_urls("Eitanli/goodreads")
    amz_review = hf_parquet_urls("cogsci13/Amazon-Reviews-2023-Books-Review")
    amz_meta = hf_parquet_urls("cogsci13/Amazon-Reviews-2023-Books-Meta")
    print(f"  shards — yelp:{len(yelp_review)} goodreads:{len(gr_review)} "
          f"amazon:{len(amz_review)}", flush=True)

    YELP_SQL = f"""
    WITH s AS (
      SELECT user_id, business_id, stars, text
      FROM read_parquet({plist(yelp_review)})
      WHERE stars > 0 LIMIT {RAW_LIMIT}
    )
    SELECT s.user_id AS user_id, s.business_id AS item_id, b.name AS item_name,
           b.categories AS category, s.stars AS rating, s.text AS review_text
    FROM s JOIN read_parquet({plist(yelp_biz)}) b
      ON s.business_id = b.business_id
    """

    GOODREADS_SQL = f"""
    WITH s AS (
      SELECT user_id, book_id, rating, review_text
      FROM read_parquet({plist(gr_review)})
      WHERE rating > 0 LIMIT {RAW_LIMIT}
    )
    SELECT s.user_id AS user_id, s.book_id AS item_id,
           COALESCE(m.Book, 'Book #' || s.book_id) AS item_name,
           'Goodreads (Book)' AS category, s.rating AS rating, s.review_text AS review_text
    FROM s LEFT JOIN read_parquet({plist(gr_meta)}) m
      ON s.book_id = regexp_extract(m.URL, '/book/show/([0-9]+)', 1)
    """

    AMAZON_SQL = f"""
    WITH s AS (
      SELECT user_id, parent_asin, rating, text
      FROM read_parquet({plist(amz_review)})
      WHERE rating > 0 AND user_id IS NOT NULL AND parent_asin IS NOT NULL
      LIMIT {RAW_LIMIT}
    )
    SELECT s.user_id AS user_id,
           s.parent_asin AS item_id,
           COALESCE(m.title, 'Amazon Book ' || s.parent_asin) AS item_name,
           'Amazon (Book)' AS category,
           s.rating AS rating,
           s.text AS review_text
    FROM s
    LEFT JOIN (SELECT parent_asin, title FROM read_parquet({plist(amz_meta)})) m
      ON s.parent_asin = m.parent_asin
    """

    print("=== Yelp ===", flush=True)
    yelp = densify(stream_query(YELP_SQL, "Yelp"), limit_users=limit_users, label="Yelp")
    print("=== Goodreads ===", flush=True)
    goodreads = densify(stream_query(GOODREADS_SQL, "Goodreads"), limit_users=limit_users, label="Goodreads")
    print("=== Amazon (Books) ===", flush=True)
    amazon = densify(stream_query(AMAZON_SQL, "Amazon"), limit_users=limit_users, label="Amazon")

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
    import io, zipfile

    token = _hf_token()
    print(f"HF token: {'found' if token else 'NOT FOUND'}")
    if not token:
        raise SystemExit("No HF token found locally — run `hf auth login` first.")

    print(f"Running dense-dataset prep on Modal (limit_users={limit_users}) ...")
    zip_bytes = prepare_data.remote(limit_users, token)

    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(".")
        names = z.namelist()
    print("\n" + "=" * 60)
    print("Dense CSVs written locally:")
    for n in names:
        sz = os.path.getsize(n) / 1024
        print(f"  {n}  ({sz:.0f} KB)")
    print("=" * 60)
