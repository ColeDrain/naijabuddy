# Implementation Plan: Integrating Real-World Yelp, Amazon, and Goodreads Datasets

Our objective is to replace the purely synthetic catalog items and ratings in **NaijaBuddy** with actual real-world subsets from Yelp, Amazon, and Goodreads. This will make our evaluation metrics (RMSE, NDCG@10) mathematically rigorous and trustworthy, proving that our output calibration and critic layers generalize to noisy, real-world data, while maintaining the unique Nigerian localization required by the hackathon brief.

## Finalized Datasets Documentation

We will stream and join public Parquet datasets directly from the Hugging Face Hub and S3 using **DuckDB**. This bypasses any gated dataset permissions or local script blocks, and allows us to pull relational records in seconds.

### 1. Yelp (Food & Spots)
* **Source Reviews:** `yashraizada/yelp-open-dataset-reviews` (train split, `0.parquet`)
* **Source Businesses:** `yashraizada/yelp-open-dataset-business` (train split, `0.parquet`)
* **Key Fields:** `user_id`, `business_id`, `stars` (rating), `text` (review text), `name` (business name), `categories`.
* **DuckDB Extraction Query:**
  ```sql
  SELECT 
      r.user_id,
      r.business_id AS item_id,
      b.name AS item_name,
      b.categories AS category,
      r.stars AS rating,
      r.text AS review_text
  FROM read_parquet('https://huggingface.co/api/datasets/yashraizada/yelp-open-dataset-reviews/parquet/default/train/0.parquet') r
  JOIN read_parquet('https://huggingface.co/api/datasets/yashraizada/yelp-open-dataset-business/parquet/default/train/0.parquet') b
  ON r.business_id = b.business_id
  LIMIT 500
  ```

### 2. Goodreads (Books)
* **Source Reviews:** `vngclinh/goodreads-reviews` (train split, `0.parquet`)
* **Source Books:** `Eitanli/goodreads` (train split, `0.parquet`)
* **Key Fields:** `user_id`, `book_id`, `rating`, `review_text`, `Book` (title), `Author`, `Description`, `URL`.
* **DuckDB Extraction Query:**
  ```sql
  SELECT 
      r.user_id,
      r.book_id AS item_id,
      m.Book AS item_name,
      'Goodreads (Book)' AS category,
      r.rating AS rating,
      r.review_text
  FROM read_parquet('https://huggingface.co/api/datasets/vngclinh/goodreads-reviews/parquet/default/train/0.parquet') r
  JOIN read_parquet('https://huggingface.co/api/datasets/Eitanli/goodreads/parquet/default/train/0.parquet') m
  ON r.book_id = regexp_extract(m.URL, '/book/show/([0-9]+)', 1)
  LIMIT 500
  ```

### 3. Amazon (Movies & Electronics)
* **Source:** ClickHouse S3 public mirror of standard 2015 Amazon reviews: `https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet`
* **Key Fields:** `customer_id`, `product_id`, `product_title`, `product_category`, `star_rating`, `review_body`.
* **DuckDB Extraction Query:**
  ```sql
  SELECT 
      customer_id AS user_id,
      product_id AS item_id,
      product_title AS item_name,
      product_category AS category,
      star_rating AS rating,
      review_body AS review_text
  FROM read_parquet('https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet')
  LIMIT 500
  ```

---

## User Review Required

> [!IMPORTANT]
> **Data Size vs. Seeding Time:** To keep the embedding generation fast (~1–2 minutes on CPU) and fit the final database inside the Docker image without bloating, we will pull:
> * **150 unique items** per domain (450 total items) to generate embeddings.
> * **500 real ratings/reviews** per domain (1500 total interactions) to populate user baselines.
>
> **Library Dependencies:** We have already successfully installed `duckdb` in our project `.venv`. The seeding script will automatically use it.

---

## Proposed Changes

We will introduce a new seeding module and modify the database, metrics, and solution paper files.

### Data Layer & Seeding

#### [NEW] [fetch_real_data.py](file:///Users/indicina/Projects/dsn-bct-hackathon/fetch_real_data.py)
A Python script that fetches real-world subsets using the DuckDB queries outlined above:
1. **Streaming & Ingestion:** Connects to DuckDB and queries public Parquet endpoints to stream subsets.
2. **Aggregated Catalog Generation:** Compile item descriptions from review summaries if needed, or use the metadata description fields.
3. **Embedding Generation:** Uses the local `BAAI/bge-small-en-v1.5` sentence-transformers model to generate 384-dimensional dense vectors for all retrieved items and writes them to `naijabuddy.db`.
4. **Localization Overlay (The Hybrid Catalog):** Retains our manually seeded Nigerian/African items (e.g., *Yellow Chilli*, *King of Boys*, Chinua Achebe's books) and overlays them on top of the real-world dataset to preserve the localized conversational features.

#### [MODIFY] [data_enricher.py](file:///Users/indicina/Projects/dsn-bct-hackathon/data_enricher.py)
Update to:
* Import and run `fetch_real_data.py` as part of the initial database setup.
* Seed our localized Nigerian personas and baseline ratings to connect them with both the real-world items and the local seeds.

---

### Core Agent & Metrics

#### [MODIFY] [calculate_metrics.py](file:///Users/indicina/Projects/dsn-bct-hackathon/calculate_metrics.py)
Modify the metrics suite to perform evaluations on the real-world dataset:
* **RMSE Evaluation:** Calculate RMSE on real-world reviews, comparing raw LLM ratings versus actual ratings, and measuring the exact error reduction provided by our Output Calibration Layer ($\alpha$-blend).
* **NDCG@10 Evaluation:** Query the vector index and compute NDCG@10 on the expanded dataset to prove the recall generalization of our dual-stage filter-then-rerank pipeline.
* **Log Performance:** Write the exact improvements directly to `metrics_results.json` so they can be referenced programmatically.

#### [MODIFY] [solution_paper.md](file:///Users/indicina/Projects/dsn-bct-hackathon/solution_paper.md)
Update Section 4 (Experiments and Ablation Study) and Section 4.2 (Metric Evaluation Results) to document the results obtained from the real-world dataset evaluation instead of the small mock values.

#### [NEW] [pre_synthesize_personas.py](file:///Users/indicina/Projects/dsn-bct-hackathon/scratch/pre_synthesize_personas.py)
A script that runs offline to pre-synthesize all Yelp/Amazon/Goodreads user personas in the database to eliminate first-query runtime latency:
* **Multi-Rating Cohort (>=2 ratings):** Uses Qwen-3B to synthesize their histories into a natural 2-sentence description (takes ~90 seconds for 60 users).
* **Single-Rating Cohort (1 rating):** Formats their single visit/rating into a clean, natural 2-sentence description via deterministic Python logic.
* **Vector Embeddings:** Automatically updates the persona text and runs `BAAI/bge-small-en-v1.5` to recompute the dense persona embeddings.

---

## Verification Plan

### Automated Tests
1. **Seeding Verification:**
   Run the new data seeder:
   ```bash
   .venv/bin/python data_enricher.py
   ```
   Verify that `items` table contains $\ge 450$ items and the `ratings` table contains $\ge 1500$ real ratings across all three domains (Yelp, Amazon, Goodreads).
2. **Pre-Synthesis Execution:**
   Run the pre-synthesis script:
   ```bash
   .venv/bin/python scratch/pre_synthesize_personas.py
   ```
   Verify that all user profiles in SQLite no longer contain placeholders (do not start with `"A real "`) and their embedding vectors are populated.
3. **Metrics Verification:**
   Run the metric calculator:
   ```bash
   .venv/bin/python calculate_metrics.py
   ```
   Verify that the RMSE of raw LLM vs. calibrated ratings is calculated successfully, and that NDCG@10 rankings are generated.

### Manual Verification
1. **API Server Test:**
   Start the server on port 8050:
   ```bash
   .venv/bin/python app.py
   ```
   Run `verify_api.py` to ensure all API endpoints (simulate, recommend, users) work seamlessly with the newly structured database.
2. **UI Inspection:**
   Open the browser interface and test recommending items in each domain (Yelp, Amazon, Goodreads) for our personas. Confirm that real-world products are displayed alongside localized recommendations.
