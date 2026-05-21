"""
Loads the pre-densified NaijaBuddy datasets from local CSV files.

The heavy data acquisition -- streaming large samples of Yelp, Goodreads and
Amazon reviews and extracting a genuine bipartite k-core from each -- is done
ONCE, off-machine, by `colab_data_prep.ipynb` running on Google Colab. That
notebook exports three CSVs into `data/`.

This module just reads those CSVs, so seeding the database is fast, fully
offline, and uses no bandwidth. The CSVs are small enough to commit to the
repo, which also makes `data_enricher.py` reproducible for the judges.

To (re)generate the CSVs:
  1. Open colab_data_prep.ipynb in Google Colab.
  2. Runtime -> Run all.
  3. Download naijabuddy_dense_data.zip and unzip it into the data/ folder.
"""

import os
import csv
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_REGEN_HINT = (
    "Generate it by running colab_data_prep.ipynb in Google Colab "
    "(Runtime -> Run all), then unzip naijabuddy_dense_data.zip into data/."
)

# Review text fields can be long; raise the csv field-size limit once.
csv.field_size_limit(16 * 1024 * 1024)


def _load_dense_csv(filename, label):
    """Read a densified CSV into the (user, item, name, category, rating, text) tuples
    that data_enricher.py expects."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{label}] missing dense dataset: {path}\n{_REGEN_HINT}"
        )

    records = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rating = float(row["rating"])
            except (ValueError, TypeError, KeyError):
                skipped += 1
                continue
            if rating <= 0:  # guard against "did not rate" sentinels
                skipped += 1
                continue
            records.append((
                row.get("user_id", ""),
                row.get("item_id", ""),
                row.get("item_name", "") or "",
                row.get("category", "") or "",
                rating,
                row.get("review_text", "") or "",
            ))

    msg = f"  [{label}] loaded {len(records)} reviews from {filename}"
    if skipped:
        msg += f" ({skipped} rows skipped)"
    print(msg)
    return records


# limit_users is accepted only for backwards compatibility with existing callers.
# The CSVs are already densified and capped by colab_data_prep.ipynb, so it is
# intentionally ignored here.

def fetch_yelp_data(limit_users=None):
    return _load_dense_csv("yelp_dense.csv", "Yelp")


def fetch_goodreads_data(limit_users=None):
    return _load_dense_csv("goodreads_dense.csv", "Goodreads")


def fetch_amazon_data(limit_users=None):
    return _load_dense_csv("amazon_dense.csv", "Amazon")


if __name__ == "__main__":
    for fn, name in [(fetch_yelp_data, "Yelp"),
                     (fetch_goodreads_data, "Goodreads"),
                     (fetch_amazon_data, "Amazon")]:
        try:
            recs = fn()
            sample = recs[0] if recs else "(none)"
            print(f"{name}: {len(recs)} records | sample: {sample}")
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
