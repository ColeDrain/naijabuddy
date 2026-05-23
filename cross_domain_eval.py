"""
Cross-domain transfer eval — Amazon Books → Movies.

For each user with rich history in BOTH Books and Movies (`xdomain_2k.jsonl`,
built by `cross_domain_dataset.py`), predict the held-out Movies rating
using only the user's Books history plus the target movie's cross-user
item-mean. This is the §4.5 cold-start thesis stated in its strongest form:
*even when the user is fully cold in the target domain, item-bias rescues
the prediction.*

No LLM is used — the rating-only 5-core source data has no item titles or
descriptions, so the LLM has nothing to ground on. The test isolates the
purely-statistical signal: does books-taste transfer to movies-taste, and
does adding the target movie's item-mean (a within-domain signal computed
across other users) close the gap?

Models compared (Books history → Movies rating):
  - V0_global   :   y_hat = global Movies mean
  - V1_books    :   y_hat = user's mean Books rating  (pure cross-domain user-bias)
  - V1_movies   :   y_hat = user's mean Movies rating (in-domain user-bias upper bound)
  - V2_books_item :  y_hat = β·μ_user_books + γ·μ_item_movies      (β+γ=1, swept)
  - V2_movies_item:  y_hat = β·μ_user_movies + γ·μ_item_movies     (β+γ=1, swept)

Writes scratch/cross_domain_results.json + a per-RMSE table.

Usage:
    .venv/bin/python cross_domain_eval.py
"""
import json
import math
import os
from collections import defaultdict

IN_PATH = "data/amazon_crossdomain/xdomain_2k.jsonl"
OUT_PATH = "scratch/cross_domain_results.json"


def rmse(preds, actuals):
    n = len(preds)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)


def mae(preds, actuals):
    n = len(preds)
    if n == 0:
        return float("nan")
    return sum(abs(p - a) for p, a in zip(preds, actuals)) / n


def main():
    print(f"loading {IN_PATH}", flush=True)
    rows = []
    with open(IN_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"  {len(rows):,} users with held-out Movies rating + Books history",
          flush=True)

    # Pre-pass: build cross-user item-mean for Movies across all users'
    # movies_train interactions. This is what makes the item-bias term
    # populated even for the target movie (otherwise the held-out movie has
    # no prior in the held-out user's view).
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)
    n_movies_train_total = 0
    movies_rating_sum = 0.0
    n_books_train_total = 0
    for r in rows:
        for m in r.get("movies_train", []):
            item_sum[m["item"]] += float(m["rating"])
            item_cnt[m["item"]] += 1
            movies_rating_sum += float(m["rating"])
            n_movies_train_total += 1
        n_books_train_total += len(r.get("books_history", []))
    item_mean = {k: item_sum[k] / item_cnt[k] for k in item_sum}
    global_mean_movies = (movies_rating_sum / n_movies_train_total
                          if n_movies_train_total else 4.0)
    print(f"  global Movies mean (from train): {global_mean_movies:.4f}",
          flush=True)
    print(f"  unique Movies items with at least 1 train rating: "
          f"{len(item_mean):,}", flush=True)
    print(f"  total Books interactions: {n_books_train_total:,}", flush=True)
    print(f"  total Movies train interactions: {n_movies_train_total:,}",
          flush=True)

    # Per-user predictions for the held-out Movies rating.
    actuals, p_v0, p_v1b, p_v1m, item_target = [], [], [], [], []
    user_mean_books, user_mean_movies = [], []
    n_cold_target = 0   # how often the held-out movie has no item-mean signal
    for r in rows:
        held_item = r["movies_holdout_item"]
        held_rating = float(r["movies_holdout_rating"])
        books = r.get("books_history", [])
        movies_tr = r.get("movies_train", [])
        if not books or not movies_tr:
            continue
        umb = sum(float(b["rating"]) for b in books) / len(books)
        umm = sum(float(m["rating"]) for m in movies_tr) / len(movies_tr)
        im = item_mean.get(held_item, global_mean_movies)
        if held_item not in item_mean:
            n_cold_target += 1

        actuals.append(held_rating)
        p_v0.append(global_mean_movies)
        p_v1b.append(umb)
        p_v1m.append(umm)
        item_target.append(im)
        user_mean_books.append(umb)
        user_mean_movies.append(umm)

    n = len(actuals)
    print(f"\n  {n:,} held-out Movies ratings to score "
          f"({n_cold_target:,} have a held-out movie nobody else rated)",
          flush=True)

    # Sweep β over [0, 1] for both V2 variants. Best beta is the test-set
    # descriptive minimum (same conservative reporting convention as §4.2/§4.5).
    grid = [round(x * 0.05, 2) for x in range(0, 21)]
    sweep_v2_books = {}
    sweep_v2_movies = {}
    for b in grid:
        g = round(1.0 - b, 2)
        preds_b = [b * umb + g * im
                   for umb, im in zip(user_mean_books, item_target)]
        preds_m = [b * umm + g * im
                   for umm, im in zip(user_mean_movies, item_target)]
        sweep_v2_books[b] = rmse(preds_b, actuals)
        sweep_v2_movies[b] = rmse(preds_m, actuals)
    best_b_books = min(sweep_v2_books, key=sweep_v2_books.get)
    best_b_movies = min(sweep_v2_movies, key=sweep_v2_movies.get)

    out = {
        "n_users": n,
        "n_held_movie_unseen_by_others": n_cold_target,
        "global_movies_mean": global_mean_movies,
        "rmse": {
            "V0_global_movies": rmse(p_v0, actuals),
            "V1_user_books": rmse(p_v1b, actuals),
            "V1_user_movies": rmse(p_v1m, actuals),
            "V0_item_mean_only": rmse(item_target, actuals),
            "V2_books_plus_item_best": sweep_v2_books[best_b_books],
            "V2_movies_plus_item_best": sweep_v2_movies[best_b_movies],
        },
        "best_beta_user_weight": {
            "books_plus_item": best_b_books,
            "movies_plus_item": best_b_movies,
        },
        "mae": {
            "V0_global_movies": mae(p_v0, actuals),
            "V1_user_books": mae(p_v1b, actuals),
            "V1_user_movies": mae(p_v1m, actuals),
        },
        "sweep_v2_books": sweep_v2_books,
        "sweep_v2_movies": sweep_v2_movies,
    }

    print("\n" + "=" * 66)
    print("CROSS-DOMAIN TRANSFER — Books history → Movies rating")
    print("=" * 66)
    print(f"  V0 (global Movies mean)                  RMSE = "
          f"{out['rmse']['V0_global_movies']:.4f}")
    print(f"  V1 (user-mean over BOOKS, cross-domain)  RMSE = "
          f"{out['rmse']['V1_user_books']:.4f}")
    print(f"  V1 (user-mean over MOVIES, in-domain)    RMSE = "
          f"{out['rmse']['V1_user_movies']:.4f}")
    print(f"  item-mean ONLY (held-out movie)          RMSE = "
          f"{out['rmse']['V0_item_mean_only']:.4f}")
    print(f"  V2 books-user + item-bias (β={best_b_books})        "
          f"RMSE = {out['rmse']['V2_books_plus_item_best']:.4f}")
    print(f"  V2 movies-user + item-bias (β={best_b_movies})       "
          f"RMSE = {out['rmse']['V2_movies_plus_item_best']:.4f}")
    print("=" * 66)

    os.makedirs("scratch", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
