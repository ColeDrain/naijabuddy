"""
Export catalogue item embeddings from a seeded naijabuddy.db into
data/item_embeddings.npz, so data_enricher.py can load them instead of
re-running the ~7-minute BGE embedding pass on every Docker build.

Each embedding is keyed by the SHA-256 of the exact text data_enricher embeds:
    f"{name} - {category}. {description}"
so the lookup round-trips deterministically from the committed CSVs.
"""
import os
import json
import sqlite3
import hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "naijabuddy.db")
OUT = os.path.join(ROOT, "data", "item_embeddings.npz")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT name, category, description, embedding FROM items").fetchall()
conn.close()

keys, vecs, skipped = [], [], 0
for r in rows:
    if not r["embedding"]:
        skipped += 1
        continue
    text = f"{r['name']} - {r['category']}. {r['description']}"
    keys.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    vecs.append(json.loads(r["embedding"]))

arr = np.asarray(vecs, dtype=np.float32)
np.savez_compressed(OUT, keys=np.array(keys), vectors=arr)
print(f"exported {len(keys)} item embeddings (skipped {skipped} with no vector)")
print(f"shape {arr.shape}  ->  {os.path.relpath(OUT, ROOT)}  "
      f"({os.path.getsize(OUT) / 1e6:.1f} MB)")
