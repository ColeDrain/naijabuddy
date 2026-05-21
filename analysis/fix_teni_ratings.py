"""
One-off surgical fix: insert Teni's 3 showcase ratings into the existing
naijabuddy.db. The data_enricher.py code is already corrected; this patches the
stale on-disk DB without a full re-seed (which would discard the synthesised
personas). Idempotent — INSERT OR IGNORE respects the UNIQUE(user_id,item_id).
"""
import os
import sqlite3

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "naijabuddy.db")

TENI = "Teni (Lagos Gen-Z Influencer)"
RATINGS = [
    ("Club Quilox (Victoria Island)", 5.0,
     "Omo! The energy at Quilox is absolutely top-tier! Afrobeats was hitting "
     "different and the aesthetics are giving premium luxury. God when will I "
     "meet my billionaire husband here? 10/10!"),
    ("The Wedding Party (Nollywood Movie)", 5.0,
     "This movie had me crying laughing! It is literally my family in a "
     "wedding. The mothers are so extra, the drama is giving pure chaotic "
     "Nigerian wedding vibes. Obsessed!"),
    ("The Place Restaurant (Yaba)", 3.0,
     "The Jollof is decent, but the queue and noise? Absolute wahala. Not "
     "aesthetic at all, do not come here if you want to take Instagram "
     "pictures. Just buy and leave."),
]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

row = cur.execute("SELECT id FROM users WHERE name = ?", (TENI,)).fetchone()
if not row:
    raise SystemExit(f"User '{TENI}' not found - DB may need a full re-seed.")
uid = row["id"]

inserted = 0
for item_name, rating, review in RATINGS:
    irow = cur.execute("SELECT id FROM items WHERE name = ?", (item_name,)).fetchone()
    if not irow:
        print(f"  [skip] item not found: {item_name}")
        continue
    cur.execute(
        "INSERT OR IGNORE INTO ratings (user_id, item_id, rating, review_text) "
        "VALUES (?, ?, ?, ?)",
        (uid, irow["id"], rating, review),
    )
    if cur.rowcount:
        inserted += 1
        print(f"  [ok] {item_name} -> {rating} stars")
    else:
        print(f"  [exists] {item_name}")
conn.commit()

n = cur.execute("SELECT COUNT(*) FROM ratings WHERE user_id = ?", (uid,)).fetchone()[0]
mean = cur.execute("SELECT AVG(rating) FROM ratings WHERE user_id = ?", (uid,)).fetchone()[0]
conn.close()
print(f"\nInserted {inserted} new rating(s). "
      f"Teni now has {n} ratings, historical mean = {mean:.2f}.")
