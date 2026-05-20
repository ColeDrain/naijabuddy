import json
import math
import os
import sqlite3

# Default SQLite database path in workspace
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "naijabuddy.db")

def get_connection():
    """Returns a connection to our SQLite database with row factory enabled and concurrency safety."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn

def initialize_db():
    """Creates the SQLite tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Items table (Yelp restaurants, Amazon movies/electronics, Goodreads books)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            domain TEXT NOT NULL,          -- 'Yelp', 'Amazon', 'Goodreads'
            description TEXT,
            average_rating REAL DEFAULT 0.0,
            embedding TEXT                 -- JSON array of floats (384 dimensions from BGE-small)
        )
    """)
    
    # 2. Users table (including historical averages and persona descriptions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            persona TEXT NOT NULL,
            user_mean_rating REAL DEFAULT 0.0,
            persona_embedding TEXT         -- JSON array of floats
        )
    """)
    
    # 3. Ratings table (simulated or real historical ratings and reviews)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            review_text TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE,
            UNIQUE(user_id, item_id)
        )
    """)
    
    conn.commit()
    conn.close()

# =====================================================================
# VECTOR SEARCH MATH (Pure Python Fallback with NumPy compatibility)
# =====================================================================

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x ** 2 for x in v))

def cosine_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

# =====================================================================
# DATABASE OPERATIONS
# =====================================================================

def get_nearest_items(query_embedding, domain, top_k=20):
    """
    Performs Stage-1 Recall. Retrieves all items in the target domain,
    calculates cosine similarity against the query embedding in Python,
    and returns the top_k most similar items.
    """
    if not query_embedding:
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # Query only items from the requested domain
    cursor.execute("SELECT id, name, category, domain, description, average_rating, embedding FROM items WHERE domain = ?", (domain,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        item_embedding_str = row["embedding"]
        if not item_embedding_str:
            continue
            
        try:
            item_embedding = json.loads(item_embedding_str)
            similarity = cosine_similarity(query_embedding, item_embedding)
            
            results.append({
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "domain": row["domain"],
                "description": row["description"],
                "average_rating": row["average_rating"],
                "similarity": similarity
            })
        except Exception as e:
            # Gracefully ignore malformed embedding strings
            continue
            
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def get_nearest_users(query_embedding, top_k=5):
    """
    Finds the top_k users whose personas are most semantically similar.
    Used for our "Cluster Mean Calibration" in cold-start scenarios.
    """
    if not query_embedding:
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, persona, user_mean_rating, persona_embedding FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        user_embedding_str = row["persona_embedding"]
        if not user_embedding_str:
            continue
            
        try:
            user_embedding = json.loads(user_embedding_str)
            similarity = cosine_similarity(query_embedding, user_embedding)
            
            results.append({
                "id": row["id"],
                "name": row["name"],
                "persona": row["persona"],
                "user_mean_rating": row["user_mean_rating"],
                "similarity": similarity
            })
        except Exception as e:
            continue
            
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def get_user_by_name(name):
    """Retrieves a user row from the database by name."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_ratings_mean(user_id):
    """Calculates and returns the historical mean rating of a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(rating) FROM ratings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else None

# =====================================================================
# SEEDING INITIALIZATION HELPER
# =====================================================================

if __name__ == "__main__":
    print(f"Initializing SQLite database at: {DB_PATH}")
    initialize_db()
    print("Database schema successfully created!")
