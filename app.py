import json
import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from agent import NaijaBuddyAgent

# 1. Initialize FastAPI App
app = FastAPI(
    title="NaijaBuddy",
    description="Unified Multi-Domain Recommender and User Modeling Agent",
    version="1.0.0"
)

# 2. Enable CORS so the web browser can communicate with our API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Agent on Startup
# The agent will attempt to load the pre-cached GGUF model or run in fallback-development mode
agent = NaijaBuddyAgent()

import threading

# 4. Lazy-load and cache the BGE-Small embedding model globally to prevent slow 12-second disk loads on every request
EMBEDDER = None
EMBEDDER_LOCK = threading.Lock()

def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        with EMBEDDER_LOCK:
            if EMBEDDER is None:
                from sentence_transformers import SentenceTransformer
                print("Loading global BGE-Small embedding model for API requests...")
                EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return EMBEDDER

# =====================================================================
# REQUEST SCHEMAS (Pydantic Models)
# =====================================================================

class SimulateRequest(BaseModel):
    user_name: str
    item_id: int
    alpha: float = 0.3

class RecommendRequest(BaseModel):
    user_name: str
    domain: str  # 'Yelp', 'Amazon', 'Goodreads'

class CreateUserRequest(BaseModel):
    name: str
    persona: str

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.post("/api/users")
def create_user(req: CreateUserRequest):
    """Registers a new custom user, generates BGE-small embeddings, and saves to SQLite."""
    conn = None
    try:
        embedder = get_embedder()
        embedding = embedder.encode(req.persona).tolist()
        
        conn = database.get_connection()
        conn.isolation_level = "IMMEDIATE"  # Upgrade to write lock instantly, preventing deadlocks
        cursor = conn.cursor()
        
        # Save to database with automatic retry for aggressive concurrent GUI clients
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                cursor.execute("""
                    INSERT INTO users (name, persona, user_mean_rating, persona_embedding)
                    VALUES (?, ?, ?, ?)
                """, (req.name, req.persona, 3.5, json.dumps(embedding)))
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.2 * (2 ** attempt))
                    continue
                raise
        
        cursor.execute("SELECT id, name, persona, user_mean_rating FROM users WHERE name = ?", (req.name,))
        row = cursor.fetchone()
        
        return dict(row)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="A persona with this name already exists.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create persona: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/api/users")
def get_users():
    """Retrieves all seeded user personas and their profile descriptions."""
    conn = None
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, persona, user_mean_rating FROM users")
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "persona": row["persona"],
                "user_mean_rating": row["user_mean_rating"]
            }
            for row in rows
        ]
    finally:
        if conn:
            conn.close()

@app.get("/api/items")
def get_items(domain: str = None):
    """Retrieves catalog items, optionally filtered by domain."""
    conn = None
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        
        if domain:
            cursor.execute("SELECT id, name, category, domain, description, average_rating FROM items WHERE domain = ?", (domain,))
        else:
            cursor.execute("SELECT id, name, category, domain, description, average_rating FROM items")
            
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "domain": row["domain"],
                "description": row["description"],
                "average_rating": row["average_rating"]
            }
            for row in rows
        ]
    finally:
        if conn:
            conn.close()

@app.post("/api/simulate")
def simulate_review(req: SimulateRequest):
    """
    POST /api/simulate (Task A)
    Simulates a localized review and calibrated star rating for a selected item.
    """
    try:
        result = agent.simulate_review(req.user_name, req.item_id, req.alpha)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference simulation error: {str(e)}")

@app.post("/api/recommend")
def recommend_items(req: RecommendRequest):
    """
    POST /api/recommend (Task B)
    Retrieves Top-10 recommended items in a specific domain, with rankings and explanations.
    """
    valid_domains = ["Yelp", "Amazon", "Goodreads"]
    if req.domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Must be one of {valid_domains}")
        
    try:
        recommendations = agent.recommend_items(req.user_name, req.domain)
        if isinstance(recommendations, dict) and "error" in recommendations:
            raise HTTPException(status_code=400, detail=recommendations["error"])
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommender agent error: {str(e)}")

# =====================================================================
# STATIC ASSETS ROUTING (For serving Web dashboard assets)
# =====================================================================

# Ensure static folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)

# Mount the static files router to serve index.html at root "/"
# StaticFiles must be mounted at the end of routing to avoid blocking API endpoints
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Read port from port or default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI App Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
