import json
import os
import sqlite3
import sys

# Configure isolated model environment variables BEFORE importing SentenceTransformer
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "hf_home")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(MODELS_DIR, "sentence_transformers")

import database
import fetch_real_data

def clean_str(val):
    """Decodes bytes to string and strips spaces/removes nulls safely."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    return str(val).strip()

def generate_user_persona(domain, history, item_details):
    """
    Dynamically constructs a descriptive persona based on the user's rated items
    and category preferences in the sample dataset, stripping explicit item names
    to prevent BGE vector search dilution.
    """
    categories = []
    reviews_summary = []
    
    for item_id, rating, review_text in history:
        item_info = item_details.get((domain, item_id))
        if item_info:
            categories.append(item_info["category"])
            if review_text.strip() and len(review_text) > 15:
                reviews_summary.append(review_text.strip()[:100])
                
    # Get unique, clean categories and de-boilerplate them
    unique_categories = []
    for cat_str in categories:
        for cat in cat_str.split(","):
            c = cat.replace("Goodreads (", "").replace("Yelp (", "").replace("Amazon (", "").replace(")", "").strip()
            if c and c not in unique_categories:
                unique_categories.append(c)
                
    categories_str = ", ".join(unique_categories[:4]) if unique_categories else "general items"
    
    persona = f"A real {domain} user interested in {categories_str}."
    if reviews_summary:
        snippet = " | ".join(reviews_summary[:2])
        persona += f" Expressed reviews: '{snippet}...'"
        
    return persona

def seed_data():
    print("=" * 60)
    print("NAIJABUDDY DATABASE DATA SEEDER")
    print("=" * 60)
    
    # Drop existing tables to start fresh cleanly
    print("Dropping existing tables to start fresh...")
    conn = database.get_connection()
    cursor = conn.cursor()
    for table in ["ratings", "users", "items"]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    
    # 1. Initialize SQLite tables
    print("Initializing SQLite database tables...")
    database.initialize_db()
    
    # 2. Check and load embedding model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Sentence-transformers package is missing. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
        
    print("Loading local BGE-Small embedding model...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model loaded successfully!")
    
    # =====================================================================
    # STREAM AND INGEST REAL-WORLD DATASETS
    # =====================================================================
    print("\n" + "-" * 50)
    print("PART 1: STREAMING REAL-WORLD DATASETS")
    print("-" * 50)
    
    yelp_records = fetch_real_data.fetch_yelp_data(limit_users=100)
    goodreads_records = fetch_real_data.fetch_goodreads_data(limit_users=100)
    amazon_records = fetch_real_data.fetch_amazon_data(limit_users=100)
    
    raw_records = []
    
    # Yelp Mapping
    for r in yelp_records:
        user_id = clean_str(r[0])
        item_id = clean_str(r[1])
        item_name = clean_str(r[2])
        category = clean_str(r[3])
        rating = float(r[4])
        review_text = clean_str(r[5])
        raw_records.append((user_id, item_id, item_name, category, "Yelp", rating, review_text))
        
    # Goodreads Mapping
    for r in goodreads_records:
        user_id = clean_str(r[0])
        item_id = clean_str(r[1])
        item_name = clean_str(r[2])
        category = clean_str(r[3])
        rating = float(r[4])
        review_text = clean_str(r[5])
        raw_records.append((user_id, item_id, item_name, category, "Goodreads", rating, review_text))
        
    # Amazon Mapping
    for r in amazon_records:
        user_id = clean_str(r[0])
        item_id = clean_str(r[1])
        item_name = clean_str(r[2])
        category = clean_str(r[3])
        rating = float(r[4])
        review_text = clean_str(r[5])
        raw_records.append((user_id, item_id, item_name, category, "Amazon", rating, review_text))
        
    # Group unique items to calculate average ratings and compile descriptions
    print("Processing unique items and compiling reviews as descriptions...")
    item_details = {}
    for user_id, item_id, item_name, category, domain, rating, review_text in raw_records:
        key = (domain, item_id)
        if key not in item_details:
            item_details[key] = {
                "name": item_name,
                "category": category,
                "ratings": [],
                "reviews": []
            }
        item_details[key]["ratings"].append(rating)
        if review_text.strip():
            item_details[key]["reviews"].append(review_text)
            
    # Process unique users to compile history for persona generation
    print("Processing unique users and compiling interaction history...")
    user_details = {} # (domain, user_id) -> list of (item_id, rating, review_text)
    for user_id, item_id, item_name, category, domain, rating, review_text in raw_records:
        key = (domain, user_id)
        if key not in user_details:
            user_details[key] = []
        user_details[key].append((item_id, rating, review_text))
        
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # Seeding Real Items with Deduplication / Suffix for collisions
    print(f"\nEncoding and seeding {len(item_details)} unique real-world items...")
    db_item_ids = {} # (domain, item_id) -> SQLite integer ID
    
    item_keys = list(item_details.keys())
    item_texts = []
    seen_names = {} # name -> domain
    
    for key in item_keys:
        info = item_details[key]
        domain, item_id = key
        
        # Compile description from review snippets
        reviews_summary = " ".join(info["reviews"][:3])[:300]
        if reviews_summary:
            description = f"Real-world {domain} item categorized under {info['category']}. Customer reviews state: '{reviews_summary}...'"
        else:
            description = f"Real-world {domain} item categorized under {info['category']}."
            
        info["description"] = description
        info["avg_rating"] = sum(info["ratings"]) / len(info["ratings"])
        
        # Deduplication check: if name is seen in another domain/category, append differentiator
        name = info["name"]
        if name in seen_names and seen_names[name] != domain:
            name = f"{name} ({info['category']})"
        seen_names[name] = domain
        info["unique_name"] = name
        
        # Build text string to represent semantic content of the item for embedding
        embedding_text = f"{name} - {info['category']}. {description}"
        item_texts.append(embedding_text)
        
    print(f"Generating embeddings for {len(item_texts)} items...")
    embeddings = embedder.encode(item_texts, show_progress_bar=True).tolist()
    
    for idx, key in enumerate(item_keys):
        info = item_details[key]
        domain, item_id = key
        embedding = embeddings[idx]
        name = info["unique_name"]
        
        try:
            cursor.execute("""
                INSERT INTO items (name, category, domain, description, average_rating, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                info["category"],
                domain,
                info["description"],
                info["avg_rating"],
                json.dumps(embedding)
            ))
            db_id = cursor.lastrowid
            db_item_ids[key] = db_id
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM items WHERE name = ?", (name,))
            row = cursor.fetchone()
            db_item_ids[key] = row[0] if row else None
            
    conn.commit()
    
    # Seeding Real Users with dynamic personas
    print(f"\nGenerating dynamic personas for {len(user_details)} unique real-world users...")
    db_user_ids = {} # (domain, user_id) -> SQLite integer ID
    
    user_keys = list(user_details.keys())
    user_personas = []
    
    for key in user_keys:
        domain, user_id = key
        history = user_details[key]
        persona = generate_user_persona(domain, history, item_details)
        user_personas.append(persona)
        
    print(f"Generating embeddings for {len(user_personas)} user personas...")
    user_embeddings = embedder.encode(user_personas, show_progress_bar=True).tolist()
    
    for idx, key in enumerate(user_keys):
        domain, user_id = key
        user_name = f"{domain} User {user_id[:8]}"
        ratings_only = [item[1] for item in user_details[key]]
        user_mean_rating = sum(ratings_only) / len(ratings_only)
        embedding = user_embeddings[idx]
        persona = user_personas[idx]
        
        try:
            cursor.execute("""
                INSERT INTO users (name, persona, user_mean_rating, persona_embedding)
                VALUES (?, ?, ?, ?)
            """, (
                user_name,
                persona,
                user_mean_rating,
                json.dumps(embedding)
            ))
            db_id = cursor.lastrowid
            db_user_ids[key] = db_id
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
            row = cursor.fetchone()
            db_user_ids[key] = row[0] if row else None
            
    conn.commit()
    
    # Seeding Real Ratings
    print("\nSeeding real-world ratings matrix...")
    seeded_ratings_count = 0
    for user_id, item_id, item_name, category, domain, rating, review_text in raw_records:
        u_key = (domain, user_id)
        i_key = (domain, item_id)
        
        db_user_id = db_user_ids.get(u_key)
        db_item_id = db_item_ids.get(i_key)
        
        if db_user_id and db_item_id:
            try:
                cursor.execute("""
                    INSERT INTO ratings (user_id, item_id, rating, review_text)
                    VALUES (?, ?, ?, ?)
                """, (db_user_id, db_item_id, rating, review_text))
                seeded_ratings_count += 1
            except sqlite3.IntegrityError:
                continue
                
    conn.commit()
    print(f"Successfully seeded {seeded_ratings_count} real-world rating rows.")
    
    # =====================================================================
    # NAIJA-ENRICHED LOCAL CATALOG OVERLAY
    # =====================================================================
    print("\n" + "-" * 50)
    print("PART 2: SEEDING NAIJA-ENRICHED LOCAL CATALOG")
    print("-" * 50)
    
    catalog_items = [
        # --- YELP DOMAIN (Restaurants, Bars, Spots) ---
        {
            "name": "Yellow Chilli Restaurant (Ikeja)",
            "category": "Yelp (Food)",
            "domain": "Yelp",
            "description": "Premium upscale Nigerian traditional restaurant. Famous for its gourmet Seafood Okra, Jollof Rice, Spicy Suya, and legendary Pepper Soup. Excellent ambiance with authentic African decorations, though music can get quite lively in the evenings.",
            "average_rating": 4.5
        },
        {
            "name": "Shiro Lagos (Victoria Island)",
            "category": "Yelp (Food)",
            "domain": "Yelp",
            "description": "High-end Japanese and Pan-Asian restaurant and lounge overlooking the ocean. Elegant, ultra-modern glassmorphic bar, serving premium sushi, Teppanyaki, and craft cocktails. Popular among Victoria Island tech founders, corporate elites, and influencers.",
            "average_rating": 4.7
        },
        {
            "name": "The Place Restaurant (Yaba)",
            "category": "Yelp (Food)",
            "domain": "Yelp",
            "description": "Fast-casual Nigerian diner. Highly affordable and popular spot for students and local tech workers. Famous for quick Jollof Rice with Fried Chicken, Yam Fries, and Asun (spicy goat meat). Extremely loud during peak hours.",
            "average_rating": 3.8
        },
        {
            "name": "Suya Spot (Gbagada)",
            "category": "Yelp (Food)",
            "domain": "Yelp",
            "description": "Authentic local roadside-style Suya joint. Serves hot, incredibly spicy beef and chicken Suya wrapped in old newspapers, heavily seasoned with Yaji pepper, sliced onions, and cabbage. No seating, takeout only.",
            "average_rating": 4.6
        },
        {
            "name": "Cozy Lagos Bookstore Café (Lekki)",
            "category": "Yelp (Spot)",
            "domain": "Yelp",
            "description": "A quiet, peaceful sanctuary. Features walls lined with African and contemporary literature. Serves premium specialty coffee, herbal teas, and fresh pastries. Perfect spot for reading, studying, or soft corporate remote work. Extremely strict silent policy.",
            "average_rating": 4.8
        },
        {
            "name": "Club Quilox (Victoria Island)",
            "category": "Yelp (Spot)",
            "domain": "Yelp",
            "description": "Vibrant, high-octane luxury nightclub. Loud Afrobeats, premium VIP sections, flashy neon strobe lights, and expensive champagne. Open until dawn. The epicentre of Lagos nightlife, but definitely not for those seeking quiet conversation.",
            "average_rating": 4.2
        },
        
        # --- AMAZON DOMAIN (Movies, Series, Electronics) ---
        {
            "name": "King of Boys (Nollywood Movie)",
            "category": "Amazon (Movie)",
            "domain": "Amazon",
            "description": "An iconic, suspenseful Nollywood political thriller directed by Kemi Adetiba. Follows the story of Alhaja Eniola Salami, a businesswoman and philanthropist with a checkered past, drawn into a struggle for political power. High tension, complex characters, and morally-grey themes.",
            "average_rating": 4.8
        },
        {
            "name": "The Wedding Party (Nollywood Movie)",
            "category": "Amazon (Movie)",
            "domain": "Amazon",
            "description": "A hilarious, fast-paced Nollywood romantic comedy. Captures the extravagant, chaotic, and colorful nature of a high-society Nigerian wedding. Packed with family drama, loud traditional mothers, and slapstick humor. Extremely entertaining and feel-good.",
            "average_rating": 4.4
        },
        {
            "name": "Aníkúlápó (Nollywood Movie)",
            "category": "Amazon (Movie)",
            "domain": "Amazon",
            "description": "An epic mystical fantasy drama set in the ancient Oyo Empire. Follows a young traveler who gains mystical powers to resurrect the dead, but is consumed by greed and lust. Rich in Yoruba folklore, gorgeous traditional costumes, and deep moral lessons.",
            "average_rating": 4.5
        },
        {
            "name": "Inception (Sci-Fi Movie)",
            "category": "Amazon (Movie)",
            "domain": "Amazon",
            "description": "A mind-bending, critically-acclaimed sci-fi action thriller directed by Christopher Nolan. Follows a thief who steals corporate secrets through dream-sharing technology, tasked with planting an idea into a CEO's mind. Complex plot, stunning visual effects, and philosophical themes.",
            "average_rating": 4.7
        },
        {
            "name": "Oraimo FreePods 4 (Wireless Earbuds)",
            "category": "Amazon (Electronics)",
            "domain": "Amazon",
            "description": "Extremely popular wireless bluetooth earbuds in Nigeria. Features active noise cancellation (ANC), heavy deep-bass tuning customized for Afrobeats, and a long-lasting battery life. Highly durable, affordable, and perfect for noisy Lagos commutes.",
            "average_rating": 4.3
        },
        
        # --- GOODREADS DOMAIN (Books, Literature) ---
        {
            "name": "Things Fall Apart (Chinua Achebe)",
            "category": "Goodreads (Book)",
            "domain": "Goodreads",
            "description": "The definitive masterpiece of African literature. Chronicles pre-colonial life in Igbo villages and the tragic clash with British colonialism and Christian missionaries, told through the tragic rise and fall of the warrior Okonkwo. Deeply historical, philosophical, and tragic.",
            "average_rating": 4.9
        },
        {
            "name": "Half of a Yellow Sun (Chimamanda Ngozi Adichie)",
            "category": "Goodreads (Book)",
            "domain": "Goodreads",
            "description": "A sweeping, heartbreaking historical fiction novel. Set during the Nigerian-Biafran Civil War of the late 1960s, it details the lives of five characters caught in the emotional and physical horrors of war, love, loyalty, and survival. Masterfully written, highly emotional.",
            "average_rating": 4.8
        },
        {
            "name": "Death and the King's Horseman (Wole Soyinka)",
            "category": "Goodreads (Book)",
            "domain": "Goodreads",
            "description": "A powerful traditional play by Nobel Laureate Wole Soyinka. Based on a real historical incident, it explores the spiritual and philosophical duty of a king's horseman to commit ritual suicide, and the catastrophic colonial intervention that disrupts the cosmic order. Rich Yoruba philosophy.",
            "average_rating": 4.6
        },
        {
            "name": "Stay with Me (Ayòbámi Adébáyò)",
            "category": "Goodreads (Book)",
            "domain": "Goodreads",
            "description": "An emotional domestic drama set in 1980s Nigeria. Follows a young married couple dealing with the intense societal and family pressure of childlessness, exploring themes of grief, jealousy, sickle-cell disease, and maternal love. Poignant and deeply moving.",
            "average_rating": 4.4
        },
        {
            "name": "The Lean Startup (Eric Ries)",
            "category": "Goodreads (Book)",
            "domain": "Goodreads",
            "description": "A highly influential business and technology book. Outlines a scientific approach to creating and managing successful startups in an age when companies need to innovate more than ever. Focuses on rapid prototyping, validated learning, and iterative product releases.",
            "average_rating": 4.5
        }
    ]
    
    print("Encoding and seeding local catalog items...")
    for item in catalog_items:
        embedding_text = f"{item['name']} - {item['category']}. {item['description']}"
        embedding = embedder.encode(embedding_text).tolist()
        
        try:
            cursor.execute("""
                INSERT INTO items (name, category, domain, description, average_rating, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item["name"],
                item["category"],
                item["domain"],
                item["description"],
                item["average_rating"],
                json.dumps(embedding)
            ))
            print(f"  [Local Item] Seeded: {item['name']}")
        except sqlite3.IntegrityError:
            continue
            
    conn.commit()
    
    # User personas
    users = [
        {
            "name": "Kunle (VI Tech Bro)",
            "persona": "A 26-year-old software engineer living in Lekki and working in Victoria Island. Tech-optimist, loves clean minimalist spaces, specialty espresso coffee, active noise cancelling tech, and fine dining like Japanese Sushi. Reads startup books, loves sci-fi movies, and has a high, generous rating scale (usually averages 4.5).",
            "user_mean_rating": 4.5
        },
        {
            "name": "Mr. Okeke (Strict Nigerian Dad)",
            "persona": "A 58-year-old retired civil servant in Enugu. Extremely traditional, conservative, and critical. Dislikes noisy crowds, neon lights, loud music, or overpriced 'modern' food. Prefers heavy traditional meals (Pounded Yam and bitterleaf soup), serious historical novels about colonialism, and tense Nollywood political drama movies. Extremely harsh rater (never gives a 5, average rating is 3.1).",
            "user_mean_rating": 3.1
        },
        {
            "name": "Teni (Lagos Gen-Z Influencer)",
            "persona": "A 22-year-old social media content creator in Lekki. Extremely dramatic, expressive, and uses heavy local youth slang ('No cap', 'God when', 'It is giving', 'Wahala'). Loves aesthetic trendy brunch spots, luxury clubs, romance movies, and modern feminist poetry or self-help books. Gives highly emotional ratings (either 5.0 or 1.0, averages 4.0).",
            "user_mean_rating": 4.0
        }
    ]
    
    print("\nEncoding and seeding local user personas...")
    for user in users:
        embedding = embedder.encode(user["persona"]).tolist()
        try:
            cursor.execute("""
                INSERT INTO users (name, persona, user_mean_rating, persona_embedding)
                VALUES (?, ?, ?, ?)
            """, (
                user["name"],
                user["persona"],
                user["user_mean_rating"],
                json.dumps(embedding)
            ))
            print(f"  [Local User] Seeded: {user['name']}")
        except sqlite3.IntegrityError:
            continue
            
    conn.commit()
    
    # Re-fetch local user/item IDs from database to overlay historical ratings
    cursor.execute("SELECT id, name FROM users")
    db_users = {row["name"]: row["id"] for row in cursor.fetchall()}
    
    cursor.execute("SELECT id, name FROM items")
    db_items = {row["name"]: row["id"] for row in cursor.fetchall()}
    
    historical_ratings = [
        # --- Kunle (VI Tech Bro) Ratings ---
        {
            "user": "Kunle (VI Tech Bro)",
            "item": "Shiro Lagos (Victoria Island)",
            "rating": 5.0,
            "review": "Absolutely mental! Shiro never disappoints. The ocean view, the elegant glassmorphic lights, and the sushi is elite. Met three other tech founders here. Best spot in VI, no cap."
        },
        {
            "user": "Kunle (VI Tech Bro)",
            "item": "The Lean Startup (Eric Ries)",
            "rating": 5.0,
            "review": "A bible for product engineering. We used the validation loop in our last app build and it completely changed our speed. Must-read for any VI founder."
        },
        {
            "user": "Kunle (VI Tech Bro)",
            "item": "Inception (Sci-Fi Movie)",
            "rating": 4.5,
            "review": "Christopher Nolan is a pure genius. The multi-layered dream logic and visual effects are breathtaking. Easily one of my favorite sci-fi movies of all time."
        },
        {
            "user": "Kunle (VI Tech Bro)",
            "item": "Oraimo FreePods 4 (Wireless Earbuds)",
            "rating": 4.0,
            "review": "The Active Noise Cancellation is surprisingly decent. Blocked out most of the yellow bus honking on my VI commute. The bass is heavily tuned for Afrobeats. Good value for money."
        },
        
        # --- Mr. Okeke (Strict Dad) Ratings ---
        {
            "user": "Mr. Okeke (Strict Nigerian Dad)",
            "item": "Things Fall Apart (Chinua Achebe)",
            "rating": 4.0,
            "review": "Chinua Achebe wrote a monumental piece of history. A profound tragedy that captures the dignity of our pre-colonial ancestors and the destructive forces of foreign intervention. Every child should read this."
        },
        {
            "user": "Mr. Okeke (Strict Nigerian Dad)",
            "item": "King of Boys (Nollywood Movie)",
            "rating": 3.5,
            "review": "A suspenseful Nollywood film with heavy moral gravity. Alhaja Salami's character shows the tragic consequences of absolute greed and political ambition. It is quite long, but teaches a good lesson."
        },
        {
            "user": "Mr. Okeke (Strict Nigerian Dad)",
            "item": "Club Quilox (Victoria Island)",
            "rating": 1.0,
            "review": "An absolute den of chaos. Loud, offensive music that rattles the eardrums, flashy neon lights that cause headaches, and young girls dressed inappropriately buying expensive drinks. Complete waste of time and money."
        },
        {
            "user": "Mr. Okeke (Strict Nigerian Dad)",
            "item": "Shiro Lagos (Victoria Island)",
            "rating": 2.0,
            "review": "Extremely overpriced and the food portions are tiny. I do not understand why our children buy small slices of raw cold fish for the price of a full goat. Ambiance is okay, but too dark."
        },
        
        # --- Teni (Lagos Gen-Z Influencer) Ratings ---
        {
            "user": "Teni (Lagos Gen-Z Influencer)",
            "item": "Club Quilox (Victoria Island)",
            "rating": 5.0,
            "review": "Omo! The energy at Quilox is absolutely top-tier! Afrobeats was hitting different and the aesthetics are giving premium luxury. God when will I meet my billionaire husband here? 10/10!"
        },
        {
            "user": "Teni (Lagos Gen-Z Influencer)",
            "item": "The Wedding Party (Nollywood Movie)",
            "rating": 5.0,
            "review": "This movie had me crying laughing! It is literally my family in a wedding. The mothers are so extra, the drama is giving pure chaotic Nigerian wedding vibes. Obsessed!"
        },
        {
            "user": "Teni (Lagos Gen-Z Influencer)",
            "item": "The Place Restaurant (Yaba)",
            "rating": 3.0,
            "review": "The Jollof is decent, but the queue and noise? Absolute wahala. Not aesthetic at all, do not come here if you want to take Instagram pictures. Just buy and leave."
        }
    ]
    
    print("\nSeeding local historical rating matrix...")
    for rating in historical_ratings:
        user_name = rating["user"]
        item_name = rating["item"]
        
        if user_name not in db_users or item_name not in db_items:
            continue
            
        user_id = db_users[user_name]
        item_id = db_items[item_name]
        
        try:
            cursor.execute("""
                INSERT INTO ratings (user_id, item_id, rating, review_text)
                VALUES (?, ?, ?, ?)
            """, (user_id, item_id, rating["rating"], rating["review"]))
            print(f"  [Local Rating] {user_name} -> {item_name} ({rating['rating']} stars)")
        except sqlite3.IntegrityError:
            continue
            
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("HYBRID CATALOG DATABASE SEEDING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    seed_data()
