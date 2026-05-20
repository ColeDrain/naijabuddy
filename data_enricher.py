import json
import os
import sys

# Configure isolated model environment variables BEFORE importing SentenceTransformer
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "hf_home")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(MODELS_DIR, "sentence_transformers")

import database

def seed_data():
    print("=" * 60)
    print("NAIJABUDDY DATABASE DATA SEEDER")
    print("=" * 60)
    
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
    # NAIJA-ENRICHED CATALOG ITEMS
    # =====================================================================
    # We define high-fidelity, descriptive items across Yelp, Amazon, and Goodreads.
    # Each item includes a name, category, domain, description, and base average rating.
    
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
    
    # Insert items into database
    conn = database.get_connection()
    cursor = conn.cursor()
    
    print("\nEncoding and seeding catalog items...")
    for item in catalog_items:
        # Build text string to represent semantic content of the item for embedding
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
            print(f"  [Item] Seeded: {item['name']}")
        except sqlite3.IntegrityError:
            # Item already exists, skip
            continue
            
    conn.commit()
    
    # =====================================================================
    # HIGH-FIDELITY USER PERSONAS
    # =====================================================================
    # We define our target Nigerian archetypes.
    
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
    
    print("\nEncoding and seeding user personas...")
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
            print(f"  [User] Seeded: {user['name']}")
        except sqlite3.IntegrityError:
            continue
            
    conn.commit()
    
    # =====================================================================
    # HISTORICAL USER-ITEM RATINGS (WARM START SEEDING)
    # =====================================================================
    # We seed historical ratings to demonstrate both warm-user and cold-start math.
    # Ratings align perfectly with user personas to ensure logical baseline averages.
    
    # Fetch user/item IDs from database
    cursor.execute("SELECT id, name FROM users")
    user_ids = {row["name"]: row["id"] for row in cursor.fetchall()}
    
    cursor.execute("SELECT id, name FROM items")
    item_ids = {row["name"]: row["id"] for row in cursor.fetchall()}
    
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
        
        # --- Teni (Gen-Z Influencer) Ratings ---
        {
            "user": "Teni (Gen-Z Influencer)",
            "item": "Club Quilox (Victoria Island)",
            "rating": 5.0,
            "review": "Omo! The energy at Quilox is absolutely top-tier! Afrobeats was hitting different and the aesthetics are giving premium luxury. God when will I meet my billionaire husband here? 10/10!"
        },
        {
            "user": "Teni (Gen-Z Influencer)",
            "item": "The Wedding Party (Nollywood Movie)",
            "rating": 5.0,
            "review": "This movie had me crying laughing! It is literally my family in a wedding. The mothers are so extra, the drama is giving pure chaotic Nigerian wedding vibes. Obsessed!"
        },
        {
            "user": "Teni (Gen-Z Influencer)",
            "item": "The Place Restaurant (Yaba)",
            "rating": 3.0,
            "review": "The Jollof is decent, but the queue and noise? Absolute wahala. Not aesthetic at all, do not come here if you want to take Instagram pictures. Just buy and leave."
        }
    ]
    
    print("\nSeeding historical rating matrix...")
    for rating in historical_ratings:
        user_name = rating["user"]
        item_name = rating["item"]
        
        if user_name not in user_ids or item_name not in item_ids:
            continue
            
        user_id = user_ids[user_name]
        item_id = item_ids[item_name]
        
        try:
            cursor.execute("""
                INSERT INTO ratings (user_id, item_id, rating, review_text)
                VALUES (?, ?, ?, ?)
            """, (user_id, item_id, rating["rating"], rating["review"]))
            print(f"  [Rating] {user_name} -> {item_name} ({rating['rating']} stars)")
        except sqlite3.IntegrityError:
            continue
            
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("HYBRID CATALOG DATABASE SEEDING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    seed_data()
