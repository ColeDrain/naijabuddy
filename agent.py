import json
import os
import sys

# Configure environment variables to find cached model folders
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "hf_home"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(MODELS_DIR, "sentence_transformers"))

import database

# Tunable blending weight hyperparameter (30% LLM reasoning, 70% statistical grounding)
ALPHA = float(os.getenv("NAIJABUDDY_ALPHA", 0.3))

class NaijaBuddyAgent:
    def __init__(self, model_path=None):
        """Initializes the Agent. Loads the local GGUF model using llama-cpp-python if available."""
        self.llm = None
        self.alpha = ALPHA
        
        # Determine the GGUF path
        if not model_path:
            model_path = os.getenv("NAIJABUDDY_MODEL_PATH") or os.path.join(MODELS_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")
            
        if os.path.exists(model_path):
            try:
                print(f"Loading local GGUF LLM from: {model_path}...")
                from llama_cpp import Llama
                
                # Attempt to initialize with macOS Metal GPU offloading & Flash Attention for 6x speedup
                # Attempt to initialize with GPU offloading
                try:
                    try:
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=2048,
                            n_gpu_layers=-1,
                            flash_attn=True,
                            verbose=False
                        )
                        print("Local GGUF LLM loaded with GPU acceleration and Flash Attention!")
                    except Exception as fa_err:
                        print(f"GPU initialization with Flash Attention failed ({fa_err}). Trying GPU without Flash Attention...")
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=2048,
                            n_gpu_layers=-1,
                            flash_attn=False,
                            verbose=False
                        )
                        print("Local GGUF LLM loaded with GPU acceleration (no Flash Attention)!")
                except Exception as gpu_err:
                    print(f"GPU initialization failed ({gpu_err}), falling back to CPU baseline...")
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=4,
                        verbose=False
                    )
                    print("Local GGUF LLM loaded on CPU baseline.")
            except Exception as e:
                print(f"Warning: Failed to load llama-cpp-python or GGUF: {e}")
                print("Agent will run in 'Mock-Fallback' mode for development UI testing.")
        else:
            print(f"Warning: GGUF model file not found at: {model_path}")
            print("Agent will run in 'Mock-Fallback' mode for development UI testing.")

    # =====================================================================
    # OUTPUT CALIBRATION LAYER (The Mathematical Defense)
    # =====================================================================
    
    def get_calibrated_rating(self, raw_llm_rating, user_id, user_name, persona_embedding_str, alpha=None):
        """
        Calculates the calibrated rating to protect our RMSE score.
        Uses warm-user historical mean or falls back to cold-start Cluster Mean.
        """
        # Determine alpha to use (request-level takes priority over agent default)
        current_alpha = alpha if alpha is not None else self.alpha
        
        # Ensure raw LLM rating is bounded
        raw_llm_rating = max(1.0, min(5.0, float(raw_llm_rating)))
        
        # 1. Warm-User Path: Retrieve historical ratings from database
        user_mean = database.get_user_ratings_mean(user_id) if user_id else None
        
        if user_mean is not None:
            calibrated_rating = (current_alpha * raw_llm_rating) + ((1.0 - current_alpha) * user_mean)
            print(f"  [Calibration] Warm User '{user_name}': Raw LLM: {raw_llm_rating:.2f} | User Mean: {user_mean:.2f} -> Calibrated: {calibrated_rating:.4f}")
            return round(calibrated_rating, 2)
            
        # 2. Cold-Start Path: Vector-based Cluster Mean Fallback
        # If user is new (has zero ratings), find similar user personas and calculate their average mean rating
        if persona_embedding_str:
            try:
                embedding = json.loads(persona_embedding_str)
                nearest_users = database.get_nearest_users(embedding, top_k=5)
                
                if nearest_users:
                    cluster_mean = sum(u["user_mean_rating"] for u in nearest_users) / len(nearest_users)
                    calibrated_rating = (current_alpha * raw_llm_rating) + ((1.0 - current_alpha) * cluster_mean)
                    print(f"  [Calibration] Cold-Start User: Raw LLM: {raw_llm_rating:.2f} | Cluster Mean (K=5): {cluster_mean:.2f} -> Calibrated: {calibrated_rating:.4f}")
                    return round(calibrated_rating, 2)
            except Exception as e:
                print(f"Error executing Cold-Start Cluster Mean: {e}")
                
        # 3. Absolute Fallback: Global baseline rating (typically 3.5)
        global_fallback = 3.5
        calibrated_rating = (current_alpha * raw_llm_rating) + ((1.0 - current_alpha) * global_fallback)
        print(f"  [Calibration] Fallback: Raw LLM: {raw_llm_rating:.2f} | Global Baseline: {global_fallback:.2f} -> Calibrated: {calibrated_rating:.4f}")
        return round(calibrated_rating, 2)

    # =====================================================================
    # CRITIC LAYER (Logical Collaborative Constraints)
    # =====================================================================
    
    def apply_critic_rules(self, ranked_items, user_persona_text):
        """
        Applies a deterministic 'Critic Layer' over the recommended items to
        filter out obvious logical anomalies or preference violations.
        """
        cleaned_items = []
        user_persona_lower = user_persona_text.lower()
        
        is_vegetarian = "vegetarian" in user_persona_lower or "vegan" in user_persona_lower
        is_strict_parent = "strict" in user_persona_lower or "parent" in user_persona_lower or "dad" in user_persona_lower or "retired" in user_persona_lower
        
        for item in ranked_items:
            item_desc_lower = item.get("description", "").lower()
            item_name_lower = item.get("name", "").lower()
            
            # Rule 1: Strict vegetarian or vegan constraint check
            if is_vegetarian:
                # If recommending a food spot that is heavily non-vegetarian
                if "suya" in item_name_lower or "meat" in item_desc_lower or "beef" in item_desc_lower:
                    print(f"  [Critic] Flagged meat item '{item['name']}' for Vegetarian persona. Pushing to bottom.")
                    item["critic_penalty"] = True
                    item["explanation"] += " (Note: Critic flagged this as heavily meat-focused, purchase with caution)."
                    
            # Rule 2: Strict/conservative parent constraint check
            if is_strict_parent:
                # Dislikes loud clubs, wild neon spots
                if "club" in item_name_lower or "quilox" in item_name_lower or "neon" in item_desc_lower or "loud afrobeats" in item_desc_lower:
                    print(f"  [Critic] Flagged loud venue '{item['name']}' for Strict Parent persona. Pushing to bottom.")
                    item["critic_penalty"] = True
                    item["explanation"] += " (Note: Critic flagged this spot as potentially too loud or chaotic)."
                    
            cleaned_items.append(item)
            
        # Re-sort to push items with 'critic_penalty' to the absolute bottom of the list
        cleaned_items.sort(key=lambda x: 1 if x.get("critic_penalty") else 0)
        return cleaned_items

    def synthesize_and_update_persona(self, user):
        """
        Uses the LLM to synthesize a natural, coherent 2-sentence character profile
        from the raw, structured interaction history, and caches it in SQLite.
        """
        print(f"  [Lazy Synthesis] Generating high-fidelity LLM persona for '{user['name']}'...")
        conn = database.get_connection()
        cursor = conn.cursor()
        
        # Retrieve this user's historical ratings and reviews to summarize
        cursor.execute("""
            SELECT i.name, i.category, r.rating, r.review_text 
            FROM ratings r
            JOIN items i ON r.item_id = i.id
            WHERE r.user_id = ?
        """, (user["id"],))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return user
            
        # Cap the history fed to the LLM so the synthesis prompt stays within
        # the context window regardless of how long a user's history is. For
        # long histories we keep the most opinionated ratings (highest and
        # lowest) so the synthesised persona captures the user's taste range.
        MAX_HISTORY = 15
        if len(rows) > MAX_HISTORY:
            ordered = sorted(rows, key=lambda r: r["rating"])
            low = ordered[: MAX_HISTORY // 2]
            high = ordered[-(MAX_HISTORY - len(low)):]
            rows = high + low

        # Formulate history narrative
        history_lines = []
        for r in rows:
            review = (r["review_text"] or "").strip().replace("\n", " ")
            if len(review) > 200:
                review = review[:200] + "..."
            history_lines.append(f"- Item: {r['name']} ({r['category']}) | Rating: {r['rating']}/5 | Review: '{review}'")
        history_str = "\n".join(history_lines)
        
        prompt = f"""<|im_start|>system
You are a expert user profiling assistant. Your goal is to synthesize a cohesive, natural, and concise 2-sentence description of a user's tastes and expectations based on their review history.
Write in a fluent, natural character description style (e.g., 'An avid reader who appreciates deep historical context...'). Do not use rigid templates or list items.

USER REVIEWS & RATING HISTORY:
{history_str}

OUTPUT FORMAT:
Return ONLY the 2-sentence persona. No markdown backticks, JSON, or extra conversational text.
<|im_end|>
<|im_start|>assistant
"""
        synthesized = user["persona"]
        if self.llm:
            try:
                response = self.llm(prompt, max_tokens=192, temperature=0.3)
                synthesized = response["choices"][0]["text"].strip()
            except Exception as e:
                print(f"  [Lazy Synthesis] LLM inference failed: {e}")
                
        # Generate new vector embedding for the synthesized persona
        if not hasattr(self, "embedder"):
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            
        print(f"  [Lazy Embedding] Generating new embedding vector for synthesized persona...")
        new_embedding = self.embedder.encode(synthesized).tolist()
        
        # Save cache to SQLite database
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET persona = ?, persona_embedding = ? 
            WHERE id = ?
        """, (synthesized, json.dumps(new_embedding), user["id"]))
        conn.commit()
        conn.close()
        print(f"  [Lazy Synthesis] Successfully updated database cache for '{user['name']}'!")
        
        # Return updated user dictionary
        updated_user = dict(user)
        updated_user["persona"] = synthesized
        updated_user["persona_embedding"] = json.dumps(new_embedding)
        return updated_user

    # =====================================================================
    # TASK A: REVIEW SIMULATION ENGINE
    # =====================================================================

    def simulate_review(self, user_name, item_id, alpha=None):
        """Generates a localized, calibrated star rating and review."""
        # 1. Fetch user and item details from database
        user = database.get_user_by_name(user_name)
        
        if user and user["persona"].startswith("A real "):
            user = self.synthesize_and_update_persona(user)
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        item_row = cursor.fetchone()
        conn.close()
        
        if not user or not item_row:
            return {"error": "User or Item not found."}
            
        item = dict(item_row)
        
        # 2. Formulate Prompt
        prompt = f"""<|im_start|>system
You are a highly advanced simulation agent modeling a specific human persona.
Your objective is to generate an authentic, context-aware star rating and written review for a product or service.

USER PERSONA PROFILE:
{user['persona']}

TARGET PRODUCT/SERVICE DETAILS:
- Name: {item['name']}
- Category: {item['category']}
- Specific Details: {item['description']}

CULTURAL STYLE GUIDELINES:
The target audience of this evaluation resides in Nigeria. Adjust your communication style, tone, vocabulary, and references to sound exactly like a real person belonging to this persona in Nigeria. Use authentic local vocabulary, slang, and cultural references naturally (e.g., "Abeg", "God when", "Wahala", "No cap", "Strict Nigerian Parent" style, "VI Tech Bro" jargon) where appropriate.

CRITICAL CONSTRAINTS FOR ROUGE OVERLAP:
1. Keep the review concise, realistic, and highly readable (2 to 4 sentences max).
2. Do not use overly poetic or academic vocabulary. Use standard everyday review keywords (e.g., "food", "service", "clean", "good", "bad", "spicy", "price").
3. Do not mention that you are an AI or an agent.

OUTPUT FORMAT (Strict JSON):
Return ONLY a valid JSON object. Do not include any markdown backticks or extra text outside the JSON.
{{
  "rating": [Generate a realistic float rating between 1.0 and 5.0 based strictly on how this persona would rate this item],
  "review": "[Write the simulated review text here]"
}}
<|im_end|>
<|im_start|>assistant
"""
        
        # 3. Execute inference using Local LLM (or Fallback if not loaded)
        raw_output = ""
        if self.llm:
            try:
                # Low temperature for deterministic, structured output conforming to constraints
                response = self.llm(prompt, max_tokens=256, temperature=0.2)
                raw_output = response["choices"][0]["text"].strip()
            except Exception as e:
                print(f"Llama-cpp inference error: {e}. Falling back to mock...")
                raw_output = self._get_mock_review_json(user_name, item["name"])
        else:
            raw_output = self._get_mock_review_json(user_name, item["name"])
            
        # Parse output JSON
        try:
            # Simple cleanup to remove any potential markdown wrapper blocks
            if "```" in raw_output:
                raw_output = raw_output.split("```json")[-1].split("```")[0].strip()
            parsed_json = json.loads(raw_output)
            raw_rating = float(parsed_json.get("rating", 4.0))
            review_text = parsed_json.get("review", "Decent experience.")
        except Exception as e:
            print(f"JSON parsing error on LLM output: {e}. Raw content was: {raw_output}")
            raw_rating = 4.0
            review_text = "Standard review text."
            
        # 4. Apply Output Calibration Layer (Mathematically adjust the score)
        if alpha is None:
            domain_lower = item.get("domain", "").lower()
            if "yelp" in domain_lower:
                alpha = 0.3
            elif "goodreads" in domain_lower:
                alpha = 0.1
            elif "amazon" in domain_lower:
                alpha = 0.2
            else:
                alpha = self.alpha

        calibrated_rating = self.get_calibrated_rating(
            raw_llm_rating=raw_rating,
            user_id=user["id"],
            user_name=user_name,
            persona_embedding_str=user["persona_embedding"],
            alpha=alpha
        )
        
        return {
            "user_name": user_name,
            "item_name": item["name"],
            "raw_rating": raw_rating,
            "calibrated_rating": calibrated_rating,
            "review": review_text
        }

    # =====================================================================
    # TASK B: RECOMENDER (Recall, Rerank, and Justify)
    # =====================================================================

    def recommend_items(self, user_name, domain, top_k=10):
        """Runs the entire Stage-1 Recall and Stage-2 Rerank recommender pipeline."""
        import time
        start_time = time.time()
        print(f"[{user_name}] Starting recommend_items...")
        
        # 1. Retrieve user
        user = database.get_user_by_name(user_name)
        if not user:
            return {"error": f"User persona '{user_name}' not found."}
            
        if user["persona"].startswith("A real "):
            user = self.synthesize_and_update_persona(user)
            
        t1 = time.time()
        print(f"[{user_name}] User retrieved in {t1 - start_time:.4f}s")
        
        # 2. Stage-1 Recall: Dense semantic vector search with Hybrid Retrieval (retrieves 10 candidates)
        user_embedding = json.loads(user["persona_embedding"])
        candidates = database.get_nearest_items(user_embedding, domain, top_k=10, user_id=user["id"])
        
        t2 = time.time()
        print(f"[{user_name}] Stage-1 Recall (Vector Search) finished in {t2 - t1:.4f}s")
        
        if not candidates:
            return {"error": f"No candidates found in domain '{domain}'."}
            
        # 3. Stage-2 Rerank: Formulate Prompt
        # Format candidates as a clean JSON catalog for the LLM to inspect
        candidates_json_str = json.dumps([
            {"id": c["id"], "name": c["name"], "category": c["category"], "description": c["description"]}
            for c in candidates
        ], indent=2)
        
        prompt = f"""<|im_start|>system
You are an elite, context-aware recommendation routing agent.
Your objective is to select, rank, and explain the top 5 most relevant items for a user from a candidate list based on their persona profile and historical preferences.

USER PERSONA PROFILE:
{user['persona']}

TARGET DOMAIN: {domain}

CANDIDATE LIST OF ITEMS (Top 10 from Semantic Search):
{candidates_json_str}

CRITICAL CONSTRAINTS:
1. Select the top 5 most relevant items from the 10 candidates and rank them from 1 (most relevant) to 5 (fifth most relevant).
2. Filter out any candidate items that violate strict user constraints (e.g., recommending pork or beef to a strict vegetarian, or loud parties to an introverted parent).
3. For each of the top 5 selected items, provide a persuasive, extremely concise (exactly 1 sentence, maximum 15 words) natural language explanation of WHY this was recommended to this persona. Highlight the exact features of the item that match the persona's core tastes.
4. Adjust your explanation tone to sound authentic to the user's cultural context (Nigeria).

OUTPUT FORMAT (Strict JSON):
Return ONLY a valid JSON array of exactly 5 objects. Do not include any extra text or markdown backticks outside the JSON.
[
  {{
    "id": [item_id],
    "name": "[item_name]",
    "rank": [1 to 5],
    "explanation": "[Write your short explanation here]"
  }},
  ...
]
<|im_end|>
<|im_start|>assistant
"""
        
        # Execute LLM Reranking
        raw_output = ""
        t3 = time.time()
        if self.llm:
            try:
                response = self.llm(prompt, max_tokens=1024, temperature=0.1)
                raw_output = response["choices"][0]["text"].strip()
            except Exception as e:
                print(f"Llama-cpp inference error: {e}. Falling back to mock reranker...")
                raw_output = self._get_mock_recommend_json(user_name, candidates)
        else:
            raw_output = self._get_mock_recommend_json(user_name, candidates)
            
        t4 = time.time()
        print(f"[{user_name}] Stage-2 Rerank (LLM Inference) finished in {t4 - t3:.4f}s")
            
        # Parse output JSON array (with truncated JSON repair defense)
        try:
            if "```" in raw_output:
                raw_output = raw_output.split("```json")[-1].split("```")[0].strip()
            
            # Defensive JSON repair: check if JSON is truncated (starts with [ but doesn't end with ])
            if raw_output.startswith("[") and not raw_output.endswith("]"):
                cleaned = raw_output.strip()
                if cleaned.endswith("}"):
                    cleaned += "]"
                    raw_output = cleaned
                else:
                    # Find the last occurrence of } to discard any incomplete trailing object
                    last_brace = cleaned.rfind("}")
                    if last_brace != -1:
                        cleaned = cleaned[:last_brace+1] + "]"
                        raw_output = cleaned
                        print(f"  [JSON Repair] Repaired truncated raw LLM response. Cleaned content ends at index {last_brace}.")
                        
            reranked_list = json.loads(raw_output)
        except Exception as e:
            print(f"JSON parsing error on Rerank output: {e}. Raw content: {raw_output}")
            # Dynamic fallback: keep the original candidate list order and generate empty explanations
            reranked_list = [
                {"id": c["id"], "name": c["name"], "rank": idx+1, "explanation": "Recommended based on your profile."}
                for idx, c in enumerate(candidates)
            ]
            
        # 4. Enrich reranked items with original catalog metadata (descriptions, categories)
        # Create helper map from candidates
        candidates_map = {c["id"]: c for c in candidates}
        enriched_recommendations = []
        
        for item in reranked_list:
            item_id = int(item.get("id"))
            if item_id in candidates_map:
                orig_item = candidates_map[item_id]
                enriched_recommendations.append({
                    "id": item_id,
                    "name": orig_item["name"],
                    "category": orig_item["category"],
                    "description": orig_item["description"],
                    "average_rating": orig_item["average_rating"],
                    "similarity": orig_item["similarity"],
                    "rank": len(enriched_recommendations) + 1,
                    "explanation": item.get("explanation", "Recommended based on semantic profile similarity.")
                })
                
        # 5. Apply Critic Layer (Deterministic Safety Filter)
        final_recommendations = self.apply_critic_rules(enriched_recommendations, user["persona"])
        
        return final_recommendations[:top_k]

    # =====================================================================
    # DEVELOPMENT Fallbacks (Zero-dependencies testing)
    # =====================================================================

    def _get_mock_review_json(self, user_name, item_name):
        """Generates localized mock reviews if local GGUF is loading or compiling."""
        if "Kunle" in user_name:
            return json.dumps({
                "rating": 5.0,
                "review": f"Honestly, {item_name} is absolutely premium. Minimalist, premium clean vibes and top-tier service. No cap, this is exactly what we need in VI."
            })
        elif "Okeke" in user_name:
            return json.dumps({
                "rating": 2.0,
                "review": f"This is what our children call modern? {item_name} is noisy, highly overpriced, and serves very small portions. A total waste of money. I do not recommend."
            })
        else: # Teni
            return json.dumps({
                "rating": 5.0,
                "review": f"Omo! {item_name} is giving everything it was supposed to give! The aesthetics are absolutely beautiful, perfect for my pictures. God when?!"
            })

    def _get_mock_recommend_json(self, user_name, candidates):
        """Generates mock reranking items if local GGUF is loading or compiling."""
        # Just return the top candidates sorted by similarity with custom persona explanations
        results = []
        for idx, c in enumerate(candidates[:10]):
            exp = "This matches your profile."
            if "Kunle" in user_name:
                exp = f"This fits your premium, high-tech, and modern aesthetic preferences in Lekki/VI."
            elif "Okeke" in user_name:
                exp = f"Recommended for its quiet environment and high educational/philosophical substance."
            elif "Teni" in user_name:
                exp = f"This is giving major aesthetic vibes, perfect for creating Instagram content and networking with premium people."
                
            results.append({
                "id": c["id"],
                "name": c["name"],
                "rank": idx + 1,
                "explanation": exp
            })
        return json.dumps(results)

if __name__ == "__main__":
    # Quick debug test of agent Fallback Mode
    agent = NaijaBuddyAgent()
    print("\nSimulating local review fallback...")
    res_a = agent.simulate_review("Kunle (VI Tech Bro)", 1) # Yellow Chilli
    print(json.dumps(res_a, indent=2))
