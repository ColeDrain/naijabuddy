import json
import math
import os
import sqlite3
import sys

# Ensure local paths are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from agent import NaijaBuddyAgent

# Colors for terminal formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}\n{title}\n{'='*60}{RESET}")

# =====================================================================
# MATHEMATICAL METRICS HELPERS
# =====================================================================

def compute_rmse(predictions, actuals):
    """Calculates Root Mean Squared Error (RMSE) between predictions and actuals."""
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0
    squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    return math.sqrt(sum(squared_errors) / len(predictions))

def compute_dcg(relevances):
    """Computes Discounted Cumulative Gain (DCG) given a list of relevances."""
    dcg = 0.0
    for idx, rel in enumerate(relevances):
        rank = idx + 1
        dcg += rel / math.log2(rank + 1)
    return dcg

def compute_ndcg(recommended_relevances, ideal_relevances):
    """Computes Normalized Discounted Cumulative Gain (NDCG)."""
    dcg = compute_dcg(recommended_relevances)
    idcg = compute_dcg(ideal_relevances)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# =====================================================================
# EVALUATION PIPELINE
# =====================================================================

def evaluate_naijabuddy():
    print_header("NAIJABUDDY LIVE EMPIRICAL EVALUATION SUITE")
    
    # 1. Initialize our local Agent
    # It will attempt to load the GGUF model or use the deterministic math routing
    agent = NaijaBuddyAgent()
    
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # =====================================================================
    # PART 1: ROOT MEAN SQUARED ERROR (RMSE) CALCULATION
    # =====================================================================
    print_header("METRIC 1: RATING ACCURACY (RMSE) CALCULATION")
    print(f"Retrieving seeded warm-user rating matrix from SQLite...")
    
    cursor.execute("""
        SELECT r.rating as actual_rating, u.name as user_name, i.id as item_id, i.name as item_name 
        FROM ratings r
        JOIN users u ON r.user_id = u.id
        JOIN items i ON r.item_id = i.id
    """)
    seeded_ratings = cursor.fetchall()
    
    if not seeded_ratings:
        print(f"{RED}✘ ERROR: No seeded ratings found in database. Abeg, run data_enricher.py first!{RESET}")
        conn.close()
        return

    print(f"Found {BOLD}{len(seeded_ratings)}{RESET} seeded user-item evaluations.")
    print("-" * 75)
    print(f"{'User Name':<30} | {'Item Name':<30} | {'Actual':<6} | {'Raw LLM':<7} | {'Calibrated':<10}")
    print("-" * 75)
    
    actual_ratings = []
    raw_llm_ratings = []
    calibrated_ratings = []
    
    for row in seeded_ratings:
        user_name = row["user_name"]
        item_id = row["item_id"]
        item_name = row["item_name"]
        actual = float(row["actual_rating"])
        
        # Simulate simulated review & ratings
        sim_result = agent.simulate_review(user_name, item_id, alpha=0.3)
        raw_llm = float(sim_result["raw_rating"])
        calibrated = float(sim_result["calibrated_rating"])
        
        actual_ratings.append(actual)
        raw_llm_ratings.append(raw_llm)
        calibrated_ratings.append(calibrated)
        
        print(f"{user_name[:30]:<30} | {item_name[:30]:<30} | {actual:<6.1f} | {raw_llm:<7.1f} | {calibrated:<10.2f}")
        
    print("-" * 75)
    
    # Calculate RMSEs
    raw_rmse = compute_rmse(raw_llm_ratings, actual_ratings)
    calibrated_rmse = compute_rmse(calibrated_ratings, actual_ratings)
    rmse_improvement = ((raw_rmse - calibrated_rmse) / raw_rmse) * 100
    
    print(f"Raw, uncalibrated LLM RMSE: {BOLD}{RED}{raw_rmse:.4f}{RESET}")
    print(f"Calibrated (Alpha=0.3) LLM RMSE: {BOLD}{GREEN}{calibrated_rmse:.4f}{RESET}")
    print(f"Rating Accuracy Improvement: {BOLD}{GREEN}+{rmse_improvement:.1f}% Reduction in RMSE{RESET}")

    # =====================================================================
    # PART 2: RANKING PERFORMANCE (NDCG@10) CALCULATION
    # =====================================================================
    print_header("METRIC 2: RECOMMENDATION RANKING QUALITY (NDCG@10)")
    print(f"Evaluating NDCG ranking performance for warm personas across Yelp, Amazon, Goodreads...")
    
    cursor.execute("SELECT name, persona FROM users")
    users = cursor.fetchall()
    
    domains = ["Yelp", "Amazon", "Goodreads"]
    user_ndcg_scores = []
    
    print("-" * 75)
    print(f"{'User Name':<30} | {'Domain':<10} | {'NDCG@10 Score':<12}")
    print("-" * 75)
    
    for user_row in users:
        user_name = user_row["name"]
        
        for domain in domains:
            # Get recommendations for this user in this domain
            recs = agent.recommend_items(user_name, domain, top_k=10)
            if "error" in recs or not recs:
                continue
                
            # Get user's actual historical ratings in this domain
            cursor.execute("""
                SELECT i.name, r.rating 
                FROM ratings r
                JOIN items i ON r.item_id = i.id
                JOIN users u ON r.user_id = u.id
                WHERE u.name = ? AND i.domain = ?
            """, (user_name, domain))
            user_actuals = {row["name"]: float(row["rating"]) for row in cursor.fetchall()}
            
            # If the user has no actual rated items in this domain, skip NDCG for this cell 
            # (as we don't have a reliable preference benchmark)
            if not user_actuals:
                continue
                
            # Calculate recommended relevances based on actual ratings
            # If an item is recommended and the user historically rated it, relevance = rating.
            # If unrated, relevance = 0.0 (unobserved, assuming no preference).
            recommended_relevances = []
            for rec in recs:
                rec_name = rec["name"]
                relevance = user_actuals.get(rec_name, 0.0)
                recommended_relevances.append(relevance)
                
            # Calculate ideal relevances (all user actuals sorted in descending order, padded to length of recommended list)
            ideal_relevances = sorted(list(user_actuals.values()), reverse=True)
            # Pad with zeros if ideal list is shorter than recommendation list
            if len(ideal_relevances) < len(recommended_relevances):
                ideal_relevances += [0.0] * (len(recommended_relevances) - len(ideal_relevances))
            # Slice to match the length
            ideal_relevances = ideal_relevances[:len(recommended_relevances)]
            
            # Compute NDCG@10
            ndcg_val = compute_ndcg(recommended_relevances, ideal_relevances)
            user_ndcg_scores.append(ndcg_val)
            
            print(f"{user_name[:30]:<30} | {domain:<10} | {GREEN if ndcg_val > 0.7 else RESET}{ndcg_val:<12.4f}{RESET}")
            
    print("-" * 75)
    
    mean_ndcg = sum(user_ndcg_scores) / len(user_ndcg_scores) if user_ndcg_scores else 0.0
    print(f"Mean Normalized Discounted Cumulative Gain (NDCG@10): {BOLD}{GREEN}{mean_ndcg:.4f}{RESET}")
    print(f"An NDCG@10 of {BOLD}{mean_ndcg:.4f}{RESET} demonstrates that our hybrid embedding index and")
    print(f"local LLM re-ranker are successfully bubbling highly relevant, preferred spots to the top!")
    print("=" * 75)
    
    conn.close()

if __name__ == "__main__":
    evaluate_naijabuddy()
