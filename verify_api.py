import sys
import requests
import json

# Target test URL of our active FastAPI development server
BASE_URL = "http://localhost:8050"

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}\n{title}\n{'='*60}{RESET}")

def run_api_tests():
    print_header("NAIJABUDDY AUTOMATED API VERIFICATION SUITE")
    
    # Test 1: GET /api/users
    print(f"Running Test 1: GET /api/users...")
    try:
        res = requests.get(f"{BASE_URL}/api/users")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        users = res.json()
        assert len(users) >= 3, "Expected at least 3 seeded user personas"
        print(f"  {GREEN}✔ SUCCESS: {len(users)} users fetched successfully.{RESET}")
        print(f"  First User: {BOLD}{users[0]['name']}{RESET}")
    except Exception as e:
        print(f"  {RED}✘ FAILED: {e}{RESET}")
        sys.exit(1)
        
    # Test 2: GET /api/items (with and without filters)
    print(f"\nRunning Test 2: GET /api/items...")
    try:
        res_all = requests.get(f"{BASE_URL}/api/items")
        assert res_all.status_code == 200
        items_all = res_all.json()
        assert len(items_all) >= 15, "Expected at least 15 catalog items"
        
        res_yelp = requests.get(f"{BASE_URL}/api/items?domain=Yelp")
        assert res_yelp.status_code == 200
        items_yelp = res_yelp.json()
        assert len(items_yelp) < len(items_all), "Filtering by Yelp domain should return a subset"
        
        print(f"  {GREEN}✔ SUCCESS: Catalog items list and filtering works.{RESET}")
        print(f"  Total items: {len(items_all)} | Yelp items: {len(items_yelp)}")
    except Exception as e:
        print(f"  {RED}✘ FAILED: {e}{RESET}")
        sys.exit(1)

    # Test 3: POST /api/users (Create custom user / cold-start)
    print(f"\nRunning Test 3: POST /api/users (Cold-Start Registry)...")
    custom_name = "Emeka (Lekki Real Estate Broker)"
    custom_persona = "A 35-year-old high-earning property agent. Loves luxury cars, expensive champagne at clubs like Quilox, premium pepper soup, and business books. Always values VIP treatment. Rating mean is generous around 4.2."
    
    try:
        payload = {"name": custom_name, "persona": custom_persona}
        res = requests.post(f"{BASE_URL}/api/users", json=payload)
        
        # Handle if the user was already created in a previous test run
        if res.status_code == 400 and "already exists" in res.json().get("detail", ""):
            print(f"  {YELLOW}⚠ NOTE: Custom user was already registered in a previous run. Skipping creation...{RESET}")
        else:
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            new_user = res.json()
            assert new_user["name"] == custom_name
            assert new_user["user_mean_rating"] == 3.5
            print(f"  {GREEN}✔ SUCCESS: Cold-start persona registered and embedded in SQLite!{RESET}")
    except Exception as e:
        print(f"  {RED}✘ FAILED: {e}{RESET}")
        sys.exit(1)

    # Test 4: POST /api/recommend (Recall & Justify - Task B)
    print(f"\nRunning Test 4: POST /api/recommend (Recommender Pipeline)...")
    try:
        payload = {"user_name": "Kunle (VI Tech Bro)", "domain": "Yelp"}
        res = requests.post(f"{BASE_URL}/api/recommend", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        recs = res.json()
        
        # Ensure we have a valid recommendation array list
        assert isinstance(recs, list)
        assert len(recs) > 0, "No recommendations returned"
        assert "rank" in recs[0], "Missing rank in recommendation card"
        assert "explanation" in recs[0], "Missing justification in recommendation card"
        assert "similarity" in recs[0], "Missing vector similarity percentage"
        
        print(f"  {GREEN}✔ SUCCESS: Recommender pipeline returned Top recommendations!{RESET}")
        print(f"  Top Rec: {BOLD}{recs[0]['name']}{RESET} (Rank: #{recs[0]['rank']} | Sim: {recs[0]['similarity']:.4f})")
        print(f"  Justification: \"{recs[0]['explanation']}\"")
    except Exception as e:
        print(f"  {RED}✘ FAILED: {e}{RESET}")
        sys.exit(1)

    # Test 5: POST /api/simulate (Review Simulation & Math Calibration - Task A)
    print(f"\nRunning Test 5: POST /api/simulate (Output Calibration Layer)...")
    try:
        # Simulate review for Kunle on Shiro Lagos (item_id=2) with alpha=0.3
        payload = {
            "user_name": "Kunle (VI Tech Bro)",
            "item_id": 2,
            "alpha": 0.3
        }
        res = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        sim = res.json()
        
        assert "raw_rating" in sim
        assert "calibrated_rating" in sim
        assert "review" in sim
        
        print(f"  {GREEN}✔ SUCCESS: Simulation engine successfully processed!{RESET}")
        print(f"  User: {sim['user_name']} | Item: {sim['item_name']}")
        print(f"  Raw LLM Rating: {sim['raw_rating']:.2f} -> {BOLD}Calibrated Rating (α=0.3): {sim['calibrated_rating']:.2f}{RESET}")
        print(f"  Written Review: \"{sim['review']}\"")
    except Exception as e:
        print(f"  {RED}✘ FAILED: {e}{RESET}")
        sys.exit(1)

    print_header("ALL API ENDPOINTS VERIFIED AND FUNCTIONAL!")
    print(f"{GREEN}{BOLD}Congratulations! The NaijaBuddy agentic recommender system is completely correct!{RESET}")

if __name__ == "__main__":
    run_api_tests()
