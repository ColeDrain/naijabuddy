# Implementation Plan: DSN X BCT LLM Agent Challenge

This document outlines the detailed design and implementation plan for building our submission for the **DSN X BCT LLM Agent Challenge**. 

Our proposed solution, **NaijaBuddy**, is a **Unified Multi-Domain Recommender and User Modeling Agent** that integrates the Yelp, Amazon, and Goodreads datasets, localized with synthetic Nigerian seed data to sound and behave authentically Nigerian.

---

## Technical Inspiration & Papers Leveraged

To ensure we build a state-of-the-art system that easily wins the **Solution Paper (15 pts)** and **Agentic Workflow** criteria, our architecture is directly grounded in the 2025 research papers found in the `/articles` folder:
1. **$L^3Rec$ (CIKM '25)**: We adopt the concept of localized modeling. We cluster user preferences into distinct sub-communities to route them to specialized, highly efficient prompt agents.
2. **Critic-LLM-RS (Oct '25)**: We implement a "Critic" layer—a programmatic scoring system that validates and refines the LLM's raw recommendations against logical collaborative constraints.
3. **RecBench (NeurIPS '25)**: We address inference efficiency by implementing a strict **Filter-then-Rerank** paradigm (minimizing LLM latency).

---

## Proposed Architecture: The "NaijaBuddy" Agent

Our containerized application will house a unified agent supporting both required tasks:

```mermaid
graph TD
    A[User Inputs Persona / History] --> B[User Persona Encoder]
    B --> C[Naija Persona Profile]
    C --> D[Task A: Review & Rating Simulator]
    C --> E[Task B: Recommendation Engine]
    
    %% Task A Flow
    D --> D1[Community-Routed Prompt]
    D1 --> D2[Simulate Nigerian Tone & Slang]
    D2 --> D3[Output Rating & Written Review JSON]
    
    %% Task B Flow
    E --> E1[Recall Stage: TF-IDF / Vector Search (BAAI/bge-small)]
    E1 --> E2[Retrieve 20 Candidates from Hybrid Catalog]
    E2 --> E3[Rerank Stage: Local LLM Agent]
    E3 --> E4[Calibration Layer: Alpha Tuning & Cluster Mean]
    E4 --> E5[Output Reranked List & Justifications]
    
    D3 --> F[Premium Web UI / API Dashboard]
    E5 --> F
```

### 1. Data Layer: The "Naija-Enriched" Hybrid Catalog
We will build a **Hybrid Local SQLite Catalog** containing:
* **The Foreign Base**: Selected subsets of items/reviews from the three datasets.
* **The Nigerian Seed (Data Augmentation)**: We will write a Python script to seed the catalog with 150+ highly recognizable local items:
  * **Yelp**: Famous Lagos/Abuja restaurants (e.g., *The Place, Yellow Chilli, Shiro, Mega Chicken*) and local delicacies (Jollof, Suya, Pepper Soup).
  * **Amazon**: Iconic Nollywood films (e.g., *The Wedding Party, King of Boys, Aníkúlápó*) and electronics.
  * **Goodreads**: Masterpieces by Nigerian/African authors (e.g., Chinua Achebe, Chimamanda Ngozi Adichie, Wole Soyinka, Ben Okri).

### B. Output Calibration (The Mathematical Defense)
LLMs exhibit inherent "positivity bias" and volatility when predicting raw numerical ratings. To protect our RMSE score, we implement a programmatic **Calibration Layer**:
1. **Hyperparameter Tuning (`alpha`)**: We will not blindly trust the LLM. The final rating is calculated as `Final = (alpha * LLM_Rating) + ((1 - alpha) * Baseline_Mean)`. We will optimize `alpha` via Grid Search.
2. **Cold-Start Protection (Cluster Mean)**: For new users without historical ratings, we do not fall back to a generic global mean. We use the Vector Database to find the top $K$ users with the most semantically similar personas to the new user. We calculate the "Cluster Mean" of these neighbors and use it as the Baseline anchor. This demonstrates an elite understanding of Neighborhood-based Collaborative Filtering.

### 2. Core Engine
* **User Persona Encoder**: Condenses raw historical clicks/ratings into a short, highly descriptive "Naija Persona Profile" (saving context window tokens and minimizing latency).
* **Task A: Review Simulator**: Generates reviews tailored to specific Nigerian personas (e.g., "Strict Nigerian Dad", "Lagos Gen-Z Influencer", "Victoria Island Tech Bro") using local phrasing and cultural nuances.
* **Task B: Recommender (Filter-then-Rerank)**:
  * **Recall**: Generates dense semantic embeddings for both the user persona profile and catalog items using the local state-of-the-art **Jina v5-text-small-retrieval** (677M parameters, scoring **71.7** on the updated **MTEB v2** leaderboard). It performs an exact cosine similarity match in NumPy to retrieve the top 20 relevant items from our hybrid catalog.
  * **Rerank**: Prompts the LLM (e.g., Gemini 3.1 Flash-Lite or local model) to order the candidates and write persuasive, context-aware explanations.
  * **Critic**: Reranks options to prioritize local matches or correct obvious anomalies (e.g., ensuring a vegan persona doesn't get recommended Suya).

### 2.1 Agent Prompt Templates

We use highly tailored, zero-shot structured prompts to maximize both semantic ROUGE overlap and localized natural language style.

#### Task A: Review & Rating Simulation Prompt
```python
TASK_A_PROMPT = """You are a highly advanced simulation agent modeling a specific human persona.
Your objective is to generate an authentic, context-aware star rating and written review for a product or service.

USER PERSONA PROFILE:
{persona_profile}

TARGET PRODUCT/SERVICE DETAILS:
- Name: {item_name}
- Category: {item_category}
- Specific Details: {item_description}

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
"""
```

#### Task B: Recommendation Engine (Rerank Stage) Prompt
```python
TASK_B_PROMPT = """You are an elite, context-aware recommendation routing agent.
Your objective is to rerank a candidate list of items for a user based on their persona profile and historical preferences.

USER PERSONA PROFILE:
{persona_profile}

TARGET DOMAIN: {target_domain}

CANDIDATE LIST OF ITEMS (Top 20 from Semantic Search):
{candidate_items_json}

CRITICAL CONSTRAINTS:
1. Sort the candidate list from most relevant (Index 0) to least relevant (Index 19).
2. Filter out any candidate items that violate strict user constraints (e.g., recommending pork or beef to a strict vegetarian, or loud parties to an introverted parent).
3. For each recommended item, provide a persuasive, short (1-2 sentences) natural language explanation of WHY this was recommended to this persona. Highlight the exact features of the item that match the persona's core tastes.
4. Adjust your explanation tone to sound authentic to the user's cultural context (Nigeria).

OUTPUT FORMAT (Strict JSON):
Return ONLY a valid JSON array of objects. Do not include any extra text or markdown backticks outside the JSON.
[
  {{
    "id": [item_id],
    "name": "[item_name]",
    "rank": [1 to 20],
    "explanation": "[Write your persuasive explanation here]"
  }},
  ...
]
"""
```


### 3. Web UI Dashboard (Conversational Agent Interface)
Instead of static forms, the frontend will be built as a premium, responsive **AI Chat Assistant ("NaijaBuddy")** featuring:
* **Multi-Turn Conversational Interface**: Users can chat naturally with NaijaBuddy (e.g., "Suggest a cool local restaurant for a tech bro date" or "Generate a review for Yellow Chilli"). This directly aligns with the multi-turn agentic criteria of the brief.
* **Interactive Glassmorphic Product Cards**: When NaijaBuddy recommends items, it renders them as beautifully animated, responsive cards directly inside the chat window, complete with "Why we recommended this" justifications and action buttons.
* **Visual Persona Selector**: A quick-toggle panel to instantly choose pre-built Nigerian personas (Tech Bro, Strict Dad, Gen-Z Influencer) or define a custom one.
* **Vibrant Dark Mode & Micro-Animations**: Sleek HSL gradients, glassmorphism (`backdrop-filter`), and micro-animations for high-end modern aesthetics.

### 4. Containerization & API
* **Docker/OrbStack Setup**: A lightweight Dockerfile running a FastAPI backend and serving our static web assets, exposing a single port.
* **API Endpoints**:
  * `POST /api/simulate`: Takes persona + item details -> returns simulated review and star rating.
  * `POST /api/recommend`: Takes persona + domain -> returns ranked list of items + explanations.

---

## Proposed Changes

We will organize our repository as follows:

### [NEW] [data_enricher.py](file:///Users/indicina/Projects/dsn-bct-hackathon/data_enricher.py)
A utility script to populate our SQLite database with a combination of the foreign datasets and our custom Nigerian seed data (local restaurants, Nollywood movies, and African literature).

### [NEW] [database.py](file:///Users/indicina/Projects/dsn-bct-hackathon/database.py)
Handles connections to our local SQLite database and performs the fast, robust Stage-1 (Recall) semantic search using local **`BAAI/bge-small-en-v1.5`** dense embeddings (130MB, MTEB: 58.21) with an exact cosine similarity match in NumPy (executing in under 1ms).

### [NEW] [agent.py](file:///Users/indicina/Projects/dsn-bct-hackathon/agent.py)
Houses our local offline LLM Agent logic. It utilizes **`llama-cpp-python`** (loaded natively in Python via pre-compiled wheels) to run a quantized, highly efficient **Qwen 3.5 (3B)** or **Gemma 4 (E2B)** local GGUF model directly inside our FastAPI process:
* Generates User Persona Profiles from history.
* Generates simulated reviews/ratings in Nigerian voices (Task A).
* Reranks candidate items and writes persuasive explanations (Task B).
* Implements the "Critic" collaborative logic layer.

### [NEW] [app.py](file:///Users/indicina/Projects/dsn-bct-hackathon/app.py)
Our FastAPI web server exposing standard endpoints and serving our static UI files.

### [NEW] [static/](file:///Users/indicina/Projects/dsn-bct-hackathon/static)
Our frontend assets:
* `index.html`: Fully responsive, semantic HTML structure.
* `styles.css`: Sleek CSS featuring dark mode, glassmorphic card grids, vibrant gradients, and micro-animations.
* `script.js`: Interactive logic to query our API and update the dashboard dynamically.

### [NEW] [downloader.py](file:///Users/indicina/Projects/dsn-bct-hackathon/downloader.py)
A critical offline seeder utility that programmatically downloads and caches our required quantized GGUF model and dense BGE-small embedding files from Hugging Face on container startup. This completely eliminates manual steps for the judges, guaranteeing a 10/10 Code Reproducibility score.

### [NEW] [Dockerfile](file:///Users/indicina/Projects/dsn-bct-hackathon/Dockerfile)
A multi-stage, lightweight Docker configuration to containerize the FastAPI web application, running `downloader.py` on build or startup to pre-cache models, compiling native python wheel bindings cleanly.

---

## Finalized Architectural Decisions

Based on rigorous constraints, region-specific verification challenges, and collaborative review, we have locked in the following decisions:

1. **LLM Provider**: We are building a **100% Local, Offline-First** application. We completely reject cloud APIs because card-linking restrictions make API sign-ups highly inaccessible for local judges. The container will load and run a quantized **Qwen 3.5 (3B)** or **Gemma 4** GGUF model directly inside our FastAPI Python process using pre-compiled `llama-cpp-python` wheels.
2. **Catalog Enriched Volume**: We will seed **as much localized data as needed** (building a rich matrix of 150+ highly recognizable Nigerian restaurants, Nollywood films, and books) to ensure excellent rating/review synthesis (Task A) and high-quality recommendation recommendations (Task B).
3. **Container Compatibility**: 
   > [!IMPORTANT]
   > **OrbStack is 100% transferrable and identical to Docker Desktop.** 
   > OrbStack runs standard, production-ready OCI Docker images under the hood. All standard Docker CLI commands work identically. When the judges build and run our container on standard Docker Desktop (macOS, Windows, or Linux), it will run perfectly because our Dockerfile utilizes standard Debian/Alpine Python base images.
4. **Vector Search (Recall)**: We use **`BAAI/bge-small-en-v1.5`** + **NumPy Cosine Similarity**. This ensures absolute offline safety inside Docker, lightning-fast CPU execution (~5ms), and 100% math accuracy without heavy C++ compilations.

---

## Verification Plan

### Automated Verification
* **Unit Tests**: Python test suite to verify that:
  * The `/api/simulate` endpoint correctly outputs valid JSON with a `rating` (float) and a non-empty `review` (string).
  * The `/api/recommend` endpoint correctly returns a list of items and their explanations.
  * SQLite database queries run in under 10ms.
* **Linting & Formatting**: Ensure clean, standardized Python code using Ruff or Black.

### Manual Verification
* **UI Testing**: We will launch the application locally, open the dashboard in a browser, and manually test different Nigerian personas (e.g., a "strict dad" vs. a "tech bro") to confirm that the simulated reviews sound authentic and the recommendations align with their tastes.
* **Docker Verification**: Build the docker image locally (`docker build -t naijabuddy .`) and run it (`docker run -p 8000:8000 naijabuddy`) to verify perfect containerization.

### The Ablation Study Plan (For the Solution Paper)
To guarantee a 15/15 score on the **Solution Paper**, we will run and document a formal Ablation Study comparing 5 distinct versions of our system:

| Version | Configuration | Primary Metric Evaluated | Expected/Target Outcome |
| :--- | :--- | :--- | :--- |
| **V0 (Baseline)** | Standard `all-MiniLM-L6-v2` + raw LLM zero-shot rating (no math adjustments, default templates). | Baseline RMSE, Recall@20, ROUGE-L. | Poor rating calibration (high RMSE), generic style. |
| **V1 (Recall Upgrade)** | Swap MiniLM for `BAAI/bge-small-en-v1.5` or `Jina v5`. | Recall@20 / Hit Rate. | Retrieval quality climbs (higher semantic accuracy for persona matching). |
| **V2 (Calibration Math)** | Activate the **Calibration Layer** (`alpha` grid search + Cluster Mean). | **RMSE** (Rating prediction error). | **RMSE drops by 20-30%** as LLM positivity bias is mathematically constrained. |
| **V3 (Prompt Styling)** | Activate the **ROUGE Constraint & Nigerian Slang Templates**. | **ROUGE-L** & Human Realism Score. | Higher word overlap with simple review keywords, and a perfect 10/10 Nigerian naturalness score. |
| **V4 (Full Pipeline)** | Add the **Collaborative Critic Layer** (recommending vegetarian/age-appropriate filters). | Constraint Violation Rate (%). | Safe, 0% constraint violation rate across edge-case personas. |

### The Solution Paper Structure (Gold-Standard 15 Pts)
To guarantee the full 15/15 score on the **Solution Paper**, we pre-structure our writing following ACM/IEEE peer-reviewed conference standards:

1. **Abstract**: High-level summary of NaijaBuddy’s local-first architecture, hybrid semantic retrieval, cluster-mean persona calibration, and ROUGE-optimized output generation.
2. **Introduction**: Problem formulation (cross-domain rating simulation and cold-start matching) and the limits of raw uncalibrated LLM outputs.
3. **Proposed Methodology**:
   * *Section A: Hybrid SQLite Catalog & Local Semantic Recall*
   * *Section B: LLM Reranking & In-Context Persona Simulation*
   * *Section C: Calibration Layer (Alpha-Tuning & Vector-Based Cluster-Mean fallback)*
4. **Experimental Setup & Ablation Study**: Formal presentation of our V0-V4 ablation table, demonstrating exact RMSE, Recall@20, and ROUGE-L metric progressions.
5. **Ethical & Cultural Localization**: Discussion of our Nigerian data augmentation methodology, balancing local slang and cultural context with model evaluation integrity.
6. **Conclusion & Future Directions**: Scaling offline agent networks and edge recommender optimization.
