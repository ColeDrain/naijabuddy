"""
generate_personas.py - Materialise the per-user LLM personas.

Two modes, chosen automatically:

  * If data/synthesized_personas.json exists, LOAD it - re-embed each persona
    with BGE-small and write it to the database. No LLM, ~1 minute. This is the
    path the Docker build takes, so the build stays fast.
  * Otherwise SYNTHESISE with the local LLM (one call per user, ~hours on CPU)
    via agent.synthesize_and_update_persona(), then export the artifact so every
    later run - and the Docker build - takes the fast path.

This mirrors how the dense datasets are handled (see colab_data_prep.ipynb): the
expensive output is generated once, committed to the repo, and loaded
thereafter. The artifact is plain text, so the synthesised personas remain
human-inspectable.

    python generate_personas.py
"""
import json
import os
import time

# Isolated model cache, set before importing SentenceTransformer.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "hf_home"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",
                      os.path.join(MODELS_DIR, "sentence_transformers"))

import database

ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "synthesized_personas.json")


def load_from_artifact():
    """Apply pre-synthesised personas from the committed artifact (no LLM)."""
    with open(ARTIFACT, encoding="utf-8") as f:
        personas = json.load(f)
    print(f"Loading {len(personas)} pre-synthesised personas from "
          f"{os.path.relpath(ARTIFACT)} (no LLM)...")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    names = list(personas)
    vectors = embedder.encode([personas[n] for n in names],
                              show_progress_bar=True).tolist()

    conn = database.get_connection()
    cursor = conn.cursor()
    applied = 0
    for name, vec in zip(names, vectors):
        cursor.execute(
            "UPDATE users SET persona = ?, persona_embedding = ? WHERE name = ?",
            (personas[name], json.dumps(vec), name))
        applied += cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Applied {applied}/{len(personas)} personas to the database.")


def export_artifact():
    """Snapshot the database's personas to the artifact for fast future loads."""
    conn = database.get_connection()
    rows = conn.execute("SELECT name, persona FROM users ORDER BY id").fetchall()
    conn.close()
    personas = {r["name"]: r["persona"] for r in rows}
    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    with open(ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=1)
    print(f"Exported {len(personas)} personas to {os.path.relpath(ARTIFACT)}")


def synthesize_with_llm():
    """
    Original path: one LLM call per user, then export the artifact.

    Resumable - only users whose persona still starts with the marker 'A real '
    are synthesised, so an interrupted run continues where it left off.
    """
    from agent import NaijaBuddyAgent
    agent = NaijaBuddyAgent()
    if agent.llm is None:
        print("ERROR: local LLM not loaded - cannot synthesise personas.")
        return

    conn = database.get_connection()
    users = conn.execute(
        "SELECT * FROM users WHERE persona LIKE 'A real %' ORDER BY id").fetchall()
    conn.close()

    total = len(users)
    print(f"Un-synthesised personas to generate with the LLM: {total}")
    if total:
        t0 = time.time()
        done = failed = 0
        for u in users:
            try:
                agent.synthesize_and_update_persona(dict(u))
                done += 1
            except Exception as e:
                failed += 1
                print(f"  [skip] user {u['id']} ({u['name']}): {e}")
            n = done + failed
            if n % 20 == 0 or n == total:
                rate = (time.time() - t0) / n
                print(f"PROGRESS {n}/{total}  {rate:.1f}s/persona  "
                      f"ETA {rate*(total-n)/60:.1f} min")
        print(f"DONE: synthesised {done}/{total} ({failed} failed)")

    export_artifact()


def main():
    print("=" * 60)
    print("NAIJABUDDY - PERSONA MATERIALISATION")
    print("=" * 60)
    if os.path.exists(ARTIFACT):
        load_from_artifact()
    else:
        print(f"No artifact at {os.path.relpath(ARTIFACT)} - synthesising with the LLM.")
        synthesize_with_llm()


if __name__ == "__main__":
    main()
