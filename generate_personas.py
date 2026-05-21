"""
generate_personas.py - Batch pre-generation of LLM-synthesised user personas.

By default NaijaBuddy synthesises a user's natural-language persona *lazily* -
on the first request that touches that user. This script does it up front for
every real user, so the database ships with materialised user models instead of
template stubs, the demo has no first-request synthesis lag, and the synthesis
feature is demonstrably real rather than claimed.

It calls agent.synthesize_and_update_persona() - the exact same code path the
live system uses - so there is no second implementation to drift.

Resumable: it only processes users whose persona still starts with the
un-synthesised marker "A real ", so re-running after an interruption continues
where it left off.

    python generate_personas.py
"""

import time
import database
from agent import NaijaBuddyAgent


def main():
    print("=" * 60)
    print("NAIJABUDDY - BATCH PERSONA SYNTHESIS")
    print("=" * 60)

    agent = NaijaBuddyAgent()
    if agent.llm is None:
        print("ERROR: local LLM not loaded - cannot synthesise personas.")
        return

    conn = database.get_connection()
    users = conn.execute(
        "SELECT * FROM users WHERE persona LIKE 'A real %' ORDER BY id"
    ).fetchall()
    conn.close()

    total = len(users)
    print(f"Un-synthesised personas to generate: {total}")
    if total == 0:
        print("Nothing to do - every persona is already synthesised.")
        return

    t0 = time.time()
    done = 0
    failed = 0
    for idx, u in enumerate(users):
        try:
            updated = agent.synthesize_and_update_persona(dict(u))
            done += 1
            # Eyeball the first couple of results to confirm quality.
            if idx < 2:
                print(f"  SAMPLE [{u['name']}]: {updated['persona']}")
        except Exception as e:
            failed += 1
            print(f"  [skip] user {u['id']} ({u['name']}): {e}")

        n = done + failed
        if n % 20 == 0 or n == total:
            rate = (time.time() - t0) / n
            eta = rate * (total - n)
            print(f"PROGRESS {n}/{total}  {rate:.1f}s/persona  ETA {eta/60:.1f} min")

    elapsed = (time.time() - t0) / 60
    print("=" * 60)
    print(f"DONE: synthesised {done}/{total} personas in {elapsed:.1f} min "
          f"({failed} failed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
