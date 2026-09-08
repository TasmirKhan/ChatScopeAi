"""
Evaluate ChatScope AI's semantic search quality.

This is the concrete proof of the project's core claim: that queries sharing
little or no vocabulary with the target message still retrieve it, because
matching happens on meaning (embedding similarity) rather than keywords.

Run after building the index:
    python3 scripts/evaluate.py

For each test case we check whether a message from the expected thread (and,
where specified, containing an expected keyword/phrase) appears in the top-K
results for a natural-language query that deliberately avoids the target
message's own wording. We report Recall@5 and Recall@10.

This is a small, hand-authored eval set (not a claim of rigorous IR
benchmarking) - it exists to give a fast, honest sanity check that retrieval
is doing its job on this specific dataset, not to substitute for a proper
labeled evaluation on a larger corpus.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.search_engine import engine

# (query, expected_thread_id, optional substring expected in a hit's message)
# Queries are deliberately paraphrased away from the target messages' actual
# wording - this is what "semantic, not keyword" search is being asked to do.
TEST_CASES = [
    # --- Trip thread: zero-overlap paraphrases of the final decision ---
    ("when did we decide on the trip?", "thread_trip_001", "Manali fix hai"),
    ("which vacation spot did the group finally agree on?", "thread_trip_001", None),
    ("how much will the vacation cost per person?", "thread_trip_001", None),
    ("what dates are we travelling on?", "thread_trip_001", None),

    # --- Project thread ---
    ("what technology did we choose for the group project?", "thread_project_001", None),
    ("who is responsible for the database", "thread_project_001", None),
    ("when is the project due?", "thread_project_001", None),

    # --- Dinner thread ---
    ("is the group dinner confirmed?", "thread_dinner_001", None),
    ("how many people are coming to dinner", "thread_dinner_001", None),
    ("what time are we meeting for dinner", "thread_dinner_001", None),
]

K_VALUES = [5, 10]


def zero_overlap(query: str, message: str) -> bool:
    """Sanity-check that a query and its target message share no non-trivial
    words, so a hit genuinely demonstrates semantic (not lexical) matching."""
    stop = {"the", "a", "an", "is", "are", "did", "we", "on", "for", "to", "what",
            "when", "how", "many", "will", "our", "of", "and", "in", "at"}
    q_words = {w.strip("?.,!").lower() for w in query.split()} - stop
    m_words = {w.strip("?.,!🔥🎉").lower() for w in message.split()} - stop
    return len(q_words & m_words) == 0


def main():
    engine.ensure_ready()

    print(f"Evaluating {len(TEST_CASES)} queries against the live index...\n")

    hits_at_k = {k: 0 for k in K_VALUES}
    max_k = max(K_VALUES)

    for query, expected_thread, expected_substring in TEST_CASES:
        results = engine.search(query, top_k=max_k)
        thread_ids = [r["thread_id"] for r in results]

        best_rank = None
        matched_message = None
        for i, r in enumerate(results, start=1):
            if r["thread_id"] == expected_thread:
                if expected_substring is None or expected_substring.lower() in r["message"].lower():
                    best_rank = i
                    matched_message = r["message"]
                    break

        status_bits = []
        for k in K_VALUES:
            hit = best_rank is not None and best_rank <= k
            hits_at_k[k] += int(hit)
            status_bits.append(f"@{k}={'HIT' if hit else 'miss'}")

        print(f"Query: \"{query}\"")
        print(f"  expected thread: {expected_thread}   {'  '.join(status_bits)}")
        if matched_message:
            overlap_note = "no shared keywords" if zero_overlap(query, matched_message) else "some shared keywords"
            print(f"  matched (rank {best_rank}, {overlap_note}): \"{matched_message[:80]}\"")
        else:
            print(f"  matched: none in top {max_k}")
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for k in K_VALUES:
        recall = hits_at_k[k] / len(TEST_CASES)
        print(f"Recall@{k}: {hits_at_k[k]}/{len(TEST_CASES)} = {recall:.0%}")


if __name__ == "__main__":
    main()
