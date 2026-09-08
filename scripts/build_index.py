"""
Precompute embeddings and build the FAISS index for ChatScope AI.

Run this once after generating the dataset (and again any time the dataset
changes):

    python3 scripts/build_index.py

This is also triggered automatically on first server startup if no cached
index is found, but running it explicitly lets you see embedding progress
and control when the (one-time, ~1-2 minute) embedding pass happens.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.search_engine import engine


def main():
    engine.build(force=True)
    stats = engine.stats()
    print("\n--- Index build complete ---")
    print(f"Total messages indexed: {stats['total_messages']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Threads: {stats['num_threads']}")


if __name__ == "__main__":
    main()
