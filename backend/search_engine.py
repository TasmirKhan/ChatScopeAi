"""
Semantic search engine for ChatScope AI.

Builds (or loads a cached) FAISS index over locally-computed sentence
embeddings of every chat message, and exposes a filterable semantic search
function used by the FastAPI routes.

Design notes:
- We use faiss.IndexFlatIP (exact brute-force inner-product search) rather
  than an approximate index (e.g. IVF/HNSW). At ~5,000 messages this is
  fast enough (a query takes low single-digit milliseconds) and gives exact
  results, which matters more for a demo/eval than the marginal speed of an
  approximate index at this scale.
- Vectors are L2-normalized at embedding time (see embeddings.py), so inner
  product is equivalent to cosine similarity.
- Metadata (sender, timestamp, thread_id, etc.) is kept in a parallel Python
  list, indexed by the same integer position used in the FAISS index.
"""

import json
import os
from typing import List, Optional, Dict, Any

import numpy as np

from backend.config import (
    DATA_PATH, INDEX_DIR, FAISS_INDEX_PATH, METADATA_PATH,
    EMBEDDINGS_CACHE_PATH, EMBEDDING_DIM,
)
from backend.embeddings import embed_texts, embed_query


class SearchEngine:
    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def build(self, force: bool = False):
        """Build the FAISS index from data/synthetic_chat.json and cache it
        to disk. If a cached index already exists and force=False, this is
        skipped (use load() instead)."""
        import faiss  # local import so the module can be imported without faiss installed

        if not force and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            return self.load()

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. Run "
                f"'python3 scripts/generate_chat.py' first."
            )

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            messages = json.load(f)

        texts = [m["message"] for m in messages]
        print(f"[search_engine] Embedding {len(texts)} messages locally (this only happens once)...")
        embeddings = embed_texts(texts, show_progress=True)

        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: model produced {embeddings.shape[1]}, "
                f"config expects {EMBEDDING_DIM}. Update EMBEDDING_DIM in config.py."
            )

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)

        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)

        self.index = index
        self.metadata = messages
        self._loaded = True
        print(f"[search_engine] Index built: {index.ntotal} vectors, dim={EMBEDDING_DIM}")

    def load(self):
        """Load a previously-built FAISS index + metadata from disk."""
        import faiss

        if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
            raise FileNotFoundError(
                "No cached index found. Call build() first, or run "
                "'python3 scripts/build_index.py'."
            )
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self._loaded = True
        print(f"[search_engine] Loaded index: {self.index.ntotal} vectors")

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def ensure_ready(self):
        if not self._loaded:
            self.build(force=False)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        sender: Optional[str] = None,
        thread_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over the message corpus.

        Filters (sender/thread_id/date range/topic) are applied by
        over-fetching a larger candidate pool from FAISS and then filtering
        in Python. This is the simplest correct approach at this dataset
        size; a metadata-aware vector DB would push filtering into the index
        itself, but that's unnecessary complexity for ~5k messages.
        """
        self.ensure_ready()

        has_filters = any([sender, thread_id, date_from, date_to, topic])
        # over-fetch when filtering so we still return top_k results after filtering
        fetch_k = min(len(self.metadata), max(top_k * 20, 200)) if has_filters else min(len(self.metadata), top_k)

        query_vec = embed_query(query)
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            msg = self.metadata[idx]

            if sender and msg["sender"] != sender:
                continue
            if thread_id and msg["thread_id"] != thread_id:
                continue
            if topic and msg["metadata"].get("topic") != topic:
                continue
            if date_from and msg["metadata"]["date"] < date_from:
                continue
            if date_to and msg["metadata"]["date"] > date_to:
                continue

            results.append({**msg, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    def get_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        self.ensure_ready()
        msgs = [m for m in self.metadata if m["thread_id"] == thread_id]
        msgs.sort(key=lambda m: m["timestamp"])
        return msgs

    def list_threads(self) -> List[Dict[str, Any]]:
        """Return a summary of every non-generic thread: id, message count,
        date span, and preview of first/last message. Useful for a UI panel
        showing 'jump to decision thread'."""
        self.ensure_ready()
        threads: Dict[str, List[Dict[str, Any]]] = {}
        for m in self.metadata:
            threads.setdefault(m["thread_id"], []).append(m)

        summaries = []
        for tid, msgs in threads.items():
            if tid.startswith("thread_general_"):
                continue  # skip the per-day background-chatter buckets
            msgs.sort(key=lambda m: m["timestamp"])
            summaries.append({
                "thread_id": tid,
                "message_count": len(msgs),
                "start_date": msgs[0]["metadata"]["date"],
                "end_date": msgs[-1]["metadata"]["date"],
                "first_message": msgs[0]["message"],
                "last_message": msgs[-1]["message"],
                "topic": msgs[0]["metadata"].get("topic"),
            })
        summaries.sort(key=lambda s: s["start_date"])
        return summaries

    def list_topics(self) -> List[str]:
        self.ensure_ready()
        topics = {m["metadata"].get("topic") for m in self.metadata if m["metadata"].get("topic")}
        return sorted(topics)

    def stats(self) -> Dict[str, Any]:
        self.ensure_ready()
        senders: Dict[str, int] = {}
        dates = []
        for m in self.metadata:
            senders[m["sender"]] = senders.get(m["sender"], 0) + 1
            dates.append(m["metadata"]["date"])
        return {
            "total_messages": len(self.metadata),
            "senders": senders,
            "date_range": {"start": min(dates), "end": max(dates)} if dates else None,
            "num_threads": len({m["thread_id"] for m in self.metadata}),
        }


# Module-level singleton used by the FastAPI app
engine = SearchEngine()
