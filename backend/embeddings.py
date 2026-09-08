"""
Local embedding utilities for ChatScope AI.

Uses HuggingFace `sentence-transformers` running entirely on-device. No network
calls are made once the model has been downloaded once by sentence-transformers
(it caches to ~/.cache/torch/sentence_transformers on first use).

Model choice: 'paraphrase-multilingual-MiniLM-L12-v2' was picked because it
- was trained on 50+ languages including Hindi, so it copes with code-mixed
  Hinglish (Latin-script Hindi + English) far better than an English-only model
- is small (~470MB) and fast enough to embed a few thousand short chat messages
  on CPU in well under a minute
- outputs 384-dim vectors, a good balance of quality vs. index size at this scale

This is a pragmatic choice, not a claim of state-of-the-art Hindi NLU: a
Latin-script transliteration of Hindi is still out-of-distribution for any
model not specifically fine-tuned on Hinglish. In practice this means recall
on paraphrased Hinglish slang can be imperfect - documented as a known
trade-off rather than hidden.
"""

from functools import lru_cache
from typing import List

import numpy as np

from backend.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_model():
    """Lazily load the sentence-transformers model once per process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts and return L2-normalized float32 vectors.

    Normalizing here lets us use a FAISS inner-product index as an exact
    cosine-similarity search, which is simpler and faster than re-normalizing
    at query time inside FAISS itself.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-length vectors -> inner product == cosine similarity
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim) float32."""
    return embed_texts([query], batch_size=1, show_progress=False)
