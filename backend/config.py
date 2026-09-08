"""Central configuration for ChatScope AI backend."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_chat.json")
INDEX_DIR = os.path.join(BASE_DIR, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "chat.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
EMBEDDINGS_CACHE_PATH = os.path.join(INDEX_DIR, "embeddings.npy")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Local sentence-transformers model. This is a small, fast, multilingual-capable
# model that handles code-mixed Hinglish reasonably well without needing any
# external API calls - everything runs on-device.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # output dimension of the model above

DEFAULT_TOP_K = 10
MAX_TOP_K = 50
