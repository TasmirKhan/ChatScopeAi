# ChatScope AI

*Search conversations by meaning, not just keywords.*

A fully local, end-to-end semantic search engine for Hinglish group-chat
conversations. Given a query like *"when did we decide on the trip?"*, it
finds the message *"bhai Manali fix hai, ab bas tickets dekhte hain 🔥"* even
though the two share zero exact words — because the search is over meaning
(dense vector embeddings + cosine similarity), not keyword overlap.

Everything runs on your machine. No OpenAI/Gemini/Anthropic API calls, no
Docker, no external services. Embeddings are computed locally with
HuggingFace `sentence-transformers`; the vector index is a local FAISS index;
the backend is FastAPI; the frontend is plain HTML/CSS/JS served by that same
backend.

---

## 1. What's in the box

```
chatscope-ai/
├── requirements.txt
├── data/
│   └── synthetic_chat.json     <- generated dataset (already included)
├── scripts/
│   ├── generate_chat.py        <- deterministic dataset generator
│   ├── build_index.py          <- precompute embeddings + FAISS index
│   └── evaluate.py             <- recall metrics proving semantic (not keyword) search works
├── backend/
│   ├── config.py                <- paths, model name, constants
│   ├── embeddings.py             <- local sentence-transformers wrapper
│   ├── search_engine.py          <- FAISS index build/load + filtered search
│   ├── models.py                 <- Pydantic request/response schemas
│   └── main.py                   <- FastAPI app + routes + static serving
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── index/                        <- generated on first run (FAISS index + cache)
```

The dataset (`data/synthetic_chat.json`) is already generated and committed,
so you can run the server immediately without regenerating it. Re-run the
generator only if you want to change the data.

---

## 2. Setup

Requires Python 3.10+.

```bash
cd chatscope-ai
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# CPU-only torch first (smaller, faster download, no CUDA needed for this
# project's scale). Skip this line and just run the pip install below if you
# already have a GPU-enabled torch you want to keep using.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

`sentence-transformers` and `torch` are the heaviest installs (a few hundred
MB total). `faiss-cpu` has no GPU dependency and installs cleanly on
Linux/macOS/Windows via pip.

### Build the search index (one-time, ~1-2 minutes on CPU)

```bash
python3 scripts/build_index.py
```

This downloads the embedding model (`paraphrase-multilingual-MiniLM-L12-v2`,
~470MB, cached by `sentence-transformers` after the first download), embeds
every message in the dataset, and writes a FAISS index + metadata cache to
`index/`. You'll see a progress bar. If you skip this step, the server will
build it automatically on first startup instead — running it explicitly just
lets you watch it happen and confirm it succeeds before starting the API.

### Run the server

```bash
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** in a browser.

### (Optional) Regenerate the dataset

```bash
python3 scripts/generate_chat.py   # writes data/synthetic_chat.json
python3 scripts/build_index.py     # re-embed and rebuild the index
```

The generator is fully deterministic (`random.seed(42)`, `numpy.random.seed(42)`)
— running it twice produces byte-identical output.

### Evaluate search quality

```bash
python3 scripts/evaluate.py
```

Runs a small hand-authored set of natural-language queries that are
deliberately paraphrased away from the target messages' actual wording (e.g.
*"when did we decide on the trip?"* → the target message is *"bhai Manali fix
hai, ab bas tickets dekhte hain 🔥"*, sharing zero non-trivial words) and
reports Recall@5 / Recall@10 against the expected decision thread. This is
the concrete, runnable proof of the project's core claim — that retrieval
works on meaning, not keyword overlap — on this dataset. It's a small
sanity-check eval set, not a rigorous IR benchmark.

---

## 3. Using it

Type a natural-language query into the search box, e.g.:

- `when did we decide on the trip?`
- `what tech stack did we pick for the project`
- `who's coming to dinner`
- `budget discussion for manali`

Results are ranked by cosine similarity between your query's embedding and
every message's embedding — there is no keyword matching happening at all.
Use the filters (sender / date range / topic) to narrow results; they're
applied after retrieval, on top of the semantic ranking.

The three chips under "jump to decision thread" open the full, chronologically
ordered anchored threads end-to-end, with the final resolution message
visually marked.

---

## 4. API reference

All routes are served by the same FastAPI app on port 8000.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/stats` | Total messages, per-sender counts, date range, thread count |
| `GET` | `/api/senders` | List of all persona names |
| `GET` | `/api/topics` | List of all topic tags present in the corpus |
| `GET` | `/api/threads` | Summary of every named (non-generic) thread |
| `GET` | `/api/thread/{thread_id}` | Full chronological message list for one thread |
| `POST` | `/api/search` | Semantic search (see below) |

`POST /api/search` body:

```json
{
  "query": "when did we decide on the trip?",
  "top_k": 10,
  "sender": null,
  "thread_id": null,
  "date_from": null,
  "date_to": null,
  "topic": null
}
```

Interactive API docs (auto-generated by FastAPI) are available at
`http://localhost:8000/docs` once the server is running.

---

## 5. Design decisions & trade-offs

**Embedding model — `paraphrase-multilingual-MiniLM-L12-v2`.** Chosen because
it was trained on 50+ languages including Hindi, so it has *some* grounding in
Hindi vocabulary and syntax, which matters for code-mixed Hinglish written in
Latin script. It is not fine-tuned specifically on Hinglish or on chat data,
so recall on heavily slang-y or transliterated phrases is imperfect — this is
a real, disclosed limitation, not a hidden one. It's also small enough
(~470MB, 384-dim output) to embed ~5,000 short messages on CPU in about a
minute, which matters for a fully local, no-GPU-required demo.

**FAISS `IndexFlatIP` (exact search), not an approximate index.** At this
corpus size (under 5,000 vectors), brute-force exact inner-product search
takes low single-digit milliseconds per query and gives exact nearest
neighbors. An approximate index (IVF/HNSW) would add tuning complexity for a
speed benefit that isn't needed until the corpus is orders of magnitude
larger.

**Filtering happens in Python, after over-fetching from FAISS.** Sender/
date/topic filters aren't pushed into the vector index itself; instead the
engine fetches a larger candidate pool (`top_k * 20`, capped at the corpus
size) and filters in Python. This is the simplest correct approach at this
scale. A production system with millions of messages would want a vector
database with native metadata filtering (e.g. a payload-filtered ANN index)
instead.

**No Docker/Redis/Celery/Kubernetes.** The whole system is a single Python
process plus a static frontend. That's a deliberate scope decision for a
locally-runnable demo, not an oversight — there's no background job queue or
multi-service orchestration to justify that infrastructure here.

**Dataset generation is template + slot-filling, not an LLM.** To keep the
generator deterministic and dependency-free (no API calls, seeded RNG only),
Hinglish messages are built from persona-specific style rules, phrase banks,
and hand-authored narrative scripts for the three anchored decision threads,
with typo/emoji/short-reply injection layered on top for realism. This means
the generic chatter is template-based rather than infinitely varied — a
trade-off documented here rather than glossed over.

---

## 6. What was verified, and what couldn't be run in this build environment

This project was built in a sandboxed environment with **no network access**,
so `pip install` for `fastapi`, `uvicorn`, `sentence-transformers`, `torch`,
and `faiss-cpu` could not be run there, and the live server was not started
end-to-end in that environment. Here's exactly what was and wasn't checked
before delivery, so you know what to double-check on your first run:

- ✅ **Dataset generator**: actually executed. Produces 4,688 deterministic
  messages (re-running gives a byte-identical file, verified via checksum),
  spanning the full 2026-01-01 to 2026-06-30 range, with all three anchored
  threads present, in chronological order, ending on their intended decision
  messages.
- ✅ **All Python files**: syntax-checked with `py_compile`.
- ✅ **Core search/filter/ranking logic in `search_engine.py`**: exercised
  with a lightweight stand-in for the FAISS index (real cosine-similarity
  math, same code paths as production, just swapping the actual FAISS/
  sentence-transformers calls for a controlled fake) — sender filters, date
  filters, thread retrieval and ordering, and stats all verified correct.
- ✅ **`scripts/evaluate.py` scoring mechanics**: verified with the same
  FAISS stand-in that rank detection, the zero-keyword-overlap sanity check,
  and Recall@K accounting are all computed correctly. What this *couldn't*
  verify offline is the real embedding model's actual retrieval quality on
  Hinglish text — that's exactly what running this script for real, after
  `pip install`, will tell you.
- ⚠️ **Not run**: the actual FastAPI server process, real sentence-transformers
  embedding calls, and the real FAISS index build. These need `pip install -r
  requirements.txt` with internet access, which this build environment didn't
  have. The code has been written carefully and matches each library's
  documented API, but you should treat your first `uvicorn` run as the first
  real end-to-end test, and open an issue/fix forward if something in a
  library API has moved since this was written.

If `python3 scripts/build_index.py` or `uvicorn backend.main:app` errors out
on your machine, it's most likely a dependency version mismatch — check the
error against `requirements.txt` first.
