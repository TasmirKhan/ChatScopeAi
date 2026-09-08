"""
ChatScope AI - FastAPI backend.

Run with:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000 in a browser.

On first startup, if no cached FAISS index is found, the full dataset is
embedded locally (via sentence-transformers) and indexed - this can take
roughly a minute or two on CPU the very first time. Subsequent restarts load
the cached index from disk in under a second.
"""
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import FRONTEND_DIR
from backend.search_engine import engine
from backend.models import (
    SearchRequest, SearchResponse, ThreadSummary, StatsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Ensuring search index is ready (building on first run)...")
    engine.ensure_ready()
    print("[startup] ChatScope AI is ready.")
    yield
    # no teardown needed - FAISS index and metadata are read-only in memory


app = FastAPI(
    title="ChatScope AI",
    description="Semantic search over Hinglish group chat conversations - runs entirely locally.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "indexed": engine.is_ready}


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    try:
        return engine.stats()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        results = engine.search(
            query=req.query,
            top_k=req.top_k,
            sender=req.sender,
            thread_id=req.thread_id,
            date_from=req.date_from,
            date_to=req.date_to,
            topic=req.topic,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"query": req.query, "count": len(results), "results": results}


@app.get("/api/threads", response_model=list[ThreadSummary])
def list_threads():
    try:
        return engine.list_threads()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/thread/{thread_id}")
def get_thread(thread_id: str):
    try:
        msgs = engine.get_thread(thread_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not msgs:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found or empty")
    return {"thread_id": thread_id, "message_count": len(msgs), "messages": msgs}


@app.get("/api/senders")
def list_senders():
    try:
        s = engine.stats()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return sorted(s["senders"].keys())


@app.get("/api/topics")
def list_topics():
    try:
        return engine.list_topics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# Frontend (plain HTML/JS/CSS, no build step)
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
