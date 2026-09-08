from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(10, ge=1, le=50)
    sender: Optional[str] = None
    thread_id: Optional[str] = None
    date_from: Optional[str] = None  # "YYYY-MM-DD"
    date_to: Optional[str] = None    # "YYYY-MM-DD"
    topic: Optional[str] = None


class SearchResult(BaseModel):
    message_id: str
    timestamp: str
    sender: str
    message: str
    message_type: str
    thread_id: str
    metadata: Dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchResult]


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    start_date: str
    end_date: str
    first_message: str
    last_message: str
    topic: Optional[str] = None


class StatsResponse(BaseModel):
    total_messages: int
    senders: Dict[str, int]
    date_range: Optional[Dict[str, str]]
    num_threads: int
