from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_pipeline import CURTRagPipeline
from memory_manager import MemoryManager


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DATABASE = BASE_DIR / "conversation_memory.sqlite3"
MEMORY_WINDOW_SIZE = 3
memory_manager = MemoryManager(MEMORY_DATABASE, window_size=MEMORY_WINDOW_SIZE)

app = FastAPI(title="CURT Internal Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[CURTRagPipeline] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = Field(default=None, max_length=128)
    # Accepted for backwards compatibility. The server-side N=3 buffer is authoritative.
    history: List[ChatMessage] = Field(default_factory=list, exclude=True)


class SourceItem(BaseModel):
    source: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    preview: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    raw_answer: Optional[str] = None
    status: str
    expanded_query: Optional[str] = None
    sources: List[SourceItem] = []
    session_id: str
    memory_messages: int


def get_pipeline() -> CURTRagPipeline:
    global pipeline
    if pipeline is None:
        pipeline = CURTRagPipeline()
    return pipeline


def extract_section_label(text: str) -> Optional[str]:
    match = re.search(r"\b(?:[A-Z]{1,3}\d+(?:\.\d+)*|IN\d+(?:\.\d+)*)\b", text or "")
    return match.group(0) if match else None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    session_id = request.session_id or str(uuid.uuid4())
    # Retrieve before adding the current message so there is no duplicate user turn.
    history = memory_manager.get_recent_history(session_id)
    result = get_pipeline().run(request.message, chat_history=history)
    memory_manager.append_message(session_id, "user", request.message)
    memory_manager.append_message(session_id, "assistant", result.get("answer", ""))

    sources: List[SourceItem] = []
    for doc in result.get("sources", []):
        metadata = getattr(doc, "metadata", {}) or {}
        preview = (getattr(doc, "page_content", "") or "")[:180]
        raw_source = metadata.get("source")
        source_name = Path(raw_source).name if raw_source else None
        sources.append(
            SourceItem(
                source=source_name,
                section=metadata.get("section") or extract_section_label(preview),
                page=metadata.get("page"),
                chunk_id=metadata.get("chunk_id"),
                preview=(getattr(doc, "page_content", "") or "")[:180],
            )
        )

    return {
        "answer": result.get("answer", ""),
        "raw_answer": result.get("raw_answer"),
        "status": result.get("status", "success"),
        "expanded_query": result.get("expanded_query"),
        "sources": sources,
        "session_id": session_id,
        "memory_messages": memory_manager.get_stats(session_id)["messages_in_memory"],
    }


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str) -> Dict[str, str]:
    memory_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
