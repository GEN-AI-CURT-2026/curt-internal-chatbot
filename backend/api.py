from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_pipeline import CURTRagPipeline


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

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
    history: List[ChatMessage] = Field(default_factory=list)


class SourceItem(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    preview: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    raw_answer: Optional[str] = None
    status: str
    expanded_query: Optional[str] = None
    sources: List[SourceItem] = []


def get_pipeline() -> CURTRagPipeline:
    global pipeline
    if pipeline is None:
        pipeline = CURTRagPipeline()
    return pipeline


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    history = [msg.model_dump() for msg in request.history]
    result = get_pipeline().run(request.message, chat_history=history)

    sources: List[SourceItem] = []
    for doc in result.get("sources", []):
        metadata = getattr(doc, "metadata", {}) or {}
        sources.append(
            SourceItem(
                source=metadata.get("source"),
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
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
