"""
memory_manager.py
=================
Hybrid Memory Architecture for CURT Gen-AI Team
-------------------------------------------------
Architecture:
    New Q&A Pair
        │
        ▼
  ┌─────────────────┐
  │  Buffer Memory  │  ◄── Stores last N=3 exchanges (verbatim, fast)
  └────────┬────────┘
           │ Overflow (oldest exchange evicted)
           ▼
  ┌─────────────────┐
  │ Summary Memory  │  ◄── LLM compresses evicted exchanges into
  └────────┬────────┘       a rolling summary (long-term context)
           │
           ▼
  ┌─────────────────┐
  │ Context Builder │  ◄── Merges summary + buffer → prompt injection
  └────────┬────────┘
           │
           ▼
      RAG Pipeline

Author : Khaled Ashraf (CURT Gen-AI Team — Memory & R&D)
Task   : Task 2 — Team Memory Documentation R&D
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Exchange:
    """One Q&A turn in the conversation."""
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return f"User: {self.question}\nAssistant: {self.answer}"


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Hybrid memory that keeps a rolling buffer of the last *buffer_size*
    exchanges and compresses older ones into a running LLM summary.

    Parameters
    ----------
    llm_model : str
        Gemini model name used for summarisation.
    api_key : str
        Google Generative AI API key.
    buffer_size : int
        Number of recent exchanges kept verbatim (default = 3, per CURT spec).
    persistence_path : str | None
        If given, memory state is saved/loaded as JSON at this path.
    """

    def __init__(
        self,
        llm_model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        buffer_size: int = 3,
        persistence_path: str | None = None,
    ):
        self.buffer_size = buffer_size
        self.persistence_path = persistence_path

        # --- Short-term buffer (last N exchanges, verbatim) ---
        self._buffer: List[Exchange] = []

        # --- Long-term summary (compressed older history) ---
        self._summary: str = ""

        # --- Statistics ---
        self._total_exchanges: int = 0
        self._summarisation_calls: int = 0

        # --- LLM for summarisation ---
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is required for MemoryManager. "
                "Pass it explicitly or set the environment variable."
            )
        self._llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.1,
            google_api_key=resolved_key,
        )

        # Load persisted state if available
        if self.persistence_path and os.path.exists(self.persistence_path):
            self._load(self.persistence_path)
            print(f"📂 Memory loaded from {self.persistence_path}")
        
        print(
            f"✅ MemoryManager initialised  |  "
            f"buffer_size={self.buffer_size}  |  "
            f"persistence={'on' if persistence_path else 'off'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_exchange(self, question: str, answer: str, metadata: dict | None = None) -> None:
        """
        Add a new Q&A exchange to memory.

        If the buffer is full the oldest exchange is evicted and folded
        into the rolling summary before the new exchange is appended.
        """
        exchange = Exchange(
            question=question,
            answer=answer,
            metadata=metadata or {},
        )
        self._total_exchanges += 1

        if len(self._buffer) >= self.buffer_size:
            # Evict oldest exchange → summarise into long-term memory
            evicted = self._buffer.pop(0)
            self._update_summary(evicted)

        self._buffer.append(exchange)

        if self.persistence_path:
            self._save(self.persistence_path)

    def get_context(self, include_summary: bool = True) -> str:
        """
        Return a formatted context string ready for prompt injection.

        Format
        ------
        [CONVERSATION SUMMARY]
        <rolling summary if it exists and include_summary=True>

        [RECENT EXCHANGES]
        Turn 1 | <timestamp>
        User: ...
        Assistant: ...
        ...
        """
        parts: List[str] = []

        if include_summary and self._summary:
            parts.append("=== CONVERSATION SUMMARY (older history) ===")
            parts.append(self._summary.strip())
            parts.append("")

        if self._buffer:
            parts.append("=== RECENT EXCHANGES (last messages) ===")
            for i, ex in enumerate(self._buffer, 1):
                parts.append(f"Turn {i} | {ex.timestamp}")
                parts.append(ex.to_text())
                parts.append("")

        return "\n".join(parts).strip()

    def get_buffer(self) -> List[Exchange]:
        """Return a copy of the current buffer (most-recent last)."""
        return list(self._buffer)

    def get_summary(self) -> str:
        """Return the current rolling summary."""
        return self._summary

    def clear(self) -> None:
        """Wipe all memory (buffer + summary)."""
        self._buffer.clear()
        self._summary = ""
        self._total_exchanges = 0
        self._summarisation_calls = 0
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)
        print("🗑️  Memory cleared.")

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "total_exchanges": self._total_exchanges,
            "buffer_length": len(self._buffer),
            "buffer_size_limit": self.buffer_size,
            "summarisation_calls": self._summarisation_calls,
            "summary_length_chars": len(self._summary),
            "has_long_term_memory": bool(self._summary),
        }

    def pretty_stats(self) -> str:
        s = self.stats()
        lines = [
            "📊 Memory Statistics",
            f"   Total exchanges   : {s['total_exchanges']}",
            f"   Buffer           : {s['buffer_length']} / {s['buffer_size_limit']}",
            f"   Summarisations   : {s['summarisation_calls']}",
            f"   Summary length   : {s['summary_length_chars']} chars",
            f"   Long-term active : {s['has_long_term_memory']}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_summary(self, evicted: Exchange) -> None:
        """
        Use the LLM to fold the evicted exchange into the running summary.
        Keeps the summary concise and factual.
        """
        self._summarisation_calls += 1

        if self._summary:
            prompt = textwrap.dedent(f"""
                You are maintaining a running summary of a conversation.
                Below is the EXISTING SUMMARY and one NEW exchange that was just evicted from 
                the recent-message buffer. Update the summary to include the key information 
                from the new exchange while keeping the result concise (≤ 120 words).
                Only output the updated summary, nothing else.

                EXISTING SUMMARY:
                {self._summary}

                NEW EXCHANGE:
                {evicted.to_text()}

                UPDATED SUMMARY:
            """).strip()
        else:
            prompt = textwrap.dedent(f"""
                Summarise the following conversation exchange in ≤ 60 words.
                Capture the main question topic and key facts from the answer.
                Only output the summary, nothing else.

                EXCHANGE:
                {evicted.to_text()}

                SUMMARY:
            """).strip()

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            self._summary = response.content.strip()
        except Exception as e:
            # Graceful fallback: append raw text so we don't lose information
            print(f"⚠️  Summarisation LLM call failed ({e}). Appending raw exchange.")
            separator = "\n---\n" if self._summary else ""
            self._summary += f"{separator}{evicted.to_text()}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        state = {
            "buffer": [asdict(ex) for ex in self._buffer],
            "summary": self._summary,
            "total_exchanges": self._total_exchanges,
            "summarisation_calls": self._summarisation_calls,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._buffer = [Exchange(**ex) for ex in state.get("buffer", [])]
        self._summary = state.get("summary", "")
        self._total_exchanges = state.get("total_exchanges", 0)
        self._summarisation_calls = state.get("summarisation_calls", 0)


# ---------------------------------------------------------------------------
# Quick standalone test  (run: python memory_manager.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    print("🧪 MemoryManager Standalone Test")
    print("=" * 55)

    mem = MemoryManager(buffer_size=3, persistence_path="test_memory_state.json")

    # Simulate 5 turns (buffer=3 → turns 1 & 2 should be summarised)
    fake_turns = [
        ("What projects has Khaled worked on?",
         "Khaled worked on 7 projects including an Autonomous RC Car, Sumo Robot, Smart Watch, and a Clinic Management System."),
        ("What programming languages does he know?",
         "He knows C, C++, Python, and Arduino C, with experience in AVR and embedded systems."),
        ("Does he have any certificates?",
         "Yes — 8+ certificates including IELTS B2, SolidWorks Basics, Embedded Systems (AVR), PLC Basic Programming, and more."),
        ("What is his education?",
         "Khaled is studying B.Eng Mechatronics at Cairo University, currently a Senior 1 Engineer, graduating July 2027."),
        ("Tell me about the Sumo Robot project.",
         "Built with SolidWorks and an ESP32 controller, the robot competed against 54 teams in ERU's competition with a custom chassis and real-time motor control."),
    ]

    for i, (q, a) in enumerate(fake_turns, 1):
        print(f"\n--- Turn {i} ---")
        mem.add_exchange(q, a)
        print(f"Buffer size now: {len(mem.get_buffer())}")

    print("\n" + "=" * 55)
    print("📋 FINAL MEMORY CONTEXT (what gets injected into the prompt):\n")
    print(mem.get_context())

    print("\n" + "=" * 55)
    print(mem.pretty_stats())
