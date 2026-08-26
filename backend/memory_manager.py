"""Persistent memory for the CURT chatbot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List


class MemoryManager:
    """Store conversations by session and expose only the last N messages to RAG."""

    def __init__(self, database_path: Path | str, window_size: int = 3):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.database_path = Path(database_path)
        self.window_size = window_size
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id "
                "ON conversation_messages(session_id, id)"
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def get_recent_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return exactly the latest N messages in chronological order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, self.window_size),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))

    def get_stats(self, session_id: str) -> Dict[str, int]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        return {
            "window_size_messages": self.window_size,
            "messages_in_memory": min(total, self.window_size),
            "total_messages_in_session": total,
        }
