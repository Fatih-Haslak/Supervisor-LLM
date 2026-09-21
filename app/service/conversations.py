"""Local, bounded conversation history for the browser chat."""

import asyncio
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from app.llm.schemas import ChatMessage


class ConversationStore(Protocol):
    async def list(self, conversation_id: str) -> list[ChatMessage]: ...
    async def append(self, conversation_id: str, user: str, assistant: str) -> None: ...
    async def delete(self, conversation_id: str) -> None: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._messages: dict[str, list[ChatMessage]] = defaultdict(list)

    async def list(self, conversation_id: str) -> list[ChatMessage]:
        return list(self._messages[conversation_id][-24:])

    async def append(self, conversation_id: str, user: str, assistant: str) -> None:
        self._messages[conversation_id].extend([
            ChatMessage(role="user", content=user),
            ChatMessage(role="assistant", content=assistant),
        ])
        self._messages[conversation_id] = self._messages[conversation_id][-24:]

    async def delete(self, conversation_id: str) -> None:
        self._messages.pop(conversation_id, None)


class SQLiteConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversation_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "conversation_id TEXT NOT NULL, role TEXT NOT NULL, "
                "content TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS conversation_messages_lookup "
                "ON conversation_messages(conversation_id, id)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _list(self, conversation_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM ("
                "SELECT id, role, content FROM conversation_messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT 24) "
                "ORDER BY id ASC", (conversation_id,),
            ).fetchall()
        return [ChatMessage(role=row[0], content=row[1]) for row in rows]

    async def list(self, conversation_id: str) -> list[ChatMessage]:
        return await asyncio.to_thread(self._list, conversation_id)

    def _append(self, conversation_id: str, user: str, assistant: str) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO conversation_messages(conversation_id, role, content) "
                "VALUES (?, ?, ?)",
                [(conversation_id, "user", user[:6000]),
                 (conversation_id, "assistant", assistant[:6000])],
            )
            connection.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ? "
                "AND id NOT IN (SELECT id FROM conversation_messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT 24)",
                (conversation_id, conversation_id),
            )

    async def append(self, conversation_id: str, user: str, assistant: str) -> None:
        await asyncio.to_thread(self._append, conversation_id, user, assistant)

    def _delete(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )

    async def delete(self, conversation_id: str) -> None:
        await asyncio.to_thread(self._delete, conversation_id)
