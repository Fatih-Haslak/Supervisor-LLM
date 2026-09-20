"""Explicit, bounded long-term memory stored in local SQLite."""

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryCategory = Literal["preference", "decision", "fact"]
_SECRET_MARKERS = re.compile(
    r"api[_ -]?key|access[_ -]?token|password|passwd|secret|private[_ -]?key|"
    r"bearer\s+\S+|şifre|parola",
    flags=re.IGNORECASE,
)


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    value: str = Field(min_length=1, max_length=500)
    category: MemoryCategory = "preference"

    @field_validator("key", "value")
    @classmethod
    def reject_secret_markers(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or _SECRET_MARKERS.search(cleaned):
            raise ValueError("Memory must not contain secret or credential data")
        return cleaned


class MemoryEntry(MemoryInput):
    updated_at: str


class SQLiteMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if self.path.is_relative_to(Path("workspace").resolve()):
            raise ValueError("Memory database must be outside workspace")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, category TEXT NOT NULL, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _save(self, item: MemoryInput) -> MemoryEntry:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memories (key, value, category) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "category=excluded.category, updated_at=CURRENT_TIMESTAMP",
                (item.key, item.value, item.category),
            )
            row = connection.execute(
                "SELECT key, value, category, updated_at FROM memories WHERE key = ?",
                (item.key,),
            ).fetchone()
        assert row is not None
        return MemoryEntry(key=row[0], value=row[1], category=row[2], updated_at=row[3])

    async def save(self, item: MemoryInput) -> MemoryEntry:
        return await asyncio.to_thread(self._save, item)

    def _list(self, limit: int) -> list[MemoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value, category, updated_at FROM memories "
                "ORDER BY updated_at DESC, key ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            MemoryEntry(key=row[0], value=row[1], category=row[2], updated_at=row[3])
            for row in rows
        ]

    async def list(self, *, limit: int = 20) -> list[MemoryEntry]:
        if not 1 <= limit <= 100:
            raise ValueError("Memory list limit must be between 1 and 100")
        return await asyncio.to_thread(self._list, limit)

    def _delete(self, key: str) -> bool:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM memories WHERE key = ?", (key,))
            return result.rowcount > 0

    async def delete(self, key: str) -> bool:
        return await asyncio.to_thread(self._delete, key)
