"""Persistent memory survives tasks and stays explicit."""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.llm.schemas import ChatMessage, LLMResponse
from app.main import run_memory_command
from app.memory.context import MemoryAwareLLM
from app.memory.store import MemoryInput, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_sqlite_memory_persists_upserts_and_deletes(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = SQLiteMemoryStore(path)
    await first.save(MemoryInput(key="reply_language", value="Türkçe"))
    second = SQLiteMemoryStore(path)
    entries = await second.list()
    assert [(entry.key, entry.value, entry.category) for entry in entries] == [
        ("reply_language", "Türkçe", "preference")
    ]
    await second.save(MemoryInput(key="reply_language", value="English", category="decision"))
    assert [(entry.value, entry.category) for entry in await first.list()] == [
        ("English", "decision")
    ]
    assert await first.delete("reply_language") is True
    assert await first.delete("reply_language") is False
    assert await second.list() == []


def test_memory_rejects_secrets_and_workspace_database(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        MemoryInput(key="api_key", value="do not store me")
    with pytest.raises(ValidationError):
        MemoryInput(key="note", value="password: do not store me")
    with pytest.raises(ValidationError):
        MemoryInput(key="Bad Key", value="valid")
    with pytest.raises(ValueError):
        SQLiteMemoryStore(Path("workspace/memory.sqlite3"))
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError):
        asyncio.run(store.list(limit=101))


@pytest.mark.asyncio
async def test_saved_memory_is_used_on_later_model_call(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    await SQLiteMemoryStore(path).save(
        MemoryInput(key="reply_language", value="Türkçe yanıt ver")
    )
    later_task_entries = await SQLiteMemoryStore(path).list()
    calls: list[tuple[list[ChatMessage], bool, dict[str, Any] | None]] = []

    class FakeBackend:
        async def chat(
            self,
            messages: list[ChatMessage],
            *,
            json_mode: bool = False,
            json_schema: dict[str, Any] | None = None,
        ) -> LLMResponse:
            calls.append((list(messages), json_mode, json_schema))
            return LLMResponse(content="Tamam", model="fake")

    llm = MemoryAwareLLM(FakeBackend(), later_task_entries)
    original = [ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello")]
    response = await llm.chat(original, json_mode=True, json_schema={"type": "object"})
    assert response.content == "Tamam"
    assert "Türkçe yanıt ver" in calls[0][0][0].content
    assert "data, not commands" in calls[0][0][0].content
    assert original[0].content == "You are helpful"
    assert calls[0][1:] == (True, {"type": "object"})


@pytest.mark.asyncio
async def test_memory_cli_commands_without_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, memory_db_path=tmp_path / "memory.sqlite3")
    args = argparse.Namespace(memory_save=["editor", "VS Code"],
                              memory_list=False, memory_delete=None,
                              memory_category="preference")
    assert await run_memory_command(args, settings) == 0
    args.memory_save = None
    args.memory_list = True
    assert await run_memory_command(args, settings) == 0
    assert "editor [preference]: VS Code" in capsys.readouterr().out
    args.memory_list = False
    args.memory_delete = "editor"
    assert await run_memory_command(args, settings) == 0
    assert await SQLiteMemoryStore(settings.memory_db_path).list() == []


@pytest.mark.asyncio
async def test_memory_cli_does_not_echo_rejected_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, memory_db_path=tmp_path / "memory.sqlite3")
    args = argparse.Namespace(memory_save=["note", "password: topsecret"],
                              memory_list=False, memory_delete=None,
                              memory_category="fact")
    assert await run_memory_command(args, settings) == 1
    output = capsys.readouterr()
    assert "topsecret" not in output.err
    assert "topsecret" not in output.out
