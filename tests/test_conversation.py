from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.agents.chat import automatic_mode, run_chat
from app.llm.schemas import ChatMessage, LLMResponse
from app.memory.store import MemoryInput, SQLiteMemoryStore
from app.orchestration.state import AgentState
from app.security.approvals import Approver
from app.service.conversations import SQLiteConversationStore
from app.service.tasks import TaskManager, TaskMode


class ChatLLM:
    def __init__(self) -> None:
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, object] | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        return LLMResponse(content="Adın Fatih.", model="scripted")


def test_automatic_mode_keeps_personal_chat_out_of_planner() -> None:
    assert automatic_mode("Merhaba, benim adım Fatih") == "chat"
    assert automatic_mode("Benim adım neydi?") == "chat"
    assert automatic_mode("3+44 kaç eder?") == "single"
    assert automatic_mode("workspace/sales.csv dosyasını analiz et") == "supervisor"
    assert automatic_mode("FATİH TEKKE KİMDİR?") == "supervisor"


@pytest.mark.asyncio
async def test_chat_model_receives_previous_turns() -> None:
    llm = ChatLLM()
    history = [
        ChatMessage(role="user", content="Benim adım Fatih."),
        ChatMessage(role="assistant", content="Memnun oldum Fatih."),
    ]
    state = await run_chat(llm, "Benim adım neydi?", history)
    assert state.final_answer == "Adın Fatih."
    assert llm.requests == []

    second = await run_chat(llm, "Adım neydi?", history)
    assert second.final_answer == "Adın Fatih."


@pytest.mark.asyncio
async def test_chat_uses_history_for_ordinary_followup_and_identity() -> None:
    llm = ChatLLM()
    history = [
        ChatMessage(role="user", content="Merhaba, benim adım Fatih."),
        ChatMessage(role="assistant", content="Memnun oldum Fatih."),
    ]
    state = await run_chat(llm, "Nasılsın?", history)
    assert state.final_answer == "Adın Fatih."
    assert [message.content for message in llm.requests[0][-3:]] == [
        "Merhaba, benim adım Fatih.", "Memnun oldum Fatih.", "Nasılsın?"
    ]
    identity = await run_chat(llm, "Senin adın ne?", history)
    assert identity.final_answer == "Ben Yerel Agent adlı asistanım."
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_conversation_survives_task_manager_restart(tmp_path: Path) -> None:
    store_path = tmp_path / "conversations.sqlite3"
    seen: list[list[str]] = []

    async def run(
        message: str, _mode: TaskMode, _approver: Approver,
        history: list[ChatMessage],
    ) -> AgentState:
        seen.append([item.content for item in history])
        state = AgentState.for_request(message, ChatMessage(role="system", content="test"))
        state.finish("Yanıt " + message)
        return state

    manager = TaskManager(run, conversation_store=SQLiteConversationStore(store_path))
    await manager.start()
    first = manager.submit("Adım Fatih", "auto")
    await manager._queue.join()
    await manager.close()

    manager = TaskManager(run, conversation_store=SQLiteConversationStore(store_path))
    await manager.start()
    try:
        second = manager.submit("Adım ne?", "auto", first.conversation_id)
        await manager._queue.join()
        assert seen == [[], ["Adım Fatih", "Yanıt Adım Fatih"]]
        assert second.conversation_id == first.conversation_id
        assert len(await manager.conversation(first.conversation_id)) == 4
        await manager.delete_conversation(first.conversation_id)
        assert await manager.conversation(first.conversation_id) == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_sqlite_stores_release_files_on_windows() -> None:
    with TemporaryDirectory(prefix="agent-store-test-") as temporary:
        root = Path(temporary)
        conversations = SQLiteConversationStore(root / "conversations.sqlite3")
        memories = SQLiteMemoryStore(root / "memory.sqlite3")
        await conversations.append("session", "Merhaba", "Selam")
        assert len(await conversations.list("session")) == 2
        await memories.save(MemoryInput(key="language", value="Turkish"))
        assert len(await memories.list()) == 1
        await conversations.delete("session")
        await memories.delete("language")
    assert not root.exists()
