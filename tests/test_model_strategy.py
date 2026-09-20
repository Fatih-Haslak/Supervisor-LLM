from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.workers import WORKER_POLICIES, build_workers
from app.config.settings import Settings
from app.llm.client import LlamaCppClient, LLMClient
from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.strategy import ModelRole, SharedModelStrategy
from app.orchestration.state import AgentState
from app.tools.registry import ToolRegistry


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "choices": [{"message": {"content": "Merhaba"}, "finish_reason": "stop"}],
        }


@pytest.mark.asyncio
async def test_all_roles_share_one_local_model_instance() -> None:
    model = CountingModel()
    async with LlamaCppClient(Settings(_env_file=None), model) as backend:
        strategy = SharedModelStrategy(backend)
        for role in ("chat", "supervisor", "planner", "router", "reviewer",
                     "general", "researcher", "coder", "file_agent"):
            client = strategy.for_role(role)
            assert client is backend
            await client.chat([ChatMessage(role="user", content="Merhaba")])
    assert model.calls == 9


class AnswerModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def chat(
        self, _messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=f'{{"action":"final_answer","answer":"{self.answer}"}}',
            model="test",
        )


@pytest.mark.asyncio
async def test_worker_factory_selects_backend_by_role() -> None:
    backends: dict[ModelRole, LLMClient] = {
        name: AnswerModel(name) for name in WORKER_POLICIES
    }
    workers = build_workers(
        backends["general"], ToolRegistry(), model_for_role=backends.__getitem__
    )
    for name in WORKER_POLICIES:
        result = await workers[name].run("Yanıt ver", AgentState(user_request="Yanıt ver"))
        assert result.answer == name
