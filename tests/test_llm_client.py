import sys
from typing import Any
from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.llm.client import LlamaCppClient, LLMError, visible_content
from app.llm.schemas import ChatMessage


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.last_kwargs = kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "Merhaba"}]
        return {
            "model": "test-model",
            "choices": [{"message": {"content": "Selam"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }


@pytest.mark.asyncio
async def test_chat_normalizes_local_response() -> None:
    model = FakeModel()
    client = LlamaCppClient(Settings(_env_file=None), model)
    response = await client.chat([ChatMessage(role="user", content="Merhaba")])
    assert response.content == "Selam"
    assert response.model == "test-model"
    assert response.usage is not None and response.usage.total_tokens == 5
    assert model.calls == 1


@pytest.mark.asyncio
async def test_json_mode_constrains_backend_output() -> None:
    model = FakeModel()
    client = LlamaCppClient(Settings(_env_file=None), model)
    await client.chat([ChatMessage(role="user", content="Merhaba")], json_mode=True)
    assert model.last_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_schema_is_forwarded_to_backend() -> None:
    model = FakeModel()
    client = LlamaCppClient(Settings(_env_file=None), model)
    schema = {"type": "object", "required": ["action"]}
    await client.chat([ChatMessage(role="user", content="Merhaba")], json_schema=schema)
    assert model.last_kwargs["response_format"] == {"type": "json_object", "schema": schema}


@pytest.mark.asyncio
async def test_missing_model_file_is_reported() -> None:
    client = LlamaCppClient(Settings(_env_file=None, model_path="missing.gguf"))
    with pytest.raises(LLMError, match="not found"):
        await client.chat([ChatMessage(role="user", content="Merhaba")])


def test_missing_llama_cpp_explains_python_environment(tmp_path: Any) -> None:
    model_path = tmp_path / "test.gguf"
    model_path.write_bytes(b"placeholder")
    client = LlamaCppClient(Settings(_env_file=None, model_path=model_path))
    with patch.dict(sys.modules, {"llama_cpp": None}):
        with pytest.raises(LLMError, match=".venv Python"):
            client._load_model()


def test_reasoning_block_is_not_user_facing() -> None:
    assert visible_content("<think>internal reasoning</think>\n\nAnkara") == "Ankara"
    assert visible_content("Ankara") == "Ankara"
