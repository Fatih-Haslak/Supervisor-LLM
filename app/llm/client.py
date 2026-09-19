"""Async interface over one in-process GGUF model instance."""

import asyncio
import os
from collections.abc import Sequence
from typing import Any, Protocol, cast

from app.config.settings import Settings
from app.llm.schemas import ChatMessage, LLMResponse, TokenUsage


class LLMError(Exception):
    """Local model loading or generation failed."""


class LLMClient(Protocol):
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class ChatBackend(Protocol):
    def create_chat_completion(self, **kwargs: Any) -> Any: ...


def visible_content(content: str) -> str:
    """Remove a completed reasoning block from models that emit one."""
    stripped = content.lstrip()
    if stripped.startswith("<think>"):
        _, separator, answer = stripped.partition("</think>")
        if separator:
            return answer.strip()
    return content


class LlamaCppClient:
    """Lazy-load one model; serialize calls because the context is shared."""

    def __init__(self, settings: Settings, model: ChatBackend | None = None) -> None:
        self._settings = settings
        self._model = model
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "LlamaCppClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        # Llama owns native memory; release it when the CLI session ends.
        self._model = None

    def _load_model(self) -> ChatBackend:
        if not self._settings.model_path.is_file():
            raise LLMError(f"GGUF model file not found: {self._settings.model_path}")
        try:
            cuda_directory = self._settings.cuda_dll_directory
            if cuda_directory is not None:
                if not cuda_directory.is_dir():
                    raise LLMError(f"CUDA DLL directory not found: {cuda_directory}")
                os.environ["PATH"] = f"{cuda_directory}{os.pathsep}{os.environ['PATH']}"
            from llama_cpp import Llama

            return cast(
                ChatBackend,
                Llama(
                    model_path=str(self._settings.model_path),
                    n_ctx=self._settings.context_length,
                    n_gpu_layers=self._settings.gpu_layers,
                    verbose=False,
                ),
            )
        except LLMError:
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "llama_cpp":
                raise LLMError(
                    "llama-cpp-python bu Python ortamında kurulu değil. "
                    "Projenin .venv Python'unu kullanın."
                ) from exc
            raise LLMError("Yerel model çalışma bağımlılığı eksik") from exc
        except Exception as exc:
            raise LLMError("Could not load the local GGUF model") from exc

    def _generate(
        self,
        messages: Sequence[ChatMessage],
        json_mode: bool,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        assert self._model is not None
        try:
            # llama-cpp-python applies the chat template embedded in the GGUF metadata.
            options: dict[str, Any] = {
                "messages": [message.model_dump() for message in messages],
                "temperature": self._settings.llm_temperature,
                "max_tokens": self._settings.llm_max_tokens,
                "stream": False,
            }
            if json_mode or json_schema is not None:
                options["response_format"] = {"type": "json_object"}
                if json_schema is not None:
                    options["response_format"]["schema"] = json_schema
            result = self._model.create_chat_completion(**options)
            choice = result["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("Invalid model content")
            raw_usage = result.get("usage")
            usage = TokenUsage.model_validate(raw_usage) if raw_usage else None
            return LLMResponse(
                content=visible_content(content),
                model=result.get("model") or self._settings.model_name,
                finish_reason=choice.get("finish_reason"),
                usage=usage,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Local model returned an invalid chat response") from exc
        except Exception as exc:
            raise LLMError("Local model generation failed") from exc

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("At least one chat message is required")
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
            return await asyncio.to_thread(self._generate, messages, json_mode, json_schema)
