"""Async interface over one in-process GGUF model instance."""

import asyncio
import os
from collections.abc import Sequence
from typing import Any, Protocol, cast

from app.config.settings import Settings
from app.llm.schemas import ChatMessage, LLMResponse, TokenUsage


class LLMError(Exception):
    """Local model loading or generation failed."""


class LLMTimeoutError(LLMError):
    """The local model exceeded its cooperative generation deadline."""


class LLMContextOverflowError(LLMError):
    """The prompt exceeded the model context window."""


def _is_context_overflow(exc: Exception) -> bool:
    message = str(exc).lower()
    return "context window" in message or "n_ctx" in message or "context size" in message


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
        self._pending: set[asyncio.Task[LLMResponse]] = set()
        self._closing = False

    async def __aenter__(self) -> "LlamaCppClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._closing = True
        if not self._pending:
            self._model = None

    def _on_done(self, task: asyncio.Task[LLMResponse]) -> None:
        self._pending.discard(task)
        if not task.cancelled():
            task.exception()  # Consume a late error after the caller has timed out.
        if self._closing and not self._pending:
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
        model = self._model
        assert model is not None
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
            result = model.create_chat_completion(**options)
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
        except LLMError:
            raise
        except TimeoutError as exc:
            raise LLMTimeoutError("Local model generation exceeded its time limit") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if _is_context_overflow(exc):
                raise LLMContextOverflowError(
                    "Prompt exceeds the local model context window"
                ) from exc
            raise LLMError("Local model returned an invalid chat response") from exc
        except Exception as exc:
            if _is_context_overflow(exc):
                raise LLMContextOverflowError(
                    "Prompt exceeds the local model context window"
                ) from exc
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
        task = asyncio.create_task(self._chat_serial(messages, json_mode, json_schema))
        self._pending.add(task)
        task.add_done_callback(self._on_done)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self._settings.llm_timeout_seconds
            )
        except TimeoutError as exc:
            raise LLMTimeoutError("Local model generation exceeded its time limit") from exc

    async def _chat_serial(
        self,
        messages: Sequence[ChatMessage],
        json_mode: bool,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
            return await asyncio.to_thread(self._generate, messages, json_mode, json_schema)
