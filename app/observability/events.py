"""Bounded structured trace events without prompts or tool payloads."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

EventName = Literal[
    "task_started", "task_completed", "task_failed", "agent_enter", "agent_exit",
    "agent_error",
    "model_call", "model_error", "model_retry", "tool_call",
    "review_verdict", "route_selected",
    "approval_requested", "approval_resolved",
]
_active: ContextVar["TraceRecorder | None"] = ContextVar("active_trace", default=None)
_safe_name = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")


def _name(value: str | None) -> str | None:
    if value is None:
        return None
    return value if _safe_name.fullmatch(value) else "unknown"


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sequence: int = Field(ge=1)
    event: EventName
    agent: str | None = None
    tool: str | None = None
    success: bool | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    prompt_chars: int | None = Field(default=None, ge=0)
    response_chars: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    error_type: str | None = None


class TraceRecorder:
    def __init__(self, *, max_events: int = 200) -> None:
        if not 1 <= max_events <= 1000:
            raise ValueError("Trace event limit must be between 1 and 1000")
        self.task_id = uuid4().hex
        self.trace_id = uuid4().hex
        self.max_events = max_events
        self.events: list[TraceEvent] = []
        self.dropped = 0

    def record(
        self, event: EventName, *, agent: str | None = None, tool: str | None = None,
        success: bool | None = None, duration_ms: float | None = None,
        prompt_chars: int | None = None, response_chars: int | None = None,
        prompt_tokens: int | None = None, completion_tokens: int | None = None,
        retry_count: int | None = None, error_type: str | None = None,
    ) -> None:
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        self.events.append(
            TraceEvent(
                task_id=self.task_id, trace_id=self.trace_id,
                sequence=len(self.events) + 1, event=event, agent=_name(agent),
                tool=_name(tool), success=success, duration_ms=duration_ms,
                prompt_chars=prompt_chars, response_chars=response_chars,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                retry_count=retry_count, error_type=_name(error_type),
            )
        )


@contextmanager
def trace_session(recorder: TraceRecorder) -> Iterator[None]:
    token = _active.set(recorder)
    try:
        yield
    finally:
        _active.reset(token)


def current_task_id() -> str:
    active = _active.get()
    return active.task_id if active is not None else uuid4().hex


def record(
    event: EventName, *, agent: str | None = None, tool: str | None = None,
    success: bool | None = None, duration_ms: float | None = None,
    prompt_chars: int | None = None, response_chars: int | None = None,
    prompt_tokens: int | None = None, completion_tokens: int | None = None,
    retry_count: int | None = None, error_type: str | None = None,
) -> None:
    active = _active.get()
    if active is not None:
        active.record(
            event, agent=agent, tool=tool, success=success,
            duration_ms=duration_ms, prompt_chars=prompt_chars,
            response_chars=response_chars, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, retry_count=retry_count,
            error_type=error_type,
        )


@contextmanager
def agent_span(name: str) -> Iterator[None]:
    start = perf_counter()
    record("agent_enter", agent=name)
    success = False
    try:
        yield
        success = True
    except Exception as exc:
        record("agent_error", agent=name, error_type=type(exc).__name__)
        raise
    finally:
        record(
            "agent_exit", agent=name, success=success,
            duration_ms=round((perf_counter() - start) * 1000, 2),
        )
