"""Route simple requests directly to one worker, with supervisor fallback."""

import json
import re
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.supervisor import Worker
from app.llm.client import LLMClient, LLMContextOverflowError
from app.llm.schemas import ChatMessage
from app.llm.structured import StructuredOutputError
from app.observability.events import agent_span, record
from app.orchestration.state import AgentOutput, AgentState, RoutingRecord


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class PlannedSupervisor(Protocol):
    async def run_planned(self, user_request: str) -> AgentState: ...


class Router:
    def __init__(
        self,
        llm: LLMClient,
        workers: Mapping[str, Worker],
        supervisor: PlannedSupervisor,
        *,
        confidence_threshold: float = 0.8,
        max_retries: int = 2,
        worker_descriptions: Mapping[str, str] | None = None,
        review_code_with_supervisor: bool = False,
    ) -> None:
        if not workers:
            raise ValueError("Router needs at least one worker")
        if not 0 <= confidence_threshold <= 1 or not 0 <= max_retries <= 5:
            raise ValueError("Router limits are outside allowed bounds")
        self._llm = llm
        self._workers = dict(workers)
        self._supervisor = supervisor
        self._threshold = confidence_threshold
        self._max_retries = max_retries
        self._review_code_with_supervisor = review_code_with_supervisor
        self._descriptions = {
            name: (worker_descriptions or {}).get(name, name) for name in sorted(workers)
        }
        self._schema = RouteDecision.model_json_schema()
        self._schema["properties"]["agent"]["enum"] = [*sorted(workers), "supervisor"]

    async def decide(self, request: str) -> RouteDecision:
        if not request.strip():
            raise ValueError("User request must not be empty")
        if len(request) > 6000:
            raise LLMContextOverflowError("Routing request exceeds the model context budget")
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a lightweight task router. /no_think\n"
                    "Choose exactly one worker for a single independent action. "
                    "Choose supervisor for requests with multiple actions, dependencies, "
                    "planning, review, or uncertainty about the right worker. "
                    "For Python code implementation choose coder; for local document "
                    "search choose researcher; for ordinary files choose file_agent; "
                    "for arithmetic choose general. Researcher has no internet access.\n"
                    "Workers: " + json.dumps(self._descriptions, ensure_ascii=False) + "\n"
                    'Return only JSON: {"agent":"...","confidence":0.0}. '
                    "Confidence is 0 to 1. Do not answer the task."
                ),
            ),
            ChatMessage(role="user", content=request),
        ]
        for attempt in range(self._max_retries + 1):
            response = await self._llm.chat(messages, json_schema=self._schema)
            try:
                decision = RouteDecision.model_validate_json(response.content)
                if decision.agent not in self._workers and decision.agent != "supervisor":
                    raise ValueError("Router selected an unavailable agent")
                return decision
            except (ValidationError, ValueError) as exc:
                record("model_retry", agent="router", retry_count=attempt + 1)
                if attempt == self._max_retries:
                    raise StructuredOutputError(
                        f"Router returned an invalid route after {attempt + 1} attempts"
                    ) from exc
                if response.content:
                    messages.append(ChatMessage(role="assistant", content=response.content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content="Choose one listed worker or supervisor. Return valid JSON only.",
                    )
                )
        raise AssertionError("Unreachable retry state")

    async def run(self, user_request: str) -> AgentState:
        request = user_request.strip()
        try:
            decision = await self.decide(request)
        except StructuredOutputError:
            decision = RouteDecision(agent="supervisor", confidence=0)
        selected = "supervisor"
        if (
            decision.agent != "supervisor"
            and decision.confidence >= self._threshold
            and not self._explicit_sequence(request)
            and not (self._review_code_with_supervisor and decision.agent == "coder")
        ):
            selected = decision.agent
        route = RoutingRecord(
            suggested_agent=decision.agent,
            confidence=decision.confidence,
            selected_agent=selected,
        )
        if selected == "supervisor":
            state = await self._supervisor.run_planned(request)
            state.route = route
            return state

        state = AgentState(
            user_request=request,
            messages=[ChatMessage(role="user", content=request)],
            current_agent=selected,
            pending_tasks=[request],
            route=route,
            step_count=1,
        )
        record("route_selected", agent=selected)
        with agent_span(selected):
            result = await self._workers[selected].run(request, state)
        state.tool_results.extend(result.tool_results)
        state.agent_outputs.append(AgentOutput(agent=selected, task=request, answer=result.answer))
        state.finish(result.answer)
        return state

    @staticmethod
    def _explicit_sequence(request: str) -> bool:
        return bool(
            re.search(
                r"\b(?:önce|ilk olarak|first)\b[\s\S]*\b(?:sonra|ardından|then)\b",
                request,
                flags=re.IGNORECASE,
            )
        )
