"""Structured, bounded task planning for the supervisor."""

import json
from collections.abc import Collection

from pydantic import ValidationError

from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage
from app.llm.structured import StructuredOutputError
from app.orchestration.state import TaskPlan


class Planner:
    def __init__(
        self, llm: LLMClient, available_agents: Collection[str], *, max_retries: int = 2
    ) -> None:
        if not available_agents:
            raise ValueError("Planner needs at least one agent")
        if not 0 <= max_retries <= 5:
            raise ValueError("Planner retry limit is outside allowed bounds")
        self._llm = llm
        self._agents = frozenset(available_agents)
        self._max_retries = max_retries
        self._schema = TaskPlan.model_json_schema()
        self._schema["$defs"]["PlannedTask"]["properties"]["agent"]["enum"] = sorted(
            self._agents
        )

    async def plan(self, request: str) -> TaskPlan:
        if not request.strip():
            raise ValueError("User request must not be empty")
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a task planner. /no_think\n"
                    "Return a concise JSON plan with 1 to 4 ordered tasks. Each task has "
                    "id, agent, task, and depends_on (list of earlier task IDs). "
                    "Use only these agent names: "
                    + json.dumps(sorted(self._agents), ensure_ascii=False)
                    + ". Preserve the user's names, exact file paths, numbers, and "
                    "constraints in task descriptions. Every requested action must be "
                    "represented. Do not copy an agent name or purpose as a task. "
                    "For a simple request use one task. Do not plan web search; "
                    "researcher can search only local workspace documents. "
                    "Use coder for Python code implementation and file_agent for "
                    "ordinary file operations. Return only the JSON object."
                ),
            ),
            ChatMessage(role="user", content=request),
        ]
        for attempt in range(self._max_retries + 1):
            response = await self._llm.chat(messages, json_schema=self._schema)
            try:
                plan = TaskPlan.model_validate_json(response.content)
                if any(task.agent not in self._agents for task in plan.tasks):
                    raise ValueError("Plan selected an unavailable agent")
                return plan
            except (ValidationError, ValueError) as exc:
                if attempt == self._max_retries:
                    raise StructuredOutputError(
                        f"Planner returned an invalid plan after {attempt + 1} attempts"
                    ) from exc
                if response.content:
                    messages.append(ChatMessage(role="assistant", content=response.content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Invalid plan. Use only listed agents, unique task IDs, and "
                            "dependencies on earlier task IDs. Return one JSON object."
                        ),
                    )
                )
        raise AssertionError("Unreachable retry state")
