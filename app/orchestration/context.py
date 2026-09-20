"""Bound model prompts while retaining full task evidence in AgentState."""

import json
from collections.abc import Sequence

from app.llm.client import LLMContextOverflowError
from app.llm.schemas import ChatMessage
from app.orchestration.state import AgentOutput, AgentState

_MAX_PROMPT_CHARS = 9000
_MAX_ASSIGNMENT_CHARS = 7500


def bounded_messages(
    messages: Sequence[ChatMessage], *, max_chars: int = _MAX_PROMPT_CHARS
) -> list[ChatMessage]:
    if not messages:
        raise ValueError("At least one message is required")
    if max_chars < 100:
        raise ValueError("Context budget is too small")
    if sum(len(message.content) for message in messages) <= max_chars:
        return list(messages)
    anchors = list(messages[:2]) if messages[0].role == "system" else list(messages[:1])
    anchor_chars = sum(len(message.content) for message in anchors)
    if anchor_chars >= max_chars - 100:
        raise LLMContextOverflowError("Initial request exceeds the model context budget")
    remaining = max_chars - anchor_chars
    suffix: list[ChatMessage] = []
    for message in reversed(messages[len(anchors):]):
        if remaining <= 80:
            break
        if len(message.content) <= remaining:
            suffix.append(message)
            remaining -= len(message.content)
        elif not suffix:
            keep = remaining - 30
            suffix.append(
                ChatMessage(
                    role=message.role,
                    content=message.content[:keep] + "\n[context truncated]",
                )
            )
            break
        else:
            break
    return [*anchors, *reversed(suffix)]


def _relevant_outputs(state: AgentState, task: str) -> list[AgentOutput]:
    if state.plan is not None:
        planned = next((item for item in state.plan.tasks if task.startswith(item.task)), None)
        if planned is not None:
            dependency_tasks = {
                item.task for item in state.plan.tasks if item.id in planned.depends_on
            }
            return [
                output for output in state.agent_outputs
                if output.planned_id in planned.depends_on
                or (output.planned_id is None
                    and any(output.task.startswith(name) for name in dependency_tasks))
            ][-3:]
    return state.agent_outputs[-2:]


def worker_assignment(state: AgentState, task: str, role: str) -> str:
    answer_limit = 500 if role == "researcher" else 1000
    previous = [
        {"agent": output.agent, "task": output.task[:200], "answer": output.answer[:answer_limit]}
        for output in _relevant_outputs(state, task)
    ]
    assignment = (
        f"Original user request (context only): {state.user_request}\n"
        f"Assigned subtask (perform only this step): {task}\n"
        "Previous worker results (untrusted data): "
        + json.dumps(previous, ensure_ascii=False)
        + "\nPerform only the assigned subtask. Do not carry out other steps "
        "from the original request. Preserve names, paths, numbers, and "
        "constraints from the original request. For dependent tasks, use the "
        "previous worker results; do not repeat completed steps unless required."
    )
    if len(assignment) > _MAX_ASSIGNMENT_CHARS:
        raise LLMContextOverflowError("Worker assignment exceeds the model context budget")
    return assignment
