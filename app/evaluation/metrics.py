"""Pure evaluation contracts and aggregate metrics."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.observability.events import TraceEvent
from app.orchestration.state import AgentState


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    task: str = Field(min_length=1, max_length=2000)
    mode: Literal["auto", "single", "supervisor", "plan", "router", "graph"] = "auto"
    prior_turns: list[str] = Field(default_factory=list, max_length=8)
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_agent: str | None = None
    expected_answer_contains: str = Field(min_length=1)
    forbidden_answer_contains: list[str] = Field(default_factory=list)
    expected_language: Literal["tr", "en"] | None = "tr"
    expected_file_contains: dict[str, str] = Field(default_factory=dict)
    fixtures: dict[str, str] = Field(default_factory=dict, max_length=5)


class EvaluationObservation(BaseModel):
    case: EvaluationCase
    state: AgentState | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    error_code: str | None = None
    files_match: bool = True


class CaseScore(BaseModel):
    id: str
    success: bool
    tools_match: bool
    route_match: bool | None = None
    error_code: str | None = None
    files_match: bool = True
    language_match: bool = True
    answer: str | None = None
    actual_tools: list[str] = Field(default_factory=list)
    actual_agents: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    case_count: int
    task_success_rate: float
    tool_selection_accuracy: float
    agent_routing_accuracy: float | None
    average_steps: float
    tool_error_rate: float | None
    average_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reviewer_pass_rate: float | None
    retry_rate: float
    cases: list[CaseScore]


def evaluate(observations: list[EvaluationObservation]) -> EvaluationReport:
    if not observations:
        raise ValueError("At least one evaluation observation is required")
    scores: list[CaseScore] = []
    routes_checked = 0
    routes_correct = 0
    tool_calls = 0
    tool_errors = 0
    reviews = 0
    passed_reviews = 0
    prompt_tokens = 0
    completion_tokens = 0
    retried_cases = 0
    for observation in observations:
        state = observation.state
        case = observation.case
        actual_tools = {call.tool for call in state.tool_results} if state else set()
        tools_match = (
            set(case.expected_tools).issubset(actual_tools)
            and not actual_tools.intersection(case.forbidden_tools)
        )
        route_match: bool | None = None
        if case.expected_agent is not None:
            routes_checked += 1
            route_match = bool(
                state and state.agent_outputs
                and state.agent_outputs[0].agent == case.expected_agent
            )
            routes_correct += int(route_match)
        answered = bool(
            state and state.final_answer and not state.pending_tasks
            and case.expected_answer_contains.casefold() in state.final_answer.casefold()
            and not any(
                phrase.casefold() in state.final_answer.casefold()
                for phrase in case.forbidden_answer_contains
            )
        )
        answer = state.final_answer if state else None
        english_words = re.findall(
            r"\b(?:the|has|been|was|were|successfully|passed|fixed|out|of|"
            r"with|and|this|that|file|function)\b",
            answer or "", flags=re.IGNORECASE,
        )
        language_match = (
            case.expected_language != "tr" or len(english_words) < 3
        )
        success = (
            answered and tools_match and route_match is not False
            and observation.files_match and language_match
            and observation.error_code is None
        )
        scores.append(
            CaseScore(
                id=case.id, success=success,
                tools_match=tools_match, route_match=route_match,
                error_code=observation.error_code, files_match=observation.files_match,
                language_match=language_match, answer=answer,
                actual_tools=[call.tool for call in state.tool_results] if state else [],
                actual_agents=[output.agent for output in state.agent_outputs] if state else [],
            )
        )
        traced_tool_calls = [
            event for event in observation.events if event.event == "tool_call"
        ]
        if traced_tool_calls:
            tool_calls += len(traced_tool_calls)
            tool_errors += sum(event.success is False for event in traced_tool_calls)
        elif state:
            tool_calls += len(state.tool_results)
            tool_errors += sum(not call.result.success for call in state.tool_results)
        if state:
            reviews += len(state.reviews)
            passed_reviews += sum(review.status == "pass" for review in state.reviews)
        retried_cases += int(any(event.event == "model_retry" for event in observation.events))
        prompt_tokens += sum(
            event.prompt_tokens or 0 for event in observation.events if event.event == "model_call"
        )
        completion_tokens += sum(
            event.completion_tokens or 0
            for event in observation.events if event.event == "model_call"
        )
    count = len(observations)
    return EvaluationReport(
        case_count=count,
        task_success_rate=sum(score.success for score in scores) / count,
        tool_selection_accuracy=sum(score.tools_match for score in scores) / count,
        agent_routing_accuracy=routes_correct / routes_checked if routes_checked else None,
        average_steps=sum(obs.state.step_count if obs.state else 0 for obs in observations) / count,
        tool_error_rate=tool_errors / tool_calls if tool_calls else None,
        average_latency_ms=sum(obs.latency_ms for obs in observations) / count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reviewer_pass_rate=passed_reviews / reviews if reviews else None,
        retry_rate=retried_cases / count,
        cases=scores,
    )
