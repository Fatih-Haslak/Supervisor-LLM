"""LangGraph orchestration over the existing supervisor, workers, and reviewer."""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal, TypedDict, cast

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.agents.reviewer import ReviewerAgent
from app.agents.supervisor import (
    DelegateDecision,
    Supervisor,
    SupervisorFinalDecision,
    SupervisorLimitError,
    Worker,
    WorkerResult,
)
from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage
from app.observability.events import agent_span, record
from app.orchestration.state import AgentState, ReviewRecord


class GraphState(TypedDict):
    agent_state: AgentState
    next_agent: str
    task: str
    last_result: WorkerResult | None
    review_attempt: int
    feedback: list[str]
    review_route: Literal["coder", "writer", "supervisor", "end"]


class GraphOrchestrator:
    """Explicit supervisor -> worker -> reviewer graph with bounded retries."""

    def __init__(
        self,
        llm: LLMClient,
        workers: Mapping[str, Worker],
        *,
        reviewer: ReviewerAgent | None = None,
        max_rounds: int = 6,
        max_json_retries: int = 2,
        max_review_retries: int = 2,
        worker_descriptions: Mapping[str, str] | None = None,
        planner_llm: LLMClient | None = None,
    ) -> None:
        self._supervisor = Supervisor(
            llm, workers, reviewer=reviewer, max_rounds=max_rounds,
            max_json_retries=max_json_retries, max_review_retries=max_review_retries,
            worker_descriptions=worker_descriptions,
            planner_llm=planner_llm,
        )
        self._reviewer = reviewer
        self._max_rounds = max_rounds
        self._max_review_retries = max_review_retries
        self._workers = dict(workers)

        builder = StateGraph(GraphState)
        builder.add_node("supervisor", RunnableLambda(self._supervisor_node))
        builder.add_edge(START, "supervisor")
        for name in self._workers:
            builder.add_node(name, RunnableLambda(self._worker_node(name)))
            if name in {"coder", "writer"} and reviewer is not None:
                builder.add_edge(name, "reviewer")
            else:
                builder.add_edge(name, "supervisor")
        reviewed_roles = {"coder", "writer"} & self._workers.keys()
        if reviewer is not None and reviewed_roles:
            builder.add_node("reviewer", RunnableLambda(self._reviewer_node))
            builder.add_conditional_edges(
                "reviewer", self._after_review,
                {**{name: name for name in reviewed_roles},
                 "supervisor": "supervisor", "end": END},
            )
        builder.add_conditional_edges(
            "supervisor", self._after_supervisor,
            {**{name: name for name in self._workers}, "end": END},
        )
        self._graph = builder.compile()

    async def _supervisor_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["agent_state"]
        if state.step_count >= self._max_rounds:
            raise SupervisorLimitError("Maximum supervisor rounds reached")
        state.step_count += 1
        decision = await self._supervisor._decide(state)
        state.messages.append(ChatMessage(role="assistant", content=decision.model_dump_json()))
        updated = graph_state.copy()
        updated["agent_state"] = state
        if isinstance(decision, SupervisorFinalDecision):
            state.finish(decision.answer)
            updated["next_agent"] = "end"
            return updated
        assert isinstance(decision, DelegateDecision)
        state.pending_tasks.append(decision.task)
        updated["next_agent"] = decision.next_agent
        updated["task"] = decision.task
        updated["review_attempt"] = 0
        updated["feedback"] = []
        return updated

    @staticmethod
    def _after_supervisor(graph_state: GraphState) -> str:
        return graph_state["next_agent"]

    def _worker_node(
        self, name: str
    ) -> Callable[[GraphState], Awaitable[GraphState]]:
        async def run(graph_state: GraphState) -> GraphState:
            state = graph_state["agent_state"]
            task = graph_state["task"]
            assignment = task
            if name in {"coder", "writer"} and graph_state["feedback"]:
                assignment += (
                    "\nReviewer feedback to fix: "
                    + json.dumps(graph_state["feedback"], ensure_ascii=False)
                    + "\nAddress this feedback in the answer or repair existing files "
                    "with overwrite=true when the task uses workspace files."
                )
            result = await self._supervisor._run_worker(
                state, name, assignment,
                complete=not (name in {"coder", "writer"} and self._reviewer is not None),
            )
            updated = graph_state.copy()
            updated["agent_state"] = state
            updated["last_result"] = result
            return updated

        return run

    async def _reviewer_node(self, graph_state: GraphState) -> GraphState:
        assert self._reviewer is not None
        result = graph_state["last_result"]
        assert result is not None
        state = graph_state["agent_state"]
        task = graph_state["task"]
        attempt = graph_state["review_attempt"] + 1
        state.current_agent = "reviewer"
        with agent_span("reviewer"):
            review = await self._reviewer.review(
                state.user_request, task, result, evidence_tools=state.tool_results
            )
        record("review_verdict", agent="reviewer", success=review.verdict.status == "pass",
               retry_count=attempt - 1)
        state.tool_results.extend(review.tool_calls)
        state.reviews.append(
            ReviewRecord(
                agent=graph_state["next_agent"], task=task, attempt=attempt,
                status=review.verdict.status, issues=review.verdict.issues,
            )
        )
        state.messages.append(
            ChatMessage(
                role="user", content="Reviewer verdict: " + review.verdict.model_dump_json()
            )
        )
        state.current_agent = "supervisor"
        if review.verdict.status == "pass":
            state.completed_tasks.append(task)
            state.pending_tasks.remove(task)
            route: Literal["coder", "writer", "supervisor", "end"] = "supervisor"
        elif attempt <= self._max_review_retries:
            route = cast(Literal["coder", "writer"], graph_state["next_agent"])
        else:
            state.fail(
                "İnceleme geçilemedi; görev tamamlanmadı. Sorunlar: "
                + "; ".join(review.verdict.issues)
            )
            route = "end"
        updated = graph_state.copy()
        updated["agent_state"] = state
        updated["review_attempt"] = attempt
        updated["feedback"] = review.verdict.issues
        updated["review_route"] = route
        return updated

    @staticmethod
    def _after_review(graph_state: GraphState) -> str:
        return graph_state["review_route"]

    async def run(
        self, user_request: str, *, history: Sequence[ChatMessage] = ()
    ) -> AgentState:
        state = self._supervisor._new_state(user_request, history)
        initial: GraphState = {
            "agent_state": state, "next_agent": "", "task": "", "last_result": None,
            "review_attempt": 0, "feedback": [], "review_route": "supervisor",
        }
        limit = self._max_rounds * (self._max_review_retries + 3) + 5
        result = await self._graph.ainvoke(initial, config={"recursion_limit": limit})
        return cast(AgentState, result["agent_state"])
