"""The task state passed between future supervisor and worker agents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.schemas import ChatMessage
from app.observability.events import current_task_id
from app.tools.base import ToolResult


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, object]
    result: ToolResult


class AgentOutput(BaseModel):
    agent: str
    task: str
    answer: str


class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    agent: str = Field(min_length=1)
    task: str = Field(min_length=1)
    depends_on: list[int] = Field(default_factory=list)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[PlannedTask] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        seen: set[int] = set()
        for task in self.tasks:
            if task.id in seen:
                raise ValueError("Plan task IDs must be unique")
            if len(task.depends_on) != len(set(task.depends_on)):
                raise ValueError("Plan dependencies must be unique")
            if any(dependency not in seen for dependency in task.depends_on):
                raise ValueError("Dependencies must refer to earlier tasks")
            seen.add(task.id)
        return self


class RoutingRecord(BaseModel):
    suggested_agent: str
    confidence: float = Field(ge=0, le=1)
    selected_agent: str


class ReviewRecord(BaseModel):
    agent: str
    task: str
    attempt: int = Field(ge=1)
    status: Literal["pass", "fail"]
    issues: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    task_id: str = Field(default_factory=current_task_id)
    user_request: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)
    current_agent: str | None = None
    completed_tasks: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    tool_results: list[ToolCallRecord] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    plan: TaskPlan | None = None
    route: RoutingRecord | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)
    final_answer: str | None = None
    step_count: int = Field(default=0, ge=0)

    @classmethod
    def for_request(cls, user_request: str, system_message: ChatMessage) -> "AgentState":
        request = user_request.strip()
        if not request:
            raise ValueError("User request must not be empty")
        return cls(
            user_request=request,
            messages=[system_message, ChatMessage(role="user", content=request)],
            current_agent="single",
            pending_tasks=[request],
        )

    def finish(self, answer: str) -> None:
        self.final_answer = answer
        self.completed_tasks.extend(self.pending_tasks)
        self.pending_tasks.clear()
        self.current_agent = None

    def fail(self, answer: str) -> None:
        """Finish with pending work preserved for an honest failure report."""
        self.final_answer = answer
        self.current_agent = None
