"""Bounded in-memory queue, task snapshots, and explicit browser approval."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.errors import ErrorInfo, describe_error
from app.llm.schemas import ChatMessage
from app.observability.events import TraceEvent, TraceRecorder, record, trace_session
from app.orchestration.state import (
    AgentOutput,
    AgentState,
    ReviewRecord,
    RoutingRecord,
    TaskPlan,
    ToolCallRecord,
)
from app.security.approvals import ApprovalRequest, Approver
from app.service.conversations import ConversationStore, InMemoryConversationStore

TaskMode = Literal["auto", "single", "supervisor", "plan", "router", "graph"]
TaskStatus = Literal["queued", "running", "waiting_approval", "completed", "failed"]
TaskRunner = Callable[[str, TaskMode, Approver, list[ChatMessage]], Awaitable[AgentState]]


class TaskView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    conversation_id: str
    message: str
    mode: TaskMode
    status: TaskStatus
    answer: str | None = None
    error: ErrorInfo | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    reviews: list[ReviewRecord] = Field(default_factory=list)
    plan: TaskPlan | None = None
    route: RoutingRecord | None = None
    pending_approval: ApprovalRequest | None = None


@dataclass
class TaskRecord:
    message: str
    mode: TaskMode
    conversation_id: str
    recorder: TraceRecorder = field(default_factory=TraceRecorder)
    status: TaskStatus = "queued"
    state: AgentState | None = None
    error: ErrorInfo | None = None
    pending_approval: ApprovalRequest | None = None
    approval_future: asyncio.Future[bool] | None = None

    def view(self) -> TaskView:
        state = self.state
        return TaskView(
            task_id=self.recorder.task_id, conversation_id=self.conversation_id,
            message=self.message, mode=self.mode,
            status=self.status,
            answer=state.final_answer if state is not None else None,
            error=self.error, events=list(self.recorder.events),
            tool_calls=list(state.tool_results) if state is not None else [],
            agent_outputs=list(state.agent_outputs) if state is not None else [],
            reviews=list(state.reviews) if state is not None else [],
            plan=state.plan if state is not None else None,
            route=state.route if state is not None else None,
            pending_approval=self.pending_approval,
        )


class BrowserApprover:
    def __init__(self, task: TaskRecord) -> None:
        self._task = task

    async def request_approval(self, request: ApprovalRequest) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._task.pending_approval = request
        self._task.approval_future = future
        self._task.status = "waiting_approval"
        record("approval_requested", tool=request.tool)
        try:
            return await asyncio.wait_for(future, timeout=300)
        except TimeoutError:
            return False
        finally:
            self._task.pending_approval = None
            self._task.approval_future = None
            self._task.status = "running"
            record("approval_resolved", tool=request.tool)


class TaskManager:
    def __init__(
        self, runner: TaskRunner, *, max_queue_size: int = 8, max_history: int = 100,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._runner = runner
        self._queue: asyncio.Queue[TaskRecord] = asyncio.Queue(maxsize=max_queue_size)
        self._records: dict[str, TaskRecord] = {}
        self._max_history = max_history
        self._conversations = conversation_store or InMemoryConversationStore()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Task manager already started")
        self._worker = asyncio.create_task(self._consume())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        for task in self._records.values():
            if task.approval_future is not None and not task.approval_future.done():
                task.approval_future.cancel()

    def submit(
        self, message: str, mode: TaskMode, conversation_id: str | None = None
    ) -> TaskView:
        if self._worker is None:
            raise RuntimeError("Task manager is not running")
        if self._queue.full():
            raise OverflowError("Task queue is full")
        for task_id, task in list(self._records.items()):
            if len(self._records) < self._max_history:
                break
            if task.status in {"completed", "failed"}:
                del self._records[task_id]
        if len(self._records) >= self._max_history:
            raise OverflowError("Task history is full")
        session = str(UUID(conversation_id)) if conversation_id else str(uuid4())
        task = TaskRecord(message=message, mode=mode, conversation_id=session)
        self._records[task.recorder.task_id] = task
        self._queue.put_nowait(task)
        return task.view()

    def get(self, task_id: str) -> TaskView | None:
        task = self._records.get(task_id)
        return task.view() if task is not None else None

    async def conversation(self, conversation_id: str) -> list[ChatMessage]:
        return await self._conversations.list(str(UUID(conversation_id)))

    async def delete_conversation(self, conversation_id: str) -> None:
        session = str(UUID(conversation_id))
        if any(task.conversation_id == session and task.status in {
            "queued", "running", "waiting_approval"
        } for task in self._records.values()):
            raise RuntimeError("Conversation has an active task")
        await self._conversations.delete(session)

    def decide(self, task_id: str, approved: bool) -> bool:
        task = self._records.get(task_id)
        if task is None or task.status != "waiting_approval":
            return False
        future = task.approval_future
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    async def _consume(self) -> None:
        while True:
            task = await self._queue.get()
            try:
                task.status = "running"
                with trace_session(task.recorder):
                    record("task_started")
                    try:
                        history = await self._conversations.list(task.conversation_id)
                        task.state = await self._runner(
                            task.message, task.mode, BrowserApprover(task), history
                        )
                        if task.state.pending_tasks:
                            review_failed = bool(
                                task.state.reviews
                                and task.state.reviews[-1].status == "fail"
                            )
                            task.status = "failed"
                            task.error = ErrorInfo(
                                code="REVIEW_FAILED" if review_failed else "INTERNAL_ERROR",
                                message=(
                                    "Reviewer incelemesi geçilemedi."
                                    if review_failed else "Görev tamamlanmadan durdu."
                                ),
                                retryable=not review_failed,
                            )
                            record(
                                "task_failed",
                                error_type="ReviewFailed" if review_failed else "IncompleteTask",
                            )
                        else:
                            task.status = "completed"
                            if task.state.final_answer:
                                await self._conversations.append(
                                    task.conversation_id, task.message,
                                    task.state.final_answer,
                                )
                            record("task_completed", success=True)
                    except Exception as exc:
                        partial = getattr(exc, "state", None)
                        if isinstance(partial, AgentState):
                            task.state = partial
                        task.error = describe_error(exc)
                        task.status = "failed"
                        record("task_failed", error_type=type(exc).__name__)
                    if task.status == "failed":
                        if task.state and task.state.final_answer:
                            failure = task.state.final_answer
                        elif task.error:
                            failure = f"Hata [{task.error.code}]: {task.error.message}"
                        else:
                            failure = "Görev tamamlanamadı."
                        try:
                            await self._conversations.append(
                                task.conversation_id, task.message, failure
                            )
                        except Exception:
                            task.error = ErrorInfo(
                                code="STORAGE_FAILURE",
                                message="Yerel dosya veya bellek erişimi başarısız.",
                                retryable=False,
                            )
            finally:
                self._queue.task_done()
