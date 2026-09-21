import asyncio

import pytest

from app.llm.schemas import ChatMessage
from app.orchestration.state import AgentState
from app.security.approvals import ApprovalRequest, Approver, ToolApprovalError
from app.service.tasks import TaskManager, TaskMode


async def wait_status(manager: TaskManager, task_id: str, status: str) -> None:
    for _ in range(100):
        view = manager.get(task_id)
        if view is not None and view.status == status:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Task never reached {status}")


@pytest.mark.asyncio
async def test_queue_runs_one_task_at_a_time() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def run(
        message: str, _mode: TaskMode, _approver: Approver,
        _history: list[ChatMessage],
    ) -> AgentState:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.set()
        await release.wait()
        state = AgentState.for_request(message, ChatMessage(role="system", content="test"))
        state.finish(message)
        active -= 1
        return state

    manager = TaskManager(run)
    await manager.start()
    try:
        first = manager.submit("ilk", "single")
        second = manager.submit("ikinci", "single")
        await started.wait()
        queued = manager.get(second.task_id)
        assert queued is not None and queued.status == "queued"
        release.set()
        await wait_status(manager, second.task_id, "completed")
        assert peak == 1
        completed = manager.get(first.task_id)
        assert completed is not None and completed.answer == "ilk"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_approval_requires_explicit_decision() -> None:
    async def run(
        _message: str, _mode: TaskMode, approver: Approver,
        _history: list[ChatMessage],
    ) -> AgentState:
        granted = await approver.request_approval(ApprovalRequest(
            tool="file_write", arguments={"path": "note.txt", "content": "Merhaba"}
        ))
        if not granted:
            raise ToolApprovalError("ApprovalDenied")
        state = AgentState(user_request="yaz")
        state.finish("Yazıldı")
        return state

    manager = TaskManager(run)
    await manager.start()
    try:
        task = manager.submit("yaz", "plan")
        await wait_status(manager, task.task_id, "waiting_approval")
        view = manager.get(task.task_id)
        assert view is not None and view.pending_approval is not None
        assert view.pending_approval.arguments["content"] == "Merhaba"
        assert manager.decide(task.task_id, False)
        assert not manager.decide(task.task_id, True)
        await wait_status(manager, task.task_id, "failed")
        final = manager.get(task.task_id)
        assert final is not None and final.error is not None
        assert final.error.code == "APPROVAL_DENIED"
    finally:
        await manager.close()
