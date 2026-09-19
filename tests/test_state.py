from app.llm.schemas import ChatMessage
from app.orchestration.state import AgentState


def test_state_tracks_request_and_completion() -> None:
    state = AgentState.for_request("  Merhaba  ", ChatMessage(role="system", content="Test"))
    assert state.task_id
    assert state.user_request == "Merhaba"
    assert state.current_agent == "single"
    assert state.pending_tasks == ["Merhaba"]
    assert state.messages[-1].content == "Merhaba"
    state.finish("Selam")
    assert state.completed_tasks == ["Merhaba"]
    assert state.pending_tasks == []
    assert state.current_agent is None
    assert state.final_answer == "Selam"


def test_state_instances_do_not_share_mutable_lists() -> None:
    first = AgentState(user_request="Bir")
    second = AgentState(user_request="İki")
    first.pending_tasks.append("Bir")
    assert second.pending_tasks == []
    assert first.task_id != second.task_id
