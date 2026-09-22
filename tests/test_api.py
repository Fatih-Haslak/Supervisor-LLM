import time

from fastapi.testclient import TestClient

from app.api import create_app
from app.llm.schemas import ChatMessage
from app.orchestration.state import AgentState
from app.security.approvals import ApprovalRequest, Approver
from app.service.tasks import TaskMode


async def fake_runner(
    message: str, _mode: TaskMode, _approver: Approver,
    _history: list[ChatMessage],
) -> AgentState:
    state = AgentState.for_request(message, ChatMessage(role="system", content="test"))
    state.finish("Yanıt: " + message)
    return state


def test_api_task_lifecycle_stream_and_ui() -> None:
    with TestClient(
        create_app(fake_runner), base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 51000),
    ) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Canlı görev haritası" in page.text
        assert 'id="flow-graph"' in page.text
        response = client.post("/tasks", json={"message": "Merhaba", "mode": "single"})
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        assert response.json()["message"] == "Merhaba"
        assert response.json()["mode"] == "single"
        for _ in range(100):
            status = client.get(f"/tasks/{task_id}").json()
            if status["status"] == "completed":
                break
            time.sleep(0.01)
        assert status["answer"] == "Yanıt: Merhaba"
        conversation_id = status["conversation_id"]
        transcript = client.get(f"/conversations/{conversation_id}")
        assert transcript.status_code == 200
        assert [item["content"] for item in transcript.json()["messages"]] == [
            "Merhaba", "Yanıt: Merhaba"
        ]
        assert client.delete(f"/conversations/{conversation_id}").status_code == 204
        assert client.get(f"/conversations/{conversation_id}").json()["messages"] == []
        events = client.get(f"/tasks/{task_id}/events")
        assert events.status_code == 200
        assert "event: update" in events.text
        assert "task_completed" in events.text


def test_api_rejects_invalid_input_and_cross_origin_post() -> None:
    with TestClient(
        create_app(fake_runner), base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 51000),
    ) as client:
        assert client.post("/tasks", json={"message": ""}).status_code == 422
        assert client.post(
            "/tasks", json={"message": "x"},
            headers={"Origin": "https://outside.example"},
        ).status_code == 403
        assert client.get("/tasks/missing").status_code == 404
        assert client.post(
            "/tasks/missing/approval", json={"approved": True}
        ).status_code == 404


def test_api_exposes_exact_approval_and_accepts_one_decision() -> None:
    async def approval_runner(
        _message: str, _mode: TaskMode, approver: Approver,
        _history: list[ChatMessage],
    ) -> AgentState:
        granted = await approver.request_approval(ApprovalRequest(
            tool="file_write", arguments={"path": "note.txt", "content": "deneme"}
        ))
        state = AgentState(user_request="dosya yaz")
        state.finish("Onaylandı" if granted else "Reddedildi")
        return state

    with TestClient(
        create_app(approval_runner), base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 51000),
    ) as client:
        task_id = client.post("/tasks", json={"message": "dosya yaz"}).json()["task_id"]
        for _ in range(100):
            status = client.get(f"/tasks/{task_id}").json()
            if status["status"] == "waiting_approval":
                break
            time.sleep(0.01)
        assert status["pending_approval"]["arguments"]["content"] == "deneme"
        assert client.post(
            f"/tasks/{task_id}/approval", json={"approved": True}
        ).status_code == 200
        assert client.post(
            f"/tasks/{task_id}/approval", json={"approved": True}
        ).status_code == 409
        for _ in range(100):
            status = client.get(f"/tasks/{task_id}").json()
            if status["status"] == "completed":
                break
            time.sleep(0.01)
        assert status["answer"] == "Onaylandı"


def test_api_followup_receives_same_conversation_and_failed_turn() -> None:
    histories: list[list[str]] = []

    async def runner(
        message: str, _mode: TaskMode, _approver: Approver,
        history: list[ChatMessage],
    ) -> AgentState:
        histories.append([item.content for item in history])
        if message == "bozuk":
            raise ValueError("scripted failure")
        state = AgentState.for_request(message, ChatMessage(role="system", content="test"))
        state.finish("Tamam")
        return state

    with TestClient(
        create_app(runner), base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 51000),
    ) as client:
        first = client.post("/tasks", json={"message": "bozuk"}).json()
        for _ in range(100):
            failed = client.get(f"/tasks/{first['task_id']}").json()
            if failed["status"] == "failed":
                break
            time.sleep(0.01)
        assert failed["error"]["code"] == "INVALID_INPUT"
        second = client.post("/tasks", json={
            "message": "tekrar dene", "conversation_id": first["conversation_id"]
        }).json()
        for _ in range(100):
            completed = client.get(f"/tasks/{second['task_id']}").json()
            if completed["status"] == "completed":
                break
            time.sleep(0.01)
        assert second["conversation_id"] == first["conversation_id"]
        assert completed["answer"] == "Tamam"
        assert histories[1][0] == "bozuk"
        assert "Hata [INVALID_INPUT]" in histories[1][1]
