"""Loopback-only FastAPI task API and event-streamed local UI."""

import argparse
import asyncio
import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config.settings import Settings
from app.memory.store import MemoryInput, SQLiteMemoryStore
from app.service.conversations import SQLiteConversationStore
from app.service.runtime import AgentRuntime
from app.service.tasks import TaskManager, TaskMode, TaskRunner, TaskView

_UI = Path(__file__).with_name("ui")


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=6000)
    mode: TaskMode = "auto"
    conversation_id: UUID | None = None


class ApprovalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool


def create_app(
    runner: TaskRunner | None = None, *, memory_store: SQLiteMemoryStore | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runner is None:
            settings = Settings()
            store = memory_store or SQLiteMemoryStore(settings.memory_db_path)
            app.state.memories = store
            async with AgentRuntime(settings, Path("workspace")) as runtime:
                manager = TaskManager(
                    runtime.run,
                    conversation_store=SQLiteConversationStore(
                        Path(".local/conversations.sqlite3")
                    ),
                )
                app.state.tasks = manager
                await manager.start()
                try:
                    yield
                finally:
                    await manager.close()
        else:
            app.state.memories = memory_store
            manager = TaskManager(runner)
            app.state.tasks = manager
            await manager.start()
            try:
                yield
            finally:
                await manager.close()

    app = FastAPI(title="Local Agent", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]
    )

    @app.middleware("http")
    async def local_request_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        peer = request.client.host if request.client is not None else ""
        try:
            is_loopback = ipaddress.ip_address(peer).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            return JSONResponse({"detail": "Local access only"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin is not None and origin != expected:
                return JSONResponse(
                    {"detail": "Cross-origin writes are blocked"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'"
        )
        return response

    def manager(request: Request) -> TaskManager:
        return cast(TaskManager, request.app.state.tasks)

    def memories(request: Request) -> SQLiteMemoryStore:
        store = getattr(request.app.state, "memories", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Memory store is unavailable")
        return cast(SQLiteMemoryStore, store)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_UI / "index.html", media_type="text/html")

    @app.get("/ui/app.js", include_in_schema=False)
    async def ui_script() -> FileResponse:
        return FileResponse(_UI / "app.js", media_type="text/javascript")

    @app.get("/ui/style.css", include_in_schema=False)
    async def ui_style() -> FileResponse:
        return FileResponse(_UI / "style.css", media_type="text/css")

    @app.post("/tasks", status_code=202, response_model=TaskView)
    async def create_task(payload: TaskInput, request: Request) -> TaskView:
        try:
            return manager(request).submit(
                payload.message, payload.mode,
                str(payload.conversation_id) if payload.conversation_id else None,
            )
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}", response_model=TaskView)
    async def get_task(task_id: str, request: Request) -> TaskView:
        view = manager(request).get(task_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return view

    @app.get("/conversations/{conversation_id}")
    async def get_conversation(conversation_id: UUID, request: Request) -> dict[str, object]:
        messages = await manager(request).conversation(str(conversation_id))
        return {"conversation_id": str(conversation_id), "messages": [
            message.model_dump() for message in messages
        ]}

    @app.get("/memories")
    async def list_memories(request: Request) -> dict[str, object]:
        return {"memories": [item.model_dump() for item in await memories(request).list()]}

    @app.post("/memories")
    async def save_memory(payload: MemoryInput, request: Request) -> dict[str, object]:
        try:
            entry = await memories(request).save(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Memory value was rejected") from exc
        return {"memory": entry.model_dump()}

    @app.delete("/memories/{key}", status_code=204)
    async def delete_memory(key: str, request: Request) -> None:
        removed = await memories(request).delete(key)
        if not removed:
            raise HTTPException(status_code=404, detail="Memory not found")

    @app.delete("/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: UUID, request: Request) -> None:
        try:
            await manager(request).delete_conversation(str(conversation_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/approval", response_model=TaskView)
    async def decide_approval(
        task_id: str, payload: ApprovalInput, request: Request
    ) -> TaskView:
        tasks = manager(request)
        if tasks.get(task_id) is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if not tasks.decide(task_id, payload.approved):
            raise HTTPException(status_code=409, detail="No approval is pending")
        view = tasks.get(task_id)
        assert view is not None
        return view

    @app.get("/tasks/{task_id}/events")
    async def stream_task(task_id: str, request: Request) -> StreamingResponse:
        tasks = manager(request)
        if tasks.get(task_id) is None:
            raise HTTPException(status_code=404, detail="Task not found")

        async def events() -> AsyncIterator[str]:
            previous = ""
            while not await request.is_disconnected():
                view = tasks.get(task_id)
                if view is None:
                    break
                data = view.model_dump_json(exclude_none=True)
                if data != previous:
                    yield f"event: update\ndata: {data}\n\n"
                    previous = data
                if view.status in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the local agent UI")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("app.api:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
