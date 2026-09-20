"""Loopback-only FastAPI task API and event-streamed local UI."""

import argparse
import asyncio
import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config.settings import Settings
from app.service.runtime import AgentRuntime
from app.service.tasks import TaskManager, TaskMode, TaskRunner, TaskView

_UI = Path(__file__).with_name("ui")


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=6000)
    mode: TaskMode = "plan"


class ApprovalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool


def create_app(runner: TaskRunner | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runner is None:
            async with AgentRuntime(Settings(), Path("workspace")) as runtime:
                manager = TaskManager(runtime.run)
                app.state.tasks = manager
                await manager.start()
                try:
                    yield
                finally:
                    await manager.close()
        else:
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
            return manager(request).submit(payload.message, payload.mode)
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}", response_model=TaskView)
    async def get_task(task_id: str, request: Request) -> TaskView:
        view = manager(request).get(task_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return view

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
