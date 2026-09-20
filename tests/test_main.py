import sys
from typing import Any

import pytest

from app import main
from app.config.settings import Settings
from app.llm.structured import StructuredOutputError


@pytest.mark.asyncio
async def test_interactive_agent_continues_after_one_failed_task(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, _settings: Settings) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    class FakeAgent:
        def __init__(self, *_args: Any) -> None:
            pass

        async def run(self, request: str) -> Any:
            calls.append(request)
            if request == "ilk":
                raise StructuredOutputError("invalid decision")
            return type("Result", (), {"answer": "Tamam", "tool_calls": []})()

    requests = iter(["ilk", "ikinci", "çık"])
    monkeypatch.setattr(main, "LlamaCppClient", FakeClient)
    monkeypatch.setattr(main, "SingleAgent", FakeAgent)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(requests))

    exit_code = await main.run_agent(None, Settings(_env_file=None), False, False)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == ["ilk", "ikinci"]
    assert "Hata [INVALID_OUTPUT]" in captured.err
    assert "Asistan> Tamam" in captured.out


def test_cli_hides_unexpected_exception_detail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken_settings() -> Settings:
        raise RuntimeError("PRIVATE_INTERNAL_DETAIL")

    monkeypatch.setattr(sys, "argv", ["app.main", "--prompt", "hello"])
    monkeypatch.setattr(main, "Settings", broken_settings)
    assert main.main() == 1
    error = capsys.readouterr().err
    assert "Hata [INTERNAL_ERROR]" in error
    assert "PRIVATE_INTERNAL_DETAIL" not in error
