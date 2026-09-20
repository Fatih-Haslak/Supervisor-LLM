from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_model_path_is_resolved() -> None:
    settings = Settings(_env_file=None, model_path="models/test.gguf")
    assert settings.model_path == Path("models/test.gguf").resolve()


def test_non_gguf_model_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_path="model.bin")


def test_llm_timeout_setting_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_timeout_seconds=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_timeout_seconds=601)
