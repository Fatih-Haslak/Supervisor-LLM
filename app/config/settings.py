"""Validated direct GGUF inference settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf")
    memory_db_path: Path = Path(".local/agent_memory.sqlite3")
    model_name: str = "qwen3-8b-q4_k_m"
    context_length: int = Field(default=4096, ge=512, le=32768)
    gpu_layers: int = Field(default=-1, ge=-1)
    cuda_dll_directory: Path | None = None
    llm_max_tokens: int = Field(default=512, ge=1, le=8192)
    llm_timeout_seconds: float = Field(default=120, gt=0, le=600)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    log_level: str = "INFO"

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: Path) -> Path:
        if value.suffix.lower() != ".gguf":
            raise ValueError("Model file must be a GGUF file")
        return value.expanduser().resolve()

    @field_validator("cuda_dll_directory")
    @classmethod
    def resolve_cuda_directory(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None

    @field_validator("memory_db_path")
    @classmethod
    def resolve_memory_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid log level")
        return level
