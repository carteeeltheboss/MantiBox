"""Runtime configuration.

All secrets are `SecretStr` so they cannot be rendered by accident in a log
line, an f-string or an exception message (SPEC §9, AGENTS.md NEVER #1).
Values are injected at runtime from the environment; nothing secret has a
working default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from maintidoc.errors import ConfigurationError

DeviceName = Literal["cuda", "mps", "cpu"]
EnvName = Literal["dev", "test", "prod"]

# The placeholder shipped in .env.example. Anything still equal to this in prod
# is a hard startup failure, not a warning.
_UNSET = "CHANGE-ME"


class _Prefixed(BaseSettings):
    """Shared behaviour: read .env, ignore unknown vars, never dump secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class AppSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="MAINTIDOC_", env_file=".env", extra="ignore")

    env: EnvName = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    secret_key: SecretStr = SecretStr(_UNSET)
    access_token_ttl_minutes: int = Field(default=60, ge=1, le=1440)
    max_upload_megabytes: int = Field(default=10, ge=1, le=100)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return level

    @model_validator(mode="after")
    def _reject_placeholder_secret_in_prod(self) -> AppSettings:
        if self.env == "prod":
            revealed = self.secret_key.get_secret_value()
            if revealed == _UNSET or len(revealed) < 32:
                msg = "MAINTIDOC_SECRET_KEY must be set to >=32 chars when MAINTIDOC_ENV=prod"
                raise ValueError(msg)
        return self


class PostgresSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "postgres"
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = "maintidoc"
    user: str = "maintidoc"
    password: SecretStr = SecretStr(_UNSET)

    @property
    def sqlalchemy_url(self) -> str:
        """psycopg3 async-free URL; the password is interpolated only here."""
        from urllib.parse import quote_plus

        pw = quote_plus(self.password.get_secret_value())
        return f"postgresql+psycopg://{self.user}:{pw}@{self.host}:{self.port}/{self.db}"


class QdrantSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = "qdrant"
    port: int = Field(default=6333, ge=1, le=65535)
    api_key: SecretStr | None = None
    text_collection: str = "maintidoc_text"
    image_collection: str = "maintidoc_images"


class MinioSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="MINIO_", env_file=".env", extra="ignore")

    endpoint: str = "minio:9000"
    access_key: SecretStr = SecretStr(_UNSET)
    secret_key: SecretStr = SecretStr(_UNSET)
    bucket: str = "maintidoc"
    secure: bool = False


class McpSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    server_url: str = "http://mcp-server:8100/mcp"
    api_key: SecretStr = SecretStr(_UNSET)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    # SPEC §9 budget: at most 3 MCP calls per user request.
    max_calls_per_request: int = Field(default=3, ge=0, le=10)


class InferenceSettings(_Prefixed):
    """Device and model selection.

    `device="auto"` probes CUDA -> MPS -> CPU at runtime (ADR 0003). The
    requested model is kept separately from the one that actually served the
    request, because the fallback ladder may step down — and that step must
    never be silent.
    """

    model_config = SettingsConfigDict(env_prefix="MAINTIDOC_", env_file=".env", extra="ignore")

    device: Literal["auto", DeviceName] = "auto"
    vlm_provider: Literal["auto", "qwen", "stub"] = "auto"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vlm_fallback_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_quantization: Literal["awq", "bnb4", "none"] = "awq"
    vlm_max_new_tokens: int = Field(default=512, ge=1, le=4096)
    vlm_max_image_pixels: int = Field(default=512 * 28 * 28, ge=28 * 28)
    vlm_min_vram_gb: float = Field(default=5.0, ge=0.0)
    vlm_cpu_offload_gb: float = Field(default=0.0, ge=0.0)
    allow_model_fallback: bool = True

    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "clip-multilingual"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_adapter: str | None = None

    # SPEC §9 budgets, enforced in code.
    max_model_calls: int = Field(default=6, ge=1, le=64)
    request_budget_seconds: float = Field(default=60.0, gt=0, le=3600)

    tracking_backend: Literal["auto", "mlflow", "null"] = "auto"
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )


class RetrievalSettings(_Prefixed):
    model_config = SettingsConfigDict(env_prefix="MAINTIDOC_", env_file=".env", extra="ignore")

    top_k_text: int = Field(default=20, ge=1, le=200)
    top_k_image: int = Field(default=20, ge=1, le=200)
    # Standard RRF constant; 60 is the value from the original RRF paper.
    rrf_k: int = Field(default=60, ge=1)
    top_n_after_fusion: int = Field(default=8, ge=1, le=50)
    min_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_max_chars: int = Field(default=1600, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=2000)


class Settings(BaseModel):
    """Root settings object. Composed once per process, then cached."""

    model_config = {"frozen": True}

    app: AppSettings
    postgres: PostgresSettings
    qdrant: QdrantSettings
    minio: MinioSettings
    mcp: McpSettings
    inference: InferenceSettings
    retrieval: RetrievalSettings


def load_settings() -> Settings:
    """Read the environment. Raises ConfigurationError, never a bare KeyError."""
    try:
        return Settings(
            app=AppSettings(),
            postgres=PostgresSettings(),
            qdrant=QdrantSettings(),
            minio=MinioSettings(),
            mcp=McpSettings(),
            inference=InferenceSettings(),
            retrieval=RetrievalSettings(),
        )
    except ValueError as exc:  # pydantic wraps env problems in ValidationError(ValueError)
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    """Test hook: drop the cached settings so a new env is picked up."""
    get_settings.cache_clear()
