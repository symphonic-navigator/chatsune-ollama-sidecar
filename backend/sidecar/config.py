"""Sidecar settings loaded from environment variables.

See SPEC.md §14. Fail-fast on missing or malformed required variables.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LogLevel = Literal["debug", "info", "warn", "error"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chatsune_backend_url: str = Field(...)
    chatsune_host_key: str = Field(...)

    ollama_url: str = Field(default="http://host.docker.internal:11434")

    sidecar_health_port: int = Field(default=8080, ge=1, le=65535)
    sidecar_log_level: LogLevel = Field(default="info")
    sidecar_max_concurrent_requests: int = Field(default=1, ge=1)

    @field_validator("chatsune_backend_url")
    @classmethod
    def _must_be_ws_scheme(cls, v: str) -> str:
        # SPEC §3 mandates wss:// for production. We also accept ws:// so
        # local development against an untERMinated backend is possible;
        # main.py logs a warning in that case.
        if not (v.startswith("wss://") or v.startswith("ws://")):
            raise ValueError(
                "CHATSUNE_BACKEND_URL must use the wss:// scheme "
                "(ws:// is permitted for local development)"
            )
        return v

    def backend_is_insecure(self) -> bool:
        return self.chatsune_backend_url.startswith("ws://")

    @field_validator("chatsune_host_key")
    @classmethod
    def _must_have_prefix(cls, v: str) -> str:
        if not v.startswith("cshost_"):
            raise ValueError("CHATSUNE_HOST_KEY must start with 'cshost_'")
        return v

    @field_validator("sidecar_log_level", mode="before")
    @classmethod
    def _lowercase(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v

    def ws_endpoint(self) -> str:
        return self.chatsune_backend_url.rstrip("/") + "/ws/sidecar"

    def host_key_tail(self) -> str:
        """Last 4 characters of the host key, safe for logs (SPEC §16)."""
        return self.chatsune_host_key[-4:]
