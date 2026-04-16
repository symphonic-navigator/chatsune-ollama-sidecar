"""vLLM engine adapter (SPEC §15.3).

vLLM exposes an OpenAI-compatible API:
  - `GET /version` for the version probe,
  - `GET /v1/models` for discovery,
  - `POST /v1/chat/completions` (with `stream: true`) for chat.

Per-model metadata that vLLM does not self-report (vision, tool calling,
reasoning capability, display name, parameter count, quantisation) comes
from the operator via the YAML layer in `vllm_models_config.py`.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from .engine import EngineStreamItem
from .frames import GenerateChatBody, ModelDescriptor
from .vllm_models_config import VllmModelMetadata


class VllmEngine:
    engine_type = "vllm"

    def __init__(
        self,
        url: str,
        *,
        metadata: dict[str, VllmModelMetadata],
        timeout: float = 10.0,
    ) -> None:
        self._base = url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )
        self._metadata = metadata
        self._warned_unknown: set[str] = set()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/version")
            r.raise_for_status()
            v = r.json().get("version")
            return v if isinstance(v, str) else "unknown"
        except (httpx.HTTPError, ValueError):
            return "unknown"

    async def list_models(self) -> list[ModelDescriptor]:
        raise NotImplementedError  # Task 7

    def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        raise NotImplementedError  # Tasks 8-12
