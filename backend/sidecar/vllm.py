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

from typing import Any, AsyncIterator

import httpx

from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
)
from .frames import (
    GenerateChatBody,
    ModelCapability,
    ModelDescriptor,
)
from .logging_setup import get_logger
from .vllm_models_config import VllmModelMetadata


log = get_logger("vllm")


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
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            payload: dict[str, Any] = r.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e
        except ValueError as e:
            raise EngineBadResponse(f"non-JSON /v1/models body: {e}") from e

        out: list[ModelDescriptor] = []
        for raw in payload.get("data") or []:
            descriptor = self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        served_id = raw.get("id")
        if not isinstance(served_id, str) or not served_id:
            return None

        context_length = raw.get("max_model_len")
        if not isinstance(context_length, int):
            log.warning(
                "vllm.model_dropped_no_context",
                id=served_id,
                reason="max_model_len missing or not an integer",
            )
            return None

        meta = self._metadata.get(served_id)
        if meta is None:
            if served_id not in self._warned_unknown:
                log.warning(
                    "vllm.model_without_metadata",
                    id=served_id,
                    message=(
                        f"No YAML metadata for model '{served_id}'. "
                        "Listing with capabilities=['text'] only."
                    ),
                )
                self._warned_unknown.add(served_id)
            capabilities: list[ModelCapability] = ["text"]
            display_name = served_id
            parameter_count: int | None = None
            quantisation: str | None = None
        else:
            capabilities = list(meta.capabilities) if meta.capabilities else ["text"]
            display_name = meta.display_name or served_id
            parameter_count = meta.parameter_count
            quantisation = meta.quantisation

        engine_metadata: dict[str, Any] = {}
        for key in ("owned_by", "root"):
            value = raw.get(key)
            if value is not None:
                engine_metadata[key] = value

        return ModelDescriptor(
            slug=served_id,
            display_name=display_name,
            parameter_count=parameter_count,
            context_length=context_length,
            quantisation=quantisation,
            capabilities=capabilities,
            engine_family="vllm",
            engine_model_id=served_id,
            engine_metadata=engine_metadata,
        )

    def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        raise NotImplementedError  # Tasks 8-12
