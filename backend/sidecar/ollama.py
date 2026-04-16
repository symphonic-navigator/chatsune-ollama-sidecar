"""Ollama engine adapter (SPEC §15.1).

Handles discovery (`/api/tags` + `/api/show`) and — in later tasks —
streaming chat (`/api/chat`).
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .engine import (
    EngineBadResponse,
    EngineUnavailable,
    ModelNotFound,
)
from .frames import ModelCapability, ModelDescriptor


# Model families known to emit <think>...</think> reasoning tokens.
# Conservative allowlist — SPEC §15.1.
_REASONING_FAMILIES = {
    "deepseek-r1",
    "deepseek_r1",
    "qwen3-thinking",
    "qwen3_thinking",
    "qwq",
}


class OllamaEngine:
    engine_type = "ollama"

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self._base = url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----- version probe --------------------------------------------------

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/api/version")
            r.raise_for_status()
            return r.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError):
            return "unknown"

    # ----- list_models ----------------------------------------------------

    async def list_models(self) -> list[ModelDescriptor]:
        try:
            tags = await self._client.get("/api/tags")
            tags.raise_for_status()
            tag_payload: dict[str, Any] = tags.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e

        out: list[ModelDescriptor] = []
        for raw in tag_payload.get("models", []):
            descriptor = await self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    async def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        name = raw.get("name")
        if not name:
            return None

        details = raw.get("details", {}) or {}
        family = (details.get("family") or "").lower()
        param_size_str = details.get("parameter_size")  # "8B"
        quantisation = details.get("quantization_level")

        try:
            show = await self._client.post("/api/show", json={"model": name})
            if show.status_code == 404:
                return None  # raced with a model deletion
            show.raise_for_status()
            show_body = show.json()
        except httpx.HTTPError:
            return None  # fail-closed per SPEC §6.5

        model_info: dict[str, Any] = show_body.get("model_info", {}) or {}
        context_length = _extract_context_length(model_info)
        if context_length is None:
            # SPEC §6.5: drop models without context_length.
            return None

        caps: list[ModelCapability] = ["text"]
        engine_caps = [c.lower() for c in show_body.get("capabilities", []) or []]
        if "tools" in engine_caps:
            caps.append("tool_calling")
        if _has_vision(model_info) or "vision" in engine_caps:
            caps.append("vision")
        if family in _REASONING_FAMILIES or "reasoning" in engine_caps:
            caps.append("reasoning")

        return ModelDescriptor(
            slug=name,
            display_name=_display_name(name),
            parameter_count=_parse_param_size(param_size_str),
            context_length=context_length,
            quantisation=quantisation,
            capabilities=caps,
            engine_family="ollama",
            engine_model_id=name,
            engine_metadata={
                "family": family or None,
                "digest": raw.get("digest"),
                "size_bytes": raw.get("size"),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_KEY_RE = re.compile(r"\.context_length$")


def _extract_context_length(model_info: dict[str, Any]) -> int | None:
    """Ollama reports `<arch>.context_length` — scan any matching key."""
    for key, value in model_info.items():
        if _CONTEXT_KEY_RE.search(key) and isinstance(value, (int, float)):
            return int(value)
    return None


def _has_vision(model_info: dict[str, Any]) -> bool:
    for key, value in model_info.items():
        if key.endswith(".vision") and value:
            return True
    return False


_PARAM_SIZE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<mag>[BMbm])$")


def _parse_param_size(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    m = _PARAM_SIZE_RE.match(raw.strip())
    if not m:
        return None
    num = float(m.group("num"))
    mag = m.group("mag").upper()
    multiplier = 1_000_000_000 if mag == "B" else 1_000_000
    return int(num * multiplier)


def _display_name(slug: str) -> str:
    try:
        base, tag = slug.split(":", 1)
        return f"{base.title()} {tag.upper()}"
    except ValueError:
        return slug
