"""Engine abstraction (SPEC §6, §15).

v1 ships one concrete engine (Ollama). This module defines the protocol
the rest of the sidecar consumes so the plumbing does not care which
engine is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from .frames import (
    GenerateChatBody,
    ModelDescriptor,
    StreamDelta,
    Usage,
)


# ---------------------------------------------------------------------------
# Exceptions — caught by the dispatcher and converted to `err` frames
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Base class. `err.code` derived by dispatcher."""


class EngineUnavailable(EngineError):
    """Local engine daemon cannot be reached at all."""


class ModelNotFound(EngineError):
    """Engine rejected the requested model."""


class ModelOutOfMemory(EngineError):
    """Engine could not load the model (typically VRAM)."""


class EngineBadResponse(EngineError):
    """Engine returned a non-success status or malformed payload."""


# ---------------------------------------------------------------------------
# Streaming output contract
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StreamTerminal:
    """Produced by the engine adapter when its stream ends cleanly.

    `finish_reason` is the SPEC §8.3 vocabulary. `usage` is optional —
    omit when the engine does not report token counts.
    """
    finish_reason: str
    usage: Usage | None = None


EngineStreamItem = StreamDelta | StreamTerminal


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------

class Engine(Protocol):
    """Minimum surface every engine adapter exposes."""

    engine_type: str  # "ollama" etc., matches SPEC EngineType

    async def probe_version(self) -> str: ...

    async def list_models(self) -> list[ModelDescriptor]: ...

    def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]: ...

    async def aclose(self) -> None: ...
