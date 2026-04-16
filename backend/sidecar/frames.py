"""Pydantic models for every CSP/1 frame type (SPEC §4, §5).

Parsing contract
----------------
- Unknown `type` values return None (SPEC §4: ignore for forward-compat).
- Missing required fields on a recognised `type` raise ValidationError
  (caller SHOULD close the connection).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------

FinishReason = Literal["stop", "length", "tool_calls", "cancelled", "error"]
ErrCode = Literal[
    "model_not_found",
    "model_oom",
    "engine_unavailable",
    "engine_error",
    "invalid_request",
    "rate_limited",
    "cancelled",
    "internal",
]
EngineType = Literal["ollama", "lmstudio", "vllm", "llamacpp"]
Capability = Literal["chat_streaming", "tool_calls", "vision", "reasoning"]
ModelCapability = Literal["text", "tool_calling", "vision", "reasoning", "json_mode"]
OpName = Literal["list_models", "generate_chat"]


# ---------------------------------------------------------------------------
# Content parts (§8.1) and messages
# ---------------------------------------------------------------------------

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image"]
    media_type: str
    data_b64: str


ContentPart = Annotated[
    Union[ContentPartText, ContentPartImage],
    Field(discriminator="type"),
]


class ToolFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolFunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    type: Literal["function"]
    function: ToolFunctionDef


class GenerateParameters(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


class GenerateOptions(BaseModel):
    reasoning: bool = False


class GenerateChatBody(BaseModel):
    model_slug: str
    messages: list[Message]
    tools: list[ToolDef] | None = None
    parameters: GenerateParameters = Field(default_factory=GenerateParameters)
    options: GenerateOptions = Field(default_factory=GenerateOptions)


# ---------------------------------------------------------------------------
# list_models types
# ---------------------------------------------------------------------------

class ModelDescriptor(BaseModel):
    slug: str
    display_name: str
    parameter_count: int | None
    context_length: int  # Required, no null (SPEC §7.2)
    quantisation: str | None
    capabilities: list[ModelCapability]
    engine_family: EngineType
    engine_model_id: str
    engine_metadata: dict[str, Any] = Field(default_factory=dict)


class ListModelsBody(BaseModel):
    models: list[ModelDescriptor]


# ---------------------------------------------------------------------------
# Streaming deltas (§8.2)
# ---------------------------------------------------------------------------

class ToolCallFragmentFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallFragment(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ToolCallFragmentFunction


class StreamDelta(BaseModel):
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallFragment] | None = None


# ---------------------------------------------------------------------------
# Usage (§8.3)
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Connection-lifecycle frames (§5.1)
# ---------------------------------------------------------------------------

class EngineInfo(BaseModel):
    type: EngineType
    version: str
    endpoint_hint: str | None = None


class HandshakeFrame(BaseModel):
    type: Literal["handshake"] = "handshake"
    csp_version: str
    sidecar_version: str
    engine: EngineInfo
    max_concurrent_requests: int = Field(ge=1)
    capabilities: list[Capability]


class HandshakeAckFrame(BaseModel):
    type: Literal["handshake_ack"] = "handshake_ack"
    csp_version: str
    homelab_id: str | None = None
    display_name: str | None = None
    accepted: bool
    notices: list[str] = Field(default_factory=list)


class PingFrame(BaseModel):
    type: Literal["ping"] = "ping"


class PongFrame(BaseModel):
    type: Literal["pong"] = "pong"


class AuthRevokedFrame(BaseModel):
    type: Literal["auth_revoked"] = "auth_revoked"


class SupersededFrame(BaseModel):
    type: Literal["superseded"] = "superseded"


# ---------------------------------------------------------------------------
# Request/response frames (§5.2)
# ---------------------------------------------------------------------------

class ReqFrame(BaseModel):
    type: Literal["req"] = "req"
    id: str
    op: OpName
    body: GenerateChatBody | None = None


class ResFrame(BaseModel):
    type: Literal["res"] = "res"
    id: str
    ok: bool
    body: ListModelsBody | None = None


class StreamFrame(BaseModel):
    type: Literal["stream"] = "stream"
    id: str
    delta: StreamDelta


class StreamEndFrame(BaseModel):
    type: Literal["stream_end"] = "stream_end"
    id: str
    finish_reason: FinishReason
    usage: Usage | None = None


class CancelFrame(BaseModel):
    type: Literal["cancel"] = "cancel"
    id: str


class ErrFrame(BaseModel):
    type: Literal["err"] = "err"
    id: str | None = None
    code: ErrCode
    message: str
    detail: str | None = None
    recoverable: bool


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

AnyFrame = Union[
    HandshakeFrame,
    HandshakeAckFrame,
    PingFrame,
    PongFrame,
    AuthRevokedFrame,
    SupersededFrame,
    ReqFrame,
    ResFrame,
    StreamFrame,
    StreamEndFrame,
    CancelFrame,
    ErrFrame,
]

_TYPE_MAP: dict[str, type[BaseModel]] = {
    "handshake": HandshakeFrame,
    "handshake_ack": HandshakeAckFrame,
    "ping": PingFrame,
    "pong": PongFrame,
    "auth_revoked": AuthRevokedFrame,
    "superseded": SupersededFrame,
    "req": ReqFrame,
    "res": ResFrame,
    "stream": StreamFrame,
    "stream_end": StreamEndFrame,
    "cancel": CancelFrame,
    "err": ErrFrame,
}


class _FrameEnvelope(BaseModel):
    type: str


def parse_frame(payload: dict[str, Any]) -> AnyFrame | None:
    """Dispatch a raw JSON frame to its model.

    Returns None for unknown `type` values (SPEC §4: forward-compat).
    Raises ValidationError for a recognised type with bad fields OR a
    missing `type` field.
    """
    env = _FrameEnvelope.model_validate(payload)
    model = _TYPE_MAP.get(env.type)
    if model is None:
        return None
    return model.model_validate(payload)
