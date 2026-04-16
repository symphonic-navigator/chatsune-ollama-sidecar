import json

import pytest
from pydantic import ValidationError

from sidecar.frames import (
    AuthRevokedFrame,
    CancelFrame,
    ContentPartImage,
    ContentPartText,
    EngineInfo,
    ErrFrame,
    GenerateChatBody,
    HandshakeAckFrame,
    HandshakeFrame,
    Message,
    ModelDescriptor,
    PingFrame,
    PongFrame,
    ReqFrame,
    ResFrame,
    StreamDelta,
    StreamEndFrame,
    StreamFrame,
    SupersededFrame,
    ToolCallFragment,
    Usage,
    parse_frame,
)


def test_handshake_roundtrip():
    f = HandshakeFrame(
        csp_version="1.0",
        sidecar_version="1.0.0",
        engine=EngineInfo(type="ollama", version="0.5.0", endpoint_hint="http://localhost:11434"),
        max_concurrent_requests=2,
        capabilities=["chat_streaming", "tool_calls"],
    )
    payload = f.model_dump(mode="json")
    assert payload["type"] == "handshake"
    parsed = parse_frame(payload)
    assert isinstance(parsed, HandshakeFrame)
    assert parsed.engine.type == "ollama"


def test_parse_handshake_ack_accepted():
    raw = {
        "type": "handshake_ack",
        "csp_version": "1.0",
        "homelab_id": "Xk7bQ2eJn9m",
        "display_name": "Wohnzimmer-GPU",
        "accepted": True,
        "notices": [],
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, HandshakeAckFrame)
    assert parsed.accepted is True


def test_ping_pong_have_no_id():
    parsed = parse_frame({"type": "ping"})
    assert isinstance(parsed, PingFrame)
    parsed = parse_frame({"type": "pong"})
    assert isinstance(parsed, PongFrame)


def test_req_list_models_no_body():
    parsed = parse_frame({"type": "req", "id": "abc-123", "op": "list_models"})
    assert isinstance(parsed, ReqFrame)
    assert parsed.op == "list_models"
    assert parsed.body is None


def test_req_generate_chat_parses_body():
    raw = {
        "type": "req",
        "id": "abc-123",
        "op": "generate_chat",
        "body": {
            "model_slug": "llama3.2:8b",
            "messages": [{"role": "user", "content": "hi"}],
            "parameters": {"temperature": 0.7},
        },
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, ReqFrame)
    body = parsed.body
    assert isinstance(body, GenerateChatBody)
    assert body.model_slug == "llama3.2:8b"
    assert body.messages[0].role == "user"
    assert body.parameters.temperature == 0.7


def test_multimodal_user_message():
    raw = {
        "type": "req",
        "id": "abc",
        "op": "generate_chat",
        "body": {
            "model_slug": "llava",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "media_type": "image/png", "data_b64": "aGVsbG8="},
                ]},
            ],
        },
    }
    parsed = parse_frame(raw)
    parts = parsed.body.messages[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], ContentPartText)
    assert isinstance(parts[1], ContentPartImage)
    assert parts[1].media_type == "image/png"


def test_assistant_tool_calls_echo():
    raw = {
        "role": "assistant",
        "content": "let me check",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"loc\":\"Vienna\"}"},
            }
        ],
    }
    msg = Message.model_validate(raw)
    assert msg.tool_calls and msg.tool_calls[0].function.name == "get_weather"


def test_cancel_frame():
    parsed = parse_frame({"type": "cancel", "id": "abc"})
    assert isinstance(parsed, CancelFrame)
    assert parsed.id == "abc"


def test_stream_frame_one_channel():
    raw = {
        "type": "stream",
        "id": "abc",
        "delta": {"content": "Hello", "reasoning": None, "tool_calls": None},
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, StreamFrame)
    assert parsed.delta.content == "Hello"


def test_stream_end_with_usage():
    raw = {
        "type": "stream_end",
        "id": "abc",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, StreamEndFrame)
    assert parsed.finish_reason == "stop"
    assert parsed.usage.total_tokens == 30


def test_stream_end_without_usage():
    parsed = parse_frame({"type": "stream_end", "id": "abc", "finish_reason": "cancelled"})
    assert isinstance(parsed, StreamEndFrame)
    assert parsed.usage is None


def test_err_frame():
    raw = {
        "type": "err",
        "id": "abc",
        "code": "model_not_found",
        "message": "Model missing.",
        "detail": None,
        "recoverable": False,
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, ErrFrame)
    assert parsed.code == "model_not_found"


def test_auth_revoked_superseded_parse():
    assert isinstance(parse_frame({"type": "auth_revoked"}), AuthRevokedFrame)
    assert isinstance(parse_frame({"type": "superseded"}), SupersededFrame)


def test_unknown_type_returns_none():
    """Per SPEC §4: unknown types MUST be ignored for forward-compatibility."""
    assert parse_frame({"type": "some_future_frame", "id": "x"}) is None


def test_missing_type_raises():
    with pytest.raises(ValidationError):
        parse_frame({"id": "x"})


def test_req_rejects_unknown_op():
    with pytest.raises(ValidationError):
        parse_frame({"type": "req", "id": "x", "op": "do_something_else"})


def test_model_descriptor_context_length_required():
    raw = {
        "slug": "llama3.2:8b",
        "display_name": "Llama",
        "parameter_count": 8_030_261_248,
        "context_length": 131072,
        "quantisation": "Q4_K_M",
        "capabilities": ["text"],
        "engine_family": "ollama",
        "engine_model_id": "llama3.2:8b",
        "engine_metadata": {},
    }
    d = ModelDescriptor.model_validate(raw)
    assert d.context_length == 131072

    del raw["context_length"]
    with pytest.raises(ValidationError):
        ModelDescriptor.model_validate(raw)


def test_stream_delta_all_channels_nullable():
    """Populating one channel and leaving the others null is legal."""
    d = StreamDelta(content="hi", reasoning=None, tool_calls=None)
    assert d.content == "hi"
    d = StreamDelta(content=None, reasoning="think", tool_calls=None)
    assert d.reasoning == "think"
    d = StreamDelta(
        content=None,
        reasoning=None,
        tool_calls=[ToolCallFragment(index=0, id="c1", type="function",
                                      function={"name": "f", "arguments": "{"})],
    )
    assert d.tool_calls[0].function.name == "f"
