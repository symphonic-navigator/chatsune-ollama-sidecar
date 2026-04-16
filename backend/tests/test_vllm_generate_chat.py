"""Unit tests for vLLM streaming chat."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from sidecar.engine import StreamTerminal
from sidecar.frames import (
    ContentPartImage,
    ContentPartText,
    GenerateChatBody,
    Message,
    StreamDelta,
)
from sidecar.vllm import VllmEngine


def _sse(events: list[dict]) -> bytes:
    """Build an OpenAI-style SSE body (each event wrapped in `data: …\n\n`)."""
    out = []
    for e in events:
        out.append(f"data: {json.dumps(e)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage_chunk(prompt: int, completion: int) -> dict:
    """vLLM sends a final event with choices: [] and a usage block when
    stream_options.include_usage is True."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


@respx.mock
async def test_plain_text_stream_and_finish_stop():
    events = [
        _chunk(content="Hello"),
        _chunk(content=" world"),
        _chunk(finish_reason="stop"),
        _usage_chunk(5, 7),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="gemma-4-26b",
        messages=[Message(role="user", content="Hi")],
    )
    items: list = []
    try:
        async for item in engine.generate_chat(body):
            items.append(item)
    finally:
        await engine.aclose()

    deltas = [i for i in items if isinstance(i, StreamDelta)]
    terms = [i for i in items if isinstance(i, StreamTerminal)]
    assert [d.content for d in deltas] == ["Hello", " world"]
    assert len(terms) == 1
    assert terms[0].finish_reason == "stop"
    assert terms[0].usage is not None
    assert terms[0].usage.prompt_tokens == 5
    assert terms[0].usage.completion_tokens == 7


@respx.mock
async def test_finish_reason_length_and_tool_calls():
    for reason in ("length", "tool_calls"):
        events = [_chunk(content="x"), _chunk(finish_reason=reason)]
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=_sse(events))
        )
        engine = VllmEngine("http://localhost:8000", metadata={})
        body = GenerateChatBody(
            model_slug="m", messages=[Message(role="user", content="x")]
        )
        term = None
        try:
            async for item in engine.generate_chat(body):
                if isinstance(item, StreamTerminal):
                    term = item
        finally:
            await engine.aclose()
        assert term is not None
        assert term.finish_reason == reason
        respx.reset()


@respx.mock
async def test_unknown_finish_reason_falls_back_to_stop():
    events = [_chunk(content="x"), _chunk(finish_reason="content_filter")]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    term = None
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamTerminal):
                term = item
    finally:
        await engine.aclose()
    assert term is not None
    assert term.finish_reason == "stop"


@respx.mock
async def test_request_payload_has_stream_true_and_include_usage():
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")]),
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="hi")],
        parameters={"temperature": 0.7, "max_tokens": 128, "top_p": 0.9, "stop": ["\n"]},
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()

    sent = captured[0]
    assert sent["model"] == "m"
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 128
    assert sent["top_p"] == 0.9
    assert sent["stop"] == ["\n"]


@respx.mock
async def test_reasoning_content_emitted_when_reasoning_on():
    events = [
        _chunk(reasoning_content="Let me "),
        _chunk(reasoning_content="think..."),
        _chunk(content="The answer is 42."),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
        options={"reasoning": True},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)
    assert reasoning == "Let me think..."
    assert content == "The answer is 42."
    for d in deltas:
        populated = sum(x is not None for x in (d.content, d.reasoning, d.tool_calls))
        assert populated <= 1


@respx.mock
async def test_reasoning_content_suppressed_when_reasoning_off():
    events = [
        _chunk(reasoning_content="secret"),
        _chunk(content="visible"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    assert "".join(d.reasoning or "" for d in deltas) == ""
    assert "".join(d.content or "" for d in deltas) == "visible"


@respx.mock
async def test_inline_think_tags_split_when_reasoning_on():
    events = [
        _chunk(content="<think>"),
        _chunk(content="pondering"),
        _chunk(content="</think>"),
        _chunk(content="answer is 42"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
        options={"reasoning": True},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)
    assert reasoning == "pondering"
    assert content == "answer is 42"


@respx.mock
async def test_inline_think_tags_dropped_when_reasoning_off():
    events = [
        _chunk(content="<think>secret</think>visible"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    assert "".join(d.reasoning or "" for d in deltas) == ""
    assert "".join(d.content or "" for d in deltas) == "visible"


@respx.mock
async def test_tool_call_fragments_progressive():
    events = [
        _chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"loc"},
        }]),
        _chunk(tool_calls=[{
            "index": 0,
            "function": {"arguments": "\":\"Vienna"},
        }]),
        _chunk(tool_calls=[{
            "index": 0,
            "function": {"arguments": "\"}"},
        }]),
        _chunk(finish_reason="tool_calls"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="weather?")],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    frags = [f for d in deltas for f in (d.tool_calls or [])]
    assert len(frags) == 3
    assert frags[0].index == 0
    assert frags[0].id == "call_1"
    assert frags[0].function.name == "get_weather"
    assert frags[0].function.arguments == "{\"loc"
    assert frags[1].function.arguments == "\":\"Vienna"
    assert frags[2].function.arguments == "\"}"
    # Reconstructed arguments must parse:
    joined = "".join(f.function.arguments or "" for f in frags)
    assert json.loads(joined) == {"loc": "Vienna"}


@respx.mock
async def test_tools_field_forwarded_to_vllm():
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")])
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="hi")],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()
    assert captured[0]["tools"][0]["function"]["name"] == "get_weather"


@respx.mock
async def test_assistant_tool_call_history_passthrough():
    """Assistant history with JSON-string arguments must reach vLLM unchanged."""
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")])
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"loc\":\"Vienna\"}"},
                }],
            ),
            Message(role="tool", content="sunny", tool_call_id="c1"),
        ],
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()

    sent = captured[0]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    args = assistant["tool_calls"][0]["function"]["arguments"]
    # Stays a JSON string — unlike Ollama, vLLM expects OpenAI-native shape.
    assert isinstance(args, str)
    assert args == "{\"loc\":\"Vienna\"}"
    tool_msg = next(m for m in sent if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
