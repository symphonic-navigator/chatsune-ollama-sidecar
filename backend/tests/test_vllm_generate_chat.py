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
