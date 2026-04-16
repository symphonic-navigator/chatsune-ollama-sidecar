import asyncio
import json
from typing import AsyncIterator

import httpx
import pytest
import respx

from sidecar.engine import EngineUnavailable, ModelNotFound, StreamTerminal
from sidecar.frames import (
    GenerateChatBody,
    Message,
    StreamDelta,
)
from sidecar.ollama import OllamaEngine


def _ndjson(lines: list[dict]) -> bytes:
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode()


@respx.mock
async def test_plain_text_stream():
    chunks = [
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"message": {"role": "assistant", "content": " world"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 7,
        },
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )

    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="llama3.2:8b",
        messages=[Message(role="user", content="Hi")],
    )
    items = []
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
async def test_reasoning_split_with_think_tags():
    chunks = [
        {"message": {"role": "assistant", "content": "<think>"}, "done": False},
        {"message": {"role": "assistant", "content": "let me ponder"}, "done": False},
        {"message": {"role": "assistant", "content": "</think>"}, "done": False},
        {"message": {"role": "assistant", "content": "The answer "}, "done": False},
        {"message": {"role": "assistant", "content": "is 42."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="deepseek-r1:7b",
        messages=[Message(role="user", content="what?")],
        options={"reasoning": True},
    )
    items: list = []
    try:
        async for item in engine.generate_chat(body):
            items.append(item)
    finally:
        await engine.aclose()

    deltas = [i for i in items if isinstance(i, StreamDelta)]
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)

    assert "let me ponder" in reasoning
    assert content == "The answer is 42."
    for d in deltas:
        populated = sum(x is not None for x in (d.content, d.reasoning, d.tool_calls))
        assert populated <= 1


@respx.mock
async def test_reasoning_suppressed_when_disabled():
    chunks = [
        {"message": {"role": "assistant", "content": "<think>secret</think>visible"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="deepseek-r1:7b",
        messages=[Message(role="user", content="q")],
        options={"reasoning": False},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    content = "".join(d.content or "" for d in deltas)
    reasoning = "".join(d.reasoning or "" for d in deltas)
    assert "secret" not in content
    assert reasoning == ""
    assert "visible" in content


@respx.mock
async def test_tool_call_fragments_passed_through():
    chunks = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"loc\":"},
                    }
                ],
            },
            "done": False,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"arguments": "\"Vienna\"}"},
                    }
                ],
            },
            "done": False,
        },
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="llama3.2:8b",
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
    assert frags[0].index == 0
    assert frags[0].id == "call_1"
    assert frags[0].function.name == "get_weather"
    assert frags[1].index == 0
    assert frags[1].function.arguments == "\"Vienna\"}"


@respx.mock
async def test_finish_reason_length_and_tool_calls():
    chunks = [
        {"message": {"role": "assistant", "content": "abc"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
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
    assert term.finish_reason == "length"


@respx.mock
async def test_model_not_found_raises():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model 'xxx' not found"})
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="xxx", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(ModelNotFound):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_connect_failure_raises_engine_unavailable():
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(EngineUnavailable):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_cancellation_stops_iteration():
    chunks = [
        {"message": {"role": "assistant", "content": f"tok{i}"}, "done": False}
        for i in range(100)
    ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    seen = 0
    gen = engine.generate_chat(body)
    try:
        async for item in gen:
            if isinstance(item, StreamDelta):
                seen += 1
                if seen >= 3:
                    break
    finally:
        await gen.aclose()
        await engine.aclose()
    assert seen == 3
