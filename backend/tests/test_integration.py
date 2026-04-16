"""End-to-end: fake Chatsune backend + mocked engine HTTP + real sidecar.

Drives:
  connect → handshake → list_models → generate_chat → cancel → ack → shutdown

Parametrised across both Ollama and vLLM adapters to prove the dispatcher
and connection plumbing are engine-agnostic.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx
import pytest
import respx
import websockets

from sidecar.config import Settings
from sidecar.connection import ConnectionManager
from sidecar.dispatcher import Dispatcher
from sidecar.engine import Engine
from sidecar.main import build_handshake_payload
from sidecar.ollama import OllamaEngine
from sidecar.vllm import VllmEngine


@asynccontextmanager
async def fake_backend(handler):
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/sidecar"


def _ndjson(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


async def _async_ndjson_stream(lines) -> AsyncIterator[bytes]:
    for line in lines:
        yield (json.dumps(line) + "\n").encode()
        await asyncio.sleep(0)


class _NdjsonStream(httpx.AsyncByteStream):
    def __init__(self, lines):
        self._lines = lines

    async def __aiter__(self):
        async for chunk in _async_ndjson_stream(self._lines):
            yield chunk

    async def aclose(self) -> None:
        pass


async def _async_sse_stream(events) -> AsyncIterator[bytes]:
    for e in events:
        yield f"data: {json.dumps(e)}\n\n".encode()
        await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"


class _SseStream(httpx.AsyncByteStream):
    def __init__(self, events):
        self._events = events

    async def __aiter__(self):
        async for chunk in _async_sse_stream(self._events):
            yield chunk

    async def aclose(self) -> None:
        pass


@dataclass
class _EngineSetup:
    name: str
    build_engine: Callable[[], Engine]
    mock_http: Callable[[], None]
    expected_slug: str
    settings_engine: str


def _setup_ollama() -> _EngineSetup:
    def mock_http():
        respx.get("http://localhost:11434/api/version").mock(
            return_value=httpx.Response(200, json={"version": "0.5.7"})
        )
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{
                    "name": "llama3.2:8b",
                    "size": 1,
                    "digest": "d",
                    "details": {
                        "family": "llama",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_K_M",
                    },
                }]
            })
        )
        respx.post("http://localhost:11434/api/show").mock(
            return_value=httpx.Response(200, json={
                "capabilities": [],
                "model_info": {"llama.context_length": 131072},
            })
        )
        ollama_chunks = [
            {"message": {"role": "assistant", "content": f"t{i}"}, "done": False}
            for i in range(1000)
        ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, stream=_NdjsonStream(ollama_chunks))
        )

    return _EngineSetup(
        name="ollama",
        build_engine=lambda: OllamaEngine("http://localhost:11434"),
        mock_http=mock_http,
        expected_slug="llama3.2:8b",
        settings_engine="ollama",
    )


def _setup_vllm() -> _EngineSetup:
    def mock_http():
        respx.get("http://localhost:8000/version").mock(
            return_value=httpx.Response(200, json={"version": "0.7.3"})
        )
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json={
                "object": "list",
                "data": [{
                    "id": "integration-model",
                    "object": "model",
                    "owned_by": "vllm",
                    "max_model_len": 8192,
                    "root": "fake/repo",
                }],
            })
        )
        vllm_events = [
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"t{i}"},
                    "finish_reason": None,
                }],
            }
            for i in range(1000)
        ] + [
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ]
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, stream=_SseStream(vllm_events))
        )

    from sidecar.vllm_models_config import VllmModelMetadata

    return _EngineSetup(
        name="vllm",
        build_engine=lambda: VllmEngine(
            "http://localhost:8000",
            metadata={
                "integration-model": VllmModelMetadata(
                    display_name="Integration Model",
                    capabilities=["text"],
                ),
            },
        ),
        mock_http=mock_http,
        expected_slug="integration-model",
        settings_engine="vllm",
    )


ENGINE_SETUPS = [_setup_ollama(), _setup_vllm()]


@pytest.mark.parametrize("setup", ENGINE_SETUPS, ids=[s.name for s in ENGINE_SETUPS])
@respx.mock
async def test_full_cycle(setup: _EngineSetup):
    setup.mock_http()

    observed: list[dict] = []
    list_models_done = asyncio.Event()
    streamed = asyncio.Event()
    finished = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "handshake"
        assert hello["engine"]["type"] == setup.name
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "homelab_id": "H",
            "display_name": "test",
            "accepted": True,
            "notices": [],
        }))

        await ws.send(json.dumps({"type": "req", "id": "r1", "op": "list_models"}))
        while not list_models_done.is_set():
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "res" and msg["id"] == "r1":
                list_models_done.set()
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        await ws.send(json.dumps({
            "type": "req", "id": "r2", "op": "generate_chat",
            "body": {
                "model_slug": setup.expected_slug,
                "messages": [{"role": "user", "content": "hi"}],
            },
        }))

        stream_count = 0
        while True:
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "stream" and msg["id"] == "r2":
                stream_count += 1
                if stream_count == 3:
                    streamed.set()
                    await ws.send(json.dumps({"type": "cancel", "id": "r2"}))
            elif msg["type"] == "stream_end" and msg["id"] == "r2":
                assert msg["finish_reason"] == "cancelled"
                finished.set()
                break
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        await ws.close()

    async with fake_backend(handler) as url:
        settings = Settings.model_construct(
            chatsune_backend_url=url.replace("/ws/sidecar", ""),
            chatsune_host_key="cshost_integration_001",
            ollama_url="http://localhost:11434",
            vllm_url="http://localhost:8000",
            vllm_models_config_path=None,
            vllm_models_overlay_path=None,
            sidecar_engine=setup.settings_engine,
            sidecar_health_port=0,
            sidecar_log_level="warn",
            sidecar_max_concurrent_requests=1,
        )

        engine = setup.build_engine()
        try:
            version = await engine.probe_version()
            handshake = build_handshake_payload(
                settings, engine_type=engine.engine_type, engine_version=version,
            )

            cm_holder: list[ConnectionManager] = []

            async def send(frame):
                await cm_holder[0].send(frame)

            dispatcher = Dispatcher(engine=engine, send=send)

            async def on_frame(frame):
                await dispatcher.handle(frame)

            cm = ConnectionManager(
                settings=settings,
                handshake_payload=handshake,
                on_frame=on_frame,
                url_override=url,
            )
            cm_holder.append(cm)
            cm._ping_interval = 1.0
            cm._pong_timeout = 5.0

            run_task = asyncio.create_task(cm.run_forever())

            await asyncio.wait_for(finished.wait(), timeout=15.0)
            cm.request_stop()
            await asyncio.wait_for(run_task, timeout=5.0)
            await dispatcher.cancel_all()
        finally:
            await engine.aclose()

    res = next(f for f in observed if f["type"] == "res")
    assert res["id"] == "r1"
    assert res["body"]["models"][0]["slug"] == setup.expected_slug
    stream_end = next(
        f for f in observed if f["type"] == "stream_end" and f["id"] == "r2"
    )
    assert stream_end["finish_reason"] == "cancelled"
