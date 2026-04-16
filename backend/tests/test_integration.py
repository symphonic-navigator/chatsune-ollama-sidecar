"""End-to-end: fake Chatsune backend + respx-mocked Ollama + real sidecar.

Drives:
  connect → handshake → list_models → generate_chat → cancel → ack → shutdown

This test is the acceptance gate for SPEC §18.5.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import pytest
import respx
import websockets

from sidecar.config import Settings
from sidecar.connection import ConnectionManager
from sidecar.dispatcher import Dispatcher
from sidecar.frames import parse_frame
from sidecar.main import build_handshake_payload
from sidecar.ollama import OllamaEngine


@asynccontextmanager
async def fake_backend(handler):
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/sidecar"


def _tags():
    return {
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
    }


def _show():
    return {
        "capabilities": [],
        "model_info": {"llama.context_length": 131072},
    }


def _ndjson(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


async def _async_ndjson_stream(lines) -> AsyncIterator[bytes]:
    """Yield each NDJSON line individually, with an event-loop yield between
    each one so that cancellation can land mid-stream."""
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


def _streaming_chat_response(lines) -> httpx.Response:
    return httpx.Response(200, stream=_NdjsonStream(lines))


@respx.mock
async def test_full_cycle():
    # --- Ollama mocks ------------------------------------------------------
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.5.7"})
    )
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=_tags())
    )
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json=_show())
    )
    # 1000-chunk stream so cancel beats natural end.
    chunks = [
        {"message": {"role": "assistant", "content": f"t{i}"}, "done": False}
        for i in range(1000)
    ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=_streaming_chat_response(chunks)
    )

    # --- Fake Chatsune backend --------------------------------------------
    observed: list[dict] = []
    list_models_done = asyncio.Event()
    streamed = asyncio.Event()
    finished = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "handshake"
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "homelab_id": "H",
            "display_name": "test",
            "accepted": True,
            "notices": [],
        }))

        # list_models
        await ws.send(json.dumps({"type": "req", "id": "r1", "op": "list_models"}))
        while not list_models_done.is_set():
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "res" and msg["id"] == "r1":
                list_models_done.set()
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        # generate_chat + cancel
        await ws.send(json.dumps({
            "type": "req", "id": "r2", "op": "generate_chat",
            "body": {
                "model_slug": "llama3.2:8b",
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
            sidecar_health_port=0,
            sidecar_log_level="warn",
            sidecar_max_concurrent_requests=1,
        )

        engine = OllamaEngine(settings.ollama_url)
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

    # Assertions
    res = next(f for f in observed if f["type"] == "res")
    assert res["id"] == "r1"
    assert res["body"]["models"][0]["slug"] == "llama3.2:8b"
    stream_end = next(
        f for f in observed if f["type"] == "stream_end" and f["id"] == "r2"
    )
    assert stream_end["finish_reason"] == "cancelled"
