import asyncio
import json
from contextlib import asynccontextmanager

import pytest
import websockets

from sidecar.config import Settings
from sidecar.connection import ConnectionManager, StopReason
from sidecar.frames import HandshakeAckFrame


@asynccontextmanager
async def fake_backend(handler):
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/sidecar"


def _mk_settings(url: str) -> Settings:
    return Settings.model_construct(
        chatsune_backend_url=url.replace("/ws/sidecar", ""),
        chatsune_host_key="cshost_test_0001",
        ollama_url="http://localhost:11434",
        sidecar_health_port=0,
        sidecar_log_level="debug",
        sidecar_max_concurrent_requests=1,
    )


async def test_happy_handshake_and_pong():
    async def handler(ws):
        # websockets >=14: handshake headers via ws.request.headers
        if ws.request.headers.get("Authorization") != "Bearer cshost_test_0001":
            await ws.close()
            return
        hello = json.loads(await ws.recv())
        assert hello["type"] == "handshake"
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "homelab_id": "Xk7bQ2eJn9m",
            "display_name": "Test",
            "accepted": True,
            "notices": [],
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ping"
        await ws.send(json.dumps({"type": "pong"}))
        await ws.close()

    async with fake_backend(handler) as url:
        s = _mk_settings(url)
        cm = ConnectionManager(
            settings=s,
            handshake_payload={
                "type": "handshake",
                "csp_version": "1.0",
                "sidecar_version": "1.0.0",
                "engine": {"type": "ollama", "version": "test", "endpoint_hint": None},
                "max_concurrent_requests": 1,
                "capabilities": ["chat_streaming"],
            },
            on_frame=lambda _: None,
            url_override=url,
        )
        cm._ping_interval = 0.05
        cm._pong_timeout = 0.5

        reason = await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert reason == StopReason.PEER_CLOSED


async def test_auth_revoked_stops_cleanly():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "accepted": True,
        }))
        await ws.send(json.dumps({"type": "auth_revoked"}))
        await ws.close()

    async with fake_backend(handler) as url:
        s = _mk_settings(url)
        cm = ConnectionManager(
            settings=s,
            handshake_payload={
                "type": "handshake",
                "csp_version": "1.0",
                "sidecar_version": "1.0.0",
                "engine": {"type": "ollama", "version": "t", "endpoint_hint": None},
                "max_concurrent_requests": 1,
                "capabilities": ["chat_streaming"],
            },
            on_frame=lambda _: None,
            url_override=url,
        )
        reason = await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert reason == StopReason.AUTH_REVOKED


async def test_handshake_rejected_stops():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "accepted": False,
            "notices": ["version_unsupported: need 2.x"],
        }))
        await ws.close()

    async with fake_backend(handler) as url:
        s = _mk_settings(url)
        cm = ConnectionManager(
            settings=s,
            handshake_payload={
                "type": "handshake",
                "csp_version": "1.0",
                "sidecar_version": "1.0.0",
                "engine": {"type": "ollama", "version": "t", "endpoint_hint": None},
                "max_concurrent_requests": 1,
                "capabilities": ["chat_streaming"],
            },
            on_frame=lambda _: None,
            url_override=url,
        )
        reason = await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert reason == StopReason.HANDSHAKE_REJECTED_HARD


def test_backoff_sequence():
    from sidecar.connection import _backoff_seconds
    seq = [_backoff_seconds(n, jitter=lambda: 1.0) for n in range(1, 10)]
    assert seq == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]


def _basic_handshake() -> dict:
    return {
        "type": "handshake",
        "csp_version": "1.0",
        "sidecar_version": "1.0.0",
        "engine": {"type": "ollama", "version": "t", "endpoint_hint": None},
        "max_concurrent_requests": 1,
        "capabilities": ["chat_streaming"],
    }


async def test_handshake_completed_false_when_peer_closes_before_ack():
    """Pre-handshake peer close must NOT reset the backoff counter."""
    async def handler(ws):
        # Read and immediately close without sending handshake_ack.
        await ws.recv()
        await ws.close()

    async with fake_backend(handler) as url:
        s = _mk_settings(url)
        cm = ConnectionManager(
            settings=s,
            handshake_payload=_basic_handshake(),
            on_frame=lambda _: None,
            url_override=url,
        )
        await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert cm._handshake_completed is False


async def test_handshake_completed_true_after_accepted_ack():
    """Post-handshake clean close MAY reset the backoff counter."""
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "accepted": True,
        }))
        await ws.close()

    async with fake_backend(handler) as url:
        s = _mk_settings(url)
        cm = ConnectionManager(
            settings=s,
            handshake_payload=_basic_handshake(),
            on_frame=lambda _: None,
            url_override=url,
        )
        await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert cm._handshake_completed is True


def test_heartbeat_deadline_matches_spec():
    """SPEC §11: sidecar closes when silence exceeds 60 s."""
    s = Settings.model_construct(
        chatsune_backend_url="wss://x.example.com",
        chatsune_host_key="cshost_t",
        ollama_url="http://localhost:11434",
        sidecar_health_port=0,
        sidecar_log_level="debug",
        sidecar_max_concurrent_requests=1,
    )
    cm = ConnectionManager(
        settings=s,
        handshake_payload=_basic_handshake(),
        on_frame=lambda _: None,
        url_override="ws://127.0.0.1:0/ws/sidecar",
    )
    assert cm._heartbeat_deadline == 60.0
