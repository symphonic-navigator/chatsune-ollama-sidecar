import asyncio

import aiohttp
import pytest

from sidecar.healthcheck import HealthcheckServer, HealthState


async def _get(port: int, path: str = "/healthz"):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{port}{path}") as resp:
            return resp.status, await resp.json()


async def test_reports_connected_as_200():
    state = HealthState()
    state.mark_engine(True, "ollama")
    state.mark_backend("connected")
    srv = HealthcheckServer(state, port=0)
    await srv.start()
    try:
        status, body = await _get(srv.port)
        assert status == 200
        assert body["ok"] is True
        assert body["backend_connection"] == "connected"
        assert body["engine"]["type"] == "ollama"
        assert body["engine"]["reachable"] is True
        assert isinstance(body["uptime_seconds"], int)
    finally:
        await srv.stop()


async def test_reports_reconnecting_as_200():
    state = HealthState()
    state.mark_engine(True, "ollama")
    state.mark_backend("reconnecting")
    srv = HealthcheckServer(state, port=0)
    await srv.start()
    try:
        status, body = await _get(srv.port)
        assert status == 200
        assert body["ok"] is True
        assert body["backend_connection"] == "reconnecting"
    finally:
        await srv.stop()


async def test_reports_disconnected_as_503():
    state = HealthState()
    state.mark_engine(True, "ollama")
    state.mark_backend("disconnected")
    srv = HealthcheckServer(state, port=0)
    await srv.start()
    try:
        status, body = await _get(srv.port)
        assert status == 503
        assert body["ok"] is False
        assert body["backend_connection"] == "disconnected"
    finally:
        await srv.stop()


async def test_unreachable_engine_still_200_when_backend_ok():
    state = HealthState()
    state.mark_engine(False, "ollama")
    state.mark_backend("connected")
    srv = HealthcheckServer(state, port=0)
    await srv.start()
    try:
        status, body = await _get(srv.port)
        assert status == 200
        assert body["engine"]["reachable"] is False
    finally:
        await srv.stop()


async def test_binds_to_loopback_only():
    state = HealthState()
    state.mark_engine(True, "ollama")
    state.mark_backend("connected")
    srv = HealthcheckServer(state, port=0)
    await srv.start()
    try:
        sockets = [s for s in srv._site._server.sockets]  # noqa: SLF001
        hosts = {s.getsockname()[0] for s in sockets}
        assert hosts <= {"127.0.0.1", "::1"}
    finally:
        await srv.stop()
