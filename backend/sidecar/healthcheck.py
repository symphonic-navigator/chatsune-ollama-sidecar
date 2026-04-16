"""Loopback healthcheck HTTP server (SPEC §13)."""
from __future__ import annotations

import time
from typing import Literal

from aiohttp import web


BackendState = Literal["connected", "reconnecting", "disconnected"]


class HealthState:
    """Mutable snapshot of the sidecar's health, shared with other tasks."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._backend: BackendState = "disconnected"
        self._engine_reachable = False
        self._engine_type = "ollama"

    def mark_backend(self, state: BackendState) -> None:
        self._backend = state

    def mark_engine(self, reachable: bool, engine_type: str) -> None:
        self._engine_reachable = reachable
        self._engine_type = engine_type

    def snapshot(self) -> dict:
        return {
            "ok": self._backend != "disconnected",
            "backend_connection": self._backend,
            "engine": {
                "type": self._engine_type,
                "reachable": self._engine_reachable,
            },
            "uptime_seconds": int(time.monotonic() - self._start),
        }


class HealthcheckServer:
    def __init__(self, state: HealthState, *, port: int = 8080) -> None:
        self._state = state
        self._requested_port = port
        self._app = web.Application()
        self._app.router.add_get("/healthz", self._handle)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._bound_port: int | None = None

    @property
    def port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("server not started")
        return self._bound_port

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host="127.0.0.1", port=self._requested_port)
        await self._site.start()
        server = self._site._server  # pyright: ignore[reportPrivateUsage]
        sock = next(iter(server.sockets or []), None)
        self._bound_port = sock.getsockname()[1] if sock else self._requested_port

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None

    async def _handle(self, request: web.Request) -> web.Response:
        body = self._state.snapshot()
        status = 200 if body["ok"] else 503
        return web.json_response(body, status=status)
