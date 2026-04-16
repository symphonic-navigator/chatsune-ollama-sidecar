"""Outbound WebSocket lifecycle (SPEC §3, §6, §11, §12).

The ConnectionManager owns one long-running task. Each "session" is
one successful connect → handshake → frame loop. Sessions end for one
of the reasons in StopReason; most trigger a reconnect with jittered
exponential backoff, two do not (auth_revoked, superseded).
"""
from __future__ import annotations

import asyncio
import enum
import json
import random
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .config import Settings
from .frames import (
    AuthRevokedFrame,
    HandshakeAckFrame,
    PingFrame,
    PongFrame,
    SupersededFrame,
    parse_frame,
)
from .logging_setup import get_logger


log = get_logger("connection")


class StopReason(enum.StrEnum):
    PEER_CLOSED = "peer_closed"
    HEARTBEAT_LOST = "heartbeat_lost"
    CONNECT_FAILED = "connect_failed"
    HANDSHAKE_REJECTED_HARD = "handshake_rejected_hard"
    AUTH_REVOKED = "auth_revoked"
    SUPERSEDED = "superseded"
    SHUTDOWN = "shutdown"


_TERMINAL = {
    StopReason.AUTH_REVOKED,
    StopReason.SUPERSEDED,
    StopReason.HANDSHAKE_REJECTED_HARD,
    StopReason.SHUTDOWN,
}

OnFrame = Callable[[Any], Awaitable[None] | None]


class ConnectionManager:
    """Drives the WS client for one sidecar process."""

    def __init__(
        self,
        *,
        settings: Settings,
        handshake_payload: dict[str, Any],
        on_frame: OnFrame,
        url_override: str | None = None,
        on_ack: Callable[[HandshakeAckFrame], None] | None = None,
        on_status_change: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._handshake = handshake_payload
        self._on_frame = on_frame
        self._on_ack = on_ack
        self._on_status = on_status_change
        self._url = url_override or settings.ws_endpoint()

        self._ping_interval = 30.0
        self._pong_timeout = 10.0

        self._send_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._active_ws: ClientConnection | None = None

    # ----- public API ------------------------------------------------------

    async def send(self, frame: dict[str, Any]) -> None:
        await self._send_q.put(frame)

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        """Main loop: connect, run a session, back off, repeat."""
        attempt = 0
        while not self._stop.is_set():
            self._status("reconnecting")
            reason = await self._run_once()
            if reason in _TERMINAL:
                log.info(
                    "ws.session_terminal",
                    reason=str(reason),
                    host_key_tail=self._settings.host_key_tail(),
                )
                if reason == StopReason.HANDSHAKE_REJECTED_HARD:
                    raise HardStop(exit_code=2)
                return

            if reason == StopReason.PEER_CLOSED:
                attempt = 0
            else:
                attempt += 1
            self._status("reconnecting")
            delay = _backoff_seconds(attempt, jitter=_uniform_jitter)
            log.info("ws.backoff", attempt=attempt, delay_seconds=delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                return  # stopped during backoff
            except asyncio.TimeoutError:
                pass

    # ----- one session -----------------------------------------------------

    async def _run_once(self) -> StopReason:
        # Drain any stale frames (including a leftover _POISON from the
        # previous session) before starting a fresh writer.
        while True:
            try:
                self._send_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        headers = {"Authorization": f"Bearer {self._settings.chatsune_host_key}"}
        try:
            async with connect(
                self._url,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                self._active_ws = ws
                try:
                    return await self._session(ws)
                finally:
                    self._active_ws = None
        except (OSError, websockets.InvalidStatus, websockets.InvalidHandshake) as e:
            log.warning("ws.connect_failed", error=str(e))
            return StopReason.CONNECT_FAILED
        except ConnectionClosed:
            return StopReason.PEER_CLOSED

    async def _session(self, ws: ClientConnection) -> StopReason:
        await ws.send(json.dumps(self._handshake))

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            return StopReason.CONNECT_FAILED

        try:
            ack = parse_frame(json.loads(raw))
        except Exception:
            log.error("ws.bad_handshake_ack", raw=str(raw)[:200])
            return StopReason.CONNECT_FAILED

        if not isinstance(ack, HandshakeAckFrame):
            log.error("ws.unexpected_first_frame", frame_type=type(ack).__name__)
            return StopReason.CONNECT_FAILED

        if not ack.accepted:
            notices_text = " | ".join(ack.notices)
            log.error("ws.handshake_rejected", notices=notices_text)
            if any("version_unsupported" in n for n in ack.notices):
                return StopReason.HANDSHAKE_REJECTED_HARD
            return StopReason.CONNECT_FAILED

        log.info(
            "ws.handshake_ok",
            homelab_id=ack.homelab_id,
            display_name=ack.display_name,
            notices=ack.notices,
        )
        if self._on_ack is not None:
            self._on_ack(ack)
        self._status("connected")

        last_pong = asyncio.get_running_loop().time()
        reader_q: asyncio.Queue[object] = asyncio.Queue()

        async def reader() -> None:
            try:
                async for message in ws:
                    if isinstance(message, (bytes, bytearray)):
                        log.error("ws.binary_frame_rejected")
                        await ws.close(code=1003, reason="binary not allowed")
                        return
                    await reader_q.put(message)
            finally:
                await reader_q.put(_CLOSED)

        async def writer() -> None:
            while True:
                frame = await self._send_q.get()
                if frame is _POISON:
                    return
                try:
                    await ws.send(json.dumps(frame))
                except ConnectionClosed:
                    return

        async def pinger() -> None:
            while True:
                await asyncio.sleep(self._ping_interval)
                await self._send_q.put({"type": "ping"})

        reader_t = asyncio.create_task(reader(), name="ws-reader")
        writer_t = asyncio.create_task(writer(), name="ws-writer")
        pinger_t = asyncio.create_task(pinger(), name="ws-pinger")

        stop_reason = StopReason.PEER_CLOSED

        try:
            while True:
                if self._stop.is_set():
                    stop_reason = StopReason.SHUTDOWN
                    break

                now = asyncio.get_running_loop().time()
                if now - last_pong > self._ping_interval + self._pong_timeout * 2 + 20:
                    log.warning("ws.heartbeat_lost", since=now - last_pong)
                    stop_reason = StopReason.HEARTBEAT_LOST
                    break

                try:
                    msg = await asyncio.wait_for(reader_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if msg is _CLOSED:
                    break

                try:
                    frame = parse_frame(json.loads(msg))
                except Exception as e:
                    log.error("ws.bad_frame", error=str(e))
                    continue
                if frame is None:
                    continue

                if isinstance(frame, PingFrame):
                    await self._send_q.put({"type": "pong"})
                    continue
                if isinstance(frame, PongFrame):
                    last_pong = asyncio.get_running_loop().time()
                    continue
                if isinstance(frame, AuthRevokedFrame):
                    log.info("ws.auth_revoked")
                    stop_reason = StopReason.AUTH_REVOKED
                    break
                if isinstance(frame, SupersededFrame):
                    log.info("ws.superseded")
                    stop_reason = StopReason.SUPERSEDED
                    break

                result = self._on_frame(frame)
                if asyncio.iscoroutine(result):
                    await result
        finally:
            await self._send_q.put(_POISON)
            for t in (reader_t, writer_t, pinger_t):
                t.cancel()
            await asyncio.gather(reader_t, writer_t, pinger_t, return_exceptions=True)
            try:
                if stop_reason in (StopReason.AUTH_REVOKED, StopReason.SUPERSEDED, StopReason.SHUTDOWN):
                    await ws.close(code=1000)
                else:
                    await ws.close(code=1001)
            except Exception:
                pass

        return stop_reason

    def _status(self, state: str) -> None:
        if self._on_status is not None:
            self._on_status(state)


_CLOSED = object()
_POISON: Any = object()


class HardStop(Exception):
    """Raised to signal the process should exit with the given code."""

    def __init__(self, exit_code: int) -> None:
        super().__init__(f"hard stop exit_code={exit_code}")
        self.exit_code = exit_code


def _uniform_jitter() -> float:
    return 1.0 + random.uniform(-0.25, 0.25)


_BACKOFF_STEPS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
_BACKOFF_STEADY = 60.0


def _backoff_seconds(attempt: int, *, jitter: Callable[[], float]) -> float:
    """SPEC §12 backoff schedule. `attempt` is 1-indexed."""
    if attempt < 1:
        return 0.0
    if attempt <= len(_BACKOFF_STEPS):
        base = _BACKOFF_STEPS[attempt - 1]
    else:
        base = _BACKOFF_STEADY
    return max(0.0, base * jitter())
