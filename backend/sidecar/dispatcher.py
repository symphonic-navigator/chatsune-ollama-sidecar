"""Operation dispatcher (SPEC §7-§10).

Holds one task per in-flight req. Cancellation cancels the task and
emits stream_end(cancelled). All outbound frames go through a single
injected `send` callback (wired to ConnectionManager.send in main).
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .engine import (
    Engine,
    EngineBadResponse,
    EngineUnavailable,
    ModelNotFound,
    ModelOutOfMemory,
    StreamTerminal,
)
from .frames import (
    CancelFrame,
    ErrFrame,
    GenerateChatBody,
    ListModelsBody,
    ReqFrame,
    ResFrame,
    StreamDelta,
    StreamEndFrame,
    StreamFrame,
)
from .logging_setup import get_logger


log = get_logger("dispatcher")


SendFn = Callable[[dict[str, Any]], Awaitable[None] | None]


class Dispatcher:
    def __init__(self, *, engine: Engine, send: SendFn) -> None:
        self._engine = engine
        self._send = send
        self._inflight: dict[str, asyncio.Task[None]] = {}

    async def handle(self, frame: Any) -> None:
        if isinstance(frame, ReqFrame):
            await self._start_req(frame)
        elif isinstance(frame, CancelFrame):
            self._cancel(frame.id)

    async def wait_idle(self) -> None:
        while self._inflight:
            await asyncio.gather(*self._inflight.values(), return_exceptions=True)

    async def cancel_all(self) -> None:
        for rid in list(self._inflight):
            self._cancel(rid)
        await self.wait_idle()

    async def _start_req(self, req: ReqFrame) -> None:
        if req.id in self._inflight:
            log.warning("dispatcher.duplicate_req_id", id=req.id)
            return

        if req.op == "list_models":
            task = asyncio.create_task(self._do_list_models(req.id), name=f"op-{req.id}")
        elif req.op == "generate_chat":
            if req.body is None:
                await self._emit(ErrFrame(
                    id=req.id, code="invalid_request",
                    message="generate_chat requires a body.",
                    recoverable=False,
                ))
                await self._emit_stream_end(req.id, "error")
                return
            task = asyncio.create_task(
                self._do_generate_chat(req.id, req.body), name=f"op-{req.id}"
            )
        else:
            await self._emit(ErrFrame(
                id=req.id, code="invalid_request",
                message=f"Unknown op '{req.op}'.",
                recoverable=False,
            ))
            return

        self._inflight[req.id] = task
        task.add_done_callback(lambda t, rid=req.id: self._inflight.pop(rid, None))

    def _cancel(self, rid: str) -> None:
        task = self._inflight.get(rid)
        if task is None:
            return
        task.cancel()

    async def _do_list_models(self, rid: str) -> None:
        try:
            models = await self._engine.list_models()
        except EngineUnavailable as e:
            await self._emit(ErrFrame(
                id=rid, code="engine_unavailable",
                message=f"{self._engine.engine_type} not reachable.",
                detail=str(e), recoverable=True,
            ))
            return
        except EngineBadResponse as e:
            await self._emit(ErrFrame(
                id=rid, code="engine_error",
                message="Engine returned a non-success response.",
                detail=str(e), recoverable=True,
            ))
            return
        await self._emit(ResFrame(
            id=rid, ok=True,
            body=ListModelsBody(models=models),
        ))

    async def _do_generate_chat(self, rid: str, body: GenerateChatBody) -> None:
        saw_terminal = False
        try:
            async for item in self._engine.generate_chat(body):
                if isinstance(item, StreamDelta):
                    await self._emit(StreamFrame(id=rid, delta=item))
                elif isinstance(item, StreamTerminal):
                    await self._emit(StreamEndFrame(
                        id=rid,
                        finish_reason=item.finish_reason,  # type: ignore[arg-type]
                        usage=item.usage,
                    ))
                    saw_terminal = True
        except asyncio.CancelledError:
            await self._emit_stream_end(rid, "cancelled")
            raise
        except ModelNotFound as e:
            await self._emit(ErrFrame(
                id=rid, code="model_not_found",
                message=f"Model '{body.model_slug}' not available on this homelab.",
                detail=str(e), recoverable=False,
            ))
            await self._emit_stream_end(rid, "error")
            return
        except ModelOutOfMemory as e:
            await self._emit(ErrFrame(
                id=rid, code="model_oom",
                message="Engine could not load the model (out of memory).",
                detail=str(e), recoverable=True,
            ))
            await self._emit_stream_end(rid, "error")
            return
        except EngineUnavailable as e:
            await self._emit(ErrFrame(
                id=rid, code="engine_unavailable",
                message=f"{self._engine.engine_type} not reachable.",
                detail=str(e), recoverable=True,
            ))
            await self._emit_stream_end(rid, "error")
            return
        except EngineBadResponse as e:
            await self._emit(ErrFrame(
                id=rid, code="engine_error",
                message="Engine returned a non-success response.",
                detail=str(e), recoverable=True,
            ))
            await self._emit_stream_end(rid, "error")
            return
        except Exception as e:
            log.exception("dispatcher.unexpected", id=rid)
            await self._emit(ErrFrame(
                id=rid, code="internal",
                message="Internal sidecar error.",
                detail=str(e), recoverable=False,
            ))
            await self._emit_stream_end(rid, "error")
            return

        if not saw_terminal:
            await self._emit_stream_end(rid, "stop")

    async def _emit(self, frame: Any) -> None:
        payload = frame.model_dump(mode="json", exclude_none=False)
        result = self._send(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _emit_stream_end(self, rid: str, reason: str) -> None:
        await self._emit(StreamEndFrame(id=rid, finish_reason=reason))  # type: ignore[arg-type]
