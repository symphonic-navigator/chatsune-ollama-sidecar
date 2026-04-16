"""Entry point. Wires everything together and handles signals (SPEC §17)."""
from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from .config import Settings
from .connection import ConnectionManager, HardStop
from .dispatcher import Dispatcher
from .engine import Engine
from .healthcheck import HealthState, HealthcheckServer
from .logging_setup import configure_logging, get_logger
from .ollama import OllamaEngine
from .vllm import VllmEngine
from .vllm_models_config import load_vllm_models_config


SIDECAR_VERSION = "1.0.0"
CSP_VERSION = "1.0"


def build_handshake_payload(
    settings: Settings, *, engine_type: str, engine_version: str
) -> dict[str, Any]:
    endpoint_hint = (
        settings.ollama_url if engine_type == "ollama" else settings.vllm_url
    )
    return {
        "type": "handshake",
        "csp_version": CSP_VERSION,
        "sidecar_version": SIDECAR_VERSION,
        "engine": {
            "type": engine_type,
            "version": engine_version,
            "endpoint_hint": endpoint_hint,
        },
        "max_concurrent_requests": settings.sidecar_max_concurrent_requests,
        "capabilities": ["chat_streaming", "tool_calls", "vision", "reasoning"],
    }


def _build_engine(settings: Settings) -> Engine:
    if settings.sidecar_engine == "ollama":
        return OllamaEngine(settings.ollama_url)
    if settings.sidecar_engine == "vllm":
        metadata = load_vllm_models_config(
            settings.vllm_models_config_path,
            settings.vllm_models_overlay_path,
        )
        return VllmEngine(settings.vllm_url, metadata=metadata)
    raise ValueError(f"unknown engine: {settings.sidecar_engine}")


async def _run(settings: Settings) -> int:
    configure_logging(settings.sidecar_log_level)
    log = get_logger("main")
    log.info(
        "sidecar.starting",
        backend_url=settings.chatsune_backend_url,
        engine=settings.sidecar_engine,
        ollama_url=settings.ollama_url,
        vllm_url=settings.vllm_url,
        health_port=settings.sidecar_health_port,
        host_key_tail=settings.host_key_tail(),
    )
    if settings.backend_is_insecure():
        log.warning(
            "sidecar.insecure_backend_scheme",
            message=(
                "CHATSUNE_BACKEND_URL uses ws:// (no TLS) — acceptable for "
                "local development only. Production MUST use wss:// per SPEC §3."
            ),
        )

    engine: Engine = _build_engine(settings)
    health = HealthState()
    health_server = HealthcheckServer(health, port=settings.sidecar_health_port)
    await health_server.start()

    version = await engine.probe_version()
    reachable = version != "unknown"
    health.mark_engine(reachable, engine.engine_type)
    log.info("sidecar.engine_probe", reachable=reachable, version=version)

    handshake_payload = build_handshake_payload(
        settings, engine_type=engine.engine_type, engine_version=version
    )

    _cm: ConnectionManager | None = None

    async def send(frame: dict[str, Any]) -> None:
        assert _cm is not None
        await _cm.send(frame)

    dispatcher = Dispatcher(engine=engine, send=send)

    async def on_frame(frame: Any) -> None:
        await dispatcher.handle(frame)

    cm = ConnectionManager(
        settings=settings,
        handshake_payload=handshake_payload,
        on_frame=on_frame,
        on_status_change=health.mark_backend,
    )
    _cm = cm

    loop = asyncio.get_running_loop()

    def _sig() -> None:
        log.info("sidecar.signal_received")
        cm.request_stop()

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    exit_code = 0
    try:
        await cm.run_forever()
    except HardStop as hs:
        exit_code = hs.exit_code
    finally:
        try:
            await asyncio.wait_for(dispatcher.cancel_all(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("sidecar.shutdown_timeout")
        await health_server.stop()
        await engine.aclose()
        log.info("sidecar.stopped", exit_code=exit_code)
    return exit_code


def main() -> int:
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as e:
        print(f"sidecar: configuration error: {e}", file=sys.stderr)
        return 2
    return asyncio.run(_run(settings))


if __name__ == "__main__":
    raise SystemExit(main())
