"""Tests for the handshake builder and engine factory in main.py."""
from __future__ import annotations

import pytest

from sidecar.config import Settings
from sidecar.main import _build_engine, build_handshake_payload
from sidecar.ollama import OllamaEngine
from sidecar.vllm import VllmEngine


def _settings(**overrides) -> Settings:
    base = dict(
        chatsune_backend_url="wss://chat.example.com",
        chatsune_host_key="cshost_abc",
        ollama_url="http://localhost:11434",
        vllm_url="http://localhost:8001",
        vllm_models_config_path=None,
        vllm_models_overlay_path=None,
        sidecar_engine="ollama",
        sidecar_health_port=0,
        sidecar_log_level="warn",
        sidecar_max_concurrent_requests=1,
    )
    base.update(overrides)
    return Settings.model_construct(**base)


def test_build_engine_returns_ollama_when_configured():
    s = _settings(sidecar_engine="ollama")
    engine = _build_engine(s)
    try:
        assert isinstance(engine, OllamaEngine)
        assert engine.engine_type == "ollama"
    finally:
        # OllamaEngine does not need async close here (no network touched).
        pass


def test_build_engine_returns_vllm_when_configured():
    s = _settings(sidecar_engine="vllm")
    engine = _build_engine(s)
    assert isinstance(engine, VllmEngine)
    assert engine.engine_type == "vllm"


def test_handshake_endpoint_hint_uses_ollama_url_for_ollama():
    s = _settings(sidecar_engine="ollama")
    payload = build_handshake_payload(s, engine_type="ollama", engine_version="0.5.7")
    assert payload["engine"]["type"] == "ollama"
    assert payload["engine"]["endpoint_hint"] == "http://localhost:11434"


def test_handshake_endpoint_hint_uses_vllm_url_for_vllm():
    s = _settings(sidecar_engine="vllm")
    payload = build_handshake_payload(s, engine_type="vllm", engine_version="0.7.3")
    assert payload["engine"]["type"] == "vllm"
    assert payload["engine"]["endpoint_hint"] == "http://localhost:8001"


def test_handshake_still_advertises_all_sidecar_capabilities():
    s = _settings()
    payload = build_handshake_payload(s, engine_type="ollama", engine_version="x")
    assert payload["capabilities"] == [
        "chat_streaming", "tool_calls", "vision", "reasoning"
    ]
