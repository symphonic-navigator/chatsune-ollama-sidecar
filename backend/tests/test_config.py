import pytest
from pydantic import ValidationError

from sidecar.config import Settings


def test_minimal_valid_settings(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.chatsune_backend_url == "wss://chat.example.com"
    assert s.chatsune_host_key == "cshost_abc123"
    assert s.ollama_url == "http://host.docker.internal:11434"
    assert s.sidecar_health_port == 8080
    assert s.sidecar_log_level == "info"
    assert s.sidecar_max_concurrent_requests == 1


def test_rejects_missing_host_key(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.delenv("CHATSUNE_HOST_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_host_key_without_prefix(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "abc123")
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_non_wss_backend(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "http://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    with pytest.raises(ValidationError):
        Settings()


def test_ws_endpoint_appends_path(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.ws_endpoint() == "wss://chat.example.com/ws/sidecar"


def test_ws_endpoint_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com/")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.ws_endpoint() == "wss://chat.example.com/ws/sidecar"


def test_log_level_case_insensitive(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    monkeypatch.setenv("SIDECAR_LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.sidecar_log_level == "debug"
