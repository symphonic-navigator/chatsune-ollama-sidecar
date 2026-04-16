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


def test_rejects_non_websocket_scheme(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "http://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    with pytest.raises(ValidationError):
        Settings()


def test_accepts_ws_scheme_for_local_dev(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "ws://backend:8000")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.chatsune_backend_url == "ws://backend:8000"
    assert s.backend_is_insecure() is True
    assert s.ws_endpoint() == "ws://backend:8000/ws/sidecar"


def test_wss_is_not_insecure(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.backend_is_insecure() is False


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


def test_sidecar_engine_defaults_to_ollama(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.sidecar_engine == "ollama"


def test_sidecar_engine_accepts_vllm(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    monkeypatch.setenv("SIDECAR_ENGINE", "vllm")
    s = Settings()
    assert s.sidecar_engine == "vllm"


def test_sidecar_engine_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    monkeypatch.setenv("SIDECAR_ENGINE", "lmstudio")
    with pytest.raises(ValidationError):
        Settings()


def test_vllm_url_default(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.vllm_url == "http://host.docker.internal:8000"


def test_vllm_url_override(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    monkeypatch.setenv("VLLM_URL", "http://localhost:8001")
    s = Settings()
    assert s.vllm_url == "http://localhost:8001"


def test_vllm_config_paths_default_none(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    s = Settings()
    assert s.vllm_models_config_path is None
    assert s.vllm_models_overlay_path is None


def test_vllm_config_paths_from_env(monkeypatch):
    monkeypatch.setenv("CHATSUNE_BACKEND_URL", "wss://chat.example.com")
    monkeypatch.setenv("CHATSUNE_HOST_KEY", "cshost_abc123")
    monkeypatch.setenv("VLLM_MODELS_CONFIG_PATH", "/etc/models.yaml")
    monkeypatch.setenv("VLLM_MODELS_OVERLAY_PATH", "/etc/models.local.yaml")
    s = Settings()
    assert s.vllm_models_config_path == "/etc/models.yaml"
    assert s.vllm_models_overlay_path == "/etc/models.local.yaml"
