from sidecar.config import Settings
from sidecar.main import build_handshake_payload


def _settings():
    return Settings.model_construct(
        chatsune_backend_url="wss://chat.example.com",
        chatsune_host_key="cshost_abc",
        ollama_url="http://localhost:11434",
        sidecar_health_port=8080,
        sidecar_log_level="info",
        sidecar_max_concurrent_requests=2,
    )


def test_handshake_payload_shape():
    s = _settings()
    payload = build_handshake_payload(s, engine_type="ollama", engine_version="0.5.7")
    assert payload["type"] == "handshake"
    assert payload["csp_version"] == "1.0"
    assert payload["sidecar_version"]
    assert payload["engine"] == {
        "type": "ollama",
        "version": "0.5.7",
        "endpoint_hint": "http://localhost:11434",
    }
    assert payload["max_concurrent_requests"] == 2
    assert "chat_streaming" in payload["capabilities"]
