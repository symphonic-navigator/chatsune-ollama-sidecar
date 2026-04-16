# Chatsune Ollama Sidecar (CSP/1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the reference CSP/1 sidecar for Ollama per `SPEC.md` §18 — a single Python 3.12 process that bridges a local Ollama engine to a remote Chatsune backend via outbound WebSocket Secure.

**Architecture:** One asyncio process. Three long-running coroutines: the WS connection loop (handshake → frame loop → reconnect), the healthcheck HTTP server, and a dispatcher that owns in-flight `req` state. The Ollama engine is a thin adapter over `httpx.AsyncClient`. All frames are Pydantic-modelled and validated on both ingress and egress. Structured JSON logging throughout.

**Tech Stack:** Python 3.12, asyncio, `websockets` (WS client), `httpx` (Ollama HTTP), `pydantic` v2 (frame models + settings), `aiohttp` (healthcheck server — asyncio-native, no uvicorn overhead), `structlog` (JSON logs), `pytest` + `pytest-asyncio` + `respx` (Ollama HTTP mocking).

**Repo layout:**

```
chatsune-ollama-sidecar/
├── .github/workflows/docker.yml
├── .gitignore
├── .env.example
├── start-backend.sh
├── compose.yml
├── README.md
├── SPEC.md                           (exists)
├── LICENSE                           (exists)
├── docs/superpowers/plans/           (this plan)
└── backend/
    ├── Dockerfile
    ├── pyproject.toml
    ├── sidecar/
    │   ├── __init__.py
    │   ├── __main__.py               # python -m sidecar
    │   ├── config.py                 # env-var Settings
    │   ├── logging_setup.py          # structlog JSON
    │   ├── frames.py                 # Pydantic frame models
    │   ├── engine.py                 # Engine protocol + types
    │   ├── ollama.py                 # Ollama adapter
    │   ├── healthcheck.py            # /healthz aiohttp server
    │   ├── connection.py             # WS client + reconnect
    │   ├── dispatcher.py             # req routing + in-flight tracking
    │   └── main.py                   # run() coordinator + signal handling
    └── tests/
        ├── __init__.py
        ├── conftest.py
        ├── fixtures/
        │   └── ollama_tags.json
        ├── test_frames.py
        ├── test_config.py
        ├── test_ollama_list_models.py
        ├── test_ollama_generate_chat.py
        ├── test_healthcheck.py
        └── test_integration.py
```

---

## Task 0: Repo scaffolding — .gitignore, start-backend.sh, GitHub Actions

No tests. These are infrastructure files the subagent just writes.

**Files:**
- Create: `.gitignore`
- Create: `start-backend.sh`
- Create: `.github/workflows/docker.yml`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/

# Environment
.env

# Docker
docker-compose.override.yml

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# uv
uv.lock

# Git worktrees
.worktrees/

# Superpowers brainstorm sessions
.superpowers/

# Obsidian state (when/if added)
obsidian/.obsidian/workspace.json
obsidian/.obsidian/workspace-mobile.json
obsidian/.obsidian/graph.json

# Runtime log files
backend/logs/
```

- [ ] **Step 2: Write `start-backend.sh`**

```bash
#! /bin/bash

cd "$(dirname "$0")/backend" || exit 1

uv sync
uv run python -m sidecar 2>&1
```

Make it executable: `chmod +x start-backend.sh`.

- [ ] **Step 3: Write `.github/workflows/docker.yml`**

Mirror `../chatsune/.github/workflows/docker.yml` but drop the frontend job and its summary row. Keep backend + report.

```yaml
name: Docker Build & Push

on:
  push:
    branches: [master]
    tags: ["v*.*.*"]
  pull_request:
    branches: [master]

env:
  REGISTRY: ghcr.io
  BACKEND_IMAGE: ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend

jobs:
  build-backend:
    name: Build Backend
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write
    outputs:
      tags: ${{ steps.meta.outputs.tags }}
      digest: ${{ steps.build-and-push.outputs.digest }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install cosign
        if: github.event_name != 'pull_request'
        uses: sigstore/cosign-installer@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log into registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.BACKEND_IMAGE }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            type=sha
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        id: build-and-push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: backend/Dockerfile
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=backend
          cache-to: type=gha,mode=max,scope=backend

      - name: Sign the published Docker image
        if: github.event_name != 'pull_request'
        env:
          TAGS: ${{ steps.meta.outputs.tags }}
          DIGEST: ${{ steps.build-and-push.outputs.digest }}
        run: echo "${TAGS}" | xargs -I {} cosign sign --yes {}@${DIGEST}

      - name: Write build manifest
        env:
          TRIGGER: ${{ github.event_name }}
        run: |
          SHORT_SHA=$(echo "$GITHUB_SHA" | cut -c1-7)
          cat > build-manifest.json << EOF
          {
            "schema": "chatsune-build/v1",
            "built_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
            "trigger": "$TRIGGER",
            "artifacts": [
              {
                "name": "chatsune-ollama-sidecar-backend",
                "type": "docker",
                "version": "$SHORT_SHA",
                "ref": "ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend:sha-$SHORT_SHA"
              }
            ]
          }
          EOF

      - name: Upload build manifest
        uses: actions/upload-artifact@v4
        with:
          name: build-manifest-backend
          path: build-manifest.json
          retention-days: 7

  report:
    name: Build Report
    runs-on: ubuntu-latest
    needs: [build-backend]
    if: always()

    steps:
      - name: Summarize built images
        run: |
          echo "## Docker Build Report" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "| Image | Tags |" >> $GITHUB_STEP_SUMMARY
          echo "|-------|------|" >> $GITHUB_STEP_SUMMARY

          BACKEND_TAGS="${{ needs.build-backend.outputs.tags }}"
          if [ -n "$BACKEND_TAGS" ]; then
            TAGS_FORMATTED="${BACKEND_TAGS//$'\n'/<br>}"
            TAGS_FORMATTED="${TAGS_FORMATTED%<br>}"
            echo "| \`backend\` | $TAGS_FORMATTED |" >> $GITHUB_STEP_SUMMARY
          else
            echo "| \`backend\` | _(not pushed — PR or failed)_ |" >> $GITHUB_STEP_SUMMARY
          fi

          echo "" >> $GITHUB_STEP_SUMMARY
          echo "Backend job: \`${{ needs.build-backend.result }}\`" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore start-backend.sh .github/workflows/docker.yml
git commit -m "Add repo scaffolding — gitignore, start script, CI workflow"
```

---

## Task 1: Python project setup — pyproject.toml, .env.example

**Files:**
- Create: `backend/pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: Write `backend/pyproject.toml`**

```toml
[project]
name = "chatsune-ollama-sidecar"
version = "1.0.0"
description = "CSP/1 sidecar bridging a local Ollama engine to a remote Chatsune backend."
requires-python = ">=3.12"
license = { text = "GPL-3.0-or-later" }
dependencies = [
    "websockets>=14.0",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "aiohttp>=3.11",
    "structlog>=24.1",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.25",
    "respx>=0.22",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.setuptools.packages.find]
include = ["sidecar*"]
```

- [ ] **Step 2: Write `.env.example`**

```dotenv
# --- Required -------------------------------------------------------------

# Base WebSocket Secure URL of the Chatsune backend. The sidecar appends
# /ws/sidecar. Must be wss:// with a certificate trusted by the system
# trust store. No self-signed shortcuts.
CHATSUNE_BACKEND_URL=wss://chat.example.com

# Host-Key issued in the Chatsune UI. MUST start with "cshost_".
# Never commit a real value.
CHATSUNE_HOST_KEY=cshost_replace_me

# --- Ollama engine --------------------------------------------------------

# URL of the local Ollama HTTP API. On Docker Desktop / Podman the
# default works out of the box; on Linux bare-metal set it to
# http://localhost:11434 or the LAN IP of the Ollama host.
OLLAMA_URL=http://host.docker.internal:11434

# --- Optional -------------------------------------------------------------

# Port for the loopback healthcheck HTTP server. Docker's HEALTHCHECK
# probes this. Must stay on 127.0.0.1.
SIDECAR_HEALTH_PORT=8080

# Log level: debug, info, warn, error. debug MAY include user content.
SIDECAR_LOG_LEVEL=info

# Override the handshake-advertised max_concurrent_requests.
# Default for the Ollama sidecar is 1 (Ollama loads one model at a time).
# SIDECAR_MAX_CONCURRENT_REQUESTS=1
```

- [ ] **Step 3: Verify uv sync works**

Run (from repo root): `cd backend && uv sync`
Expected: creates `.venv/`, resolves deps, no errors.

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml .env.example
git commit -m "Add Python project metadata and env template"
```

---

## Task 2: Config module — env-var Settings

**Files:**
- Create: `backend/sidecar/__init__.py` (empty)
- Create: `backend/sidecar/config.py`
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_config.py`:

```python
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
```

`backend/tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make `sidecar` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: ImportError / ModuleNotFoundError on `sidecar.config`.

- [ ] **Step 3: Write `backend/sidecar/config.py`**

```python
"""Sidecar settings loaded from environment variables.

See SPEC.md §14. Fail-fast on missing or malformed required variables.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LogLevel = Literal["debug", "info", "warn", "error"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chatsune_backend_url: str = Field(...)
    chatsune_host_key: str = Field(...)

    ollama_url: str = Field(default="http://host.docker.internal:11434")

    sidecar_health_port: int = Field(default=8080, ge=1, le=65535)
    sidecar_log_level: LogLevel = Field(default="info")
    sidecar_max_concurrent_requests: int = Field(default=1, ge=1)

    @field_validator("chatsune_backend_url")
    @classmethod
    def _must_be_wss(cls, v: str) -> str:
        if not v.startswith("wss://"):
            raise ValueError("CHATSUNE_BACKEND_URL must use the wss:// scheme")
        return v

    @field_validator("chatsune_host_key")
    @classmethod
    def _must_have_prefix(cls, v: str) -> str:
        if not v.startswith("cshost_"):
            raise ValueError("CHATSUNE_HOST_KEY must start with 'cshost_'")
        return v

    @field_validator("sidecar_log_level", mode="before")
    @classmethod
    def _lowercase(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v

    def ws_endpoint(self) -> str:
        return self.chatsune_backend_url.rstrip("/") + "/ws/sidecar"

    def host_key_tail(self) -> str:
        """Last 4 characters of the host key, safe for logs (SPEC §16)."""
        return self.chatsune_host_key[-4:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/__init__.py backend/sidecar/config.py backend/tests/__init__.py backend/tests/conftest.py backend/tests/test_config.py
git commit -m "Add Settings module with env-var validation"
```

---

## Task 3: Logging setup — structured JSON with secret redaction

**Files:**
- Create: `backend/sidecar/logging_setup.py`
- Test: inline smoke test in `backend/tests/test_logging.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_logging.py`:

```python
import json
import logging
from io import StringIO

from sidecar.logging_setup import configure_logging, get_logger


def test_emits_json_lines(capsys):
    configure_logging("info")
    log = get_logger("test")
    log.info("hello_event", foo="bar", id="abc-123")
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    record = json.loads(captured[0])
    assert record["event"] == "hello_event"
    assert record["level"] == "info"
    assert record["foo"] == "bar"
    assert record["id"] == "abc-123"
    assert "ts" in record


def test_respects_level(capsys):
    configure_logging("warn")
    log = get_logger("test")
    log.info("should_not_appear")
    log.warning("should_appear")
    out = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line)["event"] for line in out]
    assert "should_not_appear" not in events
    assert "should_appear" in events


def test_warn_alias_maps_to_warning(capsys):
    configure_logging("warn")
    log = get_logger("test")
    log.warning("via_warn")
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[0])["level"] == "warning"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_logging.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `backend/sidecar/logging_setup.py`**

```python
"""Structured JSON logging per SPEC §16."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str = "info") -> None:
    """Configure structlog to emit JSON lines to stdout.

    Safe to call repeatedly — idempotent. `level` is a CSP level name
    (debug/info/warn/error); the standard-library WARNING alias is also
    accepted so logger.warning() works without translation.
    """
    numeric = _LEVELS.get(level.lower(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(numeric)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.EventRenamer("event"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_logging.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/logging_setup.py backend/tests/test_logging.py
git commit -m "Add structured JSON logging"
```

---

## Task 4: Frame models — Pydantic types for all CSP/1 frames

Every wire frame from SPEC §5 becomes a Pydantic model. Parsing is a tagged-union dispatch on `type`.

**Files:**
- Create: `backend/sidecar/frames.py`
- Test: `backend/tests/test_frames.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_frames.py`:

```python
import json

import pytest
from pydantic import ValidationError

from sidecar.frames import (
    AuthRevokedFrame,
    CancelFrame,
    ContentPartImage,
    ContentPartText,
    EngineInfo,
    ErrFrame,
    GenerateChatBody,
    HandshakeAckFrame,
    HandshakeFrame,
    Message,
    ModelDescriptor,
    PingFrame,
    PongFrame,
    ReqFrame,
    ResFrame,
    StreamDelta,
    StreamEndFrame,
    StreamFrame,
    SupersededFrame,
    ToolCallFragment,
    Usage,
    parse_frame,
)


def test_handshake_roundtrip():
    f = HandshakeFrame(
        csp_version="1.0",
        sidecar_version="1.0.0",
        engine=EngineInfo(type="ollama", version="0.5.0", endpoint_hint="http://localhost:11434"),
        max_concurrent_requests=2,
        capabilities=["chat_streaming", "tool_calls"],
    )
    payload = f.model_dump(mode="json")
    assert payload["type"] == "handshake"
    parsed = parse_frame(payload)
    assert isinstance(parsed, HandshakeFrame)
    assert parsed.engine.type == "ollama"


def test_parse_handshake_ack_accepted():
    raw = {
        "type": "handshake_ack",
        "csp_version": "1.0",
        "homelab_id": "Xk7bQ2eJn9m",
        "display_name": "Wohnzimmer-GPU",
        "accepted": True,
        "notices": [],
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, HandshakeAckFrame)
    assert parsed.accepted is True


def test_ping_pong_have_no_id():
    parsed = parse_frame({"type": "ping"})
    assert isinstance(parsed, PingFrame)
    parsed = parse_frame({"type": "pong"})
    assert isinstance(parsed, PongFrame)


def test_req_list_models_no_body():
    parsed = parse_frame({"type": "req", "id": "abc-123", "op": "list_models"})
    assert isinstance(parsed, ReqFrame)
    assert parsed.op == "list_models"
    assert parsed.body is None


def test_req_generate_chat_parses_body():
    raw = {
        "type": "req",
        "id": "abc-123",
        "op": "generate_chat",
        "body": {
            "model_slug": "llama3.2:8b",
            "messages": [{"role": "user", "content": "hi"}],
            "parameters": {"temperature": 0.7},
        },
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, ReqFrame)
    body = parsed.body
    assert isinstance(body, GenerateChatBody)
    assert body.model_slug == "llama3.2:8b"
    assert body.messages[0].role == "user"
    assert body.parameters.temperature == 0.7


def test_multimodal_user_message():
    raw = {
        "type": "req",
        "id": "abc",
        "op": "generate_chat",
        "body": {
            "model_slug": "llava",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "media_type": "image/png", "data_b64": "aGVsbG8="},
                ]},
            ],
        },
    }
    parsed = parse_frame(raw)
    parts = parsed.body.messages[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], ContentPartText)
    assert isinstance(parts[1], ContentPartImage)
    assert parts[1].media_type == "image/png"


def test_assistant_tool_calls_echo():
    raw = {
        "role": "assistant",
        "content": "let me check",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"loc\":\"Vienna\"}"},
            }
        ],
    }
    msg = Message.model_validate(raw)
    assert msg.tool_calls and msg.tool_calls[0].function.name == "get_weather"


def test_cancel_frame():
    parsed = parse_frame({"type": "cancel", "id": "abc"})
    assert isinstance(parsed, CancelFrame)
    assert parsed.id == "abc"


def test_stream_frame_one_channel():
    raw = {
        "type": "stream",
        "id": "abc",
        "delta": {"content": "Hello", "reasoning": None, "tool_calls": None},
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, StreamFrame)
    assert parsed.delta.content == "Hello"


def test_stream_end_with_usage():
    raw = {
        "type": "stream_end",
        "id": "abc",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, StreamEndFrame)
    assert parsed.finish_reason == "stop"
    assert parsed.usage.total_tokens == 30


def test_stream_end_without_usage():
    parsed = parse_frame({"type": "stream_end", "id": "abc", "finish_reason": "cancelled"})
    assert isinstance(parsed, StreamEndFrame)
    assert parsed.usage is None


def test_err_frame():
    raw = {
        "type": "err",
        "id": "abc",
        "code": "model_not_found",
        "message": "Model missing.",
        "detail": None,
        "recoverable": False,
    }
    parsed = parse_frame(raw)
    assert isinstance(parsed, ErrFrame)
    assert parsed.code == "model_not_found"


def test_auth_revoked_superseded_parse():
    assert isinstance(parse_frame({"type": "auth_revoked"}), AuthRevokedFrame)
    assert isinstance(parse_frame({"type": "superseded"}), SupersededFrame)


def test_unknown_type_returns_none():
    """Per SPEC §4: unknown types MUST be ignored for forward-compatibility."""
    assert parse_frame({"type": "some_future_frame", "id": "x"}) is None


def test_missing_type_raises():
    with pytest.raises(ValidationError):
        parse_frame({"id": "x"})


def test_req_rejects_unknown_op():
    with pytest.raises(ValidationError):
        parse_frame({"type": "req", "id": "x", "op": "do_something_else"})


def test_model_descriptor_context_length_required():
    raw = {
        "slug": "llama3.2:8b",
        "display_name": "Llama",
        "parameter_count": 8_030_261_248,
        "context_length": 131072,
        "quantisation": "Q4_K_M",
        "capabilities": ["text"],
        "engine_family": "ollama",
        "engine_model_id": "llama3.2:8b",
        "engine_metadata": {},
    }
    d = ModelDescriptor.model_validate(raw)
    assert d.context_length == 131072

    del raw["context_length"]
    with pytest.raises(ValidationError):
        ModelDescriptor.model_validate(raw)


def test_stream_delta_all_channels_nullable():
    """Populating one channel and leaving the others null is legal."""
    d = StreamDelta(content="hi", reasoning=None, tool_calls=None)
    assert d.content == "hi"
    d = StreamDelta(content=None, reasoning="think", tool_calls=None)
    assert d.reasoning == "think"
    d = StreamDelta(
        content=None,
        reasoning=None,
        tool_calls=[ToolCallFragment(index=0, id="c1", type="function",
                                      function={"name": "f", "arguments": "{"})],
    )
    assert d.tool_calls[0].function.name == "f"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_frames.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `backend/sidecar/frames.py`**

```python
"""Pydantic models for every CSP/1 frame type (SPEC §4, §5).

Parsing contract
----------------
- Unknown `type` values return None (SPEC §4: ignore for forward-compat).
- Missing required fields on a recognised `type` raise ValidationError
  (caller SHOULD close the connection).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------

FinishReason = Literal["stop", "length", "tool_calls", "cancelled", "error"]
ErrCode = Literal[
    "model_not_found",
    "model_oom",
    "engine_unavailable",
    "engine_error",
    "invalid_request",
    "rate_limited",
    "cancelled",
    "internal",
]
EngineType = Literal["ollama", "lmstudio", "vllm", "llamacpp"]
Capability = Literal["chat_streaming", "tool_calls", "vision", "reasoning"]
ModelCapability = Literal["text", "tool_calling", "vision", "reasoning", "json_mode"]
OpName = Literal["list_models", "generate_chat"]


# ---------------------------------------------------------------------------
# Content parts (§8.1) and messages
# ---------------------------------------------------------------------------

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image"]
    media_type: str
    data_b64: str


ContentPart = Annotated[
    Union[ContentPartText, ContentPartImage],
    Field(discriminator="type"),
]


class ToolFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolFunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    type: Literal["function"]
    function: ToolFunctionDef


class GenerateParameters(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


class GenerateOptions(BaseModel):
    reasoning: bool = False


class GenerateChatBody(BaseModel):
    model_slug: str
    messages: list[Message]
    tools: list[ToolDef] | None = None
    parameters: GenerateParameters = Field(default_factory=GenerateParameters)
    options: GenerateOptions = Field(default_factory=GenerateOptions)


# ---------------------------------------------------------------------------
# list_models types
# ---------------------------------------------------------------------------

class ModelDescriptor(BaseModel):
    slug: str
    display_name: str
    parameter_count: int | None
    context_length: int  # Required, no null (SPEC §7.2)
    quantisation: str | None
    capabilities: list[ModelCapability]
    engine_family: EngineType
    engine_model_id: str
    engine_metadata: dict[str, Any] = Field(default_factory=dict)


class ListModelsBody(BaseModel):
    models: list[ModelDescriptor]


# ---------------------------------------------------------------------------
# Streaming deltas (§8.2)
# ---------------------------------------------------------------------------

class ToolCallFragmentFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallFragment(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ToolCallFragmentFunction


class StreamDelta(BaseModel):
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallFragment] | None = None


# ---------------------------------------------------------------------------
# Usage (§8.3)
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Connection-lifecycle frames (§5.1)
# ---------------------------------------------------------------------------

class EngineInfo(BaseModel):
    type: EngineType
    version: str
    endpoint_hint: str | None = None


class HandshakeFrame(BaseModel):
    type: Literal["handshake"] = "handshake"
    csp_version: str
    sidecar_version: str
    engine: EngineInfo
    max_concurrent_requests: int = Field(ge=1)
    capabilities: list[Capability]


class HandshakeAckFrame(BaseModel):
    type: Literal["handshake_ack"] = "handshake_ack"
    csp_version: str
    homelab_id: str | None = None
    display_name: str | None = None
    accepted: bool
    notices: list[str] = Field(default_factory=list)


class PingFrame(BaseModel):
    type: Literal["ping"] = "ping"


class PongFrame(BaseModel):
    type: Literal["pong"] = "pong"


class AuthRevokedFrame(BaseModel):
    type: Literal["auth_revoked"] = "auth_revoked"


class SupersededFrame(BaseModel):
    type: Literal["superseded"] = "superseded"


# ---------------------------------------------------------------------------
# Request/response frames (§5.2)
# ---------------------------------------------------------------------------

class ReqFrame(BaseModel):
    type: Literal["req"] = "req"
    id: str
    op: OpName
    body: GenerateChatBody | None = None


class ResFrame(BaseModel):
    type: Literal["res"] = "res"
    id: str
    ok: bool
    body: ListModelsBody | None = None


class StreamFrame(BaseModel):
    type: Literal["stream"] = "stream"
    id: str
    delta: StreamDelta


class StreamEndFrame(BaseModel):
    type: Literal["stream_end"] = "stream_end"
    id: str
    finish_reason: FinishReason
    usage: Usage | None = None


class CancelFrame(BaseModel):
    type: Literal["cancel"] = "cancel"
    id: str


class ErrFrame(BaseModel):
    type: Literal["err"] = "err"
    id: str | None = None
    code: ErrCode
    message: str
    detail: str | None = None
    recoverable: bool


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

AnyFrame = Union[
    HandshakeFrame,
    HandshakeAckFrame,
    PingFrame,
    PongFrame,
    AuthRevokedFrame,
    SupersededFrame,
    ReqFrame,
    ResFrame,
    StreamFrame,
    StreamEndFrame,
    CancelFrame,
    ErrFrame,
]

_TYPE_MAP: dict[str, type[BaseModel]] = {
    "handshake": HandshakeFrame,
    "handshake_ack": HandshakeAckFrame,
    "ping": PingFrame,
    "pong": PongFrame,
    "auth_revoked": AuthRevokedFrame,
    "superseded": SupersededFrame,
    "req": ReqFrame,
    "res": ResFrame,
    "stream": StreamFrame,
    "stream_end": StreamEndFrame,
    "cancel": CancelFrame,
    "err": ErrFrame,
}


class _FrameEnvelope(BaseModel):
    type: str


def parse_frame(payload: dict[str, Any]) -> AnyFrame | None:
    """Dispatch a raw JSON frame to its model.

    Returns None for unknown `type` values (SPEC §4: forward-compat).
    Raises ValidationError for a recognised type with bad fields OR a
    missing `type` field.
    """
    env = _FrameEnvelope.model_validate(payload)
    model = _TYPE_MAP.get(env.type)
    if model is None:
        return None
    return model.model_validate(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_frames.py -v`
Expected: all passed (18 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/frames.py backend/tests/test_frames.py
git commit -m "Add Pydantic models and parser for every CSP/1 frame"
```

---

## Task 5: Engine protocol + Ollama list_models

Define the Engine abstraction and implement Ollama's `list_models` with metadata-gap handling per SPEC §6.5/§7.

**Files:**
- Create: `backend/sidecar/engine.py`
- Create: `backend/sidecar/ollama.py`
- Create: `backend/tests/fixtures/ollama_tags.json`
- Test: `backend/tests/test_ollama_list_models.py`

- [ ] **Step 1: Create the fixture**

`backend/tests/fixtures/ollama_tags.json`:

```json
{
  "models": [
    {
      "name": "llama3.2:8b",
      "size": 4920000000,
      "digest": "abc123",
      "details": {
        "family": "llama",
        "parameter_size": "8B",
        "quantization_level": "Q4_K_M"
      }
    },
    {
      "name": "llava:7b",
      "size": 4100000000,
      "digest": "def456",
      "details": {
        "family": "llava",
        "parameter_size": "7B",
        "quantization_level": "Q4_0"
      }
    },
    {
      "name": "mystery:1b",
      "size": 500000000,
      "digest": "ghi789",
      "details": {
        "family": "mystery",
        "parameter_size": "1B",
        "quantization_level": "Q8_0"
      }
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

`backend/tests/test_ollama_list_models.py`:

```python
import json
from pathlib import Path

import httpx
import pytest
import respx

from sidecar.ollama import OllamaEngine


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _tags_body():
    return json.loads((FIXTURE_DIR / "ollama_tags.json").read_text())


def _show_body(model: str) -> dict:
    bodies = {
        "llama3.2:8b": {
            "capabilities": ["tools"],
            "model_info": {"llama.context_length": 131072},
        },
        "llava:7b": {
            "capabilities": [],
            "model_info": {
                "llama.context_length": 4096,
                "llama.vision": 1,
            },
        },
        "mystery:1b": {
            "capabilities": [],
            "model_info": {},  # no context length, MUST be dropped
        },
    }
    return bodies[model]


@respx.mock
async def test_list_models_builds_descriptors():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=_tags_body())
    )
    for name in ("llama3.2:8b", "llava:7b", "mystery:1b"):
        respx.post(
            "http://localhost:11434/api/show",
            json__eq={"model": name},
        ).mock(return_value=httpx.Response(200, json=_show_body(name)))

    engine = OllamaEngine("http://localhost:11434")
    try:
        models = await engine.list_models()
    finally:
        await engine.aclose()

    slugs = {m.slug for m in models}
    assert slugs == {"llama3.2:8b", "llava:7b"}  # mystery dropped

    llama = next(m for m in models if m.slug == "llama3.2:8b")
    assert llama.context_length == 131072
    assert llama.parameter_count == 8_000_000_000
    assert llama.quantisation == "Q4_K_M"
    assert "tool_calling" in llama.capabilities
    assert llama.engine_family == "ollama"
    assert llama.engine_model_id == "llama3.2:8b"

    llava = next(m for m in models if m.slug == "llava:7b")
    assert "vision" in llava.capabilities


@respx.mock
async def test_list_models_engine_unreachable():
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = OllamaEngine("http://localhost:11434")
    try:
        with pytest.raises(EngineUnavailable):
            await engine.list_models()
    finally:
        await engine.aclose()


@respx.mock
async def test_probe_version():
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.5.7"})
    )
    engine = OllamaEngine("http://localhost:11434")
    try:
        assert await engine.probe_version() == "0.5.7"
    finally:
        await engine.aclose()


@respx.mock
async def test_probe_version_returns_unknown_on_failure():
    respx.get("http://localhost:11434/api/version").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = OllamaEngine("http://localhost:11434")
    try:
        assert await engine.probe_version() == "unknown"
    finally:
        await engine.aclose()


# Import the exception at module load so the earlier test resolves it.
from sidecar.engine import EngineUnavailable  # noqa: E402
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_ollama_list_models.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Write `backend/sidecar/engine.py`**

```python
"""Engine abstraction (SPEC §6, §15).

v1 ships one concrete engine (Ollama). This module defines the protocol
the rest of the sidecar consumes so the plumbing does not care which
engine is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from .frames import (
    GenerateChatBody,
    ModelDescriptor,
    StreamDelta,
    Usage,
)


# ---------------------------------------------------------------------------
# Exceptions — caught by the dispatcher and converted to `err` frames
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Base class. `err.code` derived by dispatcher."""


class EngineUnavailable(EngineError):
    """Local engine daemon cannot be reached at all."""


class ModelNotFound(EngineError):
    """Engine rejected the requested model."""


class ModelOutOfMemory(EngineError):
    """Engine could not load the model (typically VRAM)."""


class EngineBadResponse(EngineError):
    """Engine returned a non-success status or malformed payload."""


# ---------------------------------------------------------------------------
# Streaming output contract
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StreamTerminal:
    """Produced by the engine adapter when its stream ends cleanly.

    `finish_reason` is the SPEC §8.3 vocabulary. `usage` is optional —
    omit when the engine does not report token counts.
    """
    finish_reason: str
    usage: Usage | None = None


EngineStreamItem = StreamDelta | StreamTerminal


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------

class Engine(Protocol):
    """Minimum surface every engine adapter exposes."""

    engine_type: str  # "ollama" etc., matches SPEC EngineType

    async def probe_version(self) -> str: ...

    async def list_models(self) -> list[ModelDescriptor]: ...

    def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]: ...

    async def aclose(self) -> None: ...
```

- [ ] **Step 5: Write initial `backend/sidecar/ollama.py` (list_models + version only)**

```python
"""Ollama engine adapter (SPEC §15.1).

Handles discovery (`/api/tags` + `/api/show`) and — in later tasks —
streaming chat (`/api/chat`).
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .engine import (
    EngineBadResponse,
    EngineUnavailable,
    ModelNotFound,
)
from .frames import ModelCapability, ModelDescriptor


# Model families known to emit <think>...</think> reasoning tokens.
# Conservative allowlist — SPEC §15.1.
_REASONING_FAMILIES = {
    "deepseek-r1",
    "deepseek_r1",
    "qwen3-thinking",
    "qwen3_thinking",
    "qwq",
}


class OllamaEngine:
    engine_type = "ollama"

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self._base = url.rstrip("/")
        # Generous total timeout but short connect; stream operations
        # override per-request.
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----- version probe --------------------------------------------------

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/api/version")
            r.raise_for_status()
            return r.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError):
            return "unknown"

    # ----- list_models ----------------------------------------------------

    async def list_models(self) -> list[ModelDescriptor]:
        try:
            tags = await self._client.get("/api/tags")
            tags.raise_for_status()
            tag_payload: dict[str, Any] = tags.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e

        out: list[ModelDescriptor] = []
        for raw in tag_payload.get("models", []):
            descriptor = await self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    async def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        name = raw.get("name")
        if not name:
            return None

        details = raw.get("details", {}) or {}
        family = (details.get("family") or "").lower()
        param_size_str = details.get("parameter_size")  # "8B"
        quantisation = details.get("quantization_level")

        try:
            show = await self._client.post("/api/show", json={"model": name})
            if show.status_code == 404:
                return None  # raced with a model deletion
            show.raise_for_status()
            show_body = show.json()
        except httpx.HTTPError:
            return None  # fail-closed per SPEC §6.5

        model_info: dict[str, Any] = show_body.get("model_info", {}) or {}
        context_length = _extract_context_length(model_info)
        if context_length is None:
            # SPEC §6.5: drop models without context_length.
            return None

        caps: list[ModelCapability] = ["text"]
        engine_caps = [c.lower() for c in show_body.get("capabilities", []) or []]
        if "tools" in engine_caps:
            caps.append("tool_calling")
        if _has_vision(model_info) or "vision" in engine_caps:
            caps.append("vision")
        if family in _REASONING_FAMILIES or "reasoning" in engine_caps:
            caps.append("reasoning")

        return ModelDescriptor(
            slug=name,
            display_name=_display_name(name),
            parameter_count=_parse_param_size(param_size_str),
            context_length=context_length,
            quantisation=quantisation,
            capabilities=caps,
            engine_family="ollama",
            engine_model_id=name,
            engine_metadata={
                "family": family or None,
                "digest": raw.get("digest"),
                "size_bytes": raw.get("size"),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_KEY_RE = re.compile(r"\.context_length$")


def _extract_context_length(model_info: dict[str, Any]) -> int | None:
    """Ollama reports `<arch>.context_length` — scan any matching key."""
    for key, value in model_info.items():
        if _CONTEXT_KEY_RE.search(key) and isinstance(value, (int, float)):
            return int(value)
    return None


def _has_vision(model_info: dict[str, Any]) -> bool:
    for key, value in model_info.items():
        if key.endswith(".vision") and value:
            return True
    return False


_PARAM_SIZE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<mag>[BMbm])$")


def _parse_param_size(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    m = _PARAM_SIZE_RE.match(raw.strip())
    if not m:
        return None
    num = float(m.group("num"))
    mag = m.group("mag").upper()
    multiplier = 1_000_000_000 if mag == "B" else 1_000_000
    return int(num * multiplier)


def _display_name(slug: str) -> str:
    # "llama3.2:8b" → "Llama3.2 8B". Falls back to the slug for anything
    # weird.
    try:
        base, tag = slug.split(":", 1)
        return f"{base.title()} {tag.upper()}"
    except ValueError:
        return slug
```

- [ ] **Step 6: Run tests**

Run: `cd backend && uv run pytest tests/test_ollama_list_models.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/sidecar/engine.py backend/sidecar/ollama.py backend/tests/fixtures/ollama_tags.json backend/tests/test_ollama_list_models.py
git commit -m "Add Ollama engine adapter with list_models and metadata-gap handling"
```

---

## Task 6: Ollama generate_chat — streaming with reasoning and tool-call parsing

Extend the Ollama adapter with `generate_chat`. Handles `<think>` parsing, OpenAI-style tool-call fragments, finish reasons, usage, and cancellation via stream close.

**Files:**
- Modify: `backend/sidecar/ollama.py` — add `generate_chat` and helpers.
- Test: `backend/tests/test_ollama_generate_chat.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_ollama_generate_chat.py`:

```python
import asyncio
import json
from typing import AsyncIterator

import httpx
import pytest
import respx

from sidecar.engine import EngineUnavailable, ModelNotFound, StreamTerminal
from sidecar.frames import (
    GenerateChatBody,
    Message,
    StreamDelta,
)
from sidecar.ollama import OllamaEngine


def _ndjson(lines: list[dict]) -> bytes:
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode()


@respx.mock
async def test_plain_text_stream():
    chunks = [
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"message": {"role": "assistant", "content": " world"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 7,
        },
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )

    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="llama3.2:8b",
        messages=[Message(role="user", content="Hi")],
    )
    items = []
    try:
        async for item in engine.generate_chat(body):
            items.append(item)
    finally:
        await engine.aclose()

    deltas = [i for i in items if isinstance(i, StreamDelta)]
    terms = [i for i in items if isinstance(i, StreamTerminal)]

    assert [d.content for d in deltas] == ["Hello", " world"]
    assert len(terms) == 1
    assert terms[0].finish_reason == "stop"
    assert terms[0].usage is not None
    assert terms[0].usage.prompt_tokens == 5
    assert terms[0].usage.completion_tokens == 7


@respx.mock
async def test_reasoning_split_with_think_tags():
    """<think>...</think> inside content becomes reasoning deltas."""
    chunks = [
        {"message": {"role": "assistant", "content": "<think>"}, "done": False},
        {"message": {"role": "assistant", "content": "let me ponder"}, "done": False},
        {"message": {"role": "assistant", "content": "</think>"}, "done": False},
        {"message": {"role": "assistant", "content": "The answer "}, "done": False},
        {"message": {"role": "assistant", "content": "is 42."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="deepseek-r1:7b",
        messages=[Message(role="user", content="what?")],
        options={"reasoning": True},
    )
    items: list = []
    try:
        async for item in engine.generate_chat(body):
            items.append(item)
    finally:
        await engine.aclose()

    deltas = [i for i in items if isinstance(i, StreamDelta)]
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)

    assert "let me ponder" in reasoning
    assert content == "The answer is 42."
    # Channel purity: no delta populates content AND reasoning.
    for d in deltas:
        populated = sum(x is not None for x in (d.content, d.reasoning, d.tool_calls))
        assert populated <= 1


@respx.mock
async def test_reasoning_suppressed_when_disabled():
    """options.reasoning=False: think-tagged text is swallowed (not folded)."""
    chunks = [
        {"message": {"role": "assistant", "content": "<think>secret</think>visible"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="deepseek-r1:7b",
        messages=[Message(role="user", content="q")],
        options={"reasoning": False},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    content = "".join(d.content or "" for d in deltas)
    reasoning = "".join(d.reasoning or "" for d in deltas)
    assert "secret" not in content
    assert reasoning == ""
    assert "visible" in content


@respx.mock
async def test_tool_call_fragments_passed_through():
    chunks = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"loc\":"},
                    }
                ],
            },
            "done": False,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"arguments": "\"Vienna\"}"},
                    }
                ],
            },
            "done": False,
        },
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="llama3.2:8b",
        messages=[Message(role="user", content="weather?")],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    frags = [f for d in deltas for f in (d.tool_calls or [])]
    assert frags[0].index == 0
    assert frags[0].id == "call_1"
    assert frags[0].function.name == "get_weather"
    # Second fragment at same index — only carries the argument slice.
    assert frags[1].index == 0
    assert frags[1].function.arguments == "\"Vienna\"}"


@respx.mock
async def test_finish_reason_length_and_tool_calls():
    chunks = [
        {"message": {"role": "assistant", "content": "abc"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length"},
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    term = None
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamTerminal):
                term = item
    finally:
        await engine.aclose()
    assert term is not None
    assert term.finish_reason == "length"


@respx.mock
async def test_model_not_found_raises():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model 'xxx' not found"})
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="xxx", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(ModelNotFound):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_connect_failure_raises_engine_unavailable():
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(EngineUnavailable):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_cancellation_stops_iteration():
    """Closing the async generator mid-stream stops consumption."""
    chunks = [
        {"message": {"role": "assistant", "content": f"tok{i}"}, "done": False}
        for i in range(100)
    ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )
    engine = OllamaEngine("http://localhost:11434")
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    seen = 0
    gen = engine.generate_chat(body)
    try:
        async for item in gen:
            if isinstance(item, StreamDelta):
                seen += 1
                if seen >= 3:
                    break
    finally:
        await gen.aclose()
        await engine.aclose()
    assert seen == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_ollama_generate_chat.py -v`
Expected: `AttributeError: 'OllamaEngine' object has no attribute 'generate_chat'`.

- [ ] **Step 3: Extend `backend/sidecar/ollama.py`**

Add at the top of the file, with the other imports:

```python
import json
from typing import AsyncIterator
```

Add these imports to the existing `from .engine import ...` line:

```python
from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
    StreamTerminal,
)
from .frames import (
    GenerateChatBody,
    ModelCapability,
    ModelDescriptor,
    StreamDelta,
    ToolCallFragment,
    ToolCallFragmentFunction,
    Usage,
)
```

Then append to the `OllamaEngine` class:

```python
    # ----- generate_chat --------------------------------------------------

    async def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        payload = self._build_chat_payload(body)
        reasoning_on = body.options.reasoning

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(None, connect=3.0),
            ) as resp:
                if resp.status_code == 404:
                    raise ModelNotFound(body.model_slug)
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    raise EngineBadResponse(f"ollama {resp.status_code}: {text}")

                parser = _ThinkTagSplitter(reasoning_on=reasoning_on)

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Ollama never interleaves non-JSON, treat as corruption.
                        raise EngineBadResponse("non-JSON line from /api/chat")

                    if chunk.get("done"):
                        terminal = _build_terminal(chunk)
                        yield terminal
                        return

                    message = chunk.get("message") or {}

                    for frag in _tool_call_fragments(message.get("tool_calls")):
                        yield StreamDelta(tool_calls=[frag])

                    content = message.get("content") or ""
                    if content:
                        for delta in parser.feed(content):
                            yield delta
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            # Timeouts, read errors, etc. — treat as unavailable so the
            # client retries.
            raise EngineUnavailable(str(e)) from e

    # ----- payload shaping -------------------------------------------------

    def _build_chat_payload(self, body: GenerateChatBody) -> dict[str, Any]:
        messages = [_message_to_ollama(m) for m in body.messages]
        options: dict[str, Any] = {}
        p = body.parameters
        if p.temperature is not None:
            options["temperature"] = p.temperature
        if p.top_p is not None:
            options["top_p"] = p.top_p
        if p.max_tokens is not None:
            options["num_predict"] = p.max_tokens
        if p.stop is not None:
            options["stop"] = p.stop

        payload: dict[str, Any] = {
            "model": body.model_slug,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        if body.tools is not None:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
        return payload
```

Append at module scope (below the class):

```python
# ---------------------------------------------------------------------------
# Chat translation helpers
# ---------------------------------------------------------------------------

def _message_to_ollama(m: Any) -> dict[str, Any]:
    """Map our Message to the Ollama /api/chat schema.

    Ollama expects string `content` plus an `images: [b64, ...]` array for
    multimodal user turns.
    """
    role = m.role
    content = m.content

    out: dict[str, Any] = {"role": role}
    images: list[str] = []

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if part.type == "text":
                text_parts.append(part.text)
            elif part.type == "image":
                images.append(part.data_b64)
        out["content"] = "".join(text_parts)
        if images:
            out["images"] = images
    else:
        out["content"] = content or ""

    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


_OLLAMA_FINISH_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "load": "stop",
    "unload": "stop",
}


def _build_terminal(chunk: dict[str, Any]) -> StreamTerminal:
    reason = _OLLAMA_FINISH_MAP.get(chunk.get("done_reason", "stop"), "stop")
    prompt = chunk.get("prompt_eval_count")
    complete = chunk.get("eval_count")
    usage: Usage | None = None
    if isinstance(prompt, int) and isinstance(complete, int):
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=complete,
            total_tokens=prompt + complete,
        )
    return StreamTerminal(finish_reason=reason, usage=usage)


def _tool_call_fragments(raw: Any) -> list[ToolCallFragment]:
    if not isinstance(raw, list):
        return []
    out: list[ToolCallFragment] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        fn = (tc.get("function") or {})
        out.append(
            ToolCallFragment(
                index=i,
                id=tc.get("id"),
                type=tc.get("type"),
                function=ToolCallFragmentFunction(
                    name=fn.get("name"),
                    arguments=fn.get("arguments"),
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# <think>…</think> splitter
# ---------------------------------------------------------------------------

class _ThinkTagSplitter:
    """Incrementally split content on <think>...</think> tags.

    Produces StreamDelta objects — either `content=...` (outside tags) or
    `reasoning=...` (inside). When `reasoning_on` is False, the inner text
    is dropped (SPEC §8.1 options.reasoning=False mandates NOT splitting;
    we choose the suppression branch).

    The splitter is resilient to a tag being chopped across chunk boundaries.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, reasoning_on: bool) -> None:
        self._reasoning_on = reasoning_on
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> list[StreamDelta]:
        self._buf += chunk
        out: list[StreamDelta] = []

        while True:
            if not self._inside:
                idx = self._buf.find(self.OPEN)
                if idx == -1:
                    # Flush everything except a trailing partial "<think"
                    flush, hold = _split_for_partial(self._buf, self.OPEN)
                    if flush:
                        out.append(StreamDelta(content=flush))
                    self._buf = hold
                    return out
                # Flush up to the opening tag as content.
                if idx > 0:
                    out.append(StreamDelta(content=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.OPEN):]
                self._inside = True
            else:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    flush, hold = _split_for_partial(self._buf, self.CLOSE)
                    if flush and self._reasoning_on:
                        out.append(StreamDelta(reasoning=flush))
                    # When reasoning_on=False we drop `flush` silently.
                    self._buf = hold
                    return out
                if idx > 0 and self._reasoning_on:
                    out.append(StreamDelta(reasoning=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.CLOSE):]
                self._inside = False


def _split_for_partial(buf: str, needle: str) -> tuple[str, str]:
    """Hold back any suffix of `buf` that could be the start of `needle`."""
    max_hold = len(needle) - 1
    for hold in range(max_hold, 0, -1):
        if needle.startswith(buf[-hold:]):
            return buf[:-hold], buf[-hold:]
    return buf, ""
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_ollama_generate_chat.py -v`
Expected: all 8 passed.

- [ ] **Step 5: Run the full test suite**

Run: `cd backend && uv run pytest -v`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add backend/sidecar/ollama.py backend/tests/test_ollama_generate_chat.py
git commit -m "Add Ollama streaming chat with reasoning and tool-call translation"
```

---

## Task 7: Healthcheck HTTP server

Tiny aiohttp app bound to 127.0.0.1 with a shared status object.

**Files:**
- Create: `backend/sidecar/healthcheck.py`
- Test: `backend/tests/test_healthcheck.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_healthcheck.py`:

```python
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
        # Resolve the actual bound address.
        sockets = [s for s in srv._site._server.sockets]  # noqa: SLF001
        hosts = {s.getsockname()[0] for s in sockets}
        assert hosts <= {"127.0.0.1", "::1"}
    finally:
        await srv.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_healthcheck.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `backend/sidecar/healthcheck.py`**

```python
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
        # Pull the actual port when 0 was requested.
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
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_healthcheck.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/healthcheck.py backend/tests/test_healthcheck.py
git commit -m "Add loopback healthcheck HTTP server"
```

---

## Task 8: WebSocket connection — handshake, heartbeat, reconnect

Owns the outbound WSS lifecycle. Delegates inbound frames to a dispatcher via a single `on_frame` callback and lets dispatcher push outbound frames through a `send` queue.

**Files:**
- Create: `backend/sidecar/connection.py`
- Test: `backend/tests/test_connection.py` (integration-style, runs a local WS server)

- [ ] **Step 1: Write the failing test**

`backend/tests/test_connection.py`:

```python
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
    # Allow ws:// for tests by bypassing validator construction.
    s = Settings.model_construct(
        chatsune_backend_url=url.replace("/ws/sidecar", ""),
        chatsune_host_key="cshost_test_0001",
        ollama_url="http://localhost:11434",
        sidecar_health_port=0,
        sidecar_log_level="debug",
        sidecar_max_concurrent_requests=1,
    )
    return s


async def test_happy_handshake_and_pong(monkeypatch):
    got_auth = asyncio.Event()

    async def handler(ws):
        # websockets >=14: handshake headers via ws.request.headers
        if ws.request.headers.get("Authorization") != "Bearer cshost_test_0001":
            await ws.close()
            return
        got_auth.set()
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
        # reply to a ping
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ping"
        await ws.send(json.dumps({"type": "pong"}))
        # Terminate the session cleanly.
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
        # Force aggressive ping for the test.
        cm._ping_interval = 0.05
        cm._pong_timeout = 0.5

        # Run one session.
        reason = await asyncio.wait_for(cm._run_once(), timeout=5.0)
    assert reason == StopReason.PEER_CLOSED


async def test_auth_revoked_stops_cleanly(monkeypatch):
    async def handler(ws):
        await ws.recv()  # handshake
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
    # Deterministic: inject fixed jitter of 1.0 (no perturbation).
    seq = [_backoff_seconds(n, jitter=lambda: 1.0) for n in range(1, 10)]
    assert seq == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_connection.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `backend/sidecar/connection.py`**

```python
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


# Reasons where we must NOT reconnect (SPEC §12).
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
            self._status("reconnecting" if attempt > 0 else "reconnecting")
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
                # A clean close after a successful handshake resets backoff.
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
        headers = {"Authorization": f"Bearer {self._settings.chatsune_host_key}"}
        try:
            async with connect(
                self._url,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                ping_interval=None,  # we do app-level ping
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
        # 1. Send handshake.
        await ws.send(json.dumps(self._handshake))

        # 2. Receive handshake_ack.
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            return StopReason.CONNECT_FAILED

        try:
            ack = parse_frame(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            log.error("ws.bad_handshake_ack", raw=raw[:200])
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

        # 3. Spawn reader / writer / pinger.
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
                # Stop was requested — clean shutdown.
                if self._stop.is_set():
                    stop_reason = StopReason.SHUTDOWN
                    break

                # Heartbeat timeout: > 60 s since last pong AND we've sent
                # at least one ping.
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
                except (json.JSONDecodeError, Exception) as e:
                    log.error("ws.bad_frame", error=str(e))
                    continue
                if frame is None:
                    continue  # unknown type — ignore (§4)

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
            # Unblock writer so it can exit.
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

    # ----- status bookkeeping ---------------------------------------------

    def _status(self, state: str) -> None:
        if self._on_status is not None:
            self._on_status(state)


# Sentinels & helpers

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
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_connection.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

Run: `cd backend && uv run pytest -v`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add backend/sidecar/connection.py backend/tests/test_connection.py
git commit -m "Add WebSocket connection manager with handshake, heartbeat and backoff"
```

---

## Task 9: Dispatcher — req routing, in-flight tracking, cancellation

Takes inbound frames from `ConnectionManager`, routes `req` to engine operations, tracks per-`id` tasks, converts engine output into outbound frames, and handles `cancel`.

**Files:**
- Create: `backend/sidecar/dispatcher.py`
- Test: `backend/tests/test_dispatcher.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_dispatcher.py`:

```python
import asyncio
from typing import AsyncIterator

import pytest

from sidecar.dispatcher import Dispatcher
from sidecar.engine import (
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
    StreamTerminal,
)
from sidecar.frames import (
    CancelFrame,
    GenerateChatBody,
    Message,
    ModelDescriptor,
    ReqFrame,
    StreamDelta,
    Usage,
)


class FakeEngine:
    engine_type = "ollama"

    def __init__(self, *, models=None, chat_script=None, raise_on_chat=None):
        self.models = models or []
        self.chat_script = chat_script or []
        self.raise_on_chat = raise_on_chat
        self.chat_entered = asyncio.Event()
        self.chat_cancelled = asyncio.Event()

    async def probe_version(self) -> str:
        return "test"

    async def list_models(self):
        return list(self.models)

    async def generate_chat(self, body) -> AsyncIterator[EngineStreamItem]:
        if self.raise_on_chat:
            raise self.raise_on_chat
        self.chat_entered.set()
        try:
            for item in self.chat_script:
                await asyncio.sleep(0)  # cooperative
                yield item
        except asyncio.CancelledError:
            self.chat_cancelled.set()
            raise

    async def aclose(self) -> None:
        pass


def _sample_model() -> ModelDescriptor:
    return ModelDescriptor(
        slug="m", display_name="M", parameter_count=None,
        context_length=4096, quantisation=None, capabilities=["text"],
        engine_family="ollama", engine_model_id="m",
    )


async def test_list_models_happy_path():
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(models=[_sample_model()]), send=sent.append)
    await d.handle(ReqFrame(id="r1", op="list_models"))
    await d.wait_idle()
    assert sent == [
        {
            "type": "res",
            "id": "r1",
            "ok": True,
            "body": {
                "models": [{
                    "slug": "m", "display_name": "M", "parameter_count": None,
                    "context_length": 4096, "quantisation": None,
                    "capabilities": ["text"], "engine_family": "ollama",
                    "engine_model_id": "m", "engine_metadata": {},
                }],
            },
        }
    ]


async def test_list_models_engine_unavailable_emits_err():
    class Boom(FakeEngine):
        async def list_models(self):
            raise EngineUnavailable("refused")

    sent: list[dict] = []
    d = Dispatcher(engine=Boom(), send=sent.append)
    await d.handle(ReqFrame(id="r1", op="list_models"))
    await d.wait_idle()
    assert len(sent) == 1
    f = sent[0]
    assert f["type"] == "err"
    assert f["code"] == "engine_unavailable"
    assert f["recoverable"] is True


async def test_generate_chat_streams_then_stream_end():
    script = [
        StreamDelta(content="Hi"),
        StreamDelta(content=" there"),
        StreamTerminal(finish_reason="stop", usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
    ]
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(chat_script=script), send=sent.append)
    body = GenerateChatBody(model_slug="m", messages=[Message(role="user", content="hi")])
    await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
    await d.wait_idle()
    assert [f["type"] for f in sent] == ["stream", "stream", "stream_end"]
    assert sent[-1]["finish_reason"] == "stop"
    assert sent[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


async def test_cancel_terminates_with_cancelled():
    # A never-finishing script so cancel is the only exit.
    async def forever():
        while True:
            await asyncio.sleep(10)

    engine = FakeEngine(chat_script=[])

    # Override generate_chat to hang until cancelled.
    async def gen_chat(self, body):
        self.chat_entered.set()
        try:
            while True:
                await asyncio.sleep(10)
                yield StreamDelta(content="x")
        except asyncio.CancelledError:
            self.chat_cancelled.set()
            raise

    FakeEngine.generate_chat = gen_chat  # monkey for this test

    sent: list[dict] = []
    d = Dispatcher(engine=engine, send=sent.append)
    body = GenerateChatBody(model_slug="m", messages=[Message(role="user", content="hi")])
    await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
    await engine.chat_entered.wait()
    await d.handle(CancelFrame(id="r1"))
    await d.wait_idle()

    assert engine.chat_cancelled.is_set()
    # Last frame is stream_end with finish_reason=cancelled
    assert sent[-1]["type"] == "stream_end"
    assert sent[-1]["finish_reason"] == "cancelled"


async def test_cancel_unknown_id_is_silent():
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(), send=sent.append)
    await d.handle(CancelFrame(id="does-not-exist"))
    await d.wait_idle()
    assert sent == []


async def test_model_not_found_emits_err_then_stream_end():
    engine = FakeEngine(raise_on_chat=ModelNotFound("nope"))
    sent: list[dict] = []
    d = Dispatcher(engine=engine, send=sent.append)
    body = GenerateChatBody(model_slug="nope", messages=[Message(role="user", content="x")])
    await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
    await d.wait_idle()
    # For streaming ops, `err` MUST be followed by `stream_end`.
    types = [f["type"] for f in sent]
    assert types == ["err", "stream_end"]
    assert sent[0]["code"] == "model_not_found"
    assert sent[0]["recoverable"] is False
    assert sent[1]["finish_reason"] == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_dispatcher.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `backend/sidecar/dispatcher.py`**

```python
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

    # ----- public API ------------------------------------------------------

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

    # ----- request starters -----------------------------------------------

    async def _start_req(self, req: ReqFrame) -> None:
        if req.id in self._inflight:
            # Duplicate id — protocol violation, but be defensive.
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
            return  # SPEC §9: silent on unknown/terminated id
        task.cancel()

    # ----- list_models ----------------------------------------------------

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

    # ----- generate_chat --------------------------------------------------

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
        except Exception as e:  # noqa: BLE001
            log.exception("dispatcher.unexpected", id=rid)
            await self._emit(ErrFrame(
                id=rid, code="internal",
                message="Internal sidecar error.",
                detail=str(e), recoverable=False,
            ))
            await self._emit_stream_end(rid, "error")
            return

        if not saw_terminal:
            # Engine iterator exhausted without emitting StreamTerminal —
            # shouldn't happen, but normalise.
            await self._emit_stream_end(rid, "stop")

    # ----- emitters --------------------------------------------------------

    async def _emit(self, frame: Any) -> None:
        payload = frame.model_dump(mode="json", exclude_none=False)
        result = self._send(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _emit_stream_end(self, rid: str, reason: str) -> None:
        await self._emit(StreamEndFrame(id=rid, finish_reason=reason))  # type: ignore[arg-type]
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_dispatcher.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run full suite**

Run: `cd backend && uv run pytest -v`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add backend/sidecar/dispatcher.py backend/tests/test_dispatcher.py
git commit -m "Add operation dispatcher with in-flight tracking and cancellation"
```

---

## Task 10: Entry point — run() coordinator and signal handling

Wires engine, dispatcher, connection, healthcheck together. Handles SIGTERM/SIGINT per SPEC §17.

**Files:**
- Create: `backend/sidecar/main.py`
- Create: `backend/sidecar/__main__.py`
- Test: `backend/tests/test_main_payload.py` (unit-test the handshake payload builder only — full wiring covered by integration test in Task 12)

- [ ] **Step 1: Write the failing test**

`backend/tests/test_main_payload.py`:

```python
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
    assert payload["sidecar_version"]  # some version string
    assert payload["engine"] == {
        "type": "ollama",
        "version": "0.5.7",
        "endpoint_hint": "http://localhost:11434",
    }
    assert payload["max_concurrent_requests"] == 2
    assert "chat_streaming" in payload["capabilities"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_main_payload.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `backend/sidecar/main.py`**

```python
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


SIDECAR_VERSION = "1.0.0"
CSP_VERSION = "1.0"


def build_handshake_payload(
    settings: Settings, *, engine_type: str, engine_version: str
) -> dict[str, Any]:
    return {
        "type": "handshake",
        "csp_version": CSP_VERSION,
        "sidecar_version": SIDECAR_VERSION,
        "engine": {
            "type": engine_type,
            "version": engine_version,
            "endpoint_hint": settings.ollama_url,
        },
        "max_concurrent_requests": settings.sidecar_max_concurrent_requests,
        # v1 must always advertise chat_streaming. Other capabilities are
        # per-model at list_models time; the handshake-level set stays minimal.
        "capabilities": ["chat_streaming", "tool_calls", "vision", "reasoning"],
    }


async def _run(settings: Settings) -> int:
    configure_logging(settings.sidecar_log_level)
    log = get_logger("main")
    log.info(
        "sidecar.starting",
        backend_url=settings.chatsune_backend_url,
        ollama_url=settings.ollama_url,
        health_port=settings.sidecar_health_port,
        host_key_tail=settings.host_key_tail(),
    )

    engine: Engine = OllamaEngine(settings.ollama_url)
    health = HealthState()
    health_server = HealthcheckServer(health, port=settings.sidecar_health_port)
    await health_server.start()

    # Probe engine version (best-effort per SPEC §17 step 3).
    version = await engine.probe_version()
    reachable = version != "unknown"
    health.mark_engine(reachable, engine.engine_type)
    log.info("sidecar.engine_probe", reachable=reachable, version=version)

    handshake_payload = build_handshake_payload(
        settings, engine_type=engine.engine_type, engine_version=version
    )

    # Late-bound send: populated once ConnectionManager exists.
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
    _cm = cm  # close over

    loop = asyncio.get_running_loop()

    def _sig() -> None:
        log.info("sidecar.signal_received")
        cm.request_stop()

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(s, _sig)
        except NotImplementedError:
            # Windows / restricted env — ignore.
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
    except Exception as e:  # pydantic ValidationError
        print(f"sidecar: configuration error: {e}", file=sys.stderr)
        return 2
    return asyncio.run(_run(settings))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write `backend/sidecar/__main__.py`**

```python
from .main import main

raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_main_payload.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run full suite**

Run: `cd backend && uv run pytest -v`
Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add backend/sidecar/main.py backend/sidecar/__main__.py backend/tests/test_main_payload.py
git commit -m "Add main entry point with signal handling and component wiring"
```

---

## Task 11: End-to-end integration test

Drive a full protocol cycle against a fake Chatsune backend and a mocked Ollama: handshake → list_models → generate_chat → cancel → disconnect.

**Files:**
- Test: `backend/tests/test_integration.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end: fake Chatsune backend + respx-mocked Ollama + real sidecar.

Drives:
  connect → handshake → list_models → generate_chat → cancel → ack → shutdown

This test is the acceptance gate for §18.5 of the SPEC.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
import respx
import websockets

from sidecar.config import Settings
from sidecar.connection import ConnectionManager
from sidecar.dispatcher import Dispatcher
from sidecar.frames import parse_frame
from sidecar.main import build_handshake_payload
from sidecar.ollama import OllamaEngine


@asynccontextmanager
async def fake_backend(handler):
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/sidecar"


def _tags():
    return {
        "models": [{
            "name": "llama3.2:8b",
            "size": 1,
            "digest": "d",
            "details": {
                "family": "llama",
                "parameter_size": "8B",
                "quantization_level": "Q4_K_M",
            },
        }]
    }


def _show():
    return {
        "capabilities": [],
        "model_info": {"llama.context_length": 131072},
    }


def _ndjson(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


@respx.mock
async def test_full_cycle():
    # --- Ollama mocks ------------------------------------------------------
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.5.7"})
    )
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=_tags())
    )
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json=_show())
    )
    # 1000-chunk stream so cancel beats natural end.
    chunks = [
        {"message": {"role": "assistant", "content": f"t{i}"}, "done": False}
        for i in range(1000)
    ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ndjson(chunks))
    )

    # --- Fake Chatsune backend --------------------------------------------
    observed: list[dict] = []
    list_models_done = asyncio.Event()
    streamed = asyncio.Event()
    finished = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "handshake"
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "homelab_id": "H",
            "display_name": "test",
            "accepted": True,
            "notices": [],
        }))

        # list_models
        await ws.send(json.dumps({"type": "req", "id": "r1", "op": "list_models"}))
        while not list_models_done.is_set():
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "res" and msg["id"] == "r1":
                list_models_done.set()
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        # generate_chat + cancel
        await ws.send(json.dumps({
            "type": "req", "id": "r2", "op": "generate_chat",
            "body": {
                "model_slug": "llama3.2:8b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        }))

        stream_count = 0
        while True:
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "stream" and msg["id"] == "r2":
                stream_count += 1
                if stream_count == 3:
                    streamed.set()
                    await ws.send(json.dumps({"type": "cancel", "id": "r2"}))
            elif msg["type"] == "stream_end" and msg["id"] == "r2":
                assert msg["finish_reason"] == "cancelled"
                finished.set()
                break
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        await ws.close()

    async with fake_backend(handler) as url:
        settings = Settings.model_construct(
            chatsune_backend_url=url.replace("/ws/sidecar", ""),
            chatsune_host_key="cshost_integration_001",
            ollama_url="http://localhost:11434",
            sidecar_health_port=0,
            sidecar_log_level="warn",
            sidecar_max_concurrent_requests=1,
        )

        engine = OllamaEngine(settings.ollama_url)
        try:
            version = await engine.probe_version()
            handshake = build_handshake_payload(
                settings, engine_type=engine.engine_type, engine_version=version,
            )
            # Late-bound send.
            cm_holder: list[ConnectionManager] = []

            async def send(frame):
                await cm_holder[0].send(frame)

            dispatcher = Dispatcher(engine=engine, send=send)

            async def on_frame(frame):
                await dispatcher.handle(frame)

            cm = ConnectionManager(
                settings=settings,
                handshake_payload=handshake,
                on_frame=on_frame,
                url_override=url,
            )
            cm_holder.append(cm)
            # Short ping so the test doesn't wait 30 s for anything.
            cm._ping_interval = 1.0
            cm._pong_timeout = 5.0

            run_task = asyncio.create_task(cm.run_forever())

            # Wait for the scripted side to observe cancellation.
            await asyncio.wait_for(finished.wait(), timeout=15.0)
            cm.request_stop()
            await asyncio.wait_for(run_task, timeout=5.0)
            await dispatcher.cancel_all()
        finally:
            await engine.aclose()

    # Assertions on observed frames.
    res = next(f for f in observed if f["type"] == "res")
    assert res["id"] == "r1"
    assert res["body"]["models"][0]["slug"] == "llama3.2:8b"
    stream_end = next(
        f for f in observed if f["type"] == "stream_end" and f["id"] == "r2"
    )
    assert stream_end["finish_reason"] == "cancelled"
```

- [ ] **Step 2: Run it**

Run: `cd backend && uv run pytest tests/test_integration.py -v`
Expected: 1 passed (will take 1-3 seconds).

- [ ] **Step 3: Run full suite one more time**

Run: `cd backend && uv run pytest -v`
Expected: all pass, total around 35-40 tests.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "Add end-to-end integration test for full protocol cycle"
```

---

## Task 12: Dockerfile and compose.yml

**Files:**
- Create: `backend/Dockerfile`
- Create: `compose.yml`

- [ ] **Step 1: Write `backend/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY backend/pyproject.toml ./backend/
RUN uv venv /app/.venv && cd backend && UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --no-install-project

COPY backend/ ./backend/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Unprivileged user per SPEC §17.
RUN useradd --system --uid 10001 --shell /usr/sbin/nologin sidecar
USER 10001:10001

# Healthcheck port (127.0.0.1 only — not exposed on the host interface,
# so no EXPOSE.)
WORKDIR /app/backend
CMD ["python", "-m", "sidecar"]
```

- [ ] **Step 2: Write `compose.yml`**

```yaml
services:
  chatsune-sidecar:
    image: ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend:latest
    restart: unless-stopped
    environment:
      CHATSUNE_BACKEND_URL: ${CHATSUNE_BACKEND_URL:?required}
      CHATSUNE_HOST_KEY: ${CHATSUNE_HOST_KEY:?required}
      OLLAMA_URL: ${OLLAMA_URL:-http://host.docker.internal:11434}
      SIDECAR_HEALTH_PORT: "8080"
      SIDECAR_LOG_LEVEL: ${SIDECAR_LOG_LEVEL:-info}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    read_only: true
    tmpfs:
      - /tmp
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz').read()"]
      interval: 30s
      timeout: 5s
      retries: 3
```

- [ ] **Step 3: Verify the image builds**

Run (repo root): `docker build -f backend/Dockerfile -t chatsune-ollama-sidecar:local .`
Expected: image builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile compose.yml
git commit -m "Add Dockerfile and reference compose stack"
```

---

## Task 13: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the stub README with complete docs**

```markdown
# chatsune-ollama-sidecar

CSP/1 sidecar for Ollama — bridges a local Ollama engine to a remote
Chatsune backend over an outbound WebSocket Secure connection.

The wire protocol, supported operations and behavioural rules are
specified in [`SPEC.md`](SPEC.md).

---

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A running Ollama daemon (local or LAN-reachable)
- A Chatsune backend with a Host-Key issued for this sidecar

## Quick start (local dev)

```bash
cp .env.example .env   # then edit CHATSUNE_HOST_KEY (at minimum)
./start-backend.sh
```

`start-backend.sh` runs `uv sync` and launches `python -m sidecar`
with autoreload disabled (this is a daemon, not an HTTP app).

## Quick start (Docker)

```bash
export CHATSUNE_BACKEND_URL=wss://chat.example.com
export CHATSUNE_HOST_KEY=cshost_yourkeyhere
docker compose up -d
docker compose logs -f
```

The image is published as
`ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend:latest` by
the `Docker Build & Push` workflow.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHATSUNE_BACKEND_URL` | *required* | Base `wss://` URL. `/ws/sidecar` is appended automatically. |
| `CHATSUNE_HOST_KEY` | *required* | Per-sidecar secret; must start with `cshost_`. |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama HTTP endpoint. |
| `SIDECAR_HEALTH_PORT` | `8080` | Loopback healthcheck port. |
| `SIDECAR_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error`. |
| `SIDECAR_MAX_CONCURRENT_REQUESTS` | `1` | Handshake-advertised concurrency. |

See [`SPEC.md §14`](SPEC.md) for authoritative semantics.

## Healthcheck

```bash
curl -s http://127.0.0.1:8080/healthz | jq
```

- `200 OK` while the backend connection is `connected` or `reconnecting`.
- `503 Service Unavailable` once the sidecar has given up.

## Development

```bash
cd backend
uv sync
uv run pytest -v
```

- `sidecar/frames.py` — every CSP/1 frame as a Pydantic model.
- `sidecar/ollama.py` — Ollama HTTP adapter (list, stream chat, cancel).
- `sidecar/connection.py` — WS client, handshake, ping/pong, reconnect.
- `sidecar/dispatcher.py` — `req`/`cancel` routing, in-flight task tracking.
- `sidecar/main.py` — entry point and signal handling.

Integration test:

```bash
uv run pytest tests/test_integration.py -v
```

## Licence

GPL-3.0-or-later — see [`LICENSE`](LICENSE).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document install, config and dev workflow"
```

---

## Final verification

- [ ] **Step 1: Full test suite**

Run: `cd backend && uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Image build sanity**

Run: `docker build -f backend/Dockerfile -t chatsune-ollama-sidecar:final .`
Expected: builds cleanly.

- [ ] **Step 3: Smoke-run (manual, only if Ollama is available)**

```bash
export CHATSUNE_BACKEND_URL=wss://chat.example.com
export CHATSUNE_HOST_KEY=cshost_smoke_test
./start-backend.sh
# In another shell:
curl http://127.0.0.1:8080/healthz
# Expected: {"ok": false, "backend_connection": "reconnecting", ...}
# (No real backend, so the connection never succeeds — but /healthz works.)
```

- [ ] **Step 4: Summary commit if anything stray**

`git status` should be clean.
