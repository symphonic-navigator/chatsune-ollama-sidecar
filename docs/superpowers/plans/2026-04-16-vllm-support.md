# vLLM Engine Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Ollama-only sidecar so it can also drive a local vLLM server, selected at deploy time via a single environment variable, with a user-maintained YAML layer for per-model metadata gaps.

**Architecture:** Keep the existing `Engine` protocol untouched. Add a second adapter `VllmEngine` in a sibling module. A slim factory in `main.py` picks between `OllamaEngine` and `VllmEngine` based on `SIDECAR_ENGINE`. Extract the shared `<think>` splitter from `ollama.py` to an internal utility so both adapters share it. The `<think>` handling — present on Ollama, also needed for vLLM without `--reasoning-parser` — is the only non-trivial cross-engine code.

**Tech Stack:** Python 3.12, asyncio, httpx (streaming HTTP), pydantic / pydantic-settings, respx (HTTP mocking in tests), pytest. New dependency: `pyyaml` for the YAML config layer.

**Spec reference:** `docs/superpowers/specs/2026-04-16-vllm-support-design.md`

**Code conventions (from user CLAUDE.md):**
- Code, comments, docstrings, commit messages → British English (`colour`, `initialise`, `behaviour`).
- Commit messages: imperative, free-form, no Conventional Commits prefix.

---

## File Structure

**New files:**
- `backend/sidecar/_reasoning.py` — shared `_ThinkTagSplitter`, `_split_for_partial` helper.
- `backend/sidecar/vllm_models_config.py` — `VllmModelMetadata`, `load_vllm_models_config`.
- `backend/sidecar/vllm.py` — `VllmEngine`.
- `backend/tests/test_vllm_models_config.py`
- `backend/tests/test_vllm_list_models.py`
- `backend/tests/test_vllm_generate_chat.py`
- `backend/tests/fixtures/vllm_models.json`

**Modified files:**
- `backend/sidecar/ollama.py` — import splitter from `_reasoning.py` instead of defining it locally.
- `backend/sidecar/config.py` — add four fields: `sidecar_engine`, `vllm_url`, `vllm_models_config_path`, `vllm_models_overlay_path`.
- `backend/sidecar/main.py` — engine factory; dynamic `endpoint_hint`.
- `backend/pyproject.toml` — add `pyyaml` to runtime dependencies.
- `backend/tests/test_integration.py` — parametrise across both engines.
- `backend/tests/test_config.py` — add coverage for the four new settings fields.
- `.env.example`, `README.md`, `compose.yml` — document the new variables and vLLM operator cookbook.

---

## Task 1: Extract `_ThinkTagSplitter` to `_reasoning.py`

Pure refactor — no behavioural change. Existing Ollama tests in `test_ollama_generate_chat.py` are the regression safety net; they must stay green before and after.

**Files:**
- Create: `backend/sidecar/_reasoning.py`
- Modify: `backend/sidecar/ollama.py`

- [ ] **Step 1: Run Ollama tests green as baseline**

```bash
cd backend && uv run pytest tests/test_ollama_generate_chat.py -v
```

Expected: all tests PASS. If any fail before you touch anything, stop and report — the baseline is broken.

- [ ] **Step 2: Create `backend/sidecar/_reasoning.py`**

```python
"""Shared reasoning-channel utilities.

The `<think>…</think>` splitter lives here because both the Ollama
adapter (legacy inline format) and the vLLM adapter (when the server
runs without `--reasoning-parser`) need the same chunk-boundary-safe
parser.
"""
from __future__ import annotations

from .frames import StreamDelta


class ThinkTagSplitter:
    """Incrementally split content on `<think>…</think>` tags.

    Produces StreamDelta objects — either `content=...` (outside tags) or
    `reasoning=...` (inside). When `reasoning_on` is False, the inner text
    is dropped.

    Resilient to a tag being chopped across chunk boundaries.
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
                    flush, hold = split_for_partial(self._buf, self.OPEN)
                    if flush:
                        out.append(StreamDelta(content=flush))
                    self._buf = hold
                    return out
                if idx > 0:
                    out.append(StreamDelta(content=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.OPEN):]
                self._inside = True
            else:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    flush, hold = split_for_partial(self._buf, self.CLOSE)
                    if flush and self._reasoning_on:
                        out.append(StreamDelta(reasoning=flush))
                    self._buf = hold
                    return out
                if idx > 0 and self._reasoning_on:
                    out.append(StreamDelta(reasoning=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.CLOSE):]
                self._inside = False


def split_for_partial(buf: str, needle: str) -> tuple[str, str]:
    """Hold back any suffix of `buf` that could be the start of `needle`."""
    max_hold = len(needle) - 1
    for hold in range(max_hold, 0, -1):
        if needle.startswith(buf[-hold:]):
            return buf[:-hold], buf[-hold:]
    return buf, ""
```

Note: the class is renamed from `_ThinkTagSplitter` (module-private) to `ThinkTagSplitter` (cross-module) because it's no longer private to `ollama.py`. The helper becomes `split_for_partial` for the same reason. The leading-underscore **module name** (`_reasoning.py`) marks the module itself as internal.

- [ ] **Step 3: Update `backend/sidecar/ollama.py` to import from the new module**

Replace the import block near the top of `ollama.py` so it reads:

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
from ._reasoning import ThinkTagSplitter
```

Delete these sections from the bottom of `ollama.py` (they now live in `_reasoning.py`):
- The `_ThinkTagSplitter` class (currently around `ollama.py:386-432`).
- The `_split_for_partial` helper (currently around `ollama.py:435-441`).
- Delete the "# <think>...</think> splitter" divider comment block that introduces them.

Update the one call site inside `generate_chat`:

```python
parser = ThinkTagSplitter(reasoning_on=reasoning_on)
```

(replace `_ThinkTagSplitter` with `ThinkTagSplitter`).

- [ ] **Step 4: Run Ollama tests to verify the refactor is transparent**

```bash
cd backend && uv run pytest tests/test_ollama_generate_chat.py -v
```

Expected: all tests PASS, same set as in Step 1.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/_reasoning.py backend/sidecar/ollama.py
git commit -m "Extract ThinkTagSplitter into a shared reasoning module"
```

---

## Task 2: Add pyyaml dependency

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Edit `backend/pyproject.toml`**

Find the `dependencies` array:

```toml
dependencies = [
    "websockets>=14.0",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "aiohttp>=3.11",
    "structlog>=24.1",
]
```

Add `pyyaml` to it so it reads:

```toml
dependencies = [
    "websockets>=14.0",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "aiohttp>=3.11",
    "structlog>=24.1",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
cd backend && uv sync
```

Expected: uv resolves and installs pyyaml alongside existing packages.

- [ ] **Step 3: Verify import works**

```bash
cd backend && uv run python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version number (e.g. `6.0.2`).

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml
git commit -m "Add pyyaml for YAML-based vLLM model metadata config"
```

Note: `uv.lock` is in this project's `.gitignore` (intentional); do not add it.

---

## Task 3: YAML model-metadata loader (`vllm_models_config.py`)

TDD: tests first, implementation after.

**Files:**
- Create: `backend/tests/test_vllm_models_config.py`
- Create: `backend/sidecar/vllm_models_config.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_vllm_models_config.py`:

```python
"""Unit tests for the vLLM model-metadata YAML loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from sidecar.vllm_models_config import (
    VllmModelMetadata,
    load_vllm_models_config,
)


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_both_paths_none_returns_empty_dict():
    assert load_vllm_models_config(None, None) == {}


def test_base_only(tmp_path: Path):
    base = _write(
        tmp_path / "base.yaml",
        """
models:
  gemma-4-26b:
    display_name: Gemma 4 26B
    capabilities: [text, vision]
    parameter_count: 26000000000
    quantisation: AWQ-4bit
""",
    )
    result = load_vllm_models_config(base, None)
    assert set(result.keys()) == {"gemma-4-26b"}
    entry = result["gemma-4-26b"]
    assert entry.display_name == "Gemma 4 26B"
    assert entry.capabilities == ["text", "vision"]
    assert entry.parameter_count == 26_000_000_000
    assert entry.quantisation == "AWQ-4bit"


def test_overlay_only(tmp_path: Path):
    overlay = _write(
        tmp_path / "overlay.yaml",
        """
models:
  some-model:
    display_name: Only-in-overlay
    capabilities: [text]
""",
    )
    result = load_vllm_models_config(None, overlay)
    assert result["some-model"].display_name == "Only-in-overlay"


def test_deep_merge_field_level(tmp_path: Path):
    """Overlay overrides individual fields; untouched fields survive."""
    base = _write(
        tmp_path / "base.yaml",
        """
models:
  gemma-4-26b:
    display_name: Gemma 4 26B
    capabilities: [text, vision]
    parameter_count: 26000000000
    quantisation: AWQ-4bit
""",
    )
    overlay = _write(
        tmp_path / "overlay.yaml",
        """
models:
  gemma-4-26b:
    capabilities: [text, vision, tool_calling]
""",
    )
    result = load_vllm_models_config(base, overlay)
    entry = result["gemma-4-26b"]
    assert entry.capabilities == ["text", "vision", "tool_calling"]
    # Fields not in overlay survive from base:
    assert entry.display_name == "Gemma 4 26B"
    assert entry.parameter_count == 26_000_000_000
    assert entry.quantisation == "AWQ-4bit"


def test_overlay_can_remove_a_capability(tmp_path: Path):
    """Lists are replaced wholesale — overlay can drop a capability."""
    base = _write(
        tmp_path / "base.yaml",
        """
models:
  m1:
    capabilities: [text, vision, tool_calling]
""",
    )
    overlay = _write(
        tmp_path / "overlay.yaml",
        """
models:
  m1:
    capabilities: [text]
""",
    )
    result = load_vllm_models_config(base, overlay)
    assert result["m1"].capabilities == ["text"]


def test_overlay_adds_model_not_in_base(tmp_path: Path):
    base = _write(
        tmp_path / "base.yaml",
        "models:\n  m1:\n    capabilities: [text]\n",
    )
    overlay = _write(
        tmp_path / "overlay.yaml",
        "models:\n  m2:\n    capabilities: [text, vision]\n",
    )
    result = load_vllm_models_config(base, overlay)
    assert set(result) == {"m1", "m2"}
    assert result["m2"].capabilities == ["text", "vision"]


def test_missing_file_raises(tmp_path: Path):
    missing = str(tmp_path / "does-not-exist.yaml")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_vllm_models_config(missing, None)
    assert "does-not-exist.yaml" in str(exc_info.value)


def test_invalid_yaml_raises(tmp_path: Path):
    bad = _write(tmp_path / "bad.yaml", "models: [this: isn't: valid")
    with pytest.raises(Exception) as exc_info:
        load_vllm_models_config(bad, None)
    assert "bad.yaml" in str(exc_info.value)


def test_schema_violation_unknown_capability(tmp_path: Path):
    bad = _write(
        tmp_path / "bad.yaml",
        "models:\n  m1:\n    capabilities: [videooo]\n",
    )
    with pytest.raises(Exception) as exc_info:
        load_vllm_models_config(bad, None)
    assert "bad.yaml" in str(exc_info.value)


def test_schema_violation_capabilities_not_a_list(tmp_path: Path):
    bad = _write(
        tmp_path / "bad.yaml",
        "models:\n  m1:\n    capabilities: vision\n",
    )
    with pytest.raises(Exception) as exc_info:
        load_vllm_models_config(bad, None)
    assert "bad.yaml" in str(exc_info.value)


def test_empty_models_section_ok(tmp_path: Path):
    f = _write(tmp_path / "empty.yaml", "models: {}\n")
    assert load_vllm_models_config(f, None) == {}


def test_metadata_model_defaults():
    m = VllmModelMetadata()
    assert m.display_name is None
    assert m.parameter_count is None
    assert m.quantisation is None
    assert m.capabilities is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_models_config.py -v
```

Expected: FAIL with `ImportError` / `ModuleNotFoundError` on `sidecar.vllm_models_config`.

- [ ] **Step 3: Implement `backend/sidecar/vllm_models_config.py`**

```python
"""YAML model-metadata layer for the vLLM adapter.

The vLLM server's `/v1/models` endpoint reports only a handful of fields
(id, max_model_len, owned_by, root). It does not tell us whether a model
supports vision, tool calling, or reasoning — so the operator supplies
that metadata via one or two YAML files, merged at field level with
the overlay winning.

Any error during loading is raised — callers are expected to let it
surface as a startup failure (exit code 2) rather than degrade silently.
See design §5.3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .frames import ModelCapability


class VllmModelMetadata(BaseModel):
    """Per-model metadata supplied by the operator.

    Every field is optional. Missing fields fall back to engine defaults
    at descriptor-build time.
    """

    display_name: str | None = None
    parameter_count: int | None = None
    quantisation: str | None = None
    capabilities: list[ModelCapability] | None = None


class _YamlRoot(BaseModel):
    """Root schema of the YAML file."""

    models: dict[str, VllmModelMetadata] = Field(default_factory=dict)


def load_vllm_models_config(
    base_path: str | None,
    overlay_path: str | None,
) -> dict[str, VllmModelMetadata]:
    """Load and merge the optional base and overlay YAML files.

    Returns a dict keyed by served_id. Fields from the overlay override
    fields from the base at per-field granularity; lists are replaced
    wholesale (no union, no subtraction).

    Raises on:
      - a configured path whose file does not exist;
      - malformed YAML;
      - schema violations (unknown capability value, wrong type, etc.).
    """
    base = _load_one(base_path)
    overlay = _load_one(overlay_path)

    merged: dict[str, VllmModelMetadata] = {}
    for key, value in base.items():
        merged[key] = value
    for key, overlay_entry in overlay.items():
        base_entry = merged.get(key)
        if base_entry is None:
            merged[key] = overlay_entry
        else:
            merged[key] = _merge_entry(base_entry, overlay_entry)
    return merged


def _load_one(path: str | None) -> dict[str, VllmModelMetadata]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"vLLM models config not found: {path}")
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"vLLM models config {path}: invalid YAML — {e}") from e
    try:
        parsed = _YamlRoot.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"vLLM models config {path}: schema error — {e}") from e
    return parsed.models


def _merge_entry(
    base: VllmModelMetadata, overlay: VllmModelMetadata
) -> VllmModelMetadata:
    """Field-level merge: overlay value wins when non-None."""
    overlay_dump = overlay.model_dump(exclude_none=True)
    base_dump = base.model_dump()
    base_dump.update(overlay_dump)
    return VllmModelMetadata.model_validate(base_dump)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_models_config.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm_models_config.py backend/tests/test_vllm_models_config.py
git commit -m "Add YAML loader for vLLM per-model metadata with overlay merging"
```

---

## Task 4: Extend Settings with four new fields

**Files:**
- Modify: `backend/sidecar/config.py`
- Modify: `backend/tests/test_config.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_config.py`**

Append these tests to the existing file:

```python
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
```

- [ ] **Step 2: Run to verify the tests fail**

```bash
cd backend && uv run pytest tests/test_config.py -v
```

Expected: seven new tests FAIL (attribute errors / validation errors).

- [ ] **Step 3: Extend `backend/sidecar/config.py`**

Modify `backend/sidecar/config.py`. Add two literal engine types to imports at the top if not already importable:

```python
SidecarEngine = Literal["ollama", "vllm"]
```

Then add four fields to the `Settings` class alongside the existing fields (place them after `ollama_url`, before the `sidecar_*` fields to keep engine fields grouped):

```python
    sidecar_engine: SidecarEngine = Field(default="ollama")

    vllm_url: str = Field(default="http://host.docker.internal:8000")
    vllm_models_config_path: str | None = Field(default=None)
    vllm_models_overlay_path: str | None = Field(default=None)
```

The final `config.py` has this ordering inside `Settings`:

```python
    chatsune_backend_url: str = Field(...)
    chatsune_host_key: str = Field(...)

    ollama_url: str = Field(default="http://host.docker.internal:11434")

    sidecar_engine: SidecarEngine = Field(default="ollama")
    vllm_url: str = Field(default="http://host.docker.internal:8000")
    vllm_models_config_path: str | None = Field(default=None)
    vllm_models_overlay_path: str | None = Field(default=None)

    sidecar_health_port: int = Field(default=8080, ge=1, le=65535)
    sidecar_log_level: LogLevel = Field(default="info")
    sidecar_max_concurrent_requests: int = Field(default=1, ge=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_config.py -v
```

Expected: all tests (old + new) PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/config.py backend/tests/test_config.py
git commit -m "Add engine selector and vLLM settings to config"
```

---

## Task 5: Create vLLM `/v1/models` fixture

**Files:**
- Create: `backend/tests/fixtures/vllm_models.json`

- [ ] **Step 1: Write the fixture**

Create `backend/tests/fixtures/vllm_models.json` with two entries: a realistic happy-path one (the user's actual `gemma-4-26b` payload) and a second one missing `max_model_len` so the drop path is exercised.

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-4-26b",
      "object": "model",
      "created": 1776354390,
      "owned_by": "vllm",
      "root": "lcu0312/gemma-4-26B-A4B-it-AWQ-4bit",
      "parent": null,
      "max_model_len": 262144,
      "permission": []
    },
    {
      "id": "no-context-model",
      "object": "model",
      "created": 1776354000,
      "owned_by": "vllm",
      "root": "some/repo",
      "parent": null,
      "permission": []
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add backend/tests/fixtures/vllm_models.json
git commit -m "Add vLLM /v1/models fixture covering happy path and drop path"
```

---

## Task 6: `VllmEngine` scaffold — `__init__`, `aclose`, `probe_version`

TDD. This task lays the shell; `list_models` and `generate_chat` follow in the next tasks.

**Files:**
- Create: `backend/sidecar/vllm.py`
- Create: `backend/tests/test_vllm_list_models.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_vllm_list_models.py` with just probe tests for now (`list_models` tests come in Task 7):

```python
"""Unit tests for the vLLM engine adapter — discovery surface."""
from __future__ import annotations

import httpx
import pytest
import respx

from sidecar.vllm import VllmEngine


@respx.mock
async def test_probe_version_returns_reported_version():
    respx.get("http://localhost:8000/version").mock(
        return_value=httpx.Response(200, json={"version": "0.7.3"})
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    try:
        assert await engine.probe_version() == "0.7.3"
    finally:
        await engine.aclose()


@respx.mock
async def test_probe_version_returns_unknown_on_connect_error():
    respx.get("http://localhost:8000/version").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    try:
        assert await engine.probe_version() == "unknown"
    finally:
        await engine.aclose()


@respx.mock
async def test_probe_version_returns_unknown_on_missing_field():
    respx.get("http://localhost:8000/version").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    try:
        assert await engine.probe_version() == "unknown"
    finally:
        await engine.aclose()


def test_engine_type_attribute():
    engine = VllmEngine("http://localhost:8000", metadata={})
    assert engine.engine_type == "vllm"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_list_models.py -v
```

Expected: FAIL with `ModuleNotFoundError` on `sidecar.vllm`.

- [ ] **Step 3: Create `backend/sidecar/vllm.py` with the scaffold**

```python
"""vLLM engine adapter (SPEC §15.3).

vLLM exposes an OpenAI-compatible API:
  - `GET /version` for the version probe,
  - `GET /v1/models` for discovery,
  - `POST /v1/chat/completions` (with `stream: true`) for chat.

Per-model metadata that vLLM does not self-report (vision, tool calling,
reasoning capability, display name, parameter count, quantisation) comes
from the operator via the YAML layer in `vllm_models_config.py`.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from .engine import EngineStreamItem
from .frames import GenerateChatBody, ModelDescriptor
from .vllm_models_config import VllmModelMetadata


class VllmEngine:
    engine_type = "vllm"

    def __init__(
        self,
        url: str,
        *,
        metadata: dict[str, VllmModelMetadata],
        timeout: float = 10.0,
    ) -> None:
        self._base = url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )
        self._metadata = metadata
        self._warned_unknown: set[str] = set()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/version")
            r.raise_for_status()
            v = r.json().get("version")
            return v if isinstance(v, str) else "unknown"
        except (httpx.HTTPError, ValueError):
            return "unknown"

    async def list_models(self) -> list[ModelDescriptor]:
        raise NotImplementedError  # Task 7

    def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        raise NotImplementedError  # Tasks 8-12
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_list_models.py -v
```

Expected: four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_list_models.py
git commit -m "Scaffold VllmEngine with version probe"
```

---

## Task 7: `VllmEngine.list_models`

**Files:**
- Modify: `backend/sidecar/vllm.py`
- Modify: `backend/tests/test_vllm_list_models.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `backend/tests/test_vllm_list_models.py`:

```python
import json
import logging
from pathlib import Path

from sidecar.engine import EngineBadResponse, EngineUnavailable
from sidecar.vllm_models_config import VllmModelMetadata


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _models_body() -> dict:
    return json.loads((FIXTURE_DIR / "vllm_models.json").read_text())


@respx.mock
async def test_list_models_with_metadata_hit():
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(200, json=_models_body())
    )
    metadata = {
        "gemma-4-26b": VllmModelMetadata(
            display_name="Gemma 4 26B",
            parameter_count=26_000_000_000,
            quantisation="AWQ-4bit",
            capabilities=["text", "vision", "tool_calling"],
        )
    }
    engine = VllmEngine("http://localhost:8000", metadata=metadata)
    try:
        models = await engine.list_models()
    finally:
        await engine.aclose()

    # The second fixture entry has no max_model_len and MUST be dropped.
    assert [m.slug for m in models] == ["gemma-4-26b"]
    m = models[0]
    assert m.display_name == "Gemma 4 26B"
    assert m.context_length == 262144
    assert m.parameter_count == 26_000_000_000
    assert m.quantisation == "AWQ-4bit"
    assert m.capabilities == ["text", "vision", "tool_calling"]
    assert m.engine_family == "vllm"
    assert m.engine_model_id == "gemma-4-26b"
    assert m.engine_metadata.get("owned_by") == "vllm"
    assert m.engine_metadata.get("root") == "lcu0312/gemma-4-26B-A4B-it-AWQ-4bit"


@respx.mock
async def test_list_models_without_metadata_uses_conservative_defaults(caplog):
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(200, json=_models_body())
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    caplog.set_level(logging.WARNING)
    try:
        models = await engine.list_models()
    finally:
        await engine.aclose()

    assert [m.slug for m in models] == ["gemma-4-26b"]
    m = models[0]
    assert m.display_name == "gemma-4-26b"
    assert m.parameter_count is None
    assert m.quantisation is None
    assert m.capabilities == ["text"]
    # One-shot warning was emitted for the unknown id.
    assert any("gemma-4-26b" in rec.getMessage() for rec in caplog.records)


@respx.mock
async def test_list_models_warn_is_one_shot_per_id(caplog):
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(200, json=_models_body())
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    caplog.set_level(logging.WARNING)
    try:
        await engine.list_models()
        caplog.clear()
        await engine.list_models()
    finally:
        await engine.aclose()

    # Second call must NOT re-emit the warning for the same id.
    warnings_for_id = [rec for rec in caplog.records if "gemma-4-26b" in rec.getMessage()]
    assert warnings_for_id == []


@respx.mock
async def test_list_models_engine_unreachable():
    respx.get("http://localhost:8000/v1/models").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    try:
        with pytest.raises(EngineUnavailable):
            await engine.list_models()
    finally:
        await engine.aclose()


@respx.mock
async def test_list_models_http_500():
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(500, text="boom")
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    try:
        with pytest.raises(EngineBadResponse):
            await engine.list_models()
    finally:
        await engine.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_list_models.py -v
```

Expected: the five new tests FAIL with `NotImplementedError`; the earlier four still PASS.

- [ ] **Step 3: Implement `list_models` in `backend/sidecar/vllm.py`**

Replace the `list_models` placeholder and add the needed imports at the top of `vllm.py`:

```python
from typing import Any, AsyncIterator

import httpx

from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
)
from .frames import (
    GenerateChatBody,
    ModelCapability,
    ModelDescriptor,
)
from .logging_setup import get_logger
from .vllm_models_config import VllmModelMetadata


log = get_logger("vllm")
```

Replace the `list_models` method body:

```python
    async def list_models(self) -> list[ModelDescriptor]:
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            payload: dict[str, Any] = r.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e
        except ValueError as e:
            raise EngineBadResponse(f"non-JSON /v1/models body: {e}") from e

        out: list[ModelDescriptor] = []
        for raw in payload.get("data") or []:
            descriptor = self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        served_id = raw.get("id")
        if not isinstance(served_id, str) or not served_id:
            return None

        context_length = raw.get("max_model_len")
        if not isinstance(context_length, int):
            log.warning(
                "vllm.model_dropped_no_context",
                id=served_id,
                reason="max_model_len missing or not an integer",
            )
            return None

        meta = self._metadata.get(served_id)
        if meta is None:
            if served_id not in self._warned_unknown:
                log.warning(
                    "vllm.model_without_metadata",
                    id=served_id,
                    message=(
                        f"No YAML metadata for model '{served_id}'. "
                        "Listing with capabilities=['text'] only."
                    ),
                )
                self._warned_unknown.add(served_id)
            capabilities: list[ModelCapability] = ["text"]
            display_name = served_id
            parameter_count: int | None = None
            quantisation: str | None = None
        else:
            capabilities = list(meta.capabilities) if meta.capabilities else ["text"]
            display_name = meta.display_name or served_id
            parameter_count = meta.parameter_count
            quantisation = meta.quantisation

        engine_metadata: dict[str, Any] = {}
        for key in ("owned_by", "root"):
            value = raw.get(key)
            if value is not None:
                engine_metadata[key] = value

        return ModelDescriptor(
            slug=served_id,
            display_name=display_name,
            parameter_count=parameter_count,
            context_length=context_length,
            quantisation=quantisation,
            capabilities=capabilities,
            engine_family="vllm",
            engine_model_id=served_id,
            engine_metadata=engine_metadata,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_list_models.py -v
```

Expected: all nine tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_list_models.py
git commit -m "Implement vLLM list_models with metadata overlay and drop semantics"
```

---

## Task 8: `VllmEngine.generate_chat` — payload, text streaming, terminal, usage

The first slice of generate_chat. Just plain text content, finish-reason mapping, usage block. Reasoning, tool calls, vision, errors follow in later tasks.

**Files:**
- Modify: `backend/sidecar/vllm.py`
- Create: `backend/tests/test_vllm_generate_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_vllm_generate_chat.py`:

```python
"""Unit tests for vLLM streaming chat."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from sidecar.engine import StreamTerminal
from sidecar.frames import (
    ContentPartImage,
    ContentPartText,
    GenerateChatBody,
    Message,
    StreamDelta,
)
from sidecar.vllm import VllmEngine


def _sse(events: list[dict]) -> bytes:
    """Build an OpenAI-style SSE body (each event wrapped in `data: …\\n\\n`)."""
    out = []
    for e in events:
        out.append(f"data: {json.dumps(e)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage_chunk(prompt: int, completion: int) -> dict:
    """vLLM sends a final event with choices: [] and a usage block when
    stream_options.include_usage is True."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


@respx.mock
async def test_plain_text_stream_and_finish_stop():
    events = [
        _chunk(content="Hello"),
        _chunk(content=" world"),
        _chunk(finish_reason="stop"),
        _usage_chunk(5, 7),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="gemma-4-26b",
        messages=[Message(role="user", content="Hi")],
    )
    items: list = []
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
async def test_finish_reason_length_and_tool_calls():
    for reason in ("length", "tool_calls"):
        events = [_chunk(content="x"), _chunk(finish_reason=reason)]
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=_sse(events))
        )
        engine = VllmEngine("http://localhost:8000", metadata={})
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
        assert term.finish_reason == reason
        respx.reset()


@respx.mock
async def test_unknown_finish_reason_falls_back_to_stop():
    events = [_chunk(content="x"), _chunk(finish_reason="content_filter")]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
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
    assert term.finish_reason == "stop"


@respx.mock
async def test_request_payload_has_stream_true_and_include_usage():
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")]),
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="hi")],
        parameters={"temperature": 0.7, "max_tokens": 128, "top_p": 0.9, "stop": ["\n"]},
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()

    sent = captured[0]
    assert sent["model"] == "m"
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 128
    assert sent["top_p"] == 0.9
    assert sent["stop"] == ["\n"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v
```

Expected: four tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `generate_chat` in `backend/sidecar/vllm.py`**

Add to the imports at the top:

```python
import json

from .engine import StreamTerminal
from .frames import (
    StreamDelta,
    Usage,
)
from ._reasoning import ThinkTagSplitter
```

Replace the `generate_chat` placeholder and add helpers:

```python
    async def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        payload = self._build_chat_payload(body)
        reasoning_on = body.options.reasoning

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(None, connect=3.0),
            ) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    raise EngineBadResponse(f"vllm {resp.status_code}: {text}")

                splitter = ThinkTagSplitter(reasoning_on=reasoning_on)
                terminal: StreamTerminal | None = None

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        raise EngineBadResponse(
                            f"non-JSON SSE line from /v1/chat/completions: {data[:80]}"
                        )

                    # Terminal-usage-only chunk: choices=[], usage=...
                    usage_block = chunk.get("usage")
                    choices = chunk.get("choices") or []
                    if not choices and usage_block:
                        if terminal is not None:
                            terminal = StreamTerminal(
                                finish_reason=terminal.finish_reason,
                                usage=_usage_from_block(usage_block),
                            )
                        continue

                    for choice in choices:
                        delta = choice.get("delta") or {}

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            for d in splitter.feed(content):
                                yield d

                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            terminal = StreamTerminal(
                                finish_reason=_map_finish_reason(finish_reason),
                                usage=(
                                    _usage_from_block(usage_block)
                                    if usage_block else None
                                ),
                            )

                if terminal is None:
                    terminal = StreamTerminal(finish_reason="stop")
                yield terminal

        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineUnavailable(str(e)) from e

    def _build_chat_payload(self, body: GenerateChatBody) -> dict[str, Any]:
        messages = [_message_to_openai(m) for m in body.messages]
        payload: dict[str, Any] = {
            "model": body.model_slug,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        p = body.parameters
        if p.temperature is not None:
            payload["temperature"] = p.temperature
        if p.top_p is not None:
            payload["top_p"] = p.top_p
        if p.max_tokens is not None:
            payload["max_tokens"] = p.max_tokens
        if p.stop is not None:
            payload["stop"] = p.stop
        if body.tools is not None:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
        return payload
```

Add module-level helpers at the bottom of the file:

```python
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
}


def _map_finish_reason(raw: str) -> str:
    return _FINISH_REASON_MAP.get(raw, "stop")


def _usage_from_block(block: dict[str, Any]) -> Usage | None:
    prompt = block.get("prompt_tokens")
    completion = block.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
    return None


def _message_to_openai(m: Any) -> dict[str, Any]:
    """Map a CSP Message to the OpenAI chat-completions schema.

    Text-only content passes through as a string. Multimodal (list) content
    becomes an array of parts with `image_url` data-URI encoding for images.
    Tool calls and tool_call_id pass through unchanged — vLLM is
    OpenAI-native there.
    """
    out: dict[str, Any] = {"role": m.role}
    content = m.content
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.type == "text":
                parts.append({"type": "text", "text": part.text})
            elif part.type == "image":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part.media_type};base64,{part.data_b64}",
                    },
                })
        out["content"] = parts
    else:
        out["content"] = content or ""

    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_generate_chat.py
git commit -m "Implement vLLM generate_chat with plain text streaming and usage"
```

---

## Task 9: Reasoning on both channels (`reasoning_content` + `<think>` tags)

Two orthogonal paths:
1. vLLM with `--reasoning-parser` emits `delta.reasoning_content`. Route to `reasoning` channel when `options.reasoning=True`; drop otherwise.
2. vLLM without the parser emits `<think>…</think>` inline in `content`. The `ThinkTagSplitter` already added in Task 8 handles this transparently.

**Files:**
- Modify: `backend/sidecar/vllm.py`
- Modify: `backend/tests/test_vllm_generate_chat.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_vllm_generate_chat.py`:

```python
@respx.mock
async def test_reasoning_content_emitted_when_reasoning_on():
    events = [
        _chunk(reasoning_content="Let me "),
        _chunk(reasoning_content="think..."),
        _chunk(content="The answer is 42."),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
        options={"reasoning": True},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)
    assert reasoning == "Let me think..."
    assert content == "The answer is 42."
    for d in deltas:
        populated = sum(x is not None for x in (d.content, d.reasoning, d.tool_calls))
        assert populated <= 1


@respx.mock
async def test_reasoning_content_suppressed_when_reasoning_off():
    events = [
        _chunk(reasoning_content="secret"),
        _chunk(content="visible"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    assert "".join(d.reasoning or "" for d in deltas) == ""
    assert "".join(d.content or "" for d in deltas) == "visible"


@respx.mock
async def test_inline_think_tags_split_when_reasoning_on():
    events = [
        _chunk(content="<think>"),
        _chunk(content="pondering"),
        _chunk(content="</think>"),
        _chunk(content="answer is 42"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
        options={"reasoning": True},
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    reasoning = "".join(d.reasoning or "" for d in deltas)
    content = "".join(d.content or "" for d in deltas)
    assert reasoning == "pondering"
    assert content == "answer is 42"


@respx.mock
async def test_inline_think_tags_dropped_when_reasoning_off():
    events = [
        _chunk(content="<think>secret</think>visible"),
        _chunk(finish_reason="stop"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="q")],
    )
    deltas: list[StreamDelta] = []
    try:
        async for item in engine.generate_chat(body):
            if isinstance(item, StreamDelta):
                deltas.append(item)
    finally:
        await engine.aclose()
    assert "".join(d.reasoning or "" for d in deltas) == ""
    assert "".join(d.content or "" for d in deltas) == "visible"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v -k reasoning
```

Expected: the two `reasoning_content` tests FAIL (adapter currently ignores that field). The two `<think>` tests PASS already (splitter is in place).

- [ ] **Step 3: Add `reasoning_content` handling in `generate_chat`**

Inside the `for choice in choices:` loop in `vllm.py`, immediately **after** the `content` handling and **before** the `finish_reason` handling, insert:

```python
                        reasoning_content = delta.get("reasoning_content")
                        if (
                            isinstance(reasoning_content, str)
                            and reasoning_content
                            and reasoning_on
                        ):
                            yield StreamDelta(reasoning=reasoning_content)
```

The full block inside the `for choice in choices:` loop should now look like:

```python
                    for choice in choices:
                        delta = choice.get("delta") or {}

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            for d in splitter.feed(content):
                                yield d

                        reasoning_content = delta.get("reasoning_content")
                        if (
                            isinstance(reasoning_content, str)
                            and reasoning_content
                            and reasoning_on
                        ):
                            yield StreamDelta(reasoning=reasoning_content)

                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            terminal = StreamTerminal(
                                finish_reason=_map_finish_reason(finish_reason),
                                usage=(
                                    _usage_from_block(usage_block)
                                    if usage_block else None
                                ),
                            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v
```

Expected: all eight tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_generate_chat.py
git commit -m "Route vLLM reasoning_content to the reasoning channel"
```

---

## Task 10: Tool-call fragment passthrough

**Files:**
- Modify: `backend/sidecar/vllm.py`
- Modify: `backend/tests/test_vllm_generate_chat.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_vllm_generate_chat.py`:

```python
@respx.mock
async def test_tool_call_fragments_progressive():
    events = [
        _chunk(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"loc"},
        }]),
        _chunk(tool_calls=[{
            "index": 0,
            "function": {"arguments": "\":\"Vienna"},
        }]),
        _chunk(tool_calls=[{
            "index": 0,
            "function": {"arguments": "\"}"},
        }]),
        _chunk(finish_reason="tool_calls"),
    ]
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
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
    assert len(frags) == 3
    assert frags[0].index == 0
    assert frags[0].id == "call_1"
    assert frags[0].function.name == "get_weather"
    assert frags[0].function.arguments == "{\"loc"
    assert frags[1].function.arguments == "\":\"Vienna"
    assert frags[2].function.arguments == "\"}"
    # Reconstructed arguments must parse:
    joined = "".join(f.function.arguments or "" for f in frags)
    assert json.loads(joined) == {"loc": "Vienna"}


@respx.mock
async def test_tools_field_forwarded_to_vllm():
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")])
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(role="user", content="hi")],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()
    assert captured[0]["tools"][0]["function"]["name"] == "get_weather"


@respx.mock
async def test_assistant_tool_call_history_passthrough():
    """Assistant history with JSON-string arguments must reach vLLM unchanged."""
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")])
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"loc\":\"Vienna\"}"},
                }],
            ),
            Message(role="tool", content="sunny", tool_call_id="c1"),
        ],
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()

    sent = captured[0]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    args = assistant["tool_calls"][0]["function"]["arguments"]
    # Stays a JSON string — unlike Ollama, vLLM expects OpenAI-native shape.
    assert isinstance(args, str)
    assert args == "{\"loc\":\"Vienna\"}"
    tool_msg = next(m for m in sent if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v -k tool
```

Expected: `test_tool_call_fragments_progressive` FAILS (adapter ignores tool_calls).
The two forwarding tests should already PASS (message mapping and tools payload forwarding are in place from Task 8).

- [ ] **Step 3: Handle tool-call fragments in `generate_chat`**

Add to the imports at the top of `vllm.py`:

```python
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

Inside the `for choice in choices:` loop in `generate_chat`, immediately **after** the `reasoning_content` handling and **before** the `finish_reason` handling, insert:

```python
                        raw_tcs = delta.get("tool_calls")
                        frags = _tool_call_fragments(raw_tcs)
                        if frags:
                            yield StreamDelta(tool_calls=frags)
```

Add at module level (bottom of `vllm.py`):

```python
def _tool_call_fragments(raw: Any) -> list[ToolCallFragment]:
    if not isinstance(raw, list):
        return []
    out: list[ToolCallFragment] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        index = tc.get("index")
        if not isinstance(index, int):
            continue
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is not None and not isinstance(args, str):
            # OpenAI-native payloads always use a string; be defensive.
            args = json.dumps(args, separators=(",", ":"))
        out.append(
            ToolCallFragment(
                index=index,
                id=tc.get("id"),
                type=tc.get("type"),
                function=ToolCallFragmentFunction(
                    name=fn.get("name"),
                    arguments=args,
                ),
            )
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v
```

Expected: all eleven tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_generate_chat.py
git commit -m "Forward vLLM tool-call fragments through to the stream"
```

---

## Task 11: Vision content-part mapping

`_message_to_openai` was already drafted in Task 8. This task confirms it against a multimodal message and locks it down with a test.

**Files:**
- Modify: `backend/tests/test_vllm_generate_chat.py`

- [ ] **Step 1: Write the test**

Append to `backend/tests/test_vllm_generate_chat.py`:

```python
@respx.mock
async def test_image_content_part_becomes_data_uri_image_url():
    captured: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, content=_sse([_chunk(content="ok"), _chunk(finish_reason="stop")])
        )

    respx.post("http://localhost:8000/v1/chat/completions").mock(side_effect=_capture)
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m",
        messages=[Message(
            role="user",
            content=[
                ContentPartText(type="text", text="What is this?"),
                ContentPartImage(type="image", media_type="image/png", data_b64="QUJDRA=="),
            ],
        )],
    )
    try:
        async for _ in engine.generate_chat(body):
            pass
    finally:
        await engine.aclose()

    sent = captured[0]["messages"]
    user_parts = sent[0]["content"]
    assert isinstance(user_parts, list)
    assert user_parts[0] == {"type": "text", "text": "What is this?"}
    image = user_parts[1]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"] == "data:image/png;base64,QUJDRA=="
```

- [ ] **Step 2: Run the test**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py::test_image_content_part_becomes_data_uri_image_url -v
```

Expected: PASS (the mapping was implemented in Task 8; this test documents the contract).

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_vllm_generate_chat.py
git commit -m "Cover vLLM vision-content mapping with a multimodal request test"
```

---

## Task 12: Error mapping and cancellation

**Files:**
- Modify: `backend/sidecar/vllm.py`
- Modify: `backend/tests/test_vllm_generate_chat.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_vllm_generate_chat.py`:

```python
from sidecar.engine import (
    EngineBadResponse,
    EngineUnavailable,
    ModelNotFound,
    ModelOutOfMemory,
)


@respx.mock
async def test_http_404_raises_model_not_found():
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"error": "model 'xxx' does not exist"})
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
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
async def test_http_400_oom_raises_model_oom():
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            400, json={"error": "CUDA out of memory, tried to allocate 2GiB"}
        )
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(ModelOutOfMemory):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_http_400_generic_raises_engine_bad_response():
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "invalid request"})
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
    body = GenerateChatBody(
        model_slug="m", messages=[Message(role="user", content="x")]
    )
    try:
        with pytest.raises(EngineBadResponse):
            async for _ in engine.generate_chat(body):
                pass
    finally:
        await engine.aclose()


@respx.mock
async def test_connect_error_raises_engine_unavailable():
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
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
async def test_cancellation_stops_iteration_cleanly():
    events = [_chunk(content=f"t{i}") for i in range(500)]
    events.append(_chunk(finish_reason="stop"))
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_sse(events))
    )
    engine = VllmEngine("http://localhost:8000", metadata={})
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

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v -k "404 or oom or 400 or connect"
```

Expected: the 404 test FAILS (currently any status >=400 raises `EngineBadResponse`, not `ModelNotFound`). The OOM test FAILS for the same reason. The generic-400 test PASSES already. The connect-error test PASSES.

The cancellation test should PASS already (httpx's `stream()` context is async-cancel-safe).

- [ ] **Step 3: Differentiate 404 and OOM in `generate_chat`**

Replace the error-status block inside `generate_chat`:

```python
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    raise EngineBadResponse(f"vllm {resp.status_code}: {text}")
```

with:

```python
                if resp.status_code == 404:
                    raise ModelNotFound(body.model_slug)
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    if _is_oom(text):
                        raise ModelOutOfMemory(text)
                    raise EngineBadResponse(f"vllm {resp.status_code}: {text}")
```

Add the `_is_oom` helper at module level (bottom of `vllm.py`):

```python
def _is_oom(text: str) -> bool:
    needle = text.lower()
    return "out of memory" in needle or "oom" in needle or "cuda" in needle and "memory" in needle
```

Add the two new exception imports near the top of `vllm.py`:

```python
from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
    ModelOutOfMemory,
    StreamTerminal,
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_vllm_generate_chat.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sidecar/vllm.py backend/tests/test_vllm_generate_chat.py
git commit -m "Map vLLM 404 to ModelNotFound and 400-OOM to ModelOutOfMemory"
```

---

## Task 13: Engine factory and dynamic `endpoint_hint`

**Files:**
- Modify: `backend/sidecar/main.py`
- Modify: `backend/tests/test_main_payload.py`

- [ ] **Step 1: Read the current test to understand the existing contract**

Run:

```bash
cd backend && cat tests/test_main_payload.py
```

Take note of what `build_handshake_payload` currently asserts.

- [ ] **Step 2: Write the failing tests**

Replace the contents of `backend/tests/test_main_payload.py` with:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_main_payload.py -v
```

Expected: tests fail (`_build_engine` does not exist yet; `endpoint_hint` is hardcoded to ollama_url).

- [ ] **Step 4: Implement the factory and dynamic hint in `backend/sidecar/main.py`**

Replace the top-of-file imports to include vLLM and config-loader:

```python
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
```

Replace `build_handshake_payload` with:

```python
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
```

Add the factory:

```python
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
```

Replace the single hardcoded line in `_run` (currently `engine: Engine = OllamaEngine(settings.ollama_url)`) with:

```python
    engine: Engine = _build_engine(settings)
```

Also update the startup log message: the line

```python
    log.info(
        "sidecar.starting",
        backend_url=settings.chatsune_backend_url,
        ollama_url=settings.ollama_url,
        health_port=settings.sidecar_health_port,
        host_key_tail=settings.host_key_tail(),
    )
```

becomes:

```python
    log.info(
        "sidecar.starting",
        backend_url=settings.chatsune_backend_url,
        engine=settings.sidecar_engine,
        ollama_url=settings.ollama_url,
        vllm_url=settings.vllm_url,
        health_port=settings.sidecar_health_port,
        host_key_tail=settings.host_key_tail(),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/test_main_payload.py -v
```

Expected: all five tests PASS.

- [ ] **Step 6: Run the full test suite as a regression check**

```bash
cd backend && uv run pytest -v
```

Expected: every test in every file PASSES.

- [ ] **Step 7: Commit**

```bash
git add backend/sidecar/main.py backend/tests/test_main_payload.py
git commit -m "Select engine via SIDECAR_ENGINE and thread endpoint_hint through"
```

---

## Task 14: Parametrise the integration test across both engines

The existing `test_integration.py` drives a full handshake/list/generate/cancel cycle. We want identical coverage against the vLLM adapter. Cleanest approach: pull the per-engine setup into small helper functions and run the same scenario twice via `pytest.mark.parametrize`.

**Files:**
- Modify: `backend/tests/test_integration.py`

- [ ] **Step 1: Understand what needs to vary per engine**

Per-engine pieces that the current test hardcodes for Ollama:
1. The respx mocks for version / discovery / chat.
2. The `engine = OllamaEngine(...)` construction.
3. The `settings` (`ollama_url` vs `vllm_url`, `sidecar_engine`).

Everything else — fake backend, WS handler, assertions — stays engine-agnostic.

- [ ] **Step 2: Refactor and parametrise**

Replace the contents of `backend/tests/test_integration.py` with:

```python
"""End-to-end: fake Chatsune backend + mocked engine HTTP + real sidecar.

Drives:
  connect → handshake → list_models → generate_chat → cancel → ack → shutdown

Parametrised across both Ollama and vLLM adapters to prove the dispatcher
and connection plumbing are engine-agnostic.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx
import pytest
import respx
import websockets

from sidecar.config import Settings
from sidecar.connection import ConnectionManager
from sidecar.dispatcher import Dispatcher
from sidecar.engine import Engine
from sidecar.main import build_handshake_payload
from sidecar.ollama import OllamaEngine
from sidecar.vllm import VllmEngine


@asynccontextmanager
async def fake_backend(handler):
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/sidecar"


def _ndjson(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


async def _async_ndjson_stream(lines) -> AsyncIterator[bytes]:
    for line in lines:
        yield (json.dumps(line) + "\n").encode()
        await asyncio.sleep(0)


class _NdjsonStream(httpx.AsyncByteStream):
    def __init__(self, lines):
        self._lines = lines

    async def __aiter__(self):
        async for chunk in _async_ndjson_stream(self._lines):
            yield chunk

    async def aclose(self) -> None:
        pass


async def _async_sse_stream(events) -> AsyncIterator[bytes]:
    for e in events:
        yield f"data: {json.dumps(e)}\n\n".encode()
        await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"


class _SseStream(httpx.AsyncByteStream):
    def __init__(self, events):
        self._events = events

    async def __aiter__(self):
        async for chunk in _async_sse_stream(self._events):
            yield chunk

    async def aclose(self) -> None:
        pass


@dataclass
class _EngineSetup:
    name: str
    build_engine: Callable[[], Engine]
    mock_http: Callable[[], None]
    expected_slug: str
    settings_engine: str


def _setup_ollama() -> _EngineSetup:
    def mock_http():
        respx.get("http://localhost:11434/api/version").mock(
            return_value=httpx.Response(200, json={"version": "0.5.7"})
        )
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={
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
            })
        )
        respx.post("http://localhost:11434/api/show").mock(
            return_value=httpx.Response(200, json={
                "capabilities": [],
                "model_info": {"llama.context_length": 131072},
            })
        )
        ollama_chunks = [
            {"message": {"role": "assistant", "content": f"t{i}"}, "done": False}
            for i in range(1000)
        ] + [{"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, stream=_NdjsonStream(ollama_chunks))
        )

    return _EngineSetup(
        name="ollama",
        build_engine=lambda: OllamaEngine("http://localhost:11434"),
        mock_http=mock_http,
        expected_slug="llama3.2:8b",
        settings_engine="ollama",
    )


def _setup_vllm() -> _EngineSetup:
    def mock_http():
        respx.get("http://localhost:8000/version").mock(
            return_value=httpx.Response(200, json={"version": "0.7.3"})
        )
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json={
                "object": "list",
                "data": [{
                    "id": "integration-model",
                    "object": "model",
                    "owned_by": "vllm",
                    "max_model_len": 8192,
                    "root": "fake/repo",
                }],
            })
        )
        vllm_events = [
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"t{i}"},
                    "finish_reason": None,
                }],
            }
            for i in range(1000)
        ] + [
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ]
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, stream=_SseStream(vllm_events))
        )

    from sidecar.vllm_models_config import VllmModelMetadata

    return _EngineSetup(
        name="vllm",
        build_engine=lambda: VllmEngine(
            "http://localhost:8000",
            metadata={
                "integration-model": VllmModelMetadata(
                    display_name="Integration Model",
                    capabilities=["text"],
                ),
            },
        ),
        mock_http=mock_http,
        expected_slug="integration-model",
        settings_engine="vllm",
    )


ENGINE_SETUPS = [_setup_ollama(), _setup_vllm()]


@pytest.mark.parametrize("setup", ENGINE_SETUPS, ids=[s.name for s in ENGINE_SETUPS])
@respx.mock
async def test_full_cycle(setup: _EngineSetup):
    setup.mock_http()

    observed: list[dict] = []
    list_models_done = asyncio.Event()
    streamed = asyncio.Event()
    finished = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "handshake"
        assert hello["engine"]["type"] == setup.name
        await ws.send(json.dumps({
            "type": "handshake_ack",
            "csp_version": "1.0",
            "homelab_id": "H",
            "display_name": "test",
            "accepted": True,
            "notices": [],
        }))

        await ws.send(json.dumps({"type": "req", "id": "r1", "op": "list_models"}))
        while not list_models_done.is_set():
            msg = json.loads(await ws.recv())
            observed.append(msg)
            if msg["type"] == "res" and msg["id"] == "r1":
                list_models_done.set()
            elif msg["type"] == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        await ws.send(json.dumps({
            "type": "req", "id": "r2", "op": "generate_chat",
            "body": {
                "model_slug": setup.expected_slug,
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
            vllm_url="http://localhost:8000",
            vllm_models_config_path=None,
            vllm_models_overlay_path=None,
            sidecar_engine=setup.settings_engine,
            sidecar_health_port=0,
            sidecar_log_level="warn",
            sidecar_max_concurrent_requests=1,
        )

        engine = setup.build_engine()
        try:
            version = await engine.probe_version()
            handshake = build_handshake_payload(
                settings, engine_type=engine.engine_type, engine_version=version,
            )

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
            cm._ping_interval = 1.0
            cm._pong_timeout = 5.0

            run_task = asyncio.create_task(cm.run_forever())

            await asyncio.wait_for(finished.wait(), timeout=15.0)
            cm.request_stop()
            await asyncio.wait_for(run_task, timeout=5.0)
            await dispatcher.cancel_all()
        finally:
            await engine.aclose()

    res = next(f for f in observed if f["type"] == "res")
    assert res["id"] == "r1"
    assert res["body"]["models"][0]["slug"] == setup.expected_slug
    stream_end = next(
        f for f in observed if f["type"] == "stream_end" and f["id"] == "r2"
    )
    assert stream_end["finish_reason"] == "cancelled"
```

- [ ] **Step 3: Run the integration test**

```bash
cd backend && uv run pytest tests/test_integration.py -v
```

Expected: two parametrised runs PASS (one `ollama`, one `vllm`).

- [ ] **Step 4: Run the full test suite**

```bash
cd backend && uv run pytest -v
```

Expected: every test across every file PASSES.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "Parametrise integration test across Ollama and vLLM engines"
```

---

## Task 15: Documentation updates

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `compose.yml`

- [ ] **Step 1: Update `.env.example`**

Insert a new `Engine selection` block between the `Required` and `Ollama engine` blocks, and a new `vLLM engine` block after the `Ollama engine` block. The final file reads:

```bash
# --- Required -------------------------------------------------------------

# Base WebSocket URL of the Chatsune backend. The sidecar appends
# /ws/sidecar. Production MUST use wss:// with a certificate trusted by
# the system trust store (no self-signed shortcuts).
# For local development against an unterminated backend, ws:// is
# accepted; the sidecar logs a startup warning in that case.
CHATSUNE_BACKEND_URL=wss://chat.example.com

# Host-Key issued in the Chatsune UI. MUST start with "cshost_".
# Never commit a real value.
CHATSUNE_HOST_KEY=cshost_replace_me

# --- Engine selection -----------------------------------------------------

# Which local inference engine to drive: "ollama" (default) or "vllm".
SIDECAR_ENGINE=ollama

# --- Ollama engine --------------------------------------------------------

# URL of the local Ollama HTTP API. On Docker Desktop / Podman the
# default works out of the box; on Linux bare-metal set it to
# http://localhost:11434 or the LAN IP of the Ollama host.
OLLAMA_URL=http://host.docker.internal:11434

# --- vLLM engine (only read when SIDECAR_ENGINE=vllm) ---------------------

# URL of the local vLLM OpenAI-compatible server.
VLLM_URL=http://host.docker.internal:8000

# Optional YAML files describing each served model's capabilities. vLLM's
# /v1/models does not report whether a model supports vision, tool
# calling, or reasoning — this layer fills the gap. Both paths are
# optional; an unknown model is listed as capabilities: [text] plus a
# warn log.
# When both are set, the overlay file overrides individual fields from
# the base file (list fields are replaced wholesale).
# VLLM_MODELS_CONFIG_PATH=/etc/chatsune-sidecar/models.yaml
# VLLM_MODELS_OVERLAY_PATH=/etc/chatsune-sidecar/models.local.yaml

# --- Optional -------------------------------------------------------------

# Port for the loopback healthcheck HTTP server. Docker's HEALTHCHECK
# probes this. Must stay on 127.0.0.1.
SIDECAR_HEALTH_PORT=8080

# Log level: debug, info, warn, error. debug MAY include user content.
SIDECAR_LOG_LEVEL=info

# Override the handshake-advertised max_concurrent_requests.
# Default for the Ollama sidecar is 1 (Ollama loads one model at a time).
# For vLLM with real batching, 4 is a sensible starting point.
# SIDECAR_MAX_CONCURRENT_REQUESTS=1
```

- [ ] **Step 2: Update `README.md`**

Two edits:

1. Extend the `Environment variables` table. Add these rows after the existing `OLLAMA_URL` row and before `SIDECAR_HEALTH_PORT`:

```markdown
| `SIDECAR_ENGINE` | `ollama` | Selects the local engine. `ollama` or `vllm`. |
| `VLLM_URL` | `http://host.docker.internal:8000` | vLLM OpenAI-compatible HTTP endpoint. Only read when `SIDECAR_ENGINE=vllm`. |
| `VLLM_MODELS_CONFIG_PATH` | *unset* | Optional path to a YAML file with per-model metadata (vision/tools/reasoning capabilities etc.). |
| `VLLM_MODELS_OVERLAY_PATH` | *unset* | Optional path to a YAML overlay that overrides individual fields from `VLLM_MODELS_CONFIG_PATH`. |
```

2. Append a new top-level section at the end of the README, just before `## Licence`:

```markdown
## vLLM engine

To run the sidecar against a local vLLM server instead of Ollama:

```bash
export SIDECAR_ENGINE=vllm
export VLLM_URL=http://localhost:8001   # wherever your vllm listens
./start-backend.sh
```

### Per-model metadata YAML

vLLM's `/v1/models` endpoint reports only `id` and `max_model_len` —
it does not declare whether a served model supports vision, tool
calling, or a dedicated reasoning channel. The sidecar fills that gap
from a small YAML file you maintain. Path is configured via
`VLLM_MODELS_CONFIG_PATH`, and an optional second file under
`VLLM_MODELS_OVERLAY_PATH` can override individual fields per
served-id.

Example:

```yaml
# models.yaml
models:
  gemma-4-26b:
    display_name: Gemma 4 26B Instruct (AWQ-4bit)
    parameter_count: 26000000000
    quantisation: AWQ-4bit
    capabilities: [text, vision, tool_calling]
  deepseek-r1-distill-qwen-32b:
    display_name: DeepSeek R1 Distill Qwen 32B
    parameter_count: 32000000000
    capabilities: [text, tool_calling, reasoning]
```

Models listed in `/v1/models` without a matching YAML entry are still
returned to the Chatsune backend, but only with `capabilities: [text]`
and the served id as display name. A warning is logged once per
unknown id.

Models without a `max_model_len` in `/v1/models` are dropped entirely
(the backend cannot safely offer a model whose context window is
unknown).

### Concurrency

For vLLM's real-batching mode, raise the concurrency the sidecar
advertises:

```bash
export SIDECAR_MAX_CONCURRENT_REQUESTS=4
```

Set it to whatever you sized the vLLM server for. Ollama users should
leave it at `1` — Ollama loads one model at a time.
```

- [ ] **Step 3: Update `compose.yml`**

Rewrite the file so it passes both engines' variables through and includes a cookbook snippet for the YAML volumes:

```yaml
services:
  chatsune-sidecar:
    image: ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend:latest
    restart: unless-stopped
    environment:
      CHATSUNE_BACKEND_URL: ${CHATSUNE_BACKEND_URL:?required}
      CHATSUNE_HOST_KEY: ${CHATSUNE_HOST_KEY:?required}
      SIDECAR_ENGINE: ${SIDECAR_ENGINE:-ollama}
      OLLAMA_URL: ${OLLAMA_URL:-http://host.docker.internal:11434}
      VLLM_URL: ${VLLM_URL:-http://host.docker.internal:8000}
      # Optional — only relevant for SIDECAR_ENGINE=vllm:
      # VLLM_MODELS_CONFIG_PATH: /etc/chatsune-sidecar/models.yaml
      # VLLM_MODELS_OVERLAY_PATH: /etc/chatsune-sidecar/models.local.yaml
      SIDECAR_HEALTH_PORT: "8080"
      SIDECAR_LOG_LEVEL: ${SIDECAR_LOG_LEVEL:-info}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    # Read-only root FS; /tmp mounted as tmpfs for any ephemeral writes
    read_only: true
    tmpfs:
      - /tmp
    # Cookbook: mount YAML metadata files when using vLLM.
    # volumes:
    #   - ./vllm-models.yaml:/etc/chatsune-sidecar/models.yaml:ro
    #   - ./vllm-models.local.yaml:/etc/chatsune-sidecar/models.local.yaml:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz').read()"]
      interval: 30s
      timeout: 5s
      retries: 3
```

- [ ] **Step 4: Run the full test suite one more time**

```bash
cd backend && uv run pytest -v
```

Expected: everything PASSES. This is the last regression check before shipping.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md compose.yml
git commit -m "Document vLLM engine configuration and per-model YAML layer"
```

---

## Done

At this point the sidecar can be deployed with either engine via a single
env-var flip, and the test suite covers both adapters end-to-end.
