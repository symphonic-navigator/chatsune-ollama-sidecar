# vLLM Engine Support — Design

**Status:** Draft (2026-04-16)
**Scope:** Extend the existing CSP/1 sidecar (currently Ollama-only) so it
can be run against a local vLLM server instead of Ollama, selected at
deploy time via a single environment variable.

---

## 1. Motivation

The sidecar architecture already anticipates multiple engines
(`Engine` protocol in `backend/sidecar/engine.py`, `EngineType` literal
covering `ollama | lmstudio | vllm | llamacpp`). SPEC §15.3 documents
vLLM as an OpenAI-compatible target. What is missing is a concrete
`VllmEngine` adapter and the wiring to select it.

vLLM in practice ships a sparser `/v1/models` response than Ollama's
`/api/show`: it reports `id`, `max_model_len`, `owned_by`, `root`, and
not much else. In particular, it does **not** tell us whether a served
model supports vision, tool calling or reasoning. This design closes
that metadata gap with a user-maintained YAML layer.

---

## 2. Non-goals

- Multi-engine multiplexing in one process (SPEC §2 forbids; one
  sidecar = one engine = one homelab).
- Auto-detection of the engine at runtime (explicit config wins; no
  magic).
- GGUF inspection, native metadata extraction for vLLM, or parsing
  `root` identifiers to infer quantisation / parameter count. Those
  would be fabrication per SPEC §15.5.
- Dynamic reload of the YAML config at runtime. Config is loaded once
  at startup; changing it requires a process restart (matches the
  sidecar's existing deployment model under Docker Compose).

---

## 3. Architecture overview

The existing `Engine` protocol stays unchanged. A new adapter
`VllmEngine` lives in `backend/sidecar/vllm.py`. A small factory in
`main.py` selects the concrete engine based on the new
`SIDECAR_ENGINE` environment variable. Dispatcher, connection,
frames, healthcheck — none of them change.

New/changed files:

| File | Change |
|---|---|
| `backend/sidecar/vllm.py` | **new** — `VllmEngine` (list_models, generate_chat) |
| `backend/sidecar/vllm_models_config.py` | **new** — YAML loader + deep-merge logic |
| `backend/sidecar/_reasoning.py` | **new** — `_ThinkTagSplitter` extracted from `ollama.py` so both adapters share it |
| `backend/sidecar/ollama.py` | imports `_ThinkTagSplitter` from `_reasoning.py`; no behavioural change |
| `backend/sidecar/config.py` | adds `sidecar_engine`, `vllm_url`, `vllm_models_config_path`, `vllm_models_overlay_path` |
| `backend/sidecar/main.py` | engine factory; `endpoint_hint` picks the active engine's URL |
| `backend/pyproject.toml` | adds `pyyaml` dependency |
| `.env.example`, `README.md`, `compose.yml` | document the new environment variables |
| `backend/tests/test_vllm_models_config.py` | **new** — merge-logic and fail-fast tests |
| `backend/tests/test_vllm_list_models.py` | **new** — `/v1/models` adapter tests |
| `backend/tests/test_vllm_generate_chat.py` | **new** — streaming-chat adapter tests |
| `backend/tests/fixtures/vllm_models.json` | **new** — recorded `/v1/models` payload |
| `backend/tests/test_integration.py` | parametrised to run once per engine |

The Docker image name stays `ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend`
for now; a rename is a separate, later step bundled with the repository
rename.

---

## 4. Settings extension

### 4.1 New fields in `config.py`

```python
sidecar_engine: Literal["ollama", "vllm"] = Field(default="ollama")

vllm_url: str = Field(default="http://host.docker.internal:8000")
vllm_models_config_path: str | None = Field(default=None)
vllm_models_overlay_path: str | None = Field(default=None)
```

### 4.2 Behaviour

- `sidecar_engine` defaults to `ollama` so that existing deployments
  keep working without changing a single variable.
- `vllm_url` uses the SPEC §14 default port `8000`. Operators point it
  at whatever port their vLLM server listens on (e.g. `:8001`).
- Both config paths are optional. Unused env vars for the inactive
  engine are ignored — no cross-field validation, no noise.
- `SIDECAR_MAX_CONCURRENT_REQUESTS` keeps its default of `1` for both
  engines. For vLLM the README recommends raising it to `4` or higher;
  this is an explicit operator decision rather than a magic default.

---

## 5. Model-metadata config layer (`vllm_models_config.py`)

### 5.1 Schema

Both files share the same schema. Every field under `models.<served_id>`
is optional.

```yaml
models:
  <served_id>:
    display_name: string              # optional, default: <served_id>
    parameter_count: int              # optional, default: null
    quantisation: string | null       # optional, default: null
    capabilities:                     # optional, default: ["text"]
      - text
      - vision
      - tool_calling
      - reasoning
      - json_mode
```

`<served_id>` is exactly the `id` vLLM returns under `/v1/models`
(e.g. `gemma-4-26b`).

### 5.2 Merge semantics

Two paths are loaded, in order: `vllm_models_config_path` (base) then
`vllm_models_overlay_path` (overlay). Merge is **deep at the
field level per served_id**:

- A served_id that appears in overlay but not base is taken as-is from
  overlay.
- A served_id that appears in both: fields present in overlay
  overwrite the base fields; fields only in base survive.
- Lists (including `capabilities`) are **replaced wholesale** — there
  is deliberately no list-union or list-subtraction operator, because
  removing a capability via overlay must remain possible.

Overlay has the last word at per-field granularity. Base is never
mutated.

### 5.3 Fail-fast behaviour

The config layer is fail-fast to match the rest of the sidecar's
startup semantics:

| Situation | Outcome |
|---|---|
| Path not set | OK — that source contributes nothing. |
| Path set, file missing | Startup error, exit code 2, path in message. |
| File present, malformed YAML | Startup error, exit code 2, path + parser error in message. |
| YAML parses but schema violated | Startup error, exit code 2, Pydantic validation output. |
| Unknown capability literal | Startup error, exit code 2. |

This aligns with `main.py:119-124` (Settings-validation fail-fast) and
with the orchestrator-friendly exit code 2 reserved for
non-restart-loop configuration errors (SPEC §12).

### 5.4 Module surface

```python
class VllmModelMetadata(BaseModel):
    display_name: str | None = None
    parameter_count: int | None = None
    quantisation: str | None = None
    capabilities: list[ModelCapability] | None = None


def load_vllm_models_config(
    base_path: str | None,
    overlay_path: str | None,
) -> dict[str, VllmModelMetadata]:
    ...
```

Returns a dict keyed by served_id. Missing keys mean “no user-supplied
metadata for this id" — the engine adapter falls back to conservative
defaults (see §6.1).

---

## 6. `VllmEngine`

Lives in `backend/sidecar/vllm.py`. Public surface matches the
`Engine` protocol.

### 6.1 `list_models`

1. `GET /v1/models` → `data[]`.
2. Per entry:
   - `slug = id`, `engine_model_id = id`.
   - `context_length = max_model_len`. If this field is missing or
     non-int, **drop the model** per SPEC §6.5. Log a warning.
   - Lookup `id` in the merged metadata dict.
     - **Hit** → use the user-supplied `display_name`,
       `parameter_count`, `quantisation`, `capabilities`.
     - **Miss** → emit a **one-shot warning per unknown id** on the
       first `list_models` call that encounters it, then populate
       with `display_name=id`, `parameter_count=None`,
       `quantisation=None`, `capabilities=["text"]`. The one-shot
       dedup lives inside the engine instance (a `set[str]` of ids
       already warned).
   - `engine_family = "vllm"`.
   - `engine_metadata` carries vLLM-native hints verbatim:
     `{"owned_by": "...", "root": "..."}`. Nothing is parsed out of
     `root` — no quantisation guessing, no parameter count guessing.
3. `EngineUnavailable` on connect error, `EngineBadResponse` on
   non-200, identical to the Ollama adapter.

### 6.2 `generate_chat`

`POST /v1/chat/completions` with:

```json
{
  "model": "<slug>",
  "messages": [...],
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": ..., "top_p": ..., "max_tokens": ..., "stop": [...],
  "tools": [...]
}
```

Unsupplied parameters are omitted (vLLM substitutes its own defaults).

**Message mapping.** OpenAI-native, so much closer to the wire than
Ollama's shape:

- Plain string `content` → passed through unchanged.
- `ContentPart[]` — `text` parts unchanged,
  `image` parts mapped to
  `{"type": "image_url", "image_url": {"url": "data:<media_type>;base64,<data_b64>"}}`.
- `tool_calls` on assistant messages: `id`, `type`, `function.name`,
  `function.arguments` (JSON string) go through unchanged — no
  arguments-to-dict round-trip like Ollama requires.
- `tool_call_id` on tool messages: unchanged.

**Stream parsing.** Lines from `resp.aiter_lines()` are SSE
`data: {...}` frames. The adapter strips the `data: ` prefix, ignores
the `[DONE]` sentinel, and decodes each chunk.

Per `chunk.choices[0]`:

- `delta.content` (string): fed through `_ThinkTagSplitter` (in
  `_reasoning.py`) unconditionally — same splitter as Ollama uses.
  When `options.reasoning=true` the splitter emits any `<think>…</think>`
  inner text as `StreamDelta(reasoning=...)`; when `false` it silently
  drops the inner text. Outside-of-tag content is always emitted as
  `StreamDelta(content=...)`. No per-model reasoning-capability lookup
  is needed — if the model does not emit `<think>` tags, the splitter
  is a passthrough.
- `delta.reasoning_content` (present when vLLM runs with
  `--reasoning-parser`): if `options.reasoning=true`, emit as
  `StreamDelta(reasoning=...)`; otherwise suppress it.
- `delta.tool_calls[]`: map each fragment to `ToolCallFragment`
  one-to-one (`index`, `id`, `type`, `function.name`,
  `function.arguments`). vLLM streams these progressively — unlike
  Ollama, which batches the full call into a single chunk — so the
  consumer sees arguments grow token-by-token.

Terminal: the chunk whose `choices[0].finish_reason` is non-null closes
the stream. Mapping:

| vLLM finish_reason | CSP finish_reason |
|---|---|
| `stop` | `stop` |
| `length` | `length` |
| `tool_calls` | `tool_calls` |
| anything else | `stop` |

The final chunk also carries `usage` (because of `include_usage: true`)
→ `StreamTerminal.usage`.

### 6.3 Error mapping

- HTTP 404 on `/v1/chat/completions` → `ModelNotFound`.
- HTTP 400 whose body mentions out-of-memory → `ModelOutOfMemory`.
- Any other non-2xx → `EngineBadResponse`.
- `httpx.ConnectError` → `EngineUnavailable`.

### 6.4 Cancellation

Identical pattern to Ollama: `asyncio.CancelledError` propagates
out of the `async with self._client.stream(...)` context, httpx closes
the socket, vLLM honours client disconnect (SPEC §9).

### 6.5 Shared splitter (`_reasoning.py`)

`_ThinkTagSplitter` moves from `ollama.py:386` to a new internal
module `backend/sidecar/_reasoning.py`. `ollama.py` and `vllm.py`
both import from there. Leading underscore signals internal utility.
No behavioural change to the splitter itself — pure relocation.

---

## 7. Handshake & main-module changes

### 7.1 Engine factory

`main.py` replaces the hardcoded `OllamaEngine(...)` with:

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

`load_vllm_models_config` raises on malformed files; `main()` already
turns uncaught configuration errors into exit code 2.

### 7.2 `endpoint_hint`

`build_handshake_payload` picks the active engine's URL:

```python
endpoint_hint = (
    settings.ollama_url if engine_type == "ollama" else settings.vllm_url
)
```

### 7.3 Handshake `capabilities`

Unchanged. The four sidecar-level capabilities
(`chat_streaming`, `tool_calls`, `vision`, `reasoning`) describe what
the sidecar is **able to translate**, not what the active models
support. Model-level capabilities stay on the `ModelDescriptor`.

### 7.4 `engine.version`

`VllmEngine.probe_version()` calls `GET /version` on vLLM and returns
the `version` field, falling back to `"unknown"` on any error.
Identical contract to `OllamaEngine.probe_version()`.

---

## 8. Test plan

### 8.1 `test_vllm_models_config.py`

- Only base, no overlay → entries loaded correctly.
- Only overlay, no base → overlay acts as sole source.
- Base + overlay deep-merge:
  - Overlay overwrites individual fields; untouched fields survive
    from base.
  - `capabilities` list in overlay replaces base's list entirely
    (no concatenation).
- Fail-fast:
  - Configured path, missing file → exception with path in message.
  - Malformed YAML → exception with path.
  - `capabilities: [videooo]` → Pydantic validation error.
  - `capabilities: "vision"` (scalar instead of list) → Pydantic
    validation error.
- Both paths `None` → empty dict, no exception.

### 8.2 `test_vllm_list_models.py`

Fixtures: `tests/fixtures/vllm_models.json` with the real
`gemma-4-26b` payload from the user plus a second entry missing
`max_model_len` to exercise the drop path.

- Happy path, config hits → all fields populated.
- Happy path, config miss → `capabilities=["text"]`,
  `display_name=id`, one-shot warning captured via caplog.
- Entry without `max_model_len` → dropped (SPEC §6.5), warning logged.
- vLLM unreachable → `EngineUnavailable`.
- HTTP 500 → `EngineBadResponse`.

### 8.3 `test_vllm_generate_chat.py`

Streaming parser tests with constructed SSE payloads, same pattern
as `test_ollama_generate_chat.py`.

- Plain content chunks → `StreamDelta(content=...)`.
- `reasoning_content` with `options.reasoning=true` →
  `StreamDelta(reasoning=...)`.
- `reasoning_content` with `options.reasoning=false` → suppressed.
- Inline `<think>...</think>` with `options.reasoning=true` → inner
  text emitted on the reasoning channel; outer text on content.
- Inline `<think>...</think>` with `options.reasoning=false` → inner
  text dropped; outer text emitted on content.
- Tool-call fragments across multiple chunks at the same `index` →
  passthrough in order; consumer can concatenate `arguments`.
- Finish-reason mapping: `stop`, `length`, `tool_calls` map through.
- `usage` block in the terminal chunk lands in `StreamTerminal.usage`.
- `ContentPartImage` → correct `image_url` with `data:<media>;base64,...`
  in the outgoing request body.
- HTTP 404 → `ModelNotFound`.
- Cancellation via `asyncio.CancelledError` → `httpx.stream` closes
  without hanging.

### 8.4 Integration test — parametrised, not new

`test_integration.py` is engine-agnostic. A `pytest.mark.parametrize`
on the engine selector drives it once per engine, with an httpx
transport mock substituted for each. No new test file.

### 8.5 Dependencies

`pyyaml` added to `backend/pyproject.toml`. Nothing else new.

---

## 9. Deployment & docs

### 9.1 `.env.example`

```bash
# --- Engine selection ----------------------------------------------------
SIDECAR_ENGINE=ollama

# --- vLLM engine (only read when SIDECAR_ENGINE=vllm) --------------------
VLLM_URL=http://host.docker.internal:8000
# VLLM_MODELS_CONFIG_PATH=/etc/chatsune-sidecar/models.yaml
# VLLM_MODELS_OVERLAY_PATH=/etc/chatsune-sidecar/models.local.yaml
```

### 9.2 `README.md`

- Environment-variable table extended with the four new variables.
- New “vLLM engine" section:
  - When to pick vLLM over Ollama.
  - Example YAML with two models (one vision-capable, one
    reasoning-capable).
  - Recommendation: raise `SIDECAR_MAX_CONCURRENT_REQUESTS` to `4` for
    vLLM deployments.

### 9.3 `compose.yml`

- Same single-service topology.
- Both `OLLAMA_URL` and `VLLM_URL` passed through (`:-` defaults).
- `SIDECAR_ENGINE` passed through.
- Commented volume-mount snippet for the two YAML files, intended as
  a cookbook recipe for vLLM operators.

### 9.4 Unchanged

- `SPEC.md` — CSP/1 is frozen; vLLM is already documented in §15.3.
- `Dockerfile` — no new system packages; pyyaml installs from wheel.
- CI — new tests run under the existing `pytest -v` step.

---

## 10. Open items / follow-ups

- If over time the list of reasoning-capable model families grows, the
  implicit allowlist in `ollama.py` (`_REASONING_FAMILIES`) may want
  the same treatment as vLLM — i.e. a user-configurable YAML. For now
  it stays as-is; it’s a separate concern.
- A future release may rename the repository and Docker image to drop
  `ollama` from the name, reflecting the multi-engine reality. Not in
  scope here.
- vLLM `GET /version` has been stable, but the field name has drifted
  on past releases. The probe falls back to `"unknown"` on any
  unexpected shape, so this is tolerable rather than urgent.
