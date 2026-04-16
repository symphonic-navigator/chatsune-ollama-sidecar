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

`start-backend.sh` runs `uv sync` inside `backend/` and launches
`python -m sidecar`. This is a daemon process, not an HTTP app —
there is no autoreload.

## Quick start (Docker)

```bash
export CHATSUNE_BACKEND_URL=wss://chat.example.com
export CHATSUNE_HOST_KEY=cshost_yourkeyhere
docker compose up -d
docker compose logs -f
```

The image is published as
`ghcr.io/symphonic-navigator/chatsune-ollama-sidecar-backend:latest`
by the Docker Build & Push workflow on pushes to `master` and on
semver tags.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHATSUNE_BACKEND_URL` | *required* | Base `wss://` URL. `/ws/sidecar` is appended automatically. |
| `CHATSUNE_HOST_KEY` | *required* | Per-sidecar secret; must start with `cshost_`. |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama HTTP endpoint. |
| `SIDECAR_ENGINE` | `ollama` | Selects the local engine. `ollama` or `vllm`. |
| `VLLM_URL` | `http://host.docker.internal:8000` | vLLM OpenAI-compatible HTTP endpoint. Only read when `SIDECAR_ENGINE=vllm`. |
| `VLLM_MODELS_CONFIG_PATH` | *unset* | Optional path to a YAML file with per-model metadata (vision/tools/reasoning capabilities etc.). |
| `VLLM_MODELS_OVERLAY_PATH` | *unset* | Optional path to a YAML overlay that overrides individual fields from `VLLM_MODELS_CONFIG_PATH`. |
| `SIDECAR_HEALTH_PORT` | `8080` | Loopback healthcheck port. |
| `SIDECAR_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error`. |
| `SIDECAR_MAX_CONCURRENT_REQUESTS` | `1` | Handshake-advertised concurrency. |

See [`SPEC.md` §14](SPEC.md) for authoritative semantics.

## Healthcheck

```bash
curl -s http://127.0.0.1:8080/healthz | jq
```

- `200 OK` while the backend connection is `connected` or `reconnecting`.
- `503 Service Unavailable` once the sidecar has given up.

The server binds to the loopback interface only; Docker's own
`HEALTHCHECK` probes it from inside the container.

## Development

```bash
cd backend
uv sync
uv run pytest -v
```

Module layout:

- `sidecar/frames.py` — every CSP/1 frame as a Pydantic model.
- `sidecar/ollama.py` — Ollama HTTP adapter (list, stream chat, cancel).
- `sidecar/connection.py` — WS client, handshake, ping/pong, reconnect.
- `sidecar/dispatcher.py` — `req`/`cancel` routing, in-flight task tracking.
- `sidecar/healthcheck.py` — loopback `/healthz` server.
- `sidecar/main.py` — entry point and signal handling.

Integration test (end-to-end cycle against a fake backend):

```bash
uv run pytest tests/test_integration.py -v
```

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

## Licence

GPL-3.0-or-later — see [`LICENSE`](LICENSE).
