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

## Licence

GPL-3.0-or-later — see [`LICENSE`](LICENSE).
