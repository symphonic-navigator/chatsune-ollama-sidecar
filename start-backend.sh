#! /bin/bash

cd "$(dirname "$0")/backend" || exit 1

uv sync
uv run python -m sidecar 2>&1
