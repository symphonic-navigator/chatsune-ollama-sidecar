# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

WORKDIR /app

# Install uv from the official distroless image — no apt packages needed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only the dependency manifest first so Docker can cache the install layer
COPY backend/pyproject.toml ./backend/

# Create the venv and install all runtime dependencies before copying source
RUN uv venv /app/.venv && \
    cd backend && \
    UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --no-install-project

# Copy the rest of the backend source
COPY backend/ ./backend/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root, no shell, no home directory needed — app runs from /app/backend
RUN useradd --system --uid 10001 --shell /usr/sbin/nologin sidecar
USER 10001:10001

WORKDIR /app/backend
CMD ["python", "-m", "sidecar"]
