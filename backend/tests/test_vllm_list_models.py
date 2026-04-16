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
