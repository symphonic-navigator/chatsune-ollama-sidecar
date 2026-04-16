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
