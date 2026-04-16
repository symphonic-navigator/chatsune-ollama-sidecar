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
