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
