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

    Null-semantics note: A field explicitly set to `null` in the overlay
    is treated as "do not override" — the base value survives. To clear a
    list, write an empty sequence (`capabilities: []`). This keeps the
    merge predictable when YAML files are hand-edited and a blank value
    might otherwise be mistaken for "remove".

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
