"""vLLM engine adapter (SPEC §15.3).

vLLM exposes an OpenAI-compatible API:
  - `GET /version` for the version probe,
  - `GET /v1/models` for discovery,
  - `POST /v1/chat/completions` (with `stream: true`) for chat.

Per-model metadata that vLLM does not self-report (vision, tool calling,
reasoning capability, display name, parameter count, quantisation) comes
from the operator via the YAML layer in `vllm_models_config.py`.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ._reasoning import ThinkTagSplitter
from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
    ModelOutOfMemory,
    StreamTerminal,
)
from .frames import (
    GenerateChatBody,
    ModelCapability,
    ModelDescriptor,
    StreamDelta,
    ToolCallFragment,
    ToolCallFragmentFunction,
    Usage,
)
from .logging_setup import get_logger
from .vllm_models_config import VllmModelMetadata


log = get_logger("vllm")


class VllmEngine:
    engine_type = "vllm"

    def __init__(
        self,
        url: str,
        *,
        metadata: dict[str, VllmModelMetadata],
        timeout: float = 10.0,
    ) -> None:
        self._base = url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )
        self._metadata = metadata
        self._warned_unknown: set[str] = set()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/version")
            r.raise_for_status()
            v = r.json().get("version")
            return v if isinstance(v, str) else "unknown"
        except (httpx.HTTPError, ValueError):
            return "unknown"

    async def list_models(self) -> list[ModelDescriptor]:
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            payload: dict[str, Any] = r.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e
        except ValueError as e:
            raise EngineBadResponse(f"non-JSON /v1/models body: {e}") from e

        out: list[ModelDescriptor] = []
        for raw in payload.get("data") or []:
            descriptor = self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        served_id = raw.get("id")
        if not isinstance(served_id, str) or not served_id:
            return None

        context_length = raw.get("max_model_len")
        if not isinstance(context_length, int):
            log.warning(
                "vllm.model_dropped_no_context",
                id=served_id,
                reason="max_model_len missing or not an integer",
            )
            return None

        meta = self._metadata.get(served_id)
        if meta is None:
            if served_id not in self._warned_unknown:
                log.warning(
                    "vllm.model_without_metadata",
                    id=served_id,
                    message=(
                        f"No YAML metadata for model '{served_id}'. "
                        "Listing with capabilities=['text'] only."
                    ),
                )
                self._warned_unknown.add(served_id)
            capabilities: list[ModelCapability] = ["text"]
            display_name = served_id
            parameter_count: int | None = None
            quantisation: str | None = None
        else:
            capabilities = list(meta.capabilities) if meta.capabilities else ["text"]
            display_name = meta.display_name or served_id
            parameter_count = meta.parameter_count
            quantisation = meta.quantisation

        engine_metadata: dict[str, Any] = {}
        for key in ("owned_by", "root"):
            value = raw.get(key)
            if value is not None:
                engine_metadata[key] = value

        return ModelDescriptor(
            slug=served_id,
            display_name=display_name,
            parameter_count=parameter_count,
            context_length=context_length,
            quantisation=quantisation,
            capabilities=capabilities,
            engine_family="vllm",
            engine_model_id=served_id,
            engine_metadata=engine_metadata,
        )

    async def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        payload = self._build_chat_payload(body)
        reasoning_on = body.options.reasoning

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(None, connect=3.0),
            ) as resp:
                if resp.status_code == 404:
                    raise ModelNotFound(body.model_slug)
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    if _is_oom(text):
                        raise ModelOutOfMemory(text)
                    raise EngineBadResponse(f"vllm {resp.status_code}: {text}")

                splitter = ThinkTagSplitter(reasoning_on=reasoning_on)
                terminal: StreamTerminal | None = None

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        raise EngineBadResponse(
                            f"non-JSON SSE line from /v1/chat/completions: {data[:80]}"
                        )

                    # Terminal-usage-only chunk: choices=[], usage=...
                    usage_block = chunk.get("usage")
                    choices = chunk.get("choices") or []
                    if not choices and usage_block:
                        if terminal is not None:
                            terminal = StreamTerminal(
                                finish_reason=terminal.finish_reason,
                                usage=_usage_from_block(usage_block),
                            )
                        continue

                    for choice in choices:
                        delta = choice.get("delta") or {}

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            for d in splitter.feed(content):
                                yield d

                        reasoning_content = delta.get("reasoning_content")
                        if (
                            isinstance(reasoning_content, str)
                            and reasoning_content
                            and reasoning_on
                        ):
                            yield StreamDelta(reasoning=reasoning_content)

                        raw_tcs = delta.get("tool_calls")
                        frags = _tool_call_fragments(raw_tcs)
                        if frags:
                            yield StreamDelta(tool_calls=frags)

                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            terminal = StreamTerminal(
                                finish_reason=_map_finish_reason(finish_reason),
                                usage=(
                                    _usage_from_block(usage_block)
                                    if usage_block else None
                                ),
                            )

                if terminal is None:
                    terminal = StreamTerminal(finish_reason="stop")
                yield terminal

        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineUnavailable(str(e)) from e

    def _build_chat_payload(self, body: GenerateChatBody) -> dict[str, Any]:
        messages = [_message_to_openai(m) for m in body.messages]
        payload: dict[str, Any] = {
            "model": body.model_slug,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        p = body.parameters
        if p.temperature is not None:
            payload["temperature"] = p.temperature
        if p.top_p is not None:
            payload["top_p"] = p.top_p
        if p.max_tokens is not None:
            payload["max_tokens"] = p.max_tokens
        if p.stop is not None:
            payload["stop"] = p.stop
        if body.tools is not None:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
        return payload


_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
}


def _map_finish_reason(raw: str) -> str:
    return _FINISH_REASON_MAP.get(raw, "stop")


def _usage_from_block(block: dict[str, Any]) -> Usage | None:
    prompt = block.get("prompt_tokens")
    completion = block.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
    return None


def _message_to_openai(m: Any) -> dict[str, Any]:
    """Map a CSP Message to the OpenAI chat-completions schema.

    Text-only content passes through as a string. Multimodal (list) content
    becomes an array of parts with `image_url` data-URI encoding for images.
    Tool calls and tool_call_id pass through unchanged — vLLM is
    OpenAI-native there.
    """
    out: dict[str, Any] = {"role": m.role}
    content = m.content
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.type == "text":
                parts.append({"type": "text", "text": part.text})
            elif part.type == "image":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part.media_type};base64,{part.data_b64}",
                    },
                })
        out["content"] = parts
    else:
        out["content"] = content or ""

    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _tool_call_fragments(raw: Any) -> list[ToolCallFragment]:
    if not isinstance(raw, list):
        return []
    out: list[ToolCallFragment] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        index = tc.get("index")
        if not isinstance(index, int):
            continue
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is not None and not isinstance(args, str):
            # OpenAI-native payloads always use a string; be defensive.
            args = json.dumps(args, separators=(",", ":"))
        out.append(
            ToolCallFragment(
                index=index,
                id=tc.get("id"),
                type=tc.get("type"),
                function=ToolCallFragmentFunction(
                    name=fn.get("name"),
                    arguments=args,
                ),
            )
        )
    return out


def _is_oom(text: str) -> bool:
    needle = text.lower()
    return "out of memory" in needle or ("cuda" in needle and "memory" in needle)
