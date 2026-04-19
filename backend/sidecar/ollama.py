"""Ollama engine adapter (SPEC §15.1).

Handles discovery (`/api/tags` + `/api/show`) and — in later tasks —
streaming chat (`/api/chat`).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, AsyncIterator

import httpx

_log = logging.getLogger(__name__)

# Opt-in payload tracing for cache-miss debugging. Enable via
# LLM_TRACE_PAYLOADS=1; shows the final /api/chat payload to Ollama.
_TRACE_PAYLOADS = os.environ.get("LLM_TRACE_PAYLOADS") == "1"

from .engine import (
    EngineBadResponse,
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
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
from ._reasoning import ThinkTagSplitter


# Model families known to emit <think>...</think> reasoning tokens.
# Conservative allowlist — SPEC §15.1.
_REASONING_FAMILIES = {
    "deepseek-r1",
    "deepseek_r1",
    "qwen3-thinking",
    "qwen3_thinking",
    "qwq",
}


class OllamaEngine:
    engine_type = "ollama"

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self._base = url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout, connect=3.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----- version probe --------------------------------------------------

    async def probe_version(self) -> str:
        try:
            r = await self._client.get("/api/version")
            r.raise_for_status()
            return r.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError):
            return "unknown"

    # ----- list_models ----------------------------------------------------

    async def list_models(self) -> list[ModelDescriptor]:
        try:
            tags = await self._client.get("/api/tags")
            tags.raise_for_status()
            tag_payload: dict[str, Any] = tags.json()
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineBadResponse(str(e)) from e

        out: list[ModelDescriptor] = []
        for raw in tag_payload.get("models", []):
            descriptor = await self._describe_one(raw)
            if descriptor is not None:
                out.append(descriptor)
        return out

    async def _describe_one(self, raw: dict[str, Any]) -> ModelDescriptor | None:
        name = raw.get("name")
        if not name:
            return None

        details = raw.get("details", {}) or {}
        family = (details.get("family") or "").lower()
        param_size_str = details.get("parameter_size")  # "8B"
        quantisation = details.get("quantization_level")

        try:
            show = await self._client.post("/api/show", json={"model": name})
            if show.status_code == 404:
                return None  # raced with a model deletion
            show.raise_for_status()
            show_body = show.json()
        except httpx.HTTPError:
            return None  # fail-closed per SPEC §6.5

        model_info: dict[str, Any] = show_body.get("model_info", {}) or {}
        context_length = _extract_context_length(model_info)
        if context_length is None:
            # SPEC §6.5: drop models without context_length.
            return None

        caps: list[ModelCapability] = ["text"]
        engine_caps = [c.lower() for c in show_body.get("capabilities", []) or []]
        if "tools" in engine_caps:
            caps.append("tool_calling")
        if _has_vision(model_info) or "vision" in engine_caps:
            caps.append("vision")
        # Ollama exposes reasoning as "thinking" in /api/show capabilities on
        # recent versions; older builds used "reasoning". The family allowlist
        # covers both older Ollama and engines that omit the flag entirely.
        if (
            family in _REASONING_FAMILIES
            or "thinking" in engine_caps
            or "reasoning" in engine_caps
        ):
            caps.append("reasoning")

        return ModelDescriptor(
            slug=name,
            display_name=_display_name(name),
            parameter_count=_parse_param_size(param_size_str),
            context_length=context_length,
            quantisation=quantisation,
            capabilities=caps,
            engine_family="ollama",
            engine_model_id=name,
            engine_metadata={
                "family": family or None,
                "digest": raw.get("digest"),
                "size_bytes": raw.get("size"),
            },
        )

    # ----- generate_chat --------------------------------------------------

    async def generate_chat(
        self, body: GenerateChatBody
    ) -> AsyncIterator[EngineStreamItem]:
        payload = self._build_chat_payload(body)
        if _TRACE_PAYLOADS:
            _log.info(
                "LLM_TRACE path=sidecar-ollama payload=%s",
                json.dumps(payload, default=str, sort_keys=True),
            )
        reasoning_on = body.options.reasoning

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(None, connect=3.0),
            ) as resp:
                if resp.status_code == 404:
                    raise ModelNotFound(body.model_slug)
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    raise EngineBadResponse(f"ollama {resp.status_code}: {text}")

                parser = ThinkTagSplitter(reasoning_on=reasoning_on)

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        raise EngineBadResponse("non-JSON line from /api/chat")

                    if chunk.get("done"):
                        yield _build_terminal(chunk)
                        return

                    message = chunk.get("message") or {}

                    for frag in _tool_call_fragments(message.get("tool_calls")):
                        yield StreamDelta(tool_calls=[frag])

                    # Modern Ollama (think: true) emits thinking on a
                    # dedicated field; older builds inline it as
                    # <think>...</think> in content (handled by the parser).
                    thinking = message.get("thinking") or ""
                    if thinking and reasoning_on:
                        yield StreamDelta(reasoning=thinking)

                    content = message.get("content") or ""
                    if content:
                        for delta in parser.feed(content):
                            yield delta
        except httpx.ConnectError as e:
            raise EngineUnavailable(str(e)) from e
        except httpx.HTTPError as e:
            raise EngineUnavailable(str(e)) from e

    # ----- payload shaping -------------------------------------------------

    def _build_chat_payload(self, body: GenerateChatBody) -> dict[str, Any]:
        messages = [_message_to_ollama(m) for m in body.messages]
        options: dict[str, Any] = {}
        p = body.parameters
        if p.temperature is not None:
            options["temperature"] = p.temperature
        if p.top_p is not None:
            options["top_p"] = p.top_p
        if p.max_tokens is not None:
            options["num_predict"] = p.max_tokens
        if p.stop is not None:
            options["stop"] = p.stop

        payload: dict[str, Any] = {
            "model": body.model_slug,
            "messages": messages,
            "stream": True,
        }
        # Ollama's top-level `think` flag enables the dedicated thinking
        # channel on capable models. Only set it when truly requested so
        # older Ollama versions (which don't recognise the parameter)
        # don't see a no-op key.
        if body.options.reasoning:
            payload["think"] = True
        if options:
            payload["options"] = options
        if body.tools is not None:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
        return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_KEY_RE = re.compile(r"\.context_length$")


def _extract_context_length(model_info: dict[str, Any]) -> int | None:
    """Ollama reports `<arch>.context_length` — scan any matching key."""
    for key, value in model_info.items():
        if _CONTEXT_KEY_RE.search(key) and isinstance(value, (int, float)):
            return int(value)
    return None


def _has_vision(model_info: dict[str, Any]) -> bool:
    for key, value in model_info.items():
        if key.endswith(".vision") and value:
            return True
    return False


_PARAM_SIZE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<mag>[BMbm])$")


def _parse_param_size(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    m = _PARAM_SIZE_RE.match(raw.strip())
    if not m:
        return None
    num = float(m.group("num"))
    mag = m.group("mag").upper()
    multiplier = 1_000_000_000 if mag == "B" else 1_000_000
    return int(num * multiplier)


def _display_name(slug: str) -> str:
    try:
        base, tag = slug.split(":", 1)
        return f"{base.title()} {tag.upper()}"
    except ValueError:
        return slug


# ---------------------------------------------------------------------------
# Chat translation helpers
# ---------------------------------------------------------------------------

def _message_to_ollama(m: Any) -> dict[str, Any]:
    """Map our Message to the Ollama /api/chat schema."""
    role = m.role
    content = m.content

    out: dict[str, Any] = {"role": role}
    images: list[str] = []

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if part.type == "text":
                text_parts.append(part.text)
            elif part.type == "image":
                images.append(part.data_b64)
        out["content"] = "".join(text_parts)
        if images:
            out["images"] = images
    else:
        out["content"] = content or ""

    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    # Ollama expects arguments as a dict in message history;
                    # CSP carries them as a JSON string. Parse back if needed.
                    "arguments": _args_to_dict(tc.function.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _args_to_dict(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args  # leave as string; Ollama may cope or error clearly
    return args if args is not None else {}


_OLLAMA_FINISH_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "load": "stop",
    "unload": "stop",
}


def _build_terminal(chunk: dict[str, Any]) -> StreamTerminal:
    reason = _OLLAMA_FINISH_MAP.get(chunk.get("done_reason", "stop"), "stop")
    prompt = chunk.get("prompt_eval_count")
    complete = chunk.get("eval_count")
    usage: Usage | None = None
    if isinstance(prompt, int) and isinstance(complete, int):
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=complete,
            total_tokens=prompt + complete,
        )
    return StreamTerminal(finish_reason=reason, usage=usage)


def _tool_call_fragments(raw: Any) -> list[ToolCallFragment]:
    if not isinstance(raw, list):
        return []
    out: list[ToolCallFragment] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        fn = (tc.get("function") or {})
        # Ollama emits arguments as a dict; SPEC §8.2 requires a JSON string
        # on the wire so fragments can be concatenated by consumers.
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, separators=(",", ":"))
        elif args is not None and not isinstance(args, str):
            args = str(args)
        out.append(
            ToolCallFragment(
                index=i,
                id=tc.get("id"),
                type=tc.get("type"),
                function=ToolCallFragmentFunction(
                    name=fn.get("name"),
                    arguments=args,
                ),
            )
        )
    return out

