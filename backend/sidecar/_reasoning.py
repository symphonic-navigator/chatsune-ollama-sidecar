"""Shared reasoning-channel utilities.

The `<think>…</think>` splitter lives here because both the Ollama
adapter (legacy inline format) and the vLLM adapter (when the server
runs without `--reasoning-parser`) need the same chunk-boundary-safe
parser.
"""
from __future__ import annotations

from .frames import StreamDelta


class ThinkTagSplitter:
    """Incrementally split content on `<think>…</think>` tags.

    Produces StreamDelta objects — either `content=...` (outside tags) or
    `reasoning=...` (inside). When `reasoning_on` is False, the inner text
    is dropped.

    Resilient to a tag being chopped across chunk boundaries.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, reasoning_on: bool) -> None:
        self._reasoning_on = reasoning_on
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> list[StreamDelta]:
        self._buf += chunk
        out: list[StreamDelta] = []

        while True:
            if not self._inside:
                idx = self._buf.find(self.OPEN)
                if idx == -1:
                    flush, hold = split_for_partial(self._buf, self.OPEN)
                    if flush:
                        out.append(StreamDelta(content=flush))
                    self._buf = hold
                    return out
                if idx > 0:
                    out.append(StreamDelta(content=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.OPEN):]
                self._inside = True
            else:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    flush, hold = split_for_partial(self._buf, self.CLOSE)
                    if flush and self._reasoning_on:
                        out.append(StreamDelta(reasoning=flush))
                    self._buf = hold
                    return out
                if idx > 0 and self._reasoning_on:
                    out.append(StreamDelta(reasoning=self._buf[:idx]))
                self._buf = self._buf[idx + len(self.CLOSE):]
                self._inside = False


def split_for_partial(buf: str, needle: str) -> tuple[str, str]:
    """Hold back any suffix of `buf` that could be the start of `needle`."""
    max_hold = len(needle) - 1
    for hold in range(max_hold, 0, -1):
        if needle.startswith(buf[-hold:]):
            return buf[:-hold], buf[-hold:]
    return buf, ""
