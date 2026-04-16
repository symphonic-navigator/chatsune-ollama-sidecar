import asyncio
from typing import AsyncIterator

import pytest

from sidecar.dispatcher import Dispatcher
from sidecar.engine import (
    EngineStreamItem,
    EngineUnavailable,
    ModelNotFound,
    StreamTerminal,
)
from sidecar.frames import (
    CancelFrame,
    GenerateChatBody,
    Message,
    ModelDescriptor,
    ReqFrame,
    StreamDelta,
    Usage,
)


class FakeEngine:
    engine_type = "ollama"

    def __init__(self, *, models=None, chat_script=None, raise_on_chat=None):
        self.models = models or []
        self.chat_script = chat_script or []
        self.raise_on_chat = raise_on_chat
        self.chat_entered = asyncio.Event()
        self.chat_cancelled = asyncio.Event()

    async def probe_version(self) -> str:
        return "test"

    async def list_models(self):
        return list(self.models)

    async def generate_chat(self, body) -> AsyncIterator[EngineStreamItem]:
        if self.raise_on_chat:
            raise self.raise_on_chat
        self.chat_entered.set()
        try:
            for item in self.chat_script:
                await asyncio.sleep(0)
                yield item
        except asyncio.CancelledError:
            self.chat_cancelled.set()
            raise

    async def aclose(self) -> None:
        pass


def _sample_model() -> ModelDescriptor:
    return ModelDescriptor(
        slug="m", display_name="M", parameter_count=None,
        context_length=4096, quantisation=None, capabilities=["text"],
        engine_family="ollama", engine_model_id="m",
    )


async def test_list_models_happy_path():
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(models=[_sample_model()]), send=sent.append)
    await d.handle(ReqFrame(id="r1", op="list_models"))
    await d.wait_idle()
    assert sent == [
        {
            "type": "res",
            "id": "r1",
            "ok": True,
            "body": {
                "models": [{
                    "slug": "m", "display_name": "M", "parameter_count": None,
                    "context_length": 4096, "quantisation": None,
                    "capabilities": ["text"], "engine_family": "ollama",
                    "engine_model_id": "m", "engine_metadata": {},
                }],
            },
        }
    ]


async def test_list_models_engine_unavailable_emits_err():
    class Boom(FakeEngine):
        async def list_models(self):
            raise EngineUnavailable("refused")

    sent: list[dict] = []
    d = Dispatcher(engine=Boom(), send=sent.append)
    await d.handle(ReqFrame(id="r1", op="list_models"))
    await d.wait_idle()
    assert len(sent) == 1
    f = sent[0]
    assert f["type"] == "err"
    assert f["code"] == "engine_unavailable"
    assert f["recoverable"] is True


async def test_generate_chat_streams_then_stream_end():
    script = [
        StreamDelta(content="Hi"),
        StreamDelta(content=" there"),
        StreamTerminal(finish_reason="stop", usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
    ]
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(chat_script=script), send=sent.append)
    body = GenerateChatBody(model_slug="m", messages=[Message(role="user", content="hi")])
    await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
    await d.wait_idle()
    assert [f["type"] for f in sent] == ["stream", "stream", "stream_end"]
    assert sent[-1]["finish_reason"] == "stop"
    assert sent[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


async def test_cancel_terminates_with_cancelled():
    engine = FakeEngine(chat_script=[])
    original_generate_chat = FakeEngine.generate_chat

    async def gen_chat(self, body):
        self.chat_entered.set()
        try:
            while True:
                await asyncio.sleep(10)
                yield StreamDelta(content="x")
        except asyncio.CancelledError:
            self.chat_cancelled.set()
            raise

    FakeEngine.generate_chat = gen_chat

    try:
        sent: list[dict] = []
        d = Dispatcher(engine=engine, send=sent.append)
        body = GenerateChatBody(model_slug="m", messages=[Message(role="user", content="hi")])
        await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
        await engine.chat_entered.wait()
        await d.handle(CancelFrame(id="r1"))
        await d.wait_idle()

        assert engine.chat_cancelled.is_set()
        assert sent[-1]["type"] == "stream_end"
        assert sent[-1]["finish_reason"] == "cancelled"
    finally:
        FakeEngine.generate_chat = original_generate_chat


async def test_cancel_unknown_id_is_silent():
    sent: list[dict] = []
    d = Dispatcher(engine=FakeEngine(), send=sent.append)
    await d.handle(CancelFrame(id="does-not-exist"))
    await d.wait_idle()
    assert sent == []


async def test_model_not_found_emits_err_then_stream_end():
    engine = FakeEngine(raise_on_chat=ModelNotFound("nope"))
    sent: list[dict] = []
    d = Dispatcher(engine=engine, send=sent.append)
    body = GenerateChatBody(model_slug="nope", messages=[Message(role="user", content="x")])
    await d.handle(ReqFrame(id="r1", op="generate_chat", body=body))
    await d.wait_idle()
    types = [f["type"] for f in sent]
    assert types == ["err", "stream_end"]
    assert sent[0]["code"] == "model_not_found"
    assert sent[0]["recoverable"] is False
    assert sent[1]["finish_reason"] == "error"
