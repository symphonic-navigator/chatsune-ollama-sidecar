import json
import logging
from io import StringIO

from sidecar.logging_setup import configure_logging, get_logger


def test_emits_json_lines(capsys):
    configure_logging("info")
    log = get_logger("test")
    log.info("hello_event", foo="bar", id="abc-123")
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    record = json.loads(captured[0])
    assert record["event"] == "hello_event"
    assert record["level"] == "info"
    assert record["foo"] == "bar"
    assert record["id"] == "abc-123"
    assert "ts" in record


def test_respects_level(capsys):
    configure_logging("warn")
    log = get_logger("test")
    log.info("should_not_appear")
    log.warning("should_appear")
    out = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line)["event"] for line in out]
    assert "should_not_appear" not in events
    assert "should_appear" in events


def test_warn_alias_maps_to_warning(capsys):
    configure_logging("warn")
    log = get_logger("test")
    log.warning("via_warn")
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[0])["level"] == "warning"


def test_exception_traceback_rendered_in_output(capsys):
    """log.exception() must surface a full traceback in the `exception` key."""
    configure_logging("debug")
    log = get_logger("test")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("unexpected_failure", id="abc")
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    record = json.loads(out[0])
    assert record["event"] == "unexpected_failure"
    assert record["id"] == "abc"
    assert "exception" in record
    assert "RuntimeError: boom" in record["exception"]
    assert "Traceback" in record["exception"]
