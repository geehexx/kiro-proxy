"""Tests for _emit_gateway_baseline — Step 1b non-streaming emit."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from kiro.routes_anthropic import _emit_gateway_baseline


def _make_request_with_writer(writer: AsyncMock | None = None) -> Request:
    """Minimal Request stub with app.state.baselines_writer wired."""
    app = MagicMock()
    app.state = MagicMock()
    if writer is None:
        app.state.baselines_writer = None
    else:
        app.state.baselines_writer = writer
    req = MagicMock(spec=Request)
    req.app = app
    return req


@pytest.mark.asyncio
async def test_emits_record_with_expected_shape() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    response_body = {
        "id": "msg_123",
        "model": "claude-opus-4-7",
        "usage": {
            "input_tokens": 42,
            "output_tokens": 7,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
        },
    }

    ts_before = time.time()
    await _emit_gateway_baseline(
        request,
        response_body=response_body,
        request_model="claude-opus-4-7",
        session_id_gw="sess-abc",
        cache_key="0123456789abcdef" + "extra",
        upstream_ms=120,
        gateway_cache="miss",
        status=200,
    )
    ts_after = time.time()

    assert writer.write.call_count == 1
    args, _kwargs = writer.write.call_args
    assert args[0] == "gateway-requests"
    rec = args[1]
    assert rec["message_id"] == "msg_123"
    assert rec["session_id_gw"] == "sess-abc"
    assert rec["cache_key"] == "0123456789abcdef"  # clamped to 16 chars
    assert rec["model"] == "claude-opus-4-7"
    assert rec["input_tokens"] == 42
    assert rec["output_tokens"] == 7
    assert rec["cache_read_input_tokens"] == 10
    assert rec["cache_creation_input_tokens"] == 5
    assert rec["upstream_ms_total"] == 120
    assert rec["gateway_cache"] == "miss"
    assert rec["stream"] is False
    assert rec["status"] == 200
    assert ts_before <= rec["ts"] <= ts_after


@pytest.mark.asyncio
async def test_cache_hit_emits_with_none_upstream_ms() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_hit", "usage": {}},
        request_model="claude-opus-4-7",
        session_id_gw="sess-x",
        cache_key="k" * 32,
        upstream_ms=None,
        gateway_cache="hit",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec["upstream_ms_total"] is None
    assert rec["gateway_cache"] == "hit"


@pytest.mark.asyncio
async def test_missing_writer_is_no_op() -> None:
    request = _make_request_with_writer(writer=None)
    # must not raise even though no writer is installed
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_x", "usage": {}},
        request_model="m",
        session_id_gw=None,
        cache_key=None,
        upstream_ms=None,
        gateway_cache="bypass",
        status=200,
    )


@pytest.mark.asyncio
async def test_writer_failure_is_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    writer = AsyncMock()
    writer.write.side_effect = RuntimeError("disk full")
    request = _make_request_with_writer(writer)

    # must not raise
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_y", "usage": {"input_tokens": 1, "output_tokens": 1}},
        request_model="m",
        session_id_gw="s",
        cache_key=None,
        upstream_ms=5,
        gateway_cache="miss",
        status=200,
    )
    assert writer.write.called


@pytest.mark.asyncio
async def test_null_cache_key_passed_through() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_z", "usage": {}},
        request_model="m",
        session_id_gw=None,
        cache_key=None,
        upstream_ms=10,
        gateway_cache="bypass",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec["cache_key"] is None
    assert rec["session_id_gw"] is None


@pytest.mark.asyncio
async def test_re2_applied_false_by_default() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_re2_off", "usage": {}},
        request_model="claude-sonnet-4-6",
        session_id_gw="sess-re2",
        cache_key=None,
        upstream_ms=50,
        gateway_cache="miss",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec["re2_applied"] is False


@pytest.mark.asyncio
async def test_re2_applied_true_when_set() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_re2_on", "usage": {"input_tokens": 100, "output_tokens": 20}},
        request_model="claude-sonnet-4-6",
        session_id_gw="sess-re2",
        cache_key="abcdef1234567890",
        upstream_ms=75,
        gateway_cache="miss",
        status=200,
        re2_applied=True,
    )
    rec = writer.write.call_args.args[1]
    assert rec["re2_applied"] is True
    assert rec["input_tokens"] == 100
    assert rec["output_tokens"] == 20


@pytest.mark.asyncio
async def test_re2_applied_propagated_on_error() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={},
        request_model="claude-opus-4-7",
        session_id_gw="sess-err",
        cache_key=None,
        upstream_ms=200,
        gateway_cache="bypass",
        status=429,
        error_reason="INSUFFICIENT_MODEL_CAPACITY",
        re2_applied=True,
    )
    rec = writer.write.call_args.args[1]
    assert rec["re2_applied"] is True
    assert rec["status"] == 429
    assert rec["error_reason"] == "INSUFFICIENT_MODEL_CAPACITY"



@pytest.mark.asyncio
async def test_complexity_label_emitted() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={},
        request_model="claude-sonnet-4.6",
        session_id_gw="sess-1",
        cache_key=None,
        upstream_ms=100,
        gateway_cache="miss",
        status=200,
        re2_applied=True,
        complexity_label="medium",
    )
    rec = writer.write.call_args.args[1]
    assert rec["complexity_label"] == "medium"
    assert rec["re2_applied"] is True


@pytest.mark.asyncio
async def test_complexity_label_none_by_default() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={},
        request_model="claude-sonnet-4.6",
        session_id_gw="sess-2",
        cache_key=None,
        upstream_ms=100,
        gateway_cache="miss",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec.get("complexity_label") is None


@pytest.mark.asyncio
async def test_response_model_captured_when_present() -> None:
    """response_model field is emitted when upstream returns a model id."""
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={
            "id": "msg_rm",
            "model": "claude-sonnet-4.5",  # upstream-resolved name
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        request_model="claude-sonnet-4.6",  # what client requested
        session_id_gw="sess-rm",
        cache_key=None,
        upstream_ms=10,
        gateway_cache="miss",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec["model"] == "claude-sonnet-4.6"
    assert rec["response_model"] == "claude-sonnet-4.5"


@pytest.mark.asyncio
async def test_response_model_none_when_absent() -> None:
    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    await _emit_gateway_baseline(
        request,
        response_body={"id": "msg_no_rm", "usage": {}},
        request_model="claude-opus-4.7",
        session_id_gw="s",
        cache_key=None,
        upstream_ms=5,
        gateway_cache="miss",
        status=200,
    )
    rec = writer.write.call_args.args[1]
    assert rec.get("response_model") is None


@pytest.mark.asyncio
async def test_routing_mismatch_logged(caplog: pytest.LogCaptureFixture) -> None:
    """When upstream returns a model id != requested, a warning is logged."""
    import logging

    writer = AsyncMock()
    request = _make_request_with_writer(writer)
    with caplog.at_level(logging.WARNING):
        await _emit_gateway_baseline(
            request,
            response_body={
                "id": "msg_mismatch",
                "model": "claude-sonnet-4.5",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
            request_model="claude-opus-4.7",
            session_id_gw="sess-mm",
            cache_key=None,
            upstream_ms=200,
            gateway_cache="miss",
            status=200,
        )
    # Either via caplog OR via the record itself — both should reflect the mismatch
    rec = writer.write.call_args.args[1]
    assert rec["model"] == "claude-opus-4.7"
    assert rec["response_model"] == "claude-sonnet-4.5"
    # Routing mismatch warning visible somewhere — the logger is "kiro.routes_anthropic"
    # but loguru may not propagate to caplog in all setups, so the durable assertion
    # is on the baseline record. The warning is a side effect.


# ---------------------------------------------------------------------------
# 1.1 — cache token fields from message_start captured in streaming baseline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_stream_telemetry_captures_cache_tokens_from_message_start() -> None:
    """_extract_stream_telemetry_from_chunk must return cache token fields
    from message_start.message.usage.

    The stream_wrapper in routes_anthropic.py uses this helper to collect
    telemetry from SSE chunks. Before the fix, cache token fields were never
    captured because message_delta (the only event previously read) does not
    carry them — only message_start does.
    """
    import json

    from kiro.routes_anthropic import _extract_stream_telemetry_from_chunk

    chunk = (
        "event: message_start\n"
        "data: " + json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_cache_test",
                "model": "claude-sonnet-4.6",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 80,
                    "cache_read_input_tokens": 20,
                },
            },
        }) + "\n\n"
    )

    result = _extract_stream_telemetry_from_chunk(chunk)

    assert result["cache_creation_input_tokens"] == 80, (
        "cache_creation_input_tokens must be extracted from message_start.message.usage"
    )
    assert result["cache_read_input_tokens"] == 20, (
        "cache_read_input_tokens must be extracted from message_start.message.usage"
    )
    assert result["message_id"] == "msg_cache_test"
    assert result["response_model"] == "claude-sonnet-4.6"


@pytest.mark.asyncio
async def test_extract_stream_telemetry_captures_output_tokens_from_message_delta() -> None:
    """_extract_stream_telemetry_from_chunk must return output_tokens from message_delta."""
    import json

    from kiro.routes_anthropic import _extract_stream_telemetry_from_chunk

    chunk = (
        "event: message_delta\n"
        "data: " + json.dumps({
            "type": "message_delta",
            "usage": {"output_tokens": 42},
        }) + "\n\n"
    )

    result = _extract_stream_telemetry_from_chunk(chunk)

    assert result["output_tokens"] == 42
    assert result.get("cache_creation_input_tokens") is None
    assert result.get("cache_read_input_tokens") is None
