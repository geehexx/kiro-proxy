"""
Phase 0 gateway smoke-test — telemetry-phases.md §Phase 0

Verifies the current state of three pass-through features required before
Phase 2 (emitter) can ship. Each test documents whether the feature is
preserved, missing, or needs a fix, producing the result table:

    Feature                                  | Status
    -----------------------------------------|--------
    x-claude-code-agent-id header forwarded  | needs-fix (not forwarded)
    output_format json_schema pass-through   | needs-fix (stripped/ignored)
    gen_ai.response.model captured from SSE  | yes (record_request emits it)

Tests are intentionally written to PASS in the current codebase so they
serve as a regression baseline. When a feature is added, the corresponding
xfail marker is removed and the assertion is flipped.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Probe 1 — x-claude-code-agent-id / x-claude-code-parent-agent-id
# ---------------------------------------------------------------------------

class TestAgentIdHeaderForwarding:
    """
    Phase 0 probe: does the gateway forward x-claude-code-agent-id and
    x-claude-code-parent-agent-id from the inbound CC request upstream?

    Current state: NOT forwarded. get_kiro_headers() builds a fixed header
    dict from auth credentials only; inbound request headers are not
    threaded through to the upstream call.

    When the feature is implemented:
    1. Remove the xfail marker.
    2. Flip the assertion to `assert agent_id_forwarded`.
    """

    @pytest.mark.xfail(
        reason="agent-id header forwarding not yet implemented — Phase 0 gap",
        strict=True,
    )
    def test_agent_id_header_is_forwarded_to_upstream(self) -> None:
        """x-claude-code-agent-id must appear in the upstream request headers."""
        from kiro.utils import get_kiro_headers
        import inspect

        sig = inspect.signature(get_kiro_headers)
        param_names = list(sig.parameters.keys())

        # The feature requires a third parameter (inbound_headers or similar)
        # so the agent-id can be threaded through.
        agent_id_forwarded = "inbound_headers" in param_names or "agent_id" in param_names
        assert agent_id_forwarded, (
            "get_kiro_headers() has no inbound_headers/agent_id parameter — "
            "x-claude-code-agent-id cannot be forwarded upstream. "
            "Fix: add optional inbound_headers: dict param and copy "
            "x-claude-code-agent-id / x-claude-code-parent-agent-id through."
        )

    def test_agent_id_gap_is_documented(self) -> None:
        """Confirms the gap exists so CI stays green until the fix lands."""
        from kiro.utils import get_kiro_headers
        import inspect

        sig = inspect.signature(get_kiro_headers)
        param_names = list(sig.parameters.keys())
        assert "inbound_headers" not in param_names and "agent_id" not in param_names, (
            "get_kiro_headers() now accepts inbound headers — "
            "remove xfail on test_agent_id_header_is_forwarded_to_upstream "
            "and flip its assertion."
        )


# ---------------------------------------------------------------------------
# Probe 2 — output_format: { type: "json_schema", schema: {...} }
# ---------------------------------------------------------------------------

class TestOutputFormatJsonSchemaPassThrough:
    """
    Phase 0 probe: does the gateway preserve output_format in the request
    body when converting OpenAI → Anthropic format?

    Current state: NOT handled. converters_openai.py does not reference
    output_format. The field is silently dropped.

    When the feature is implemented:
    1. Remove the xfail marker.
    2. Flip the assertion to `assert output_format_preserved`.
    """

    @pytest.mark.xfail(
        reason="output_format json_schema pass-through not yet implemented — Phase 0 gap",
        strict=True,
    )
    def test_output_format_json_schema_is_preserved(self) -> None:
        """output_format must survive OpenAI→Anthropic conversion."""
        from kiro import converters_openai

        openai_body: dict[str, Any] = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "output_format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            },
        }

        converted = converters_openai.openai_to_anthropic(openai_body)
        output_format_preserved = (
            "output_format" in converted
            or any("json" in b for b in converted.get("betas", []))
        )
        assert output_format_preserved, (
            "output_format was stripped during OpenAI→Anthropic conversion. "
            "Fix: detect output_format.type == 'json_schema' in converters_openai "
            "and map it to the Anthropic structured-output field."
        )

    def test_output_format_gap_is_documented(self) -> None:
        """Confirms the gap exists so CI stays green until the fix lands."""
        from kiro import converters_openai
        import inspect

        src = inspect.getsource(converters_openai)
        assert "output_format" not in src, (
            "converters_openai.py now references output_format — "
            "remove xfail on test_output_format_json_schema_is_preserved "
            "and flip its assertion."
        )


# ---------------------------------------------------------------------------
# Probe 3 — gen_ai.response.model captured from message_start SSE event
# ---------------------------------------------------------------------------

class TestResponseModelCapturedFromSSE:
    """
    Phase 0 probe: does the gateway capture gen_ai.response.model from the
    upstream message_start SSE event and emit it in telemetry?

    Current state: YES — routes_anthropic.py extracts response_model from
    message_start and passes it to record_request(), which emits
    gen_ai.response.model in the Logfire span.
    """

    def test_record_request_emits_response_model_attribute(self) -> None:
        """record_request() must set gen_ai.response.model when response_model is given."""
        import kiro.telemetry as telemetry

        captured_attrs: dict[str, Any] = {}

        mock_logfire = MagicMock()

        def capture_span(name: str, **kwargs: Any) -> Any:
            captured_attrs.update(kwargs)
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=None)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        mock_logfire.span.side_effect = capture_span

        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    telemetry.record_request(
                        model="claude-sonnet-4.6",
                        stream=True,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=800,
                        status=200,
                        input_tokens=10_000,
                        output_tokens=300,
                        response_model="claude-sonnet-4.6",
                    )

        assert "gen_ai.response.model" in captured_attrs, (
            "record_request() did not emit gen_ai.response.model — "
            "check telemetry.py record_request() response_model branch."
        )
        assert captured_attrs["gen_ai.response.model"] == "claude-sonnet-4.6"

    def test_routing_mismatch_flagged_when_response_model_differs(self) -> None:
        """When upstream returns a different model, routing_mismatch must be True."""
        import kiro.telemetry as telemetry

        captured_attrs: dict[str, Any] = {}

        mock_logfire = MagicMock()

        def capture_span(name: str, **kwargs: Any) -> Any:
            captured_attrs.update(kwargs)
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=None)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        mock_logfire.span.side_effect = capture_span

        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    telemetry.record_request(
                        model="claude-opus-4.7",
                        stream=False,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=2000,
                        status=200,
                        input_tokens=20_000,
                        response_model="claude-sonnet-4.6",  # silent downgrade
                    )

        assert captured_attrs.get("kiro.gateway.routing_mismatch") is True, (
            "routing_mismatch not flagged when request model != response model."
        )

    def test_streaming_anthropic_extracts_response_model_from_message_start(self) -> None:
        """routes_anthropic._extract_stream_telemetry_from_chunk must yield
        response_model from a message_start SSE chunk."""
        from kiro.routes_anthropic import _extract_stream_telemetry_from_chunk

        message_start_data = json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_01abc",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6-20251101",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        })
        sse_chunk = f"event: message_start\ndata: {message_start_data}\n\n"

        result = _extract_stream_telemetry_from_chunk(sse_chunk)
        assert result.get("response_model") == "claude-sonnet-4.6-20251101", (
            f"response_model not extracted from message_start chunk; got: {result}"
        )


# ---------------------------------------------------------------------------
# Phase 0 result table
# ---------------------------------------------------------------------------

class TestPhase0ResultTable:
    """Prints the Phase 0 result table on every run for easy CI log scanning."""

    def test_print_result_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        table = (
            "\nPhase 0 — Gateway smoke-test result table\n"
            "==========================================\n"
            "Feature                                        | Status\n"
            "-----------------------------------------------|------------------\n"
            "x-claude-code-agent-id header forwarded        | needs-fix\n"
            "output_format json_schema pass-through         | needs-fix\n"
            "gen_ai.response.model captured from SSE        | yes (PASS)\n"
            "==========================================\n"
            "Next step: implement header forwarding + output_format conversion\n"
            "before Phase 2 (emitter) starts.\n"
        )
        print(table)
        captured = capsys.readouterr()
        assert "needs-fix" in captured.out
        assert "yes (PASS)" in captured.out
