"""Unit tests for kiro/telemetry.py — covers the logfire-disabled path."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import kiro.telemetry as telemetry
from kiro.telemetry import (
    _cost_usd,
    _trunc,
    emit_cache_event,
    emit_upstream_call,
    gateway_request_span,
    instrument_fastapi,
    record_model_resolution,
    record_request,
    setup_logfire,
    user_request_span,
)


class TestTrunc:
    def test_none_returns_empty(self):
        assert _trunc(None) == ""

    def test_short_string_unchanged(self):
        assert _trunc("hello") == "hello"

    def test_long_string_truncated(self):
        s = "x" * 300
        result = _trunc(s, max_chars=200)
        assert len(result) < 300
        assert "…[+" in result

    def test_exact_length_unchanged(self):
        s = "x" * 200
        assert _trunc(s, max_chars=200) == s

    def test_non_string_converted(self):
        assert _trunc(42) == "42"
        assert _trunc(3.14) == "3.14"

    def test_custom_max_chars(self):
        result = _trunc("hello world", max_chars=5)
        assert result.startswith("hello")
        assert "…[+" in result


class TestCostUsd:
    def test_flat_rate(self):
        cost = _cost_usd(is_overage=False)
        assert cost > 0
        assert cost < 1.0  # sanity: less than $1 per invocation

    def test_overage_rate_higher(self):
        flat = _cost_usd(is_overage=False)
        overage = _cost_usd(is_overage=True)
        assert overage > flat


class TestSetupLogfire:
    def test_returns_false_when_logfire_unavailable(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", False):
            with patch.object(telemetry, "_configured", False):
                result = setup_logfire()
        assert result is False

    def test_returns_false_in_test_env(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
            with patch.object(telemetry, "_configured", False):
                with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_foo"}):
                    result = setup_logfire()
        assert result is False

    def test_returns_false_without_token(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
            with patch.object(telemetry, "_configured", False):
                env = {k: v for k, v in os.environ.items() if k != "LOGFIRE_TOKEN"}
                env.pop("PYTEST_CURRENT_TEST", None)
                env.pop("CI", None)
                with patch.dict(os.environ, env, clear=True):
                    result = setup_logfire()
        assert result is False

    def test_returns_true_if_already_configured(self):
        with patch.object(telemetry, "_configured", True):
            result = setup_logfire()
        assert result is True


class TestRecordRequest:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            # Should not raise
            record_request(
                model="claude-sonnet-4.6",
                stream=True,
                gateway_cache="miss",
                re2_applied=False,
                upstream_ms=1000,
                status=200,
                input_tokens=5000,
            )

    def test_no_op_when_logfire_unavailable(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", False):
            record_request(
                model="claude-sonnet-4.6",
                stream=False,
                gateway_cache="hit",
                re2_applied=True,
                upstream_ms=None,
                status=200,
            )

    def test_skips_low_token_requests(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-haiku-4.5",
                        stream=False,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=100,
                        status=200,
                        input_tokens=10,  # below LOGFIRE_MIN_INPUT_TOKENS
                    )
        mock_logfire.span.assert_not_called()

    def test_emits_span_for_errors_regardless_of_tokens(self):
        mock_logfire = MagicMock()
        mock_span = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-sonnet-4.6",
                        stream=False,
                        gateway_cache="bypass",
                        re2_applied=False,
                        upstream_ms=500,
                        status=429,
                        input_tokens=10,
                        error_reason="INSUFFICIENT_MODEL_CAPACITY",
                    )
        mock_logfire.span.assert_called_once()


class TestRecordModelResolution:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            record_model_resolution(
                raw_model="claude-sonnet-4-6",
                resolved_model="claude-sonnet-4.6",
                resolution_source="alias",
            )

    def test_no_op_when_same_model(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_model_resolution(
                        raw_model="claude-sonnet-4.6",
                        resolved_model="claude-sonnet-4.6",
                        resolution_source="direct",
                    )
        mock_logfire.info.assert_not_called()

    def test_emits_when_models_differ(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_model_resolution(
                        raw_model="claude-sonnet-4-6",
                        resolved_model="claude-sonnet-4.6",
                        resolution_source="normalize",
                    )
        mock_logfire.info.assert_called_once()


class TestUserRequestSpan:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            with user_request_span(model="m", stream=True, message_count=2):
                pass

    def test_no_op_when_logfire_unavailable(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", False):
            with user_request_span(model="m", stream=True, message_count=2):
                pass

    def test_creates_span_with_optional_attrs(self):
        mock_logfire = MagicMock()
        # span() must return a context manager
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    with user_request_span(
                        model="claude-opus-4.7",
                        stream=True,
                        message_count=5,
                        session_id="abc-very-long-session-id-truncated",
                        last_user_message_preview="hello world",
                    ):
                        pass
        mock_logfire.span.assert_called_once()
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["gen_ai.request.model"] == "claude-opus-4.7"
        assert attrs["kiro.request.message_count"] == 5
        assert attrs["kiro.session.id"] == "abc-very-long-se"  # 16-char clamp
        assert "hello world" in attrs["kiro.request.prompt_preview"]

    def test_swallows_inner_exception(self):
        """If logfire.span itself raises, the context manager still yields."""
        mock_logfire = MagicMock()
        mock_logfire.span.side_effect = RuntimeError("logfire blew up")
        executed = []
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    with user_request_span(model="m", stream=False, message_count=1):
                        executed.append(True)
        assert executed == [True]


class TestGatewayRequestSpan:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            with gateway_request_span(
                model="m",
                stream=True,
                re2_applied=False,
                cache_result="miss",
            ):
                pass

    def test_emits_span_with_overage_flag(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    with gateway_request_span(
                        model="claude-sonnet-4.6",
                        stream=False,
                        re2_applied=True,
                        cache_result="hit",
                        upstream_ms=None,
                        status=200,
                        input_tokens=1000,
                        output_tokens=200,
                        is_overage=True,
                    ):
                        pass
        mock_logfire.span.assert_called_once()
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["kiro.gateway.cache"] == "hit"
        assert attrs["kiro.cost.is_overage"] is True

    def test_includes_error_reason_when_set(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    with gateway_request_span(
                        model="claude-opus-4.7",
                        stream=True,
                        re2_applied=False,
                        cache_result="bypass",
                        status=429,
                        error_reason="INSUFFICIENT_MODEL_CAPACITY",
                        retry_count=3,
                    ):
                        pass
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["kiro.gateway.status"] == 429
        assert "CAPACITY" in attrs["kiro.gateway.error_reason"]
        assert attrs["kiro.gateway.retry_count"] == 3

    def test_swallows_inner_exception(self):
        mock_logfire = MagicMock()
        mock_logfire.span.side_effect = RuntimeError("oops")
        executed = []
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    with gateway_request_span(
                        model="m", stream=True, re2_applied=False, cache_result="miss"
                    ):
                        executed.append(True)
        assert executed == [True]


class TestEmitCacheEvent:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            emit_cache_event(event="hit", model="m")

    def test_no_op_when_logfire_unavailable(self):
        with patch.object(telemetry, "_LOGFIRE_AVAILABLE", False):
            emit_cache_event(event="miss", model="m")

    def test_emits_log_event(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    emit_cache_event(event="hit", model="claude-opus-4.7", cache_key_prefix="abc12345")
        mock_logfire.info.assert_called_once()
        kwargs = mock_logfire.info.call_args.kwargs
        assert kwargs["kiro.cache.event"] == "hit"
        assert kwargs["kiro.cache.key"] == "abc12345"

    def test_swallows_internal_exception(self):
        mock_logfire = MagicMock()
        mock_logfire.info.side_effect = RuntimeError("nope")
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    # Must not raise
                    emit_cache_event(event="bypass", model="m")


class TestEmitUpstreamCall:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            emit_upstream_call(model="m", upstream_ms=100, status=200)

    def test_emits_warning_for_4xx(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    emit_upstream_call(
                        model="claude-haiku-4.5",
                        upstream_ms=200,
                        status=429,
                        error_reason="rate_limit",
                    )
        # warning getattr branch was used
        assert mock_logfire.warning.called or mock_logfire.info.called

    def test_emits_info_for_2xx(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    emit_upstream_call(model="m", upstream_ms=300, status=200)
        assert mock_logfire.info.called

    def test_swallows_internal_exception(self):
        mock_logfire = MagicMock()
        mock_logfire.info.side_effect = RuntimeError("disk")
        mock_logfire.warning.side_effect = RuntimeError("disk")
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    emit_upstream_call(model="m", upstream_ms=10, status=200)


class TestInstrumentFastapi:
    def test_no_op_when_not_configured(self):
        with patch.object(telemetry, "_configured", False):
            instrument_fastapi(MagicMock())  # must not raise

    def test_calls_instrumentation_when_configured(self):
        mock_logfire = MagicMock()
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    instrument_fastapi(MagicMock())
        assert mock_logfire.instrument_fastapi.called
        assert mock_logfire.instrument_httpx.called

    def test_swallows_instrumentation_failure(self):
        mock_logfire = MagicMock()
        mock_logfire.instrument_fastapi.side_effect = RuntimeError("nope")
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    instrument_fastapi(MagicMock())  # must not raise


class TestRecordRequestNewFields:
    """Phase 1.5 — response_model + routing_mismatch attributes."""

    def test_response_model_emitted(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-opus-4.7",
                        stream=True,
                        gateway_cache="miss",
                        re2_applied=True,
                        upstream_ms=1500,
                        status=200,
                        input_tokens=20_000,
                        output_tokens=400,
                        response_model="claude-opus-4.7",
                    )
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["gen_ai.response.model"] == "claude-opus-4.7"
        assert "kiro.gateway.routing_mismatch" not in attrs

    def test_routing_mismatch_flag_set(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-opus-4.7",
                        stream=False,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=2000,
                        status=200,
                        input_tokens=5000,
                        response_model="claude-sonnet-4.5",  # silent downgrade
                    )
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["gen_ai.response.model"] == "claude-sonnet-4.5"
        assert attrs["kiro.gateway.routing_mismatch"] is True

    def test_dedup_hit_attribute(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-haiku-4.5",
                        stream=False,
                        gateway_cache="bypass",
                        re2_applied=False,
                        upstream_ms=100,
                        status=200,
                        input_tokens=500,
                        dedup_hit=True,
                    )
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["kiro.gateway.dedup_hit"] is True

    def test_complexity_label_attribute(self):
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=None)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    record_request(
                        model="claude-sonnet-4.6",
                        stream=True,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=800,
                        status=200,
                        input_tokens=10_000,
                        complexity_label="medium",
                        session_id="0123456789abcdefghij",
                    )
        attrs = mock_logfire.span.call_args.kwargs
        assert attrs["kiro.gateway.complexity_label"] == "medium"
        # session_id clamped to 16 chars
        assert attrs["kiro.conversation.id"] == "0123456789abcdef"

    def test_record_request_swallows_exception(self):
        """If span construction raises, record_request returns silently."""
        mock_logfire = MagicMock()
        mock_logfire.span.side_effect = RuntimeError("logfire blew up")
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    # must not raise
                    record_request(
                        model="m",
                        stream=False,
                        gateway_cache="miss",
                        re2_applied=False,
                        upstream_ms=100,
                        status=200,
                        input_tokens=5000,
                    )

    def test_record_model_resolution_swallows_exception(self):
        mock_logfire = MagicMock()
        mock_logfire.info.side_effect = RuntimeError("disk full")
        with patch.object(telemetry, "_configured", True):
            with patch.object(telemetry, "_LOGFIRE_AVAILABLE", True):
                with patch.object(telemetry, "_logfire", mock_logfire):
                    # must not raise
                    record_model_resolution(
                        raw_model="x", resolved_model="y", resolution_source="alias"
                    )
