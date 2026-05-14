"""Unit tests for kiro/telemetry.py — covers the logfire-disabled path."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import kiro.telemetry as telemetry
from kiro.telemetry import _cost_usd, _trunc, record_model_resolution, record_request, setup_logfire


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
