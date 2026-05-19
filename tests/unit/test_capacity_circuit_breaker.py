"""
Tests for capacity-exhaustion fast circuit-breaker.

Verifies that INSUFFICIENT_MODEL_CAPACITY 429s are classified as FATAL
(not RECOVERABLE) so kiro-proxy fails fast instead of cycling accounts.

Also includes an integration test that exercises the full route handler
and verifies the 503 + Retry-After + X-Kiro-Capacity-Exhausted response.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from kiro.account_errors import ErrorType, classify_error


class TestCapacityCircuitBreaker:
    """Tests for INSUFFICIENT_MODEL_CAPACITY fast-fail classification."""

    def test_insufficient_model_capacity_is_fatal(self):
        """
        INSUFFICIENT_MODEL_CAPACITY 429 must be FATAL — cycling accounts
        won't help since all accounts share the same upstream capacity pool.
        """
        result = classify_error(status_code=429, reason="INSUFFICIENT_MODEL_CAPACITY")
        assert result == ErrorType.FATAL, (
            "INSUFFICIENT_MODEL_CAPACITY should be FATAL (fast circuit-breaker). "
            "Cycling accounts wastes retries against the same capacity ceiling."
        )

    def test_other_429_still_recoverable(self):
        """
        Non-capacity 429s (per-account rate limits) must remain RECOVERABLE
        so account cycling still works for rate-limit errors.
        """
        result = classify_error(status_code=429, reason=None)
        assert result == ErrorType.RECOVERABLE, (
            "Generic 429 (no reason) should remain RECOVERABLE — "
            "per-account rate limits benefit from account cycling."
        )

    def test_rate_limit_reason_still_recoverable(self):
        """
        A 429 with a non-capacity reason should remain RECOVERABLE.
        """
        result = classify_error(status_code=429, reason="RATE_LIMIT_EXCEEDED")
        assert result == ErrorType.RECOVERABLE, (
            "RATE_LIMIT_EXCEEDED 429 should remain RECOVERABLE."
        )

    def test_capacity_exhausted_reason_case_sensitive(self):
        """
        The reason string comparison is case-sensitive. A differently-cased
        reason should NOT trigger the fast circuit-breaker.
        """
        result = classify_error(status_code=429, reason="insufficient_model_capacity")
        # lowercase variant should fall through to generic 429 -> RECOVERABLE
        assert result == ErrorType.RECOVERABLE, (
            "Lowercase reason should not match INSUFFICIENT_MODEL_CAPACITY."
        )

    def test_non_429_capacity_error_unaffected(self):
        """
        A 503 with no reason should still be FATAL (existing behaviour).
        The circuit-breaker only applies to 429 + INSUFFICIENT_MODEL_CAPACITY.
        """
        result = classify_error(status_code=503, reason=None)
        assert result == ErrorType.FATAL, (
            "503 should remain FATAL regardless of reason."
        )


class TestCapacityCircuitBreakerRouteIntegration:
    """
    Integration tests: upstream 429 INSUFFICIENT_MODEL_CAPACITY -> route returns
    503 with Retry-After: 60 and X-Kiro-Capacity-Exhausted headers.
    """

    def test_upstream_capacity_429_returns_503_with_headers(
        self, test_client, valid_proxy_api_key
    ):
        """
        When the upstream Kiro API returns a 429 with reason
        INSUFFICIENT_MODEL_CAPACITY, the proxy must:
          - respond with status 503
          - include Retry-After: 60
          - include X-Kiro-Capacity-Exhausted set to the resolved model name

        Note: the route normalizes the requested model name before setting the
        header (e.g. "claude-opus-4-5" -> "claude-opus-4.5"), so we assert
        against the normalized form.
        """
        # Use the already-normalized form so the assertion is stable.
        # normalize_model_name("claude-opus-4-5") == "claude-opus-4.5"
        model_input = "claude-opus-4-5"
        model_resolved = "claude-opus-4.5"

        # Build the upstream 429 error body that Kiro returns
        upstream_error_body = json.dumps(
            {"message": "Model capacity exhausted.", "reason": "INSUFFICIENT_MODEL_CAPACITY"}
        ).encode()

        # Mock the HTTP client response that request_with_retry returns
        mock_upstream = AsyncMock()
        mock_upstream.status_code = 429
        mock_upstream.aread = AsyncMock(return_value=upstream_error_body)
        mock_upstream.aclose = AsyncMock()

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            return_value=mock_upstream,
        ):
            response = test_client.post(
                "/v1/messages",
                headers={"x-api-key": valid_proxy_api_key},
                json={
                    "model": model_input,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 503, (
            f"Expected 503 for capacity-exhausted upstream 429, got {response.status_code}"
        )
        assert "Retry-After" in response.headers, (
            "Response must include Retry-After header"
        )
        assert response.headers["Retry-After"] == "60", (
            f"Retry-After must be '60', got {response.headers['Retry-After']!r}"
        )
        assert "X-Kiro-Capacity-Exhausted" in response.headers, (
            "Response must include X-Kiro-Capacity-Exhausted header"
        )
        assert response.headers["X-Kiro-Capacity-Exhausted"] == model_resolved, (
            f"X-Kiro-Capacity-Exhausted must be '{model_resolved}', "
            f"got {response.headers['X-Kiro-Capacity-Exhausted']!r}"
        )
