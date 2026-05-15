"""
Tests for capacity-exhaustion fast circuit-breaker.

Verifies that INSUFFICIENT_MODEL_CAPACITY 429s are classified as FATAL
(not RECOVERABLE) so kiro-proxy fails fast instead of cycling accounts.
"""

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
        # lowercase variant should fall through to generic 429 → RECOVERABLE
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
