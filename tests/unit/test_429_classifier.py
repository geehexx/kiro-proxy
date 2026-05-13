"""
Unit tests for 429 capacity-aware backoff classification.

Verifies that:
- A 429 with reason=INSUFFICIENT_MODEL_CAPACITY is classified correctly
  and carries a retry_after_hint.
- A standard rate-limit 429 (no reason / different reason) is NOT
  classified as a capacity error.
- The KiroErrorInfo.retry_after_hint field is populated only for
  INSUFFICIENT_MODEL_CAPACITY.
"""

from kiro.kiro_errors import KiroErrorInfo, enhance_kiro_error


class TestCapacity429Classification:
    """Tests for INSUFFICIENT_MODEL_CAPACITY 429 classification."""

    def test_capacity_error_sets_retry_after_hint(self):
        """
        What it does: Verifies INSUFFICIENT_MODEL_CAPACITY sets retry_after_hint.
        Purpose: Ensure the http_client can detect capacity errors and apply
                 a longer backoff than the standard rate-limit path.
        """
        error_json = {
            "message": "Model capacity temporarily unavailable.",
            "reason": "INSUFFICIENT_MODEL_CAPACITY",
        }
        error_info = enhance_kiro_error(error_json)

        assert error_info.retry_after_hint is not None
        assert error_info.retry_after_hint > 0
        assert error_info.reason == "INSUFFICIENT_MODEL_CAPACITY"

    def test_capacity_error_hint_is_30_seconds(self):
        """
        What it does: Verifies the default retry_after_hint is 30 seconds.
        Purpose: Confirm the documented default backoff hint value.
        """
        error_json = {
            "message": "Model capacity temporarily unavailable.",
            "reason": "INSUFFICIENT_MODEL_CAPACITY",
        }
        error_info = enhance_kiro_error(error_json)

        assert error_info.retry_after_hint == 30.0

    def test_capacity_error_user_message_is_friendly(self):
        """
        What it does: Verifies the user-facing message is clear and non-technical.
        Purpose: Ensure end users understand the error without technical jargon.
        """
        error_json = {
            "message": "Model capacity temporarily unavailable.",
            "reason": "INSUFFICIENT_MODEL_CAPACITY",
        }
        error_info = enhance_kiro_error(error_json)

        assert "capacity" in error_info.user_message.lower()
        assert "retry" in error_info.user_message.lower()
        # Should NOT expose the raw reason code to end users
        assert "INSUFFICIENT_MODEL_CAPACITY" not in error_info.user_message

    def test_standard_rate_limit_429_no_hint(self):
        """
        What it does: Verifies a standard rate-limit 429 does NOT set retry_after_hint.
        Purpose: Ensure the capacity path is not triggered for ordinary rate limits.
        """
        error_json = {
            "message": "Too many requests.",
            "reason": "RATE_LIMIT_EXCEEDED",
        }
        error_info = enhance_kiro_error(error_json)

        # Standard rate-limit falls through to the generic handler
        assert error_info.retry_after_hint is None

    def test_monthly_limit_429_no_hint(self):
        """
        What it does: Verifies MONTHLY_REQUEST_COUNT does NOT set retry_after_hint.
        Purpose: Monthly quota exhaustion is not a transient capacity issue.
        """
        error_json = {
            "message": "You have reached the limit.",
            "reason": "MONTHLY_REQUEST_COUNT",
        }
        error_info = enhance_kiro_error(error_json)

        assert error_info.retry_after_hint is None

    def test_no_reason_field_no_hint(self):
        """
        What it does: Verifies missing reason field does NOT set retry_after_hint.
        Purpose: Ensure graceful handling of errors without a reason field.
        """
        error_json = {"message": "An error occurred."}
        error_info = enhance_kiro_error(error_json)

        assert error_info.retry_after_hint is None

    def test_content_length_error_no_hint(self):
        """
        What it does: Verifies CONTENT_LENGTH_EXCEEDS_THRESHOLD does NOT set retry_after_hint.
        Purpose: Context-limit errors are not retryable with backoff.
        """
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
        }
        error_info = enhance_kiro_error(error_json)

        assert error_info.retry_after_hint is None

    def test_kiro_error_info_default_hint_is_none(self):
        """
        What it does: Verifies KiroErrorInfo.retry_after_hint defaults to None.
        Purpose: Ensure backward compatibility — existing callers that don't
                 check retry_after_hint are unaffected.
        """
        info = KiroErrorInfo(
            reason="SOME_REASON",
            user_message="Some message",
            original_message="Original",
        )
        assert info.retry_after_hint is None

    def test_kiro_error_info_hint_can_be_set(self):
        """
        What it does: Verifies KiroErrorInfo.retry_after_hint can be set explicitly.
        Purpose: Ensure the field is writable for future use.
        """
        info = KiroErrorInfo(
            reason="INSUFFICIENT_MODEL_CAPACITY",
            user_message="Capacity exhausted.",
            original_message="Original.",
            retry_after_hint=60.0,
        )
        assert info.retry_after_hint == 60.0
