"""Unit tests for body-content retry classifier — Pattern 3 backport."""
from __future__ import annotations

import json

import pytest

from kiro.retry_classifier import RetryDecision, RetryKind, classify


def test_status_429_with_no_body_is_throttle():
    d = classify(429, b"")
    assert d.kind is RetryKind.THROTTLE
    assert "429" in d.reason


def test_status_500_with_no_body_is_standard():
    d = classify(500, b"")
    assert d.kind is RetryKind.STANDARD
    assert "500" in d.reason


def test_monthly_request_count_in_429_is_no_retry():
    body = json.dumps({"reason": "MONTHLY_REQUEST_COUNT exceeded"}).encode()
    d = classify(429, body)
    assert d.kind is RetryKind.NO_RETRY
    assert "MONTHLY_REQUEST_COUNT" in d.reason


def test_monthly_request_count_inline_text_is_no_retry():
    body = b"You have hit your MONTHLY_REQUEST_COUNT limit"
    d = classify(429, body)
    assert d.kind is RetryKind.NO_RETRY


def test_high_load_500_is_throttle_not_standard():
    body = b"Encountered unexpectedly high load -- please retry"
    d = classify(500, body)
    assert d.kind is RetryKind.THROTTLE
    assert "high load" in d.reason


def test_service_unavailable_in_5xx_is_throttle():
    body = json.dumps({"reason": "ServiceUnavailableException"}).encode()
    d = classify(503, body)
    assert d.kind is RetryKind.THROTTLE


def test_throttling_exception_in_429_is_throttle():
    body = json.dumps({"reason": "ThrottlingException"}).encode()
    d = classify(429, body)
    assert d.kind is RetryKind.THROTTLE


def test_insufficient_model_capacity_is_throttle():
    body = json.dumps({"reason": "INSUFFICIENT_MODEL_CAPACITY"}).encode()
    d = classify(429, body)
    assert d.kind is RetryKind.THROTTLE
    assert "INSUFFICIENT_MODEL_CAPACITY" in d.reason


def test_malformed_json_falls_back_to_status_default():
    d = classify(429, b"\xff\xfe not json at all")
    # Falls back to status-based default (still throttle for 429)
    assert d.kind is RetryKind.THROTTLE


def test_none_body_falls_back_to_status_default():
    d = classify(500, None)
    assert d.kind is RetryKind.STANDARD


def test_no_retry_takes_precedence_over_throttle():
    # If a body somehow carries BOTH markers, no-retry wins (hard quota).
    body = b"MONTHLY_REQUEST_COUNT exceeded; the model was overloaded too"
    d = classify(429, body)
    assert d.kind is RetryKind.NO_RETRY


def test_200_with_clean_body_is_standard():
    d = classify(200, b'{"ok": true}')
    assert d.kind is RetryKind.STANDARD


def test_decision_is_immutable():
    d = classify(429, b"")
    with pytest.raises(Exception):
        d.kind = RetryKind.NO_RETRY  # type: ignore[misc]


def test_quota_exceeded_lowercase_marker_misses():
    """Markers are case-sensitive — guards against false positives in
    user prose (e.g. a doc mentioning 'request quota exceeded' shouldn't
    block legitimate retries unless the upstream genuinely reports it).
    Today: case-sensitive match means 'Request Quota Exceeded' would
    miss; documenting the choice."""
    body = b"Request Quota Exceeded"
    d = classify(429, body)
    # NB: case-sensitive — falls back to status-default (throttle for 429)
    assert d.kind is RetryKind.THROTTLE
