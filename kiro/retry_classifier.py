"""Body-content retry classifier — backport from amazon-q-cli.

Status code alone is wrong both ways:

  - Some 429s carry MONTHLY_REQUEST_COUNT — a hard quota, retrying just
    burns the rest of the retry budget without a chance of success.
  - Some 500s carry "Encountered unexpectedly high load" or
    ServiceUnavailableException — capacity throttling masquerading as a
    server error.  These should be treated like a 429 (throttle backoff
    + adaptive bucket record_throttle), not a generic 5xx retry.

This classifier reads the response body and returns a RetryDecision
the route handler uses to pick the right backoff strategy.

Source spec: data/basic-memory/research/2026-05-16-restart-recovery-and-tier1-hardening/
B-amazon-q-cli-backport-patterns.md §Pattern 3
(amazon-q-developer-cli crates/chat-cli/src/api_client/retry_classifier.rs)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class RetryKind(str, Enum):
    """How the route handler should treat this response."""

    NO_RETRY = "no_retry"          # hard quota / non-retryable — break loop
    THROTTLE = "throttle"          # capacity / rate-limit — backoff + bucket throttle
    STANDARD = "standard"          # generic transient — exponential backoff


@dataclass(frozen=True)
class RetryDecision:
    kind: RetryKind
    reason: str


# Body markers — string-match against the decoded response body.
_NO_RETRY_MARKERS: tuple[str, ...] = (
    "MONTHLY_REQUEST_COUNT",
    "MONTHLY_TOKEN_COUNT",
    "request quota exceeded",
)

_THROTTLE_MARKERS: tuple[str, ...] = (
    "Encountered unexpectedly high load",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
    "INSUFFICIENT_MODEL_CAPACITY",
    "model is currently overloaded",
)


def _decode_body(body: bytes | None) -> str:
    if not body:
        return ""
    try:
        return body.decode("utf-8", errors="replace")
    except (UnicodeDecodeError, AttributeError):
        return ""


def _extract_reason_field(body_str: str) -> str | None:
    """Pull `.reason` out of a JSON body if present, else None."""
    try:
        obj = json.loads(body_str)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict):
        reason = obj.get("reason")
        if isinstance(reason, str):
            return reason
    return None


def classify(status: int, body: bytes | None) -> RetryDecision:
    """Decide retry strategy from status code + body content.

    The classifier never raises; on parse error it falls back to the
    status-code-based default (preserving today's behaviour).
    """
    body_str = _decode_body(body)
    json_reason = _extract_reason_field(body_str)

    # Hard-quota markers take precedence — never retry these.
    for marker in _NO_RETRY_MARKERS:
        if marker in body_str or (json_reason and marker in json_reason):
            return RetryDecision(RetryKind.NO_RETRY, f"hard quota: {marker}")

    # Throttle markers (any status code).  A 500 carrying
    # ServiceUnavailableException should be treated as a throttle.
    for marker in _THROTTLE_MARKERS:
        if marker in body_str or (json_reason and marker in json_reason):
            return RetryDecision(RetryKind.THROTTLE, f"throttle: {marker}")

    # Status-code-based defaults — back-compat with today's behaviour.
    if status == 429:
        return RetryDecision(RetryKind.THROTTLE, "status 429 (no body markers)")
    if 500 <= status < 600:
        return RetryDecision(RetryKind.STANDARD, f"status {status} (generic 5xx)")

    return RetryDecision(RetryKind.STANDARD, f"status {status}")
