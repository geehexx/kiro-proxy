
"""
Live smoke tests against a running kiro-gateway.

These tests are EXCLUDED from the default pytest run. They exercise the
gateway end-to-end over HTTP, including the upstream Kiro API, so they
require a running gateway, real credentials, and network access.

## Running

Prerequisites:
- kiro-gateway running at $KIRO_GATEWAY_URL (default http://127.0.0.1:8765)
- $KIRO_GATEWAY_API_KEY set to the proxy API key the gateway accepts
- Network egress to the gateway

Invocation:

    KIRO_GATEWAY_URL=http://127.0.0.1:8765 \
    KIRO_GATEWAY_API_KEY=<proxy-api-key> \
    .venv/bin/python -m pytest tests/live -v -m live

Or via lefthook ad-hoc:

    lefthook run live-smoke

## Design notes

- No upstream mocking — this is the ONE surface where we confirm the
  real request path works. Unit tests mock everything below the route
  handler and miss integration issues (e.g. INT-73, which was a
  runtime import error that passed all 1800+ unit tests).
- Tests are tolerant of upstream slowness: each Claude call has a 60s
  timeout. If the gateway is down or upstream is throttling, the
  failure message is unambiguous.
- No per-test account mutation — these tests do not modify state.json,
  credentials.json, or any upstream account. They only exercise the
  read path (/v1/models) and a single completion call.

## What to run before a release

See `docs/release-checklist.md` (future doc). At minimum:
1. /v1/models returns the expected model shape, including display_name
2. A tiny completion round-trips (non-streaming and streaming)
3. count_tokens computes a plausible number
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("KIRO_GATEWAY_URL", "http://127.0.0.1:8765")
API_KEY = os.environ.get("KIRO_GATEWAY_API_KEY")


def _skip_if_missing_api_key() -> None:
    if not API_KEY:
        pytest.skip(
            "KIRO_GATEWAY_API_KEY not set — live smoke tests need the "
            "proxy API key that the running gateway accepts."
        )


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        base_url=GATEWAY_URL,
        headers={"x-api-key": API_KEY or ""},
        timeout=httpx.Timeout(timeout),
    )


# ---------------------------------------------------------------------------
# Health + /v1/models
# ---------------------------------------------------------------------------


class TestLiveHealth:
    def test_health_endpoint_returns_200(self) -> None:
        """Gateway is alive on the expected port."""
        response = httpx.get(f"{GATEWAY_URL}/health", timeout=5.0)
        assert response.status_code == 200

    def test_root_returns_ok(self) -> None:
        """Root endpoint responds (shape varies by version)."""
        response = httpx.get(GATEWAY_URL, timeout=5.0)
        assert response.status_code in (200, 405)


# ---------------------------------------------------------------------------
# /v1/models parity + humanised naming
# ---------------------------------------------------------------------------


class TestLiveModelsEndpoint:
    def test_list_models_returns_expected_shape(self) -> None:
        _skip_if_missing_api_key()
        with _client() as client:
            response = client.get("/v1/models")
        assert response.status_code == 200, response.text
        body = response.json()

        assert body.get("object") == "list"
        assert isinstance(body.get("data"), list)
        assert len(body["data"]) > 0, "Expected at least one model in /v1/models"

    def test_each_model_has_required_fields(self) -> None:
        _skip_if_missing_api_key()
        with _client() as client:
            response = client.get("/v1/models")
        assert response.status_code == 200, response.text
        for model in response.json()["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "owned_by" in model

    def test_claude_models_have_humanised_display_name(self) -> None:
        """After the humanised-naming change, every Claude model has display_name."""
        _skip_if_missing_api_key()
        with _client() as client:
            response = client.get("/v1/models")
        claude_models = [
            m for m in response.json()["data"] if m["id"].lower().startswith("claude-")
        ]
        assert claude_models, "Expected at least one Claude model in the picker"
        for model in claude_models:
            assert "display_name" in model, f"Missing display_name: {model}"
            assert model["display_name"], f"Empty display_name: {model}"
            assert model["display_name"].startswith("Claude "), (
                f"display_name should start with 'Claude ': {model}"
            )

    def test_model_ids_use_canonical_dotted_form(self) -> None:
        """No mixed-format ids like claude-opus-4-7 (should be claude-opus-4.7)."""
        _skip_if_missing_api_key()
        with _client() as client:
            response = client.get("/v1/models")
        for model in response.json()["data"]:
            mid = model["id"].lower()
            is_claude_family = (
                mid.startswith("claude-opus")
                or mid.startswith("claude-sonnet")
                or mid.startswith("claude-haiku")
            )
            if not is_claude_family:
                continue
            parts = model["id"].split("-")
            # Reject trailing -N-N shape
            if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
                pytest.fail(
                    f"Model id {model['id']!r} uses mixed dashed-minor-version "
                    f"form; should be canonical dotted form"
                )


# ---------------------------------------------------------------------------
# count_tokens round trip
# ---------------------------------------------------------------------------


class TestLiveCountTokens:
    def test_count_tokens_computes_positive_integer(self) -> None:
        _skip_if_missing_api_key()
        with _client() as client:
            response = client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "claude-sonnet-4.5",
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body.get("input_tokens"), int)
        assert body["input_tokens"] > 0


# ---------------------------------------------------------------------------
# Tiny completion round trip (non-streaming)
# ---------------------------------------------------------------------------


class TestLiveCompletion:
    def test_non_streaming_completion_round_trips(self) -> None:
        """
        A single-turn completion returns a well-formed Anthropic Messages
        response with usage.input_tokens + usage.output_tokens > 0.

        Model is pinned to Haiku to keep cost and latency low.
        """
        _skip_if_missing_api_key()
        with _client(timeout=60.0) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4.5",
                    "max_tokens": 50,
                    "messages": [
                        {"role": "user", "content": "Reply with just the word 'ok'."}
                    ],
                    "stream": False,
                },
            )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()

        # Schema
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert isinstance(body["content"], list)
        assert isinstance(body.get("usage"), dict)
        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0

        # Cache-status header should be set when cache is enabled; either
        # value is acceptable.
        cache_header = response.headers.get("x-kiro-cache")
        assert cache_header in (None, "hit", "miss")
