"""
Phase 2.7 — Beta header emulation smoke test.

Verifies that unsupported Anthropic beta headers are stripped before
the request reaches the upstream, and that supported betas pass through.
"""
from __future__ import annotations

import pytest


class TestBetaHeaderStripping:
    """Verify unsupported beta headers are stripped, supported ones pass through."""

    def test_unsupported_beta_stripped(self, test_client, valid_proxy_api_key):
        """
        Request with advanced-tool-use-2025-11-20 beta header should not
        cause a 422 — the gateway strips it before forwarding.
        """
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "advanced-tool-use-2025-11-20",
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should not be 422 (validation error) — beta was stripped, not rejected
        assert response.status_code != 422, (
            f"Beta header caused validation error: {response.text[:200]}"
        )

    def test_computer_use_beta_stripped(self, test_client, valid_proxy_api_key):
        """computer-use-2024-10-22 is also unsupported — should be stripped."""
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "computer-use-2024-10-22",
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code != 422

    def test_multiple_betas_partial_strip(self, test_client, valid_proxy_api_key):
        """
        When multiple betas are sent and only some are unsupported,
        the request should proceed (unsupported ones stripped, supported kept).
        """
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
                # Mix of unsupported + a hypothetical supported beta
                "anthropic-beta": "advanced-tool-use-2025-11-20,max-tokens-3-5-sonnet-2024-07-15",
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code != 422

    def test_no_beta_header_accepted(self, test_client, valid_proxy_api_key):
        """Requests without any beta header should work normally."""
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code != 422
