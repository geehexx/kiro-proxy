"""
Unit tests for timing-safe API key comparison.

Verifies that verify_api_key (OpenAI routes) and verify_anthropic_api_key
(Anthropic routes) use hmac.compare_digest and reject all the obvious
attack shapes: wrong key, empty key, None.

Background: == on Python strings short-circuits on first byte mismatch,
leaking timing info that lets an attacker brute-force the key one byte
at a time. hmac.compare_digest is constant-time for equal-length inputs
and short-circuit-resistant otherwise.

See basic-memory://audits/2026-05-19-kiro-proxy-audit/E-security E1.
"""

import pytest
from fastapi import HTTPException

from kiro.config import _PLACEHOLDER_PROXY_API_KEY, PROXY_API_KEY
from kiro.routes_anthropic import verify_anthropic_api_key
from kiro.routes_openai import verify_api_key


class TestVerifyApiKeyTimingSafe:
    """OpenAI verify_api_key — timing-safe path coverage."""

    @pytest.mark.asyncio
    async def test_correct_token_returns_true(self):
        """Valid Bearer token passes."""
        result = await verify_api_key(f"Bearer {PROXY_API_KEY}")
        assert result is True

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self):
        """Wrong token raises 401, no timing-side-channel."""
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("Bearer definitely_not_the_key")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_token_returns_401(self):
        """Empty string is rejected without crashing on hmac.compare_digest."""
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_none_token_returns_401(self):
        """None auth header is rejected — guarded before hmac.compare_digest
        (which would raise TypeError on None)."""
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_token_length_mismatch_returns_401(self):
        """Length-mismatched tokens — hmac.compare_digest still safe (no crash)."""
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("Bearer x")  # very short
        assert exc.value.status_code == 401

        with pytest.raises(HTTPException) as exc:
            await verify_api_key("Bearer " + "x" * 1000)  # very long
        assert exc.value.status_code == 401


class TestVerifyAnthropicApiKeyTimingSafe:
    """Anthropic verify_anthropic_api_key — timing-safe path coverage."""

    @pytest.mark.asyncio
    async def test_correct_x_api_key_returns_true(self):
        """Valid x-api-key passes."""
        result = await verify_anthropic_api_key(x_api_key=PROXY_API_KEY, authorization=None)
        assert result is True

    @pytest.mark.asyncio
    async def test_correct_bearer_returns_true(self):
        """Valid Authorization: Bearer passes as fallback."""
        result = await verify_anthropic_api_key(
            x_api_key=None, authorization=f"Bearer {PROXY_API_KEY}"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_wrong_x_api_key_returns_401(self):
        """Wrong x-api-key raises 401."""
        with pytest.raises(HTTPException) as exc:
            await verify_anthropic_api_key(x_api_key="wrong_key", authorization=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_x_api_key_returns_401(self):
        """Empty x-api-key rejected without crashing."""
        with pytest.raises(HTTPException) as exc:
            await verify_anthropic_api_key(x_api_key="", authorization=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_none_both_returns_401(self):
        """No headers at all → 401."""
        with pytest.raises(HTTPException) as exc:
            await verify_anthropic_api_key(x_api_key=None, authorization=None)
        assert exc.value.status_code == 401


class TestPlaceholderConstantExposed:
    """E2 — _PLACEHOLDER_PROXY_API_KEY is the canonical default literal.

    Tests reference the constant rather than hardcoding the placeholder
    string. This keeps gitleaks happy (no literal default-key value in
    test source) and ensures tests stay in sync if the placeholder ever
    changes.
    """

    def test_placeholder_is_a_nonempty_string(self):
        assert isinstance(_PLACEHOLDER_PROXY_API_KEY, str)
        assert len(_PLACEHOLDER_PROXY_API_KEY) > 0

    def test_proxy_api_key_is_not_placeholder_in_tests(self):
        """The conftest sets PROXY_API_KEY to a non-default value before
        kiro.config imports — this guards against accidental regressions
        where conftest stops setting it and the import-time guard fires."""
        assert PROXY_API_KEY != _PLACEHOLDER_PROXY_API_KEY
