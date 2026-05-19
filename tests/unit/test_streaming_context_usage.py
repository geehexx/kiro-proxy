"""Unit tests for forwarding contextUsageEvent to clients (Fix 4).

Background: Kiro emits a contextUsageEvent at end of stream with the percentage
of context window used. The proxy parses it for token estimation but didn't
forward it to clients. Threading it into the Anthropic message_delta.usage and
non-streaming usage object lets Claude Code display real context usage.

See /tmp/prompt.md §5 Fix 4.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro.streaming_anthropic import (
    collect_anthropic_response,
    stream_kiro_to_anthropic,
)
from kiro.streaming_core import KiroEvent, StreamResult


@pytest.fixture
def mock_model_cache():
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    return MagicMock()


@pytest.fixture
def mock_response():
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


def _extract_message_delta_usage(events: list[str]) -> dict | None:
    """Pull the usage payload out of the message_delta SSE event."""
    for raw in events:
        if "message_delta" not in raw:
            continue
        for line in raw.strip().split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if payload.get("type") == "message_delta":
                    return payload.get("usage", {})
    return None


class TestStreamingContextUsageForwarded:
    """Streaming path — context_usage_percentage flows into message_delta.usage."""

    @pytest.mark.asyncio
    async def test_forwards_context_usage_percentage_in_message_delta(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """When Kiro emits contextUsageEvent, message_delta.usage carries the value."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=37.5)

        events: list[str] = []
        with patch("kiro.streaming_anthropic.parse_kiro_stream", mock_parse_kiro_stream), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            async for event in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            ):
                events.append(event)

        usage = _extract_message_delta_usage(events)
        assert usage is not None, "message_delta event missing"
        assert usage.get("context_usage_percentage") == 37.5

    @pytest.mark.asyncio
    async def test_rounds_context_usage_to_two_decimals(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Float values from Kiro are rounded to 2dp before forwarding."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")
            yield KiroEvent(type="context_usage", context_usage_percentage=42.123456)

        events: list[str] = []
        with patch("kiro.streaming_anthropic.parse_kiro_stream", mock_parse_kiro_stream), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            async for event in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            ):
                events.append(event)

        usage = _extract_message_delta_usage(events)
        assert usage["context_usage_percentage"] == 42.12

    @pytest.mark.asyncio
    async def test_omits_context_usage_when_kiro_did_not_emit(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """No contextUsageEvent in stream → field absent from usage payload."""
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")
            # no context_usage event

        events: list[str] = []
        with patch("kiro.streaming_anthropic.parse_kiro_stream", mock_parse_kiro_stream), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            async for event in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            ):
                events.append(event)

        usage = _extract_message_delta_usage(events)
        assert usage is not None
        assert "context_usage_percentage" not in usage


class TestNonStreamingContextUsageForwarded:
    """Non-streaming path — context_usage_percentage flows into the response usage."""

    @pytest.mark.asyncio
    async def test_non_streaming_includes_context_usage(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """collect_anthropic_response() forwards context_usage_percentage in usage."""
        result = StreamResult(
            content="hello",
            thinking_content="",
            tool_calls=[],

            usage=None,
            context_usage_percentage=12.34,
        )
        with patch("kiro.streaming_anthropic.collect_stream_to_result", AsyncMock(return_value=result)), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            response = await collect_anthropic_response(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            )
        assert response["usage"]["context_usage_percentage"] == 12.34

    @pytest.mark.asyncio
    async def test_non_streaming_rounds_to_two_decimals(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Non-streaming float rounding matches the streaming path."""
        result = StreamResult(
            content="hi",
            thinking_content="",
            tool_calls=[],

            usage=None,
            context_usage_percentage=88.987654,
        )
        with patch("kiro.streaming_anthropic.collect_stream_to_result", AsyncMock(return_value=result)), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            response = await collect_anthropic_response(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            )
        assert response["usage"]["context_usage_percentage"] == 88.99

    @pytest.mark.asyncio
    async def test_non_streaming_omits_field_when_unavailable(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """No upstream context usage → field absent from non-streaming usage."""
        result = StreamResult(
            content="hi",
            thinking_content="",
            tool_calls=[],

            usage=None,
            context_usage_percentage=None,
        )
        with patch("kiro.streaming_anthropic.collect_stream_to_result", AsyncMock(return_value=result)), \
             patch("kiro.streaming_anthropic.parse_bracket_tool_calls", return_value=[]):
            response = await collect_anthropic_response(
                mock_response, "claude-sonnet-4.6", mock_model_cache, mock_auth_manager
            )
        assert "context_usage_percentage" not in response["usage"]
