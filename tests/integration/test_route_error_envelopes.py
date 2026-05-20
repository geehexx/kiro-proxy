"""
Integration tests: upstream error envelopes and streaming paths.

Covers the 5 highest-impact uncovered code paths in routes_anthropic.py
and routes_openai.py:

1. Anthropic route -- upstream non-200 error envelope shape
2. Anthropic route -- 429 capacity-exhausted -> 503 + X-Kiro-Capacity-Exhausted
3. Anthropic route -- streaming path (SSE chunks forwarded)
4. OpenAI route -- upstream non-200 -> OpenAI error envelope shape
5. OpenAI route -- streaming path ([DONE] sentinel present)

All tests mock the upstream AWS Q HTTP layer at the KiroHttpClient boundary.
No real network calls are made.

IMPORTANT: Each test uses a unique message string (prefixed with the test name)
to avoid response-cache collisions with other tests that use the same
model+messages combination.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_sse_chunks(content: str = "Hi") -> list[bytes]:
    """Minimal valid Anthropic SSE byte chunks."""
    return [
        b'data: {"type":"message_start","message":{"id":"msg_t1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        ('data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"' + content + '"}}\n\n').encode(),
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
        b'data: [DONE]\n\n',
    ]


def _make_openai_sse_chunks(content: str = "Hi") -> list[bytes]:
    """Minimal valid OpenAI-format SSE byte chunks."""
    return [
        ('data: {"id":"chatcmpl-t1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"' + content + '"},"finish_reason":null}]}\n\n').encode(),
        b'data: {"id":"chatcmpl-t1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]


def _mock_upstream_response(
    status_code: int = 200,
    body: bytes = b"",
    sse_chunks: list[bytes] | None = None,
) -> MagicMock:
    """Build a mock httpx.Response for the upstream Kiro API."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.aread = AsyncMock(return_value=body)

    chunks = sse_chunks or []

    async def _aiter_bytes():
        for chunk in chunks:
            yield chunk

    mock_resp.aiter_bytes = _aiter_bytes
    mock_resp.aclose = AsyncMock()
    return mock_resp


# ---------------------------------------------------------------------------
# Anthropic route -- upstream error envelopes
# ---------------------------------------------------------------------------

class TestAnthropicUpstreamErrorEnvelope:
    """routes_anthropic.py lines 1373-1462: non-200 upstream -> Anthropic error shape."""

    @pytest.mark.slow
    def test_upstream_500_returns_anthropic_error_envelope(
        self, test_client, valid_proxy_api_key
    ):
        """When upstream returns 500, route must return {"type":"error","error":{...}}."""
        error_body = json.dumps({"message": "Internal server error"}).encode()
        mock_resp = _mock_upstream_response(status_code=500, body=error_body)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/messages",
                headers={
                    "x-api-key": valid_proxy_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    # Unique content prevents cache collision with other tests
                    "messages": [{"role": "user", "content": "test_upstream_500_anthropic_unique_xq7k"}],
                },
            )

        assert response.status_code == 500
        data = response.json()
        assert "type" in data, f"Missing 'type' key in: {data}"
        assert data["type"] == "error"
        assert "error" in data
        assert "message" in data["error"]

    @pytest.mark.slow
    def test_upstream_429_capacity_exhausted_returns_503(
        self, test_client, valid_proxy_api_key
    ):
        """
        When upstream returns 429 with INSUFFICIENT_MODEL_CAPACITY reason,
        route must return 503 with X-Kiro-Capacity-Exhausted header.
        (routes_anthropic.py lines 1438-1455)
        """
        capacity_error = json.dumps({
            "message": "Insufficient model capacity",
            "reason": "INSUFFICIENT_MODEL_CAPACITY",
        }).encode()
        mock_resp = _mock_upstream_response(status_code=429, body=capacity_error)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/messages",
                headers={
                    "x-api-key": valid_proxy_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    # Unique content prevents cache collision with other tests
                    "messages": [{"role": "user", "content": "test_429_capacity_exhausted_unique_m3p9"}],
                },
            )

        # Capacity-exhausted 429 must be converted to 503
        assert response.status_code == 503
        data = response.json()
        assert data.get("type") == "error"
        assert "error" in data
        # Header must be present (case-insensitive)
        header_names = {k.lower() for k in response.headers}
        assert "x-kiro-capacity-exhausted" in header_names, (
            f"Missing X-Kiro-Capacity-Exhausted header. Headers: {dict(response.headers)}"
        )

    @pytest.mark.slow
    def test_upstream_401_returns_anthropic_error_envelope(
        self, test_client, valid_proxy_api_key
    ):
        """Upstream 401 must be forwarded as Anthropic error envelope, not raw text."""
        error_body = json.dumps({"message": "Unauthorized"}).encode()
        mock_resp = _mock_upstream_response(status_code=401, body=error_body)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/messages",
                headers={
                    "x-api-key": valid_proxy_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    # Unique content prevents cache collision with other tests
                    "messages": [{"role": "user", "content": "test_401_unauthorized_unique_n8v2"}],
                },
            )

        assert response.status_code == 401
        data = response.json()
        assert data.get("type") == "error"
        assert "error" in data
        assert "message" in data["error"]


# ---------------------------------------------------------------------------
# Anthropic route -- streaming path
# ---------------------------------------------------------------------------

class TestAnthropicStreamingPath:
    """routes_anthropic.py lines 1464-1592: streaming SSE path."""

    @pytest.mark.slow
    def test_streaming_returns_event_stream_content_type(
        self, test_client, valid_proxy_api_key
    ):
        """Streaming requests must return Content-Type: text/event-stream."""
        chunks = _make_anthropic_sse_chunks("Hello streaming")
        mock_resp = _mock_upstream_response(status_code=200, sse_chunks=chunks)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/messages",
                headers={
                    "x-api-key": valid_proxy_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    "stream": True,
                    # stream=True bypasses response cache, no unique suffix needed
                    "messages": [{"role": "user", "content": "test_streaming_content_type_anthropic"}],
                },
            )

        assert response.status_code == 200
        ct = response.headers.get("content-type", "")
        assert "text/event-stream" in ct, f"Expected text/event-stream, got: {ct}"

    @pytest.mark.slow
    def test_streaming_chunks_contain_data_prefix(
        self, test_client, valid_proxy_api_key
    ):
        """Each SSE chunk forwarded to the client must start with 'data: '."""
        chunks = _make_anthropic_sse_chunks("chunk test")
        mock_resp = _mock_upstream_response(status_code=200, sse_chunks=chunks)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/messages",
                headers={
                    "x-api-key": valid_proxy_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "test_streaming_data_prefix_anthropic"}],
                },
            )

        assert response.status_code == 200
        body = response.text
        assert "data:" in body, f"No SSE data lines in response body: {body[:200]}"


# ---------------------------------------------------------------------------
# OpenAI route -- upstream error envelopes
# ---------------------------------------------------------------------------

class TestOpenAIUpstreamErrorEnvelope:
    """routes_openai.py lines 875-916: non-200 upstream -> OpenAI error shape."""

    @pytest.mark.slow
    def test_upstream_500_returns_openai_error_envelope(
        self, test_client, valid_proxy_api_key
    ):
        """
        When upstream returns 500, the OpenAI route must return
        {"error": {"message": ..., "type": "kiro_api_error", "code": 500}}.
        """
        error_body = json.dumps({"message": "Upstream failure"}).encode()
        mock_resp = _mock_upstream_response(status_code=500, body=error_body)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "test_openai_500_unique_w5j1"}],
                },
            )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data, f"Missing 'error' key in: {data}"
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert data["error"]["type"] == "kiro_api_error"
        assert data["error"]["code"] == 500

    @pytest.mark.slow
    def test_upstream_503_returns_openai_error_envelope(
        self, test_client, valid_proxy_api_key
    ):
        """Upstream 503 must be forwarded as OpenAI error envelope with code=503."""
        error_body = json.dumps({"message": "Service unavailable"}).encode()
        mock_resp = _mock_upstream_response(status_code=503, body=error_body)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "test_openai_503_unique_r2h8"}],
                },
            )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 503

    @pytest.mark.slow
    def test_upstream_error_with_non_json_body(
        self, test_client, valid_proxy_api_key
    ):
        """
        When upstream returns a non-JSON error body, the route must still
        return a valid OpenAI error envelope (not crash).
        """
        mock_resp = _mock_upstream_response(status_code=502, body=b"Bad Gateway")

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "test_openai_non_json_body_unique_k6t4"}],
                },
            )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "Bad Gateway" in data["error"]["message"]


# ---------------------------------------------------------------------------
# OpenAI route -- streaming path
# ---------------------------------------------------------------------------

class TestOpenAIStreamingPath:
    """routes_openai.py lines 923-981: streaming SSE path."""

    @pytest.mark.slow
    def test_streaming_returns_event_stream_content_type(
        self, test_client, valid_proxy_api_key
    ):
        """OpenAI streaming requests must return Content-Type: text/event-stream."""
        chunks = _make_openai_sse_chunks("Hello")
        mock_resp = _mock_upstream_response(status_code=200, sse_chunks=chunks)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "test_openai_streaming_content_type"}],
                },
            )

        assert response.status_code == 200
        ct = response.headers.get("content-type", "")
        assert "text/event-stream" in ct, f"Expected text/event-stream, got: {ct}"

    @pytest.mark.slow
    def test_streaming_body_contains_done_sentinel(
        self, test_client, valid_proxy_api_key
    ):
        """OpenAI streaming response body must contain 'data: [DONE]' sentinel."""
        chunks = _make_openai_sse_chunks("test")
        mock_resp = _mock_upstream_response(status_code=200, sse_chunks=chunks)

        with patch(
            "kiro.http_client.KiroHttpClient.request_with_retry",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = mock_resp
            response = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "test_openai_streaming_done_sentinel"}],
                },
            )

        assert response.status_code == 200
        body = response.text
        assert "[DONE]" in body, f"Missing [DONE] sentinel in: {body[:300]}"
