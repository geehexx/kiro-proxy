"""
Targeted tests for top coverage gaps identified by:
  uv run pytest --cov=kiro --cov-report=term-missing -q

Top 5 uncovered modules (by miss count):
  1. routes_anthropic.py  387 missed (44%)
  2. routes_openai.py     239 missed (43%)
  3. account_manager.py   106 missed (76%)
  4. streaming_anthropic.py 69 missed (76%)
  5. mcp_tools.py          44 missed (76%)

These tests target the highest-value reachable paths without real upstream calls.
"""


# ===========================================================================
# 1. routes_anthropic.py — count_tokens endpoint + system prompt + multi-turn
# ===========================================================================

class TestCountTokensEndpoint:
    """POST /v1/messages/count_tokens — pure local estimation, no upstream."""

    def test_count_tokens_simple(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] > 0

    def test_count_tokens_with_system_string(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["input_tokens"] > 0

    def test_count_tokens_with_system_blocks(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "system": [{"type": "text", "text": "You are a helpful assistant."}],
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["input_tokens"] > 0

    def test_count_tokens_multi_turn(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["input_tokens"] > 0

    def test_count_tokens_with_tools(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["input_tokens"] > 0

    def test_count_tokens_more_tokens_with_longer_message(self, test_client, valid_proxy_api_key):
        short_resp = test_client.post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        long_resp = test_client.post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hi " * 200}],
            },
        )
        assert short_resp.status_code == 200
        assert long_resp.status_code == 200
        assert long_resp.json()["input_tokens"] > short_resp.json()["input_tokens"]

    def test_count_tokens_requires_auth(self, test_client):
        response = test_client.post(
            "/v1/messages/count_tokens",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401


# ===========================================================================
# 2. routes_anthropic.py — /v1/messages request shape variants
# ===========================================================================

class TestAnthropicMessagesRequestVariants:
    """Exercises request-shape branches in the /v1/messages handler."""

    def test_system_prompt_string(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code not in (401, 422), response.text

    def test_system_prompt_blocks(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "system": [{"type": "text", "text": "You are a helpful assistant."}],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code not in (401, 422), response.text

    def test_multi_turn_conversation(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        assert response.status_code not in (401, 422), response.text

    def test_content_blocks_in_user_message(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello from a content block"}],
                    }
                ],
            },
        )
        assert response.status_code not in (401, 422), response.text

    def test_with_anthropic_tools(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
            },
        )
        assert response.status_code not in (401, 422), response.text

    def test_missing_messages_rejected(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-5", "max_tokens": 50},
        )
        assert response.status_code == 422

    def test_empty_messages_rejected(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": []},
        )
        assert response.status_code == 422


# ===========================================================================
# 3. routes_openai.py — /v1/chat/completions request shape variants
# ===========================================================================

class TestOpenAIChatCompletionsVariants:
    """Exercises request-shape branches in the /v1/chat/completions handler."""

    def test_system_plus_user(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert response.status_code not in (401, 404, 422), response.text

    def test_multi_turn(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        assert response.status_code not in (401, 404, 422), response.text

    def test_with_tools(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
            },
        )
        assert response.status_code not in (401, 404, 422), response.text

    def test_with_temperature_and_max_tokens(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "max_tokens": 100,
            },
        )
        assert response.status_code not in (401, 404, 422), response.text

    def test_stream_false_explicit(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code not in (401, 404, 422), response.text

    def test_missing_model_rejected(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 422

    def test_empty_messages_rejected(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4-5", "messages": []},
        )
        assert response.status_code == 422

    def test_invalid_api_key_rejected(self, test_client):
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key-xyz"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401


# ===========================================================================
# 4. streaming_anthropic.py — _extract_stream_telemetry_from_chunk (unit)
# ===========================================================================

class TestExtractStreamTelemetry:
    """Unit tests for the SSE telemetry extraction helper."""

    def _extract(self, chunk: str) -> dict:
        from kiro.routes_anthropic import _extract_stream_telemetry_from_chunk
        return _extract_stream_telemetry_from_chunk(chunk)

    def test_message_start_chunk(self):
        chunk = (
            'data: {"type":"message_start","message":{"id":"msg_abc","type":"message",'
            '"role":"assistant","content":[],"model":"claude-sonnet-4-5",'
            '"stop_reason":null,"stop_sequence":null,'
            '"usage":{"input_tokens":42,"output_tokens":0}}}\n\n'
        )
        result = self._extract(chunk)
        assert result.get("response_model") == "claude-sonnet-4-5"
        assert result.get("message_id") == "msg_abc"
        assert result.get("input_tokens") == 42

    def test_message_delta_chunk(self):
        chunk = (
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":17}}\n\n'
        )
        result = self._extract(chunk)
        assert result.get("output_tokens") == 17

    def test_content_block_delta_ignored(self):
        chunk = (
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        )
        result = self._extract(chunk)
        assert result == {}

    def test_done_sentinel_ignored(self):
        result = self._extract("data: [DONE]\n\n")
        assert result == {}

    def test_no_data_prefix_ignored(self):
        result = self._extract("event: ping\n\n")
        assert result == {}

    def test_malformed_json_ignored(self):
        result = self._extract("data: {not valid json}\n\n")
        assert result == {}

    def test_cache_tokens_extracted(self):
        chunk = (
            'data: {"type":"message_start","message":{"id":"msg_x","model":"claude-opus-4-5",'
            '"usage":{"input_tokens":10,"cache_read_input_tokens":5,'
            '"cache_creation_input_tokens":3,"output_tokens":0}}}\n\n'
        )
        result = self._extract(chunk)
        assert result.get("cache_read_input_tokens") == 5
        assert result.get("cache_creation_input_tokens") == 3


# ===========================================================================
# 5. account_manager.py — get_account_stats, get_all_available_models
# ===========================================================================

class TestAccountManagerCoverage:
    """Exercises AccountManager paths reachable via the health and models endpoints."""

    def test_health_returns_account_list(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "accounts" in data
        assert isinstance(data["accounts"], list)

    def test_models_endpoint_returns_non_empty_list(self, test_client, valid_proxy_api_key):
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0

    def test_models_include_required_fields(self, test_client, valid_proxy_api_key):
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert response.status_code == 200
        for model in response.json()["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"

    def test_models_include_context_window(self, test_client, valid_proxy_api_key):
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert response.status_code == 200
        for model in response.json()["data"]:
            assert "context_window" in model
            assert model["context_window"] > 0

    def test_models_1m_variants_present(self, test_client, valid_proxy_api_key):
        """[1m] bracket variants must appear in the listing for 1M-capable models."""
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert response.status_code == 200
        ids = {m["id"] for m in response.json()["data"]}
        # At least one [1m] variant should be present
        bracket_models = [mid for mid in ids if "[1m]" in mid]
        assert len(bracket_models) > 0, f"No [1m] models found in listing: {ids}"

    def test_models_1m_variants_have_1m_context_window(self, test_client, valid_proxy_api_key):
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert response.status_code == 200
        for model in response.json()["data"]:
            if "[1m]" in model["id"]:
                assert model["context_window"] == 1_000_000, (
                    f"{model['id']} should have 1M context window, got {model['context_window']}"
                )
