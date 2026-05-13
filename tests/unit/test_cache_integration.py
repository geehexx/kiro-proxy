
"""
Tests for kiro/cache_integration.py — the glue between the route
handler and the ResponseCache singleton.

Focus: key-derivation contract, cache hit/miss control flow, and the
Hypothesis invariants that must survive any future keying-scheme change.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kiro.cache_integration import (
    compute_cache_key,
    derive_session_id,
    entry_to_response_body,
    store_cache,
    store_stream_cache,
    try_cache_lookup,
    try_stream_cache_lookup,
)
from kiro.response_cache import ResponseCache

# ---------------------------------------------------------------------------
# derive_session_id
# ---------------------------------------------------------------------------


class TestDeriveSessionId:
    def test_client_header_with_api_key_is_deterministic(self) -> None:
        # Same api_key + same header → same session ID
        sid1 = derive_session_id(api_key="key-abc", client_header="session-xyz")
        sid2 = derive_session_id(api_key="key-abc", client_header="session-xyz")
        assert sid1 == sid2

    def test_different_api_keys_same_header_get_different_sessions(self) -> None:
        # Security fix: two clients with different API keys but the same
        # X-Kiro-Session-Id header must NOT share a cache bucket.
        sid_a = derive_session_id(api_key="key-a", client_header="session-xyz")
        sid_b = derive_session_id(api_key="key-b", client_header="session-xyz")
        assert sid_a != sid_b

    def test_no_api_key_header_only_is_deterministic(self) -> None:
        # When no api_key is present, header alone determines the session.
        sid1 = derive_session_id(None, "session-xyz")
        sid2 = derive_session_id(None, "session-xyz")
        assert sid1 == sid2

    def test_api_key_hashes_deterministically(self) -> None:
        assert derive_session_id("key-abc", None) == derive_session_id("key-abc", None)

    def test_different_keys_different_sessions(self) -> None:
        assert derive_session_id("key-a", None) != derive_session_id("key-b", None)

    def test_empty_falls_through_to_anonymous(self) -> None:
        assert derive_session_id(None, None) == "anonymous"

    def test_empty_string_treated_as_none(self) -> None:
        assert derive_session_id("", None) == "anonymous"


# ---------------------------------------------------------------------------
# compute_cache_key
# ---------------------------------------------------------------------------


SAMPLE = dict(
    session_id="sess-a",
    system="You are helpful.",
    messages=[{"role": "user", "content": "hi"}],
    model="claude-opus-4.7",
    max_tokens=1024,
    tools=None,
)


class TestComputeCacheKey:
    def test_deterministic(self) -> None:
        assert compute_cache_key(**SAMPLE) == compute_cache_key(**SAMPLE)

    def test_session_id_affects_key(self) -> None:
        a = compute_cache_key(**SAMPLE)
        b = compute_cache_key(**{**SAMPLE, "session_id": "sess-b"})
        assert a != b, "Session isolation broken — cross-tenant leakage risk"

    def test_model_affects_key(self) -> None:
        a = compute_cache_key(**SAMPLE)
        b = compute_cache_key(**{**SAMPLE, "model": "claude-sonnet-4.5"})
        assert a != b

    def test_max_tokens_affects_key(self) -> None:
        a = compute_cache_key(**SAMPLE)
        b = compute_cache_key(**{**SAMPLE, "max_tokens": 2048})
        assert a != b

    def test_trailing_user_turn_affects_key(self) -> None:
        """Regression gate — prefix-only keying would cause cross-answer collisions."""
        prefix = [{"role": "user", "content": "what is 2+2?"}]
        assistant_answer = [{"role": "assistant", "content": "4"}]
        a = compute_cache_key(
            **{**SAMPLE, "messages": prefix + assistant_answer + [{"role": "user", "content": "and 3+3?"}]}
        )
        b = compute_cache_key(
            **{**SAMPLE, "messages": prefix + assistant_answer + [{"role": "user", "content": "and 5+5?"}]}
        )
        assert a != b

    def test_tool_signature_affects_key(self) -> None:
        a = compute_cache_key(**SAMPLE)
        b = compute_cache_key(**{**SAMPLE, "tools": [{"name": "calc", "input_schema": {}}]})
        assert a != b

    def test_thinking_config_affects_key(self) -> None:
        """thinking-on and thinking-off requests must not share a cache bucket."""
        a = compute_cache_key(**SAMPLE)
        b = compute_cache_key(**{**SAMPLE, "thinking": {"type": "enabled", "budget_tokens": 8000}})
        assert a != b, "thinking config must be part of the cache key"

    def test_thinking_config_is_stable(self) -> None:
        """Same thinking config must produce the same key."""
        thinking = {"type": "enabled", "budget_tokens": 8000}
        a = compute_cache_key(**{**SAMPLE, "thinking": thinking})
        b = compute_cache_key(**{**SAMPLE, "thinking": thinking})
        assert a == b


# ---------------------------------------------------------------------------
# try_cache_lookup / store_cache / entry_to_response_body
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_cache() -> ResponseCache:
    return ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)


class TestCacheLookupAndStore:
    def test_lookup_miss_when_disabled(self) -> None:
        assert try_cache_lookup(None, "any-key") is None

    def test_store_noop_when_disabled(self) -> None:
        assert store_cache(None, "any-key", {"a": 1}) is False

    def test_round_trip(self, fresh_cache: ResponseCache) -> None:
        body = {"id": "msg_01", "type": "message", "usage": {"input_tokens": 10, "output_tokens": 20}}
        assert store_cache(fresh_cache, "k1", body) is True
        entry = try_cache_lookup(fresh_cache, "k1")
        assert entry is not None
        assert entry_to_response_body(entry) == body

    def test_miss_returns_none(self, fresh_cache: ResponseCache) -> None:
        assert try_cache_lookup(fresh_cache, "does-not-exist") is None

    def test_store_rejects_unserialisable(self, fresh_cache: ResponseCache) -> None:
        # Functions are not JSON-serialisable.
        body = {"callback": lambda x: x}
        assert store_cache(fresh_cache, "k2", body) is False

    def test_store_oversize_rejected(self) -> None:
        cache = ResponseCache(
            max_entries=10,
            max_bytes=1024 * 1024,
            ttl_seconds=60,
            max_entry_bytes=1024,
        )
        big_body = {"big": "a" * 2048}
        assert store_cache(cache, "k3", big_body) is False


# ---------------------------------------------------------------------------
# Hypothesis invariants
# ---------------------------------------------------------------------------


def _bounded_text() -> st.SearchStrategy[str]:
    return st.text(max_size=40)


class TestInvariants:
    @given(session=_bounded_text(), model=_bounded_text(), max_tokens=st.integers(min_value=1, max_value=200_000))
    def test_cache_key_is_deterministic(self, session: str, model: str, max_tokens: int) -> None:
        """compute_cache_key(x) == compute_cache_key(x) for all inputs."""
        args = dict(
            session_id=session,
            system=None,
            messages=[{"role": "user", "content": "x"}],
            model=model,
            max_tokens=max_tokens,
            tools=None,
        )
        assert compute_cache_key(**args) == compute_cache_key(**args)

    @given(a=_bounded_text(), b=_bounded_text())
    def test_different_sessions_different_keys(self, a: str, b: str) -> None:
        """If sessions differ, keys differ (cross-tenant isolation)."""
        if a == b:
            return
        key_a = compute_cache_key(
            session_id=a,
            system=None,
            messages=[{"role": "user", "content": "x"}],
            model="m",
            max_tokens=10,
        )
        key_b = compute_cache_key(
            session_id=b,
            system=None,
            messages=[{"role": "user", "content": "x"}],
            model="m",
            max_tokens=10,
        )
        assert key_a != key_b

    @given(content=_bounded_text())
    def test_round_trip_serialisation(self, content: str) -> None:
        """JSON round-trip preserves the response dict."""
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        body = {"content": [{"type": "text", "text": content}], "usage": {"input_tokens": 0, "output_tokens": 0}}
        key = "rt-" + str(hash(content) & 0xFFFFFFFF)
        store_cache(cache, key, body)
        entry = try_cache_lookup(cache, key)
        assert entry is not None
        assert entry_to_response_body(entry) == body


# ---------------------------------------------------------------------------
# Streaming cache helpers
# ---------------------------------------------------------------------------


class TestStreamCache:
    def test_lookup_miss_when_disabled(self) -> None:
        assert try_stream_cache_lookup(None, "any-key") is None

    def test_store_noop_when_disabled(self) -> None:
        assert store_stream_cache(None, "any-key", [b"chunk"]) is False

    def test_store_noop_on_empty_chunks(self) -> None:
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        assert store_stream_cache(cache, "k1", []) is False

    def test_round_trip(self) -> None:
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        chunks = [b"event: message_start\ndata: {}\n\n", b"event: message_stop\ndata: {}\n\n"]
        assert store_stream_cache(cache, "k1", chunks) is True
        result = try_stream_cache_lookup(cache, "k1")
        assert result == b"".join(chunks)

    def test_miss_returns_none(self) -> None:
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        assert try_stream_cache_lookup(cache, "does-not-exist") is None

    def test_oversize_rejected(self) -> None:
        cache = ResponseCache(
            max_entries=10,
            max_bytes=1024 * 1024,
            ttl_seconds=60,
            max_entry_bytes=1024,
        )
        big_chunks = [b"x" * 2048]
        assert store_stream_cache(cache, "k2", big_chunks) is False

    def test_idempotent_store(self) -> None:
        """Storing the same key twice replaces the entry (idempotency invariant)."""
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        chunks_v1 = [b"version-1"]
        chunks_v2 = [b"version-2"]
        store_stream_cache(cache, "k1", chunks_v1)
        store_stream_cache(cache, "k1", chunks_v2)
        result = try_stream_cache_lookup(cache, "k1")
        assert result == b"version-2"

    def test_stream_and_non_stream_share_same_key_space(self) -> None:
        """A streaming store must not collide with a non-streaming store on the same key."""
        cache = ResponseCache(max_entries=10, max_bytes=1024 * 1024, ttl_seconds=60)
        body = {"id": "msg_01", "type": "message"}
        chunks = [b"event: message_start\ndata: {}\n\n"]
        store_cache(cache, "k1", body)
        store_stream_cache(cache, "k2", chunks)
        # Each key is independent
        assert try_cache_lookup(cache, "k1") is not None
        assert try_stream_cache_lookup(cache, "k2") is not None
        assert try_stream_cache_lookup(cache, "k1") is not None  # raw bytes of JSON body
        assert try_cache_lookup(cache, "k2") is not None  # raw bytes of SSE chunks
