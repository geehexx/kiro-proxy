
"""
Unit tests for kiro/response_cache.py.

Covers: key determinism, tool signature, LRU eviction, byte budget, TTL
expiry, session invalidation, thread safety, oversize rejection.
"""

import threading
import time

import pytest

from kiro.response_cache import (
    MAX_ENTRY_BYTES_HARDCAP,
    ResponseCache,
    _canonical,
    _tool_signature,
    make_key,
)


class TestKeyDeterminism:
    """make_key must be deterministic and sensitive to all inputs."""

    def _args(self, **overrides):
        base = dict(
            session_id="sess-abc",
            system=[{"type": "text", "text": "You are helpful."}],
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ],
            model="claude-sonnet-4-5",
            max_tokens=4096,
            tools=None,
        )
        base.update(overrides)
        return base

    def test_same_inputs_produce_same_key(self):
        args = self._args()
        assert make_key(**args) == make_key(**args)

    def test_trailing_user_turn_included_in_key(self):
        """Trailing user turn MUST affect the key (correctness).

        Prefix-only keying returned the same cached response for different
        trailing questions — wrong answer bug. Full-request keying is the
        only correctness-safe option for response caching without upstream
        KV-cache support.
        """
        args_a = self._args()
        args_b = self._args(
            messages=args_a["messages"][:-1]
            + [{"role": "user", "content": "something totally different"}]
        )
        assert make_key(**args_a) != make_key(**args_b)

    def test_trailing_assistant_message_affects_key(self):
        """Assistant trailing turn affects the key (all messages are part of the hash)."""
        args_a = self._args(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello A"},
            ]
        )
        args_b = self._args(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello B"},
            ]
        )
        assert make_key(**args_a) != make_key(**args_b)

    def test_session_id_affects_key(self):
        a = self._args(session_id="s-one")
        b = self._args(session_id="s-two")
        assert make_key(**a) != make_key(**b)

    def test_system_affects_key(self):
        a = self._args(system=[{"type": "text", "text": "A"}])
        b = self._args(system=[{"type": "text", "text": "B"}])
        assert make_key(**a) != make_key(**b)

    def test_model_affects_key(self):
        assert make_key(**self._args(model="m1")) != make_key(
            **self._args(model="m2")
        )

    def test_max_tokens_affects_key(self):
        assert make_key(**self._args(max_tokens=100)) != make_key(
            **self._args(max_tokens=200)
        )

    def test_dict_ordering_does_not_affect_key(self):
        """Canonical JSON must normalise key order."""
        a = self._args(
            system=[{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]
        )
        b = self._args(
            system=[{"cache_control": {"type": "ephemeral"}, "text": "hi", "type": "text"}]
        )
        assert make_key(**a) == make_key(**b)

    def test_message_prefix_change_affects_key(self):
        """Changing earlier messages (the cacheable prefix) invalidates the key."""
        a = self._args()
        b = self._args(
            messages=[
                {"role": "user", "content": "different opener"},
                a["messages"][1],
                a["messages"][2],
            ]
        )
        assert make_key(**a) != make_key(**b)


class TestToolSignature:
    def test_no_tools_constant_signature(self):
        assert _tool_signature(None) == "no-tools"
        assert _tool_signature([]) == "no-tools"

    def test_tool_order_independent(self):
        tools_a = [
            {"name": "Bash", "description": "Run shell", "input_schema": {}},
            {"name": "Read", "description": "Read file", "input_schema": {}},
        ]
        tools_b = [tools_a[1], tools_a[0]]
        assert _tool_signature(tools_a) == _tool_signature(tools_b)

    def test_description_change_invalidates(self):
        a = [{"name": "Bash", "description": "Run shell", "input_schema": {}}]
        b = [{"name": "Bash", "description": "Run shell v2", "input_schema": {}}]
        assert _tool_signature(a) != _tool_signature(b)

    def test_schema_change_invalidates(self):
        a = [{"name": "Bash", "description": "x", "input_schema": {"type": "object"}}]
        b = [
            {
                "name": "Bash",
                "description": "x",
                "input_schema": {"type": "object", "required": ["command"]},
            }
        ]
        assert _tool_signature(a) != _tool_signature(b)

    def test_tool_signature_integrated_into_key(self):
        args = dict(
            session_id="s",
            system=None,
            messages=[{"role": "user", "content": "q"}],
            model="m",
            max_tokens=10,
        )
        k_empty = make_key(tools=None, **args)
        k_with_tool = make_key(
            tools=[{"name": "Bash", "description": "d", "input_schema": {}}],
            **args,
        )
        assert k_empty != k_with_tool


class TestLRUEviction:
    def test_max_entries_evicts_oldest(self):
        cache = ResponseCache(max_entries=3, max_bytes=10_000_000, ttl_seconds=3600)
        cache.put("k1", b"a")
        cache.put("k2", b"b")
        cache.put("k3", b"c")
        cache.put("k4", b"d")  # forces eviction of k1
        assert cache.get("k1") is None  # miss
        assert cache.get("k2").body == b"b"
        assert cache.get("k3").body == b"c"
        assert cache.get("k4").body == b"d"
        assert cache.stats()["evictions"] == 1

    def test_get_bumps_entry_to_newest(self):
        cache = ResponseCache(max_entries=2, max_bytes=10_000_000, ttl_seconds=3600)
        cache.put("old", b"x")
        cache.put("mid", b"y")
        # Access "old" to bump it.
        cache.get("old")
        cache.put("new", b"z")  # should evict "mid", not "old"
        assert cache.get("old") is not None
        assert cache.get("mid") is None
        assert cache.get("new") is not None

    def test_byte_budget_evicts(self):
        cache = ResponseCache(max_entries=100, max_bytes=3_000, ttl_seconds=3600)
        cache.put("k1", b"x" * 1_000)
        cache.put("k2", b"y" * 1_000)
        cache.put("k3", b"z" * 1_000)
        cache.put("k4", b"w" * 1_000)  # total 4000 > 3000 -> evicts k1
        assert cache.get("k1") is None
        assert cache.stats()["total_bytes"] <= 3_000


class TestOversize:
    def test_entry_over_max_entry_bytes_rejected(self):
        cache = ResponseCache(
            max_entries=100,
            max_bytes=10_000_000,
            ttl_seconds=3600,
            max_entry_bytes=1_024,
        )
        # 2 KiB payload, max_entry_bytes=1 KiB -> reject.
        ok = cache.put("big", b"x" * 2_048)
        assert ok is False
        assert cache.get("big") is None

    def test_hardcap_on_max_entry_bytes(self):
        with pytest.raises(ValueError):
            ResponseCache(max_entry_bytes=MAX_ENTRY_BYTES_HARDCAP + 1)


class TestTTL:
    def test_expired_entry_returns_none(self):
        cache = ResponseCache(ttl_seconds=1)
        cache.put("k", b"v")
        # Back-date the entry by tampering with stored entry timestamp.
        with cache._lock:  # noqa: SLF001 - test-only access
            cache._entries["k"].created_at = time.time() - 2
        assert cache.get("k") is None
        assert cache.stats()["misses"] == 1

    def test_fresh_entry_hits(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("k", b"v")
        assert cache.get("k").body == b"v"
        assert cache.stats()["hits"] == 1


class TestSessionInvalidation:
    def test_invalidate_clears_all_phase1(self):
        cache = ResponseCache()
        cache.put("k1", b"a")
        cache.put("k2", b"b")
        cleared = cache.invalidate_session("any-session-id")
        assert cleared == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_entries": 0},
            {"max_bytes": 100},  # < 1024
            {"ttl_seconds": 0},
            {"max_entry_bytes": 10},  # < 1024
        ],
    )
    def test_invalid_constructor_args(self, kwargs):
        with pytest.raises(ValueError):
            ResponseCache(**kwargs)


class TestThreadSafety:
    def test_concurrent_puts_no_corruption(self):
        cache = ResponseCache(max_entries=500, max_bytes=10_000_000)

        def worker(thread_id: int) -> None:
            for i in range(50):
                cache.put(f"t{thread_id}-{i}", b"x" * 10)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.stats()
        # Shouldn't crash, shouldn't over-count bytes.
        assert stats["entries"] <= 500
        assert stats["total_bytes"] <= 10_000_000

    def test_concurrent_get_put(self):
        cache = ResponseCache()
        cache.put("k", b"hello")
        failures: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(200):
                    cache.get("k")
            except Exception as exc:  # noqa: BLE001 - capture in test
                failures.append(exc)

        def writer() -> None:
            try:
                for i in range(200):
                    cache.put(f"k{i}", b"v" * 10)
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)] + [
            threading.Thread(target=writer) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not failures


class TestHeaders:
    def test_headers_roundtrip(self):
        cache = ResponseCache()
        cache.put("k", b"body", headers={"Content-Type": "application/json"})
        entry = cache.get("k")
        assert entry.headers == {"Content-Type": "application/json"}


class TestCanonical:
    def test_canonical_is_sorted(self):
        s = _canonical({"b": 1, "a": 2})
        assert s == '{"a":2,"b":1}'


class TestClear:
    def test_clear_resets_all_state(self):
        cache = ResponseCache()
        cache.put("a", b"x")
        cache.get("a")  # bump hits
        cache.get("missing")  # bump misses
        cache.clear()
        stats = cache.stats()
        assert stats == {
            "entries": 0,
            "total_bytes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "stream_hits": 0,
            "stream_misses": 0,
            "nonstream_hits": 0,
            "nonstream_misses": 0,
        }


class TestCacheKeyNormalization:
    """make_key normalizes whitespace so minor formatting differences don't cause misses."""

    def _base_args(self):
        return dict(
            session_id="sess-1",
            model="claude-sonnet-4.6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    def test_system_trailing_newline_same_key(self):
        args = self._base_args()
        k1 = make_key(**args, system="You are helpful.")
        k2 = make_key(**args, system="You are helpful.\n")
        assert k1 == k2

    def test_system_leading_whitespace_same_key(self):
        args = self._base_args()
        k1 = make_key(**args, system="You are helpful.")
        k2 = make_key(**args, system="  You are helpful.")
        assert k1 == k2

    def test_system_internal_whitespace_collapsed(self):
        args = self._base_args()
        k1 = make_key(**args, system="You are  helpful.")
        k2 = make_key(**args, system="You are helpful.")
        assert k1 == k2

    def test_message_text_trailing_whitespace_same_key(self):
        k1 = make_key(session_id="sess-1", model="claude-sonnet-4.6", max_tokens=1024,
                      system=None, messages=[{"role": "user", "content": "Hello"}])
        k2 = make_key(session_id="sess-1", model="claude-sonnet-4.6", max_tokens=1024,
                      system=None, messages=[{"role": "user", "content": "Hello  "}])
        assert k1 == k2

    def test_different_content_different_key(self):
        args = self._base_args()
        k1 = make_key(**args, system="You are helpful.")
        k2 = make_key(**args, system="You are not helpful.")
        assert k1 != k2

    def test_system_list_text_block_normalized(self):
        args = self._base_args()
        k1 = make_key(**args, system=[{"type": "text", "text": "Be concise."}])
        k2 = make_key(**args, system=[{"type": "text", "text": "Be concise.\n"}])
        assert k1 == k2

    def test_message_content_list_text_block_normalized(self):
        msgs1 = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        msgs2 = [{"role": "user", "content": [{"type": "text", "text": "Hello\n"}]}]
        k1 = make_key(session_id="sess-1", model="claude-sonnet-4.6", max_tokens=1024,
                      system=None, messages=msgs1)
        k2 = make_key(session_id="sess-1", model="claude-sonnet-4.6", max_tokens=1024,
                      system=None, messages=msgs2)
        assert k1 == k2

class TestPerTypeStats:
    """E5: streaming vs non-streaming get/put split — populated when callers
    pass `streaming=` to get(). stats() exposes stream_* and nonstream_*
    counters separately from the aggregate hits/misses."""

    def test_streaming_hit_increments_stream_counters(self):
        cache = ResponseCache()
        cache.put("k", b"v")
        cache.get("k", streaming=True)
        s = cache.stats()
        assert s["hits"] == 1
        assert s["stream_hits"] == 1
        assert s["nonstream_hits"] == 0

    def test_nonstream_default_increments_nonstream_counter(self):
        cache = ResponseCache()
        cache.put("k", b"v")
        cache.get("k")  # default: streaming=False
        s = cache.stats()
        assert s["hits"] == 1
        assert s["nonstream_hits"] == 1
        assert s["stream_hits"] == 0

    def test_miss_increments_correct_per_type_miss_counter(self):
        cache = ResponseCache()
        cache.get("missing", streaming=True)
        cache.get("missing", streaming=False)
        s = cache.stats()
        assert s["misses"] == 2
        assert s["stream_misses"] == 1
        assert s["nonstream_misses"] == 1

    def test_aggregate_equals_sum_of_per_type(self):
        cache = ResponseCache()
        cache.put("k1", b"v")
        cache.get("k1", streaming=True)
        cache.get("k1", streaming=False)
        cache.get("missing-stream", streaming=True)
        cache.get("missing-non", streaming=False)
        s = cache.stats()
        assert s["hits"] == s["stream_hits"] + s["nonstream_hits"]
        assert s["misses"] == s["stream_misses"] + s["nonstream_misses"]

    def test_clear_zeroes_per_type_counters(self):
        cache = ResponseCache()
        cache.put("k", b"v")
        cache.get("k", streaming=True)
        cache.get("missing", streaming=False)
        cache.clear()
        s = cache.stats()
        assert s["stream_hits"] == 0
        assert s["stream_misses"] == 0
        assert s["nonstream_hits"] == 0
        assert s["nonstream_misses"] == 0


class TestMutationKilling:
    """Targeted tests to kill surviving mutants in response_cache.py."""

    # --- TTL expiry counter accumulation (misses += 1, not = 1) ---

    def test_expired_entry_accumulates_misses(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("k1", b"v1")
        cache.put("k2", b"v2")
        # Back-date both entries
        with cache._lock:  # noqa: SLF001
            cache._entries["k1"].created_at = time.time() - 7200
            cache._entries["k2"].created_at = time.time() - 7200
        cache.get("k1")
        cache.get("k2")
        assert cache.stats()["misses"] == 2

    def test_expired_stream_miss_accumulates(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("k", b"v")
        with cache._lock:  # noqa: SLF001
            cache._entries["k"].created_at = time.time() - 7200
        cache.get("k", streaming=True)
        cache.get("k", streaming=True)  # second miss after re-put would need re-put; just miss twice
        assert cache.stats()["stream_misses"] >= 1

    # --- put() size calculation: + not - ---

    def test_put_size_includes_header_bytes(self):
        cache = ResponseCache(max_entries=100, max_bytes=10_000_000, max_entry_bytes=1024)
        # body=10 bytes, header key+val = 30 bytes → total 40 bytes, fits in 1024
        ok = cache.put("k", b"x" * 10, headers={"Content-Type": "application/json"})
        assert ok is True
        assert cache.stats()["total_bytes"] == 10 + len("Content-Type") + len("application/json")

    def test_put_size_boundary_exact_max_entry_bytes_allowed(self):
        # size == max_entry_bytes should be allowed (> not >=)
        cache = ResponseCache(max_entries=100, max_bytes=10_000_000, max_entry_bytes=1024)
        ok = cache.put("k", b"x" * 1024)
        assert ok is True

    def test_put_size_one_over_max_entry_bytes_rejected(self):
        cache = ResponseCache(max_entries=100, max_bytes=10_000_000, max_entry_bytes=1024)
        ok = cache.put("k", b"x" * 1025)
        assert ok is False

    # --- invalidate_session sets total_bytes=0, not None ---

    def test_invalidate_session_resets_total_bytes_to_zero(self):
        cache = ResponseCache()
        cache.put("k", b"hello")
        cache.invalidate_session("any")
        stats = cache.stats()
        assert stats["total_bytes"] == 0
        assert stats["entries"] == 0

    # --- _tool_signature: non-dict tools skipped with continue not break ---

    def test_tool_signature_skips_non_dict_tools(self):
        from kiro.response_cache import _tool_signature
        tools_with_non_dict = [
            "not-a-dict",
            {"name": "Bash", "description": "run", "input_schema": {}},
        ]
        sig = _tool_signature(tools_with_non_dict)
        sig_clean = _tool_signature([{"name": "Bash", "description": "run", "input_schema": {}}])
        assert sig == sig_clean

    def test_tool_signature_missing_name_uses_empty_string(self):
        from kiro.response_cache import _tool_signature
        sig = _tool_signature([{"description": "x", "input_schema": {}}])
        assert isinstance(sig, str) and len(sig) == 16

    # --- _normalize_text: replacement is single space ---

    def test_normalize_text_collapses_to_single_space(self):
        from kiro.response_cache import _normalize_text
        assert _normalize_text("a  b") == "a b"
        assert _normalize_text("a\t\tb") == "a b"
        assert _normalize_text("  hello  ") == "hello"

    # --- _normalize_system: AND not OR for dict+type check ---

    def test_normalize_system_non_dict_block_not_normalized(self):
        from kiro.response_cache import _normalize_system
        result = _normalize_system(["not-a-dict", {"type": "text", "text": "hi  "}])
        assert result[0] == "not-a-dict"
        assert result[1]["text"] == "hi"

    # --- _normalize_messages: non-dict messages preserved as-is ---

    def test_normalize_messages_preserves_non_dict_messages(self):
        from kiro.response_cache import _normalize_messages
        result = _normalize_messages(["not-a-dict", {"role": "user", "content": "hi  "}])
        assert result[0] == "not-a-dict"
        assert result[1]["content"] == "hi"

    def test_normalize_messages_text_block_normalized_not_replaced_with_none(self):
        from kiro.response_cache import _normalize_messages
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello  "}]}]
        result = _normalize_messages(msgs)
        block = result[0]["content"][0]
        assert block is not None
        assert block["text"] == "hello"

    # --- _strip_nondeterministic: UUID/timestamp replacement ---

    def test_strip_nondeterministic_replaces_uuid(self):
        from kiro.response_cache import _strip_nondeterministic
        text = "id=550e8400-e29b-41d4-a716-446655440000 done"
        result = _strip_nondeterministic(text)
        assert "<uuid>" in result
        assert "550e8400" not in result

    def test_strip_nondeterministic_replaces_iso_timestamp(self):
        from kiro.response_cache import _strip_nondeterministic
        text = "at 2024-01-15T10:30:00Z completed"
        result = _strip_nondeterministic(text)
        assert "<timestamp>" in result

    def test_strip_nondeterministic_replaces_unix_ts(self):
        from kiro.response_cache import _strip_nondeterministic
        # regex matches 11-digit unix ms timestamps in range 16000000000-19999999999
        text = "ts=17000000000 done"
        result = _strip_nondeterministic(text)
        assert "<unix_ts>" in result

    def test_strip_nondeterministic_preserves_other_content(self):
        from kiro.response_cache import _strip_nondeterministic
        text = "hello world 42"
        assert _strip_nondeterministic(text) == text

    # --- make_key: tool_result content stripped of non-deterministic fields ---

    def test_make_key_tool_result_uuid_normalized(self):
        msgs_a = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x",
             "content": "id=550e8400-e29b-41d4-a716-446655440000"}
        ]}]
        msgs_b = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x",
             "content": "id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
        ]}]
        k1 = make_key(session_id="s", system=None, messages=msgs_a, model="m", max_tokens=10)
        k2 = make_key(session_id="s", system=None, messages=msgs_b, model="m", max_tokens=10)
        assert k1 == k2


class TestPropertyBased:
    """Property-based tests using Hypothesis for make_key determinism and normalization."""

    def test_make_key_deterministic(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(
            session_id=st.text(min_size=1, max_size=20),
            system=st.one_of(st.none(), st.text(max_size=50)),
            content=st.text(min_size=1, max_size=50),
            model=st.sampled_from(["claude-sonnet-4.6", "claude-haiku-4.5"]),
            max_tokens=st.integers(min_value=1, max_value=8192),
        )
        @settings(max_examples=100)
        def inner(session_id: str, system: str | None, content: str, model: str, max_tokens: int) -> None:
            msgs = [{"role": "user", "content": content}]
            k1 = make_key(session_id=session_id, system=system, messages=msgs,
                          model=model, max_tokens=max_tokens)
            k2 = make_key(session_id=session_id, system=system, messages=msgs,
                          model=model, max_tokens=max_tokens)
            assert k1 == k2

        inner()

    def test_make_key_whitespace_normalized_system(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(
            base=st.text(min_size=1, max_size=50, alphabet=st.characters(
                blacklist_categories=("Cs",), blacklist_characters="\x00"
            )),
            leading=st.text(min_size=0, max_size=5, alphabet=" \t"),
            trailing=st.text(min_size=0, max_size=5, alphabet=" \n"),
        )
        @settings(max_examples=100)
        def inner(base, leading, trailing):
            msgs = [{"role": "user", "content": "hello"}]
            k1 = make_key(session_id="s", system=base, messages=msgs, model="m", max_tokens=10)
            k2 = make_key(session_id="s", system=leading + base + trailing,
                          messages=msgs, model="m", max_tokens=10)
            # Keys should be equal when the only difference is leading/trailing whitespace
            # (normalize_text strips both ends)
            assert k1 == k2

        inner()

