# -*- coding: utf-8 -*-

"""
Unit tests for kiro/prefix_cache.py.

Covers: key determinism, tool signature, LRU eviction, byte budget, TTL
expiry, session invalidation, thread safety, oversize rejection.
"""

import threading
import time

import pytest

from kiro.prefix_cache import (
    MAX_ENTRY_BYTES_HARDCAP,
    PrefixCache,
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

    def test_trailing_user_turn_excluded_from_key(self):
        """Two requests differing only in trailing user message must share a key."""
        args_a = self._args()
        args_b = self._args(
            messages=args_a["messages"][:-1]
            + [{"role": "user", "content": "something totally different"}]
        )
        assert make_key(**args_a) == make_key(**args_b)

    def test_trailing_assistant_turn_kept_in_key(self):
        """If last message is NOT user (e.g. tool_result flow), all messages matter."""
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
        cache = PrefixCache(max_entries=3, max_bytes=10_000_000, ttl_seconds=3600)
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
        cache = PrefixCache(max_entries=2, max_bytes=10_000_000, ttl_seconds=3600)
        cache.put("old", b"x")
        cache.put("mid", b"y")
        # Access "old" to bump it.
        cache.get("old")
        cache.put("new", b"z")  # should evict "mid", not "old"
        assert cache.get("old") is not None
        assert cache.get("mid") is None
        assert cache.get("new") is not None

    def test_byte_budget_evicts(self):
        cache = PrefixCache(max_entries=100, max_bytes=3_000, ttl_seconds=3600)
        cache.put("k1", b"x" * 1_000)
        cache.put("k2", b"y" * 1_000)
        cache.put("k3", b"z" * 1_000)
        cache.put("k4", b"w" * 1_000)  # total 4000 > 3000 -> evicts k1
        assert cache.get("k1") is None
        assert cache.stats()["total_bytes"] <= 3_000


class TestOversize:
    def test_entry_over_max_entry_bytes_rejected(self):
        cache = PrefixCache(
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
            PrefixCache(max_entry_bytes=MAX_ENTRY_BYTES_HARDCAP + 1)


class TestTTL:
    def test_expired_entry_returns_none(self):
        cache = PrefixCache(ttl_seconds=1)
        cache.put("k", b"v")
        # Back-date the entry by tampering with stored entry timestamp.
        with cache._lock:  # noqa: SLF001 - test-only access
            cache._entries["k"].created_at = time.time() - 2
        assert cache.get("k") is None
        assert cache.stats()["misses"] == 1

    def test_fresh_entry_hits(self):
        cache = PrefixCache(ttl_seconds=3600)
        cache.put("k", b"v")
        assert cache.get("k").body == b"v"
        assert cache.stats()["hits"] == 1


class TestSessionInvalidation:
    def test_invalidate_clears_all_phase1(self):
        cache = PrefixCache()
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
            PrefixCache(**kwargs)


class TestThreadSafety:
    def test_concurrent_puts_no_corruption(self):
        cache = PrefixCache(max_entries=500, max_bytes=10_000_000)

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
        cache = PrefixCache()
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
        cache = PrefixCache()
        cache.put("k", b"body", headers={"Content-Type": "application/json"})
        entry = cache.get("k")
        assert entry.headers == {"Content-Type": "application/json"}


class TestCanonical:
    def test_canonical_is_sorted(self):
        s = _canonical({"b": 1, "a": 2})
        assert s == '{"a":2,"b":1}'


class TestClear:
    def test_clear_resets_all_state(self):
        cache = PrefixCache()
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
        }
