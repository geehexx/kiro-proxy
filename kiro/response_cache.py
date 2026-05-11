
"""
Response cache for kiro-gateway.

Caches upstream responses keyed by request prefix (system + messages except
the trailing user turn + model + max_tokens + tool signature). Session-scoped
via session_id to prevent cross-session context leaks.

In-memory LRU, bounded by entry count and total bytes.

Why this exists: Anthropic cache_control beta is dropped by this gateway
(converters_anthropic.py lines 87 and 105), and AWS Q upstream does not
honour Anthropic prompt caching. The gateway-side response cache is the
only way to get prompt-cache-like behaviour on this stack.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

MAX_ENTRY_BYTES_HARDCAP = 10 * 1024 * 1024


@dataclass
class CacheEntry:
    """Single cached upstream response plus metadata for replay."""

    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0


def _tool_signature(tools: list[dict[str, Any]] | None) -> str:
    """Stable signature of the tool list."""
    if not tools:
        return "no-tools"
    normalised = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        normalised.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {}),
            }
        )
    normalised.sort(key=lambda t: t["name"])
    raw = json.dumps(normalised, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def make_key(
    *,
    session_id: str,
    system: Any,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    thinking: Any = None,
) -> str:
    """Derive a deterministic SHA256 cache key.

    All messages are included in the key (full-request caching).
    Prefix-only keying is valid only for upstream KV-cache activation
    (Anthropic cache_control, which AWS Q does not support).
    """
    # All messages are included in the key (full-request caching).
    # Excluding the trailing turn is tempting for "prefix reuse" but is
    # incorrect for full-response caching: two conversations with the same
    # prefix but different trailing turns would return the same answer.
    # Prefix-only keying is valid only for upstream KV-cache activation
    # (Anthropic cache_control, which AWS Q does not support).
    key_material = {
        "session_id": session_id,
        "system": system,
        "messages": list(messages),
        "model": model,
        "max_tokens": max_tokens,
        "tool_signature": _tool_signature(tools),
        "thinking": thinking,
    }
    serialised = _canonical(key_material).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


class ResponseCache:
    """Thread-safe LRU response cache with TTL and byte budget."""

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        max_bytes: int = 500 * 1024 * 1024,
        ttl_seconds: int = 3600,
        max_entry_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1024:
            raise ValueError("max_bytes must be >= 1024")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        if max_entry_bytes < 1024 or max_entry_bytes > MAX_ENTRY_BYTES_HARDCAP:
            raise ValueError(
                f"max_entry_bytes must be between 1024 and {MAX_ENTRY_BYTES_HARDCAP}"
            )
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._max_entry_bytes = max_entry_bytes
        self._lock = RLock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if time.time() - entry.created_at > self._ttl:
                self._entries.pop(key)
                self._total_bytes -= entry.size_bytes
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(
        self, key: str, body: bytes, headers: dict[str, str] | None = None
    ) -> bool:
        size = len(body) + sum(
            len(k) + len(v) for k, v in (headers or {}).items()
        )
        if size > self._max_entry_bytes:
            return False

        with self._lock:
            if key in self._entries:
                old = self._entries.pop(key)
                self._total_bytes -= old.size_bytes

            entry = CacheEntry(
                body=body,
                headers=dict(headers or {}),
                size_bytes=size,
            )
            self._entries[key] = entry
            self._total_bytes += size

            while (
                len(self._entries) > self._max_entries
                or self._total_bytes > self._max_bytes
            ):
                oldest_key, oldest_entry = self._entries.popitem(last=False)
                self._total_bytes -= oldest_entry.size_bytes
                self.evictions += 1
                if oldest_key == key:
                    return False
            return True

    def invalidate_session(self, session_id: str) -> int:
        """Evict all entries for a session_id.

        Currently clears the whole cache because session_id is baked into
        the hash and cannot be targeted without a secondary index.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._total_bytes = 0
            return count

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_bytes": self._total_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
