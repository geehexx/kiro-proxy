
"""Response cache for kiro-proxy.

Caches upstream responses keyed by request prefix (system + messages except
the trailing user turn + model + max_tokens + tool signature). Session-scoped
via session_id to prevent cross-session context leaks.

In-memory LRU, bounded by entry count and total bytes.
Persisted to disk on shutdown and loaded on startup so cache survives restarts.

Why this exists: Anthropic cache_control beta is dropped by this gateway
(converters_anthropic.py lines 87 and 105), and AWS Q upstream does not
honour Anthropic prompt caching. The gateway-side response cache is the
only way to get prompt-cache-like behaviour on this stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle  # nosec B403 — local cache file written by this process, not untrusted input
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

MAX_ENTRY_BYTES_HARDCAP = 10 * 1024 * 1024

# Pickle format version — bump when CacheEntry fields change to force cache invalidation
_PICKLE_VERSION = 1

_logger = logging.getLogger(__name__)


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


import re as _re  # noqa: E402 — placed after _tool_signature to keep related cache-key helpers together

# Patterns for non-deterministic fields in tool_result content that vary
# across agent-loop replays but carry no semantic meaning for cache keying.
_UUID_RE = _re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    _re.IGNORECASE,
)
_ISO_TS_RE = _re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_UNIX_TS_RE = _re.compile(r"\b1[6-9]\d{9}(?:\.\d+)?\b")  # unix timestamps 2023-2033


def _strip_nondeterministic(text: str) -> str:
    """Replace UUIDs and timestamps in tool_result text with stable placeholders.

    Conservative: only replaces patterns that are structurally non-deterministic
    (UUIDs, ISO-8601 timestamps, unix epoch seconds). Does not touch numeric IDs
    or other content that may be semantically load-bearing.
    """
    text = _UUID_RE.sub("<uuid>", text)
    text = _ISO_TS_RE.sub("<timestamp>", text)
    text = _UNIX_TS_RE.sub("<unix_ts>", text)
    return text



def _normalize_text(value: Any) -> Any:
    """Normalize string whitespace for stable cache keys.

    Strips leading/trailing whitespace and collapses internal runs of
    whitespace to a single space. Applied to system prompts and message
    text content so minor formatting differences don't cause cache misses.
    """
    if isinstance(value, str):
        return _re.sub(r"\s+", " ", value.strip())
    return value


def _normalize_system(system: Any) -> Any:
    """Normalize system prompt for cache key stability."""
    if isinstance(system, str):
        return _normalize_text(system)
    if isinstance(system, list):
        result = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                result.append({**block, "text": _normalize_text(block.get("text", ""))})
            else:
                result.append(block)
        return result
    return system


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: C901
    """Normalize message text content for cache key stability."""
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            result.append({**msg, "content": _normalize_text(content)})
        elif isinstance(content, list):
            normalized_blocks = []
            for block in content:
                if not isinstance(block, dict):
                    normalized_blocks.append(block)
                elif block.get("type") == "text":
                    normalized_blocks.append({**block, "text": _normalize_text(block.get("text", ""))})
                elif block.get("type") == "tool_result":
                    # Strip non-deterministic fields (UUIDs, timestamps) from tool_result
                    # content so agent-loop replays hit the cache on identical tool calls.
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        inner = _strip_nondeterministic(inner)
                    elif isinstance(inner, list):
                        stripped = []
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                stripped.append({**ib, "text": _strip_nondeterministic(ib.get("text", ""))})
                            else:
                                stripped.append(ib)
                        inner = stripped
                    normalized_blocks.append({**block, "content": inner})
                else:
                    normalized_blocks.append(block)
            result.append({**msg, "content": normalized_blocks})
        else:
            result.append(msg)
    return result


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def make_key(  # noqa: PLR0913
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
        "system": _normalize_system(system),
        "messages": _normalize_messages(list(messages)),
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
        # Cap max_entry_bytes to max_bytes so a single entry can never exceed the total budget.
        # This avoids a configuration where put() would evict the entire cache and still fail.
        max_entry_bytes = min(max_entry_bytes, max_bytes)
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
        # Per-type counters (E5 — break out streaming vs non-streaming).
        # Aggregate hits/misses are preserved for back-compat; per-type
        # counters are populated when callers pass `streaming=` to
        # get()/put(). Default False keeps behaviour unchanged.
        self.stream_hits = 0
        self.stream_misses = 0
        self.nonstream_hits = 0
        self.nonstream_misses = 0

    def get(self, key: str, *, streaming: bool = False) -> CacheEntry | None:
        """Return a cached entry by key, or None on miss or TTL expiry.

        Args:
            key: Cache key string.
            streaming: Pass True for streaming responses to track per-type counters.

        Returns:
            A copy of the CacheEntry, or None if not found or expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                if streaming:
                    self.stream_misses += 1
                else:
                    self.nonstream_misses += 1
                return None
            if time.time() - entry.created_at > self._ttl:
                self._entries.pop(key)
                self._total_bytes -= entry.size_bytes
                self.misses += 1
                if streaming:
                    self.stream_misses += 1
                else:
                    self.nonstream_misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            if streaming:
                self.stream_hits += 1
            else:
                self.nonstream_hits += 1
            # Return a copy so callers cannot mutate the cached headers dict.
            return CacheEntry(
                body=entry.body,
                headers=dict(entry.headers),
                created_at=entry.created_at,
                size_bytes=entry.size_bytes,
            )

    def put(
        self, key: str, body: bytes, headers: dict[str, str] | None = None
    ) -> bool:
        """Store a response in the cache.

        Args:
            key: Cache key string.
            body: Raw response bytes.
            headers: Optional response headers to cache alongside the body.

        Returns:
            True if the entry was stored, False if it exceeded max_entry_bytes.
        """
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

    def save(self, path: Path) -> bool:
        """Persist cache to disk using pickle.

        Saves entries, config, and version tag. Returns True on success.
        Failures are logged but never raised — cache persistence must not
        break the hot path.
        """
        try:
            with self._lock:
                entries: list[tuple[str, CacheEntry]] = list(self._entries.items())
                payload = {
                    "version": _PICKLE_VERSION,
                    "saved_at": time.time(),
                    "config": {
                        "max_entries": self._max_entries,
                        "max_bytes": self._max_bytes,
                        "ttl": self._ttl,
                        "max_entry_bytes": self._max_entry_bytes,
                    },
                    "entries": entries,
                    "total_bytes": self._total_bytes,
                    "hits": self.hits,
                    "misses": self.misses,
                    "evictions": self.evictions,
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301 — local cache file written by this process, not untrusted input
            tmp.replace(path)
            _logger.info(f"Cache saved: {len(entries)} entries → {path}")
            return True
        except Exception as e:
            _logger.warning(f"Cache save failed (non-fatal): {e}")
            return False

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        max_entries: int = 1000,
        max_bytes: int = 500 * 1024 * 1024,
        ttl_seconds: int = 3600,
        max_entry_bytes: int = 5 * 1024 * 1024,
    ) -> "ResponseCache":
        """Load a previously saved cache from disk.

        Returns a fresh empty cache on any error (corruption, version mismatch,
        missing file). Never raises.
        """
        cache = cls(
            max_entries=max_entries,
            max_bytes=max_bytes,
            ttl_seconds=ttl_seconds,
            max_entry_bytes=max_entry_bytes,
        )
        if not path.exists():
            return cache
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)  # nosec B301 — local cache file written by this process, not untrusted input
            if payload.get("version") != _PICKLE_VERSION:
                _logger.info(f"Cache version mismatch — starting fresh (got {payload.get('version')}, want {_PICKLE_VERSION})")
                return cache
            now = time.time()
            loaded = 0
            expired = 0
            for key, entry in payload.get("entries", []):
                if now - entry.created_at > ttl_seconds:
                    expired += 1
                    continue
                if entry.size_bytes > max_entry_bytes:
                    continue
                cache._entries[key] = entry
                cache._total_bytes += entry.size_bytes
                loaded += 1
            cache.hits = payload.get("hits", 0)
            cache.misses = payload.get("misses", 0)
            cache.evictions = payload.get("evictions", 0)
            _logger.info(f"Cache loaded: {loaded} entries ({expired} expired) from {path}")
        except Exception as e:
            _logger.warning(f"Cache load failed — starting fresh: {e}")
            cache = cls(
                max_entries=max_entries,
                max_bytes=max_bytes,
                ttl_seconds=ttl_seconds,
                max_entry_bytes=max_entry_bytes,
            )
        return cache

    def invalidate_session(self, _session_id: str) -> int:
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
        """Return a snapshot of cache counters (entries, bytes, hits, misses, evictions)."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_bytes": self._total_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                # E5: per-type breakdown — populated when callers pass
                # streaming= to get()/put(). Sum of stream_* + nonstream_*
                # equals hits/misses only after all callers have migrated.
                "stream_hits": self.stream_hits,
                "stream_misses": self.stream_misses,
                "nonstream_hits": self.nonstream_hits,
                "nonstream_misses": self.nonstream_misses,
            }

    def clear(self) -> None:
        """Evict all entries and reset all counters. Only call when the app is idle."""
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.stream_hits = 0
            self.stream_misses = 0
            self.nonstream_hits = 0
            self.nonstream_misses = 0
