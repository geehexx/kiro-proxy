
"""
In-flight request deduplication (request coalescing).

When N concurrent requests have the same cache key, only one executes
upstream; the others await the shared Future and receive identical
responses.

Correctness-safe: identical inputs -> identical outputs, no semantic
guess work. Catches retry storms (tool-use loops, CI re-tries, duplicate
clicks) that the exact-request response cache cannot help with until
the first call has fully completed and populated the cache.

Complements ResponseCache: dedup runs FIRST (coalesce concurrent dupes),
then ResponseCache.put stores the result for the next identical request.

SoTA: LiteLLM / Portkey / vLLM request coalescing. Typical production
contribution: 1-3% additional invocation reduction, higher under retry
bursts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class InFlightDedup:
    """Per-key Future registry. Subsequent callers with the same key
    await the first caller's Future instead of firing their own upstream
    call.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def coalesce(
        self, key: str, upstream: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Execute `upstream` once per key for concurrent callers.

        First caller for a key runs upstream and records the result on
        the shared Future. Subsequent concurrent callers await that
        Future. Once upstream completes, the Future is dropped so the
        next call starts fresh (no stale cache).
        """
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                self.hits += 1
                # Release the lock BEFORE awaiting so other coalescers
                # can find the same future. Awaiting inside the lock
                # would serialise followers and defeat the dedup.
                existing_future = existing
                misses_branch = False
            else:
                future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
                self.misses += 1
                existing_future = None
                misses_branch = True

        if not misses_branch:
            return await existing_future

        try:
            result = await upstream()
            future.set_result(result)
            return result
        except asyncio.CancelledError:
            # CancelledError is BaseException in Python 3.11+; cancel the shared
            # future so followers get a fresh attempt rather than inheriting the
            # cancellation silently.
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                # Only drop if the Future we set is still the registered one.
                # A later coalesce() on the same key after this finished will
                # have installed a new Future for a fresh upstream call.
                if self._inflight.get(key) is future:
                    del self._inflight[key]

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "inflight": len(self._inflight),
        }

    def clear(self) -> None:
        self._inflight.clear()
        self.hits = 0
        self.misses = 0
