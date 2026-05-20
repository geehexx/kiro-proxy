"""Per-session concurrency limiter.

Caps the number of in-flight upstream calls per caller (session) so that
one session making many parallel tool calls cannot starve the shared
httpx connection pool for other sessions.

Usage (async-with, for work that completes inside a single handler):
    limiter = SessionLimiter(default_concurrency=8)
    async with limiter.acquire(session_id):
        response = await upstream.call(...)

Usage (explicit slot, for work that outlives the handler — e.g. the
body of a StreamingResponse which runs AFTER the handler returns):
    slot = await limiter.acquire_slot(session_id)
    try:
        initial = await upstream.call(...)
        async def stream_wrapper():
            try:
                async for chunk in initial.aiter_bytes(): yield chunk
            finally:
                slot.release()
        return StreamingResponse(stream_wrapper(), ...)
    except Exception:
        slot.release()
        raise

Semantics:
- One `asyncio.Semaphore(N)` per session_id. Sessions are created on
  first acquire() and never evicted (see §Limitations).
- Shared across event loops is NOT supported — the limiter and its
  semaphores must be created on the serving event loop.
- Order: acquire this AFTER the in-flight dedup check so dedup still
  collapses identical concurrent requests before the semaphore is
  contested.

Limitations:
- No eviction. Session dict grows monotonically. At 1000 sessions this is
  ~1 MB of semaphores — fine. If you need eviction, wrap with a TTL
  sweep (e.g., drop sessions idle >1h in a periodic task).
- No config-driven per-session override yet — all sessions get the same
  ceiling. Per-tenant overrides are a future add.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SessionLimiterSlot:
    """A held semaphore slot from SessionLimiter.acquire_slot().

    Call release() exactly once. Idempotent — second release is a no-op,
    so a `finally: slot.release()` after a success-path release is safe.
    """

    __slots__ = ("_sem", "_released")

    def __init__(self, sem: asyncio.Semaphore) -> None:
        self._sem = sem
        self._released = False

    def release(self) -> None:
        """Release the held semaphore slot. Idempotent — safe to call multiple times."""
        if not self._released:
            self._sem.release()
            self._released = True


class SessionLimiter:
    """Per-session asyncio.Semaphore registry.

    Thread/task-safe for the single event loop it was created on. Create
    one instance at app startup, attach to ``app.state.session_limiter``.
    """

    def __init__(self, default_concurrency: int = 8) -> None:
        if default_concurrency < 1:
            raise ValueError(
                f"default_concurrency must be >= 1, got {default_concurrency}"
            )
        self._default = default_concurrency
        self._sems: dict[str, asyncio.Semaphore] = {}
        # Dict mutations happen on the serving event loop so no lock is
        # strictly needed. Keeping a lock anyway for clarity — cheap on
        # the uncontested path.
        self._lock = asyncio.Lock()
        self.acquires = 0
        self.waits = 0

    async def _get_or_create(self, session_id: str) -> asyncio.Semaphore:
        async with self._lock:
            sem = self._sems.get(session_id)
            if sem is None:
                sem = asyncio.Semaphore(self._default)
                self._sems[session_id] = sem
            return sem

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Context manager that blocks until the session has capacity.

        Yields once the semaphore slot is held. Releases on __aexit__
        even if the caller raises.
        """
        sem = await self._get_or_create(session_id)
        self.acquires += 1
        if sem.locked():
            # Track only contested acquires so the hit-rate stat is meaningful.
            self.waits += 1
        async with sem:
            yield

    async def acquire_slot(self, session_id: str) -> SessionLimiterSlot:
        """Acquire a slot and return a releasable handle.

        Use this for work whose lifetime exceeds the current handler
        scope — e.g. StreamingResponse bodies that run AFTER the handler
        returns. Caller MUST ensure release() is eventually called
        (typically in a finally: block in the stream generator).
        """
        sem = await self._get_or_create(session_id)
        self.acquires += 1
        if sem.locked():
            self.waits += 1
        await sem.acquire()
        return SessionLimiterSlot(sem)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of limiter counters (sessions, acquires, waits, default_concurrency)."""
        return {
            "sessions": len(self._sems),
            "acquires": self.acquires,
            "waits": self.waits,
            "default_concurrency": self._default,
        }

    def clear(self) -> None:
        """Drop all session state. Only call when the app is idle."""
        self._sems.clear()
        self.acquires = 0
        self.waits = 0
