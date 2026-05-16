"""Adaptive retry token bucket — backport of aws-sdk RetryConfig::adaptive mode.

Behaviour: a single process-wide shared token bucket. Each retry attempt
calls ``acquire()`` which consumes a token. Tokens refill at a steady rate
(default 1/second). On throttling (429 / capacity error), the refill rate
HALVES — concurrent retries become rarer until the upstream recovers.
On a successful request, the refill rate doubles back toward normal.

Why: under fan-out (50+ concurrent sub-agent requests), fixed exponential
backoff causes a thundering herd — every request retries at exactly t+1s,
+2s, +4s. The token bucket smooths retry concurrency: when the bucket is
empty, requests wait, with the wait time growing as throttling persists.

This is a backport of the pattern described in
``data/basic-memory/research/2026-05-16-restart-recovery-and-tier1-hardening
/B-amazon-q-cli-backport-patterns.md`` §Pattern 2 (source:
``crates/chat-cli/src/api_client/mod.rs:679-683`` in aws/amazon-q-developer-cli
v1.19.7).

Defaults match aws-sdk-rust's ``RetryConfig::adaptive()`` semantics where
practical for an asyncio context: tokens=10 capacity, refill=1/s, halve-on-
throttle, double-on-success, min refill 0.1/s, max refill 1/s.

Disable via ``ADAPTIVE_RETRY_ENABLED=false`` to fall back to the existing
fixed-exponential path; the bucket is gated at the call site.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULT_CAPACITY = 10.0
_DEFAULT_REFILL_PER_SEC = 1.0
_MIN_REFILL_PER_SEC = 0.1
_MAX_REFILL_PER_SEC = 1.0
# Safety bound for the contention loop in acquire().  Real callers exit on
# the first or (rarely) second iteration; the bound only triggers under
# pathological conditions - e.g. a frozen test clock that never advances or
# a buggy caller holding the lock far longer than the refill rate.  Bounded
# at 100 to prevent infinite loops while sitting well above the real-world
# worst case under heavy concurrent fan-out.
_MAX_ACQUIRE_ATTEMPTS = 100


@dataclass
class BucketState:
    tokens: float
    refill_per_sec: float
    last_refill_at: float


class AdaptiveRetryBucket:
    """Process-wide token bucket for retry pacing.

    Thread-/coroutine-safe via an internal asyncio.Lock. Designed for one
    instance per process (the gateway has one HTTP client → one bucket).
    """

    def __init__(
        self,
        *,
        capacity: float = _DEFAULT_CAPACITY,
        refill_per_sec: float = _DEFAULT_REFILL_PER_SEC,
        clock: callable = time.monotonic,  # type: ignore[assignment]
    ) -> None:
        self._capacity = capacity
        self._initial_refill = refill_per_sec
        self._clock = clock
        self._lock = asyncio.Lock()
        self._state = BucketState(
            tokens=capacity,
            refill_per_sec=refill_per_sec,
            last_refill_at=clock(),
        )

    async def acquire(self, *, jitter: bool = True) -> float:
        """Consume one token, blocking if bucket empty.

        Returns the cumulative wait time in seconds (>=0).  With jitter the
        per-iteration wait gets +/-25% randomisation so concurrent callers
        don't unblock in lockstep when tokens trickle in.

        Loops internally until a token is consumed (or the safety bound is
        hit), so a single ``await acquire()`` always corresponds to exactly
        one consumed token.  Without the loop, contention with a sibling
        caller could let acquire() return without consuming - the caller
        would then issue an unpaced retry, defeating the bucket.
        """
        total_wait = 0.0
        for attempt in range(_MAX_ACQUIRE_ATTEMPTS):
            async with self._lock:
                self._refill_locked()
                if self._state.tokens >= 1.0:
                    self._state.tokens -= 1.0
                    return total_wait
                # Compute wait time to next token under the current refill rate.
                shortfall = 1.0 - self._state.tokens
                wait = shortfall / max(self._state.refill_per_sec, _MIN_REFILL_PER_SEC)

            if jitter:
                wait *= 1.0 + random.uniform(-0.25, 0.25)
            wait = max(wait, 0.0)

            await asyncio.sleep(wait)
            total_wait += wait

        # Safety bound hit.  This indicates either a frozen test clock that
        # never advances or a real-world starvation condition we want to
        # surface rather than spin on forever.  Log + return without consuming
        # so the caller can retry; in practice the test fixture should mock
        # the clock advance or the production caller should be diagnosed.
        logger.warning(
            "AdaptiveRetryBucket.acquire hit safety bound after "
            "%d attempts (total_wait=%.2fs); returning without token consume. "
            "Likely cause: frozen clock in tests or stuck refill loop.",
            _MAX_ACQUIRE_ATTEMPTS,
            total_wait,
        )
        return total_wait

    async def record_success(self) -> None:
        """Successful upstream response — restore refill rate toward normal."""
        async with self._lock:
            self._state.refill_per_sec = min(
                self._state.refill_per_sec * 2.0, _MAX_REFILL_PER_SEC
            )

    async def record_throttle(self) -> None:
        """Upstream returned 429 / capacity / throttle — halve refill rate."""
        async with self._lock:
            self._state.refill_per_sec = max(
                self._state.refill_per_sec / 2.0, _MIN_REFILL_PER_SEC
            )

    async def stats(self) -> dict:
        async with self._lock:
            self._refill_locked()
            return {
                "tokens": round(self._state.tokens, 3),
                "capacity": self._capacity,
                "refill_per_sec": round(self._state.refill_per_sec, 4),
                "initial_refill_per_sec": self._initial_refill,
            }

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._state.last_refill_at
        if elapsed > 0:
            self._state.tokens = min(
                self._capacity,
                self._state.tokens + elapsed * self._state.refill_per_sec,
            )
            self._state.last_refill_at = now


_singleton: AdaptiveRetryBucket | None = None


def get_singleton() -> AdaptiveRetryBucket:
    """Return the process-wide bucket. Created on first call."""
    global _singleton
    if _singleton is None:
        _singleton = AdaptiveRetryBucket()
    return _singleton


def reset_singleton() -> None:
    """Test hook — drops the singleton so tests get a fresh bucket."""
    global _singleton
    _singleton = None
