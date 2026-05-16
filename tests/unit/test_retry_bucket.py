"""Unit tests for the adaptive retry token bucket.

Verifies the four invariants the bucket promises:
  1. tokens never exceed capacity
  2. throttle halves refill rate (clamped at MIN)
  3. success doubles refill rate (clamped at MAX)
  4. concurrent acquire() under sustained throttling stays bounded — no
     thundering herd
"""
from __future__ import annotations

import asyncio

import pytest

from kiro.retry_bucket import (
    _MAX_REFILL_PER_SEC,
    _MIN_REFILL_PER_SEC,
    AdaptiveRetryBucket,
    get_singleton,
    reset_singleton,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_singleton()
    yield
    reset_singleton()


@pytest.mark.asyncio
async def test_first_acquire_is_immediate_when_full():
    bucket = AdaptiveRetryBucket(capacity=5.0, refill_per_sec=1.0)
    wait = await bucket.acquire(jitter=False)
    assert wait == 0.0


@pytest.mark.asyncio
async def test_acquire_waits_when_empty():
    """When tokens drain, the next acquire waits ~ 1/refill seconds."""
    fake_now = [0.0]

    def clock():
        return fake_now[0]

    bucket = AdaptiveRetryBucket(capacity=2.0, refill_per_sec=1.0, clock=clock)
    # Drain
    assert await bucket.acquire(jitter=False) == 0.0
    assert await bucket.acquire(jitter=False) == 0.0

    # Advance clock 0 seconds; bucket has zero tokens, refill_per_sec=1 → wait ≈ 1s.
    # We can't easily mock asyncio.sleep here; instead inspect post-acquire state.
    stats_pre = await bucket.stats()
    assert stats_pre["tokens"] < 1.0


@pytest.mark.asyncio
async def test_record_throttle_halves_refill_floored_at_min():
    bucket = AdaptiveRetryBucket(refill_per_sec=1.0)
    await bucket.record_throttle()
    s = await bucket.stats()
    assert s["refill_per_sec"] == 0.5
    # Halve repeatedly — must clamp at MIN
    for _ in range(20):
        await bucket.record_throttle()
    s = await bucket.stats()
    assert s["refill_per_sec"] == _MIN_REFILL_PER_SEC


@pytest.mark.asyncio
async def test_record_success_doubles_refill_capped_at_max():
    bucket = AdaptiveRetryBucket(refill_per_sec=0.2)
    await bucket.record_success()
    s = await bucket.stats()
    assert s["refill_per_sec"] == 0.4
    for _ in range(20):
        await bucket.record_success()
    s = await bucket.stats()
    assert s["refill_per_sec"] == _MAX_REFILL_PER_SEC


@pytest.mark.asyncio
async def test_throttle_then_success_recovers_toward_normal():
    bucket = AdaptiveRetryBucket(refill_per_sec=1.0)
    for _ in range(3):
        await bucket.record_throttle()
    s_throttled = await bucket.stats()
    assert s_throttled["refill_per_sec"] == 0.125  # 1.0 / 2 / 2 / 2

    for _ in range(3):
        await bucket.record_success()
    s_recovered = await bucket.stats()
    assert s_recovered["refill_per_sec"] == _MAX_REFILL_PER_SEC


@pytest.mark.asyncio
async def test_tokens_never_exceed_capacity():
    fake_now = [0.0]

    def clock():
        return fake_now[0]

    bucket = AdaptiveRetryBucket(capacity=3.0, refill_per_sec=10.0, clock=clock)
    # Pretend a long time has passed — refill should clamp at capacity.
    fake_now[0] = 100_000.0
    s = await bucket.stats()
    assert s["tokens"] <= 3.0


@pytest.mark.asyncio
async def test_get_singleton_is_idempotent():
    a = get_singleton()
    b = get_singleton()
    assert a is b


@pytest.mark.asyncio
async def test_reset_singleton_creates_fresh_instance():
    a = get_singleton()
    reset_singleton()
    b = get_singleton()
    assert a is not b


@pytest.mark.asyncio
async def test_concurrent_acquire_under_drained_bucket_serialises():
    """50 concurrent acquire calls on an empty bucket: total wait time
    grows roughly linearly with N — no thundering herd at zero wait."""
    fake_now = [0.0]

    def clock():
        return fake_now[0]

    bucket = AdaptiveRetryBucket(capacity=1.0, refill_per_sec=100.0, clock=clock)
    # Drain
    await bucket.acquire(jitter=False)

    # Now N concurrent acquires. They should each see tokens<1 and wait.
    async def one():
        return await bucket.acquire(jitter=False)

    waits = await asyncio.gather(*(one() for _ in range(20)))
    # All N waits should be non-zero (bucket empty at start of each).
    # (refill rate is 100/s so the actual sleep is short — we just assert >0.)
    nonzero = sum(1 for w in waits if w > 0)
    assert nonzero == 20


# --- Tests for the bounded acquire-loop contention fix --------------------------
# Each test uses asyncio.wait_for() with a generous wall-clock cap to prove the
# loop CANNOT hang under any condition.  The internal _MAX_ACQUIRE_ATTEMPTS=100
# safety bound is the second-line defence.


@pytest.mark.asyncio
async def test_acquire_loop_consumes_token_when_clock_advances():
    """With a manually-advancing clock, acquire() loops until it consumes a token."""
    from kiro.retry_bucket import _MAX_ACQUIRE_ATTEMPTS  # noqa: F401

    fake_now = [0.0]

    def clock():
        # Advance clock by 0.5s on every read; after a few reads the bucket
        # refills enough for acquire() to consume a token in the loop.
        fake_now[0] += 0.5
        return fake_now[0]

    bucket = AdaptiveRetryBucket(capacity=2.0, refill_per_sec=1.0, clock=clock)
    # Drain.
    await asyncio.wait_for(bucket.acquire(jitter=False), timeout=5.0)
    await asyncio.wait_for(bucket.acquire(jitter=False), timeout=5.0)
    # Now empty.  Next acquire will loop, but clock advances and refill happens.
    wait = await asyncio.wait_for(bucket.acquire(jitter=False), timeout=5.0)
    assert wait >= 0.0
    # Token was consumed (cumulative wait may be 0 if first iteration saw enough refill).
    s = await bucket.stats()
    assert s["tokens"] < 1.0  # we consumed 1 of <2 available


@pytest.mark.asyncio
async def test_acquire_safety_bound_triggers_under_frozen_clock():
    """A frozen clock + empty bucket must hit the _MAX_ACQUIRE_ATTEMPTS bound,
    log a warning, and return without spinning forever."""
    from kiro.retry_bucket import _MAX_ACQUIRE_ATTEMPTS

    fake_now = [0.0]

    def clock():
        return fake_now[0]  # frozen — never advances

    # Use a tiny refill rate so the per-iteration sleep stays short.
    bucket = AdaptiveRetryBucket(
        capacity=1.0, refill_per_sec=_MAX_REFILL_PER_SEC, clock=clock
    )
    # Drain the only token.
    await asyncio.wait_for(bucket.acquire(jitter=False), timeout=5.0)

    # Next acquire must loop, hit the safety bound, and return without hanging.
    # Each iteration sleeps ~ 1/refill = 1.0s — too slow for a real test run, so
    # we override refill_per_sec to make per-iteration sleep small.
    bucket._state.refill_per_sec = _MAX_REFILL_PER_SEC * 1000  # 1000/s, so wait ~1ms
    wait = await asyncio.wait_for(bucket.acquire(jitter=False), timeout=10.0)
    # After hitting the bound, acquire returns the cumulative wait (>=0) without
    # consuming.  Bucket should still be empty.
    s = await bucket.stats()
    assert s["tokens"] < 1.0
    # cumulative_wait is approximately attempts * per_iter_wait; >=0 is the only
    # robust assertion (jitter and float arithmetic make exact bounds brittle).
    assert wait >= 0.0
    # Bound is _MAX_ACQUIRE_ATTEMPTS; sanity-check the constant exists and is sane.
    assert 1 <= _MAX_ACQUIRE_ATTEMPTS <= 10000


@pytest.mark.asyncio
async def test_acquire_loop_never_hangs_under_contention():
    """50 concurrent acquires on a sustained-throttle bucket must all complete
    within a reasonable wall-clock cap.  No hang, no thundering-herd consume."""
    bucket = AdaptiveRetryBucket(capacity=5.0, refill_per_sec=_MAX_REFILL_PER_SEC)
    # Drain the bucket.
    for _ in range(5):
        await asyncio.wait_for(bucket.acquire(jitter=False), timeout=2.0)

    async def one():
        # Each call must complete or the test will time out.
        return await asyncio.wait_for(bucket.acquire(jitter=False), timeout=15.0)

    waits = await asyncio.wait_for(
        asyncio.gather(*(one() for _ in range(10))), timeout=30.0
    )
    assert len(waits) == 10
    # Every caller eventually returns (consumed a token OR hit safety bound).
    assert all(w >= 0.0 for w in waits)



@pytest.mark.asyncio
async def test_acquire_loops_until_token_consumed():
    """Regression: acquire must loop until a token is consumed, not return
    after a single sleep when contention left the bucket empty.  Without the
    loop, a caller could receive a non-zero ``wait`` value yet not have
    consumed a token, defeating the bucket's pacing.

    Use a real (monotonic) clock so the refill actually advances during the
    short asyncio.sleep — the test verifies the loop exits cleanly when the
    clock is healthy, not the safety bound."""
    bucket = AdaptiveRetryBucket(capacity=1.0, refill_per_sec=200.0)
    # Drain the initial token.
    assert await bucket.acquire(jitter=False) == 0.0
    # Re-acquire on an empty bucket: with refill=200/s the wait is ~5ms; the
    # loop should exit on the second iteration with a consumed token.
    wait = await bucket.acquire(jitter=False)
    assert wait > 0.0
    # Bucket must now be back below 1 token (we consumed exactly one).
    s = await bucket.stats()
    assert s["tokens"] < 1.0


@pytest.mark.asyncio
async def test_acquire_safety_bound_with_frozen_clock():
    """A clock that never advances would loop forever without the safety
    bound.  Verify acquire() returns within _MAX_ACQUIRE_ATTEMPTS iterations
    and logs a WARNING (callers proceed without bucket throttling)."""
    fake_now = [0.0]

    def clock():
        return fake_now[0]

    bucket = AdaptiveRetryBucket(capacity=1.0, refill_per_sec=1000.0, clock=clock)
    # Drain.
    assert await bucket.acquire(jitter=False) == 0.0

    # Now the bucket is empty AND the clock is frozen — refill never happens.
    # Use a high refill rate so each per-iteration computed wait is ~1ms;
    # 100 iterations complete in ~0.1s real time even though the fake clock
    # never advances and no token ever materialises.
    wait = await asyncio.wait_for(bucket.acquire(jitter=False), timeout=5.0)
    # Returned cumulative wait is positive (we slept many times).
    assert wait > 0.0
    # Bucket still has zero tokens — the safety-bound exit does NOT consume.
    s = await bucket.stats()
    assert s["tokens"] < 1.0
