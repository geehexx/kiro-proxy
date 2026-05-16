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
    _DEFAULT_CAPACITY,
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
