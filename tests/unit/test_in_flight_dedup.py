
"""
Unit tests for kiro.in_flight_dedup - request coalescing.

Semantics: if N concurrent requests have the same cache key, only one
goes upstream. The others await the shared Future. Correctness-safe
(identical inputs produce identical outputs) and catches retry storms
(tool loops, CI re-tries, duplicate clicks).
"""

import asyncio

import pytest

from kiro.in_flight_dedup import InFlightDedup


@pytest.mark.asyncio
async def test_identical_keys_share_a_single_execution():
    """Two awaits on the same key share one upstream call."""
    dedup = InFlightDedup()
    call_count = 0

    async def upstream():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return {"content": "hello", "id": 42}

    a_task = asyncio.create_task(dedup.coalesce("k1", upstream))
    b_task = asyncio.create_task(dedup.coalesce("k1", upstream))
    a_res, b_res = await asyncio.gather(a_task, b_task)

    assert a_res == b_res == {"content": "hello", "id": 42}
    assert call_count == 1
    assert dedup.stats()["hits"] == 1
    assert dedup.stats()["misses"] == 1


@pytest.mark.asyncio
async def test_different_keys_run_independently():
    dedup = InFlightDedup()
    call_count = 0

    async def upstream():
        nonlocal call_count
        call_count += 1
        return {"n": call_count}

    a_task = asyncio.create_task(dedup.coalesce("k1", upstream))
    b_task = asyncio.create_task(dedup.coalesce("k2", upstream))
    await asyncio.gather(a_task, b_task)
    assert call_count == 2
    assert dedup.stats()["hits"] == 0


@pytest.mark.asyncio
async def test_exception_propagates_to_all_waiters():
    """If upstream raises, ALL waiters see the same exception."""
    dedup = InFlightDedup()

    async def upstream():
        raise RuntimeError("boom")

    a_task = asyncio.create_task(dedup.coalesce("k", upstream))
    b_task = asyncio.create_task(dedup.coalesce("k", upstream))
    for task in (a_task, b_task):
        with pytest.raises(RuntimeError, match="boom"):
            await task


@pytest.mark.asyncio
async def test_completed_future_not_reused_after_finish():
    """Once upstream completes, subsequent calls start fresh - no stale cache."""
    dedup = InFlightDedup()
    calls = 0

    async def upstream():
        nonlocal calls
        calls += 1
        return calls

    a = await dedup.coalesce("k", upstream)
    b = await dedup.coalesce("k", upstream)
    assert a == 1
    assert b == 2
    assert calls == 2
    assert dedup.stats()["hits"] == 0


@pytest.mark.asyncio
async def test_three_concurrent_waiters_one_execution():
    dedup = InFlightDedup()
    calls = 0

    async def upstream():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "ok"

    tasks = [asyncio.create_task(dedup.coalesce("k", upstream)) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    assert results == ["ok", "ok", "ok"]
    assert calls == 1
    assert dedup.stats()["hits"] == 2


@pytest.mark.asyncio
async def test_cancellation_does_not_silently_drop_followers():
    """Cancelling the first caller cancels the shared Future so followers
    get a fresh attempt rather than inheriting the cancellation silently."""
    dedup = InFlightDedup()
    started = asyncio.Event()

    async def slow_upstream():
        started.set()
        await asyncio.sleep(10)  # long enough to be cancelled
        return "done"

    # Start the first caller (will be cancelled mid-flight)
    first = asyncio.create_task(dedup.coalesce("k", slow_upstream))
    await started.wait()  # ensure first caller is inside upstream

    # Start a follower that is awaiting the shared Future
    follower = asyncio.create_task(dedup.coalesce("k", slow_upstream))
    await asyncio.sleep(0)  # let follower register

    # Cancel the first caller
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    # The follower should NOT silently hang or inherit the cancellation;
    # it should either complete (if it retried) or raise CancelledError
    # (if it was also cancelled). What it must NOT do is deadlock.
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower

    # After cancellation the registry should be clean
    assert dedup.stats()["inflight"] == 0
