"""
Unit tests for kiro.session_limiter — per-session concurrency cap.

Covers: basic limit enforcement, per-session isolation, release on raise,
stats accounting, config validation.
"""

from __future__ import annotations

import asyncio

import pytest

from kiro.session_limiter import SessionLimiter


@pytest.mark.asyncio
async def test_limiter_caps_concurrent_acquires_per_session() -> None:
    """With N=2, only 2 callers can hold a slot concurrently for one session."""
    limiter = SessionLimiter(default_concurrency=2)
    in_flight_peak = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal in_flight_peak, in_flight
        async with limiter.acquire("session-a"):
            async with lock:
                in_flight += 1
                in_flight_peak = max(in_flight_peak, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert in_flight_peak == 2, f"peak {in_flight_peak} exceeded ceiling 2"
    assert in_flight == 0


@pytest.mark.asyncio
async def test_different_sessions_do_not_block_each_other() -> None:
    """Session-B should not wait on session-A's slots."""
    limiter = SessionLimiter(default_concurrency=1)
    a_started = asyncio.Event()
    a_can_finish = asyncio.Event()
    b_finished = asyncio.Event()

    async def slow_a() -> None:
        async with limiter.acquire("session-a"):
            a_started.set()
            await a_can_finish.wait()

    async def quick_b() -> None:
        await a_started.wait()
        async with limiter.acquire("session-b"):
            b_finished.set()

    a_task = asyncio.create_task(slow_a())
    b_task = asyncio.create_task(quick_b())

    # B must finish while A is still holding its slot.
    await asyncio.wait_for(b_finished.wait(), timeout=0.5)
    assert not a_task.done()
    a_can_finish.set()
    await asyncio.gather(a_task, b_task)


@pytest.mark.asyncio
async def test_semaphore_released_on_exception() -> None:
    """A raising caller must not leak its slot."""
    limiter = SessionLimiter(default_concurrency=1)

    with pytest.raises(RuntimeError, match="boom"):
        async with limiter.acquire("session-a"):
            raise RuntimeError("boom")

    # Next acquire must proceed without blocking.
    async with asyncio.timeout(0.5):
        async with limiter.acquire("session-a"):
            pass


@pytest.mark.asyncio
async def test_stats_report_waits_only_on_contention() -> None:
    """`waits` increments only when acquire actually blocks."""
    limiter = SessionLimiter(default_concurrency=1)

    # Uncontested — waits stays 0
    async with limiter.acquire("a"):
        pass
    async with limiter.acquire("a"):
        pass
    s = limiter.stats()
    assert s["acquires"] == 2
    assert s["waits"] == 0

    # Contested — second acquirer should record a wait
    hold = asyncio.Event()
    start = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with limiter.acquire("b"):
            start.set()
            await release.wait()

    async def contender() -> None:
        await start.wait()
        async with limiter.acquire("b"):
            hold.set()

    hold_task = asyncio.create_task(holder())
    cont_task = asyncio.create_task(contender())
    await start.wait()
    # Yield a moment so contender enters acquire() and sees the locked semaphore
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(hold_task, cont_task)
    s = limiter.stats()
    assert s["waits"] >= 1, s


def test_invalid_concurrency_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        SessionLimiter(default_concurrency=0)
    with pytest.raises(ValueError, match=">= 1"):
        SessionLimiter(default_concurrency=-5)


@pytest.mark.asyncio
async def test_clear_resets_sessions_and_stats() -> None:
    limiter = SessionLimiter(default_concurrency=3)
    async with limiter.acquire("a"):
        pass
    async with limiter.acquire("b"):
        pass
    assert limiter.stats()["sessions"] == 2

    limiter.clear()
    s = limiter.stats()
    assert s["sessions"] == 0
    assert s["acquires"] == 0
    assert s["waits"] == 0


@pytest.mark.asyncio
async def test_session_sem_reused_across_acquires() -> None:
    """Second acquire for the same session must reuse the existing semaphore."""
    limiter = SessionLimiter(default_concurrency=1)
    async with limiter.acquire("a"):
        pass
    # Re-acquire same key — should find existing semaphore, not create a new one.
    async with limiter.acquire("a"):
        pass
    assert limiter.stats()["sessions"] == 1
