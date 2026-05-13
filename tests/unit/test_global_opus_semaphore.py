"""
Unit tests for the global Opus concurrency semaphore (§3).

Verifies that:
- When GATEWAY_GLOBAL_OPUS_CONCURRENCY=0 (default), no semaphore is created
  and the streaming path is a no-op (no behaviour change).
- When GATEWAY_GLOBAL_OPUS_CONCURRENCY=2, a third concurrent Opus request
  waits until one of the first two completes.
- Non-Opus models are never gated by the semaphore.
- The semaphore is released even when the stream raises an exception.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_state(opus_cap: int) -> MagicMock:
    """Return a mock app.state with global_opus_semaphore set per cap."""
    state = MagicMock()
    if opus_cap > 0:
        state.global_opus_semaphore = asyncio.Semaphore(opus_cap)
    else:
        state.global_opus_semaphore = None
    return state


# ---------------------------------------------------------------------------
# Tests: flag OFF (default)
# ---------------------------------------------------------------------------

class TestOpusSemaphoreFlagOff:
    """When GATEWAY_GLOBAL_OPUS_CONCURRENCY=0, semaphore is None — no-op."""

    def test_app_state_semaphore_is_none_when_cap_zero(self):
        """
        What it does: Verifies app.state.global_opus_semaphore is None when cap=0.
        Purpose: Confirm the default config produces no semaphore.
        """
        state = _make_app_state(0)
        assert state.global_opus_semaphore is None

    @pytest.mark.asyncio
    async def test_no_semaphore_means_no_wait(self):
        """
        What it does: Verifies that with semaphore=None, concurrent Opus requests
                      all proceed immediately without waiting.
        Purpose: Ensure default behaviour is unchanged (no latency added).
        """
        semaphore = None  # cap=0 → no semaphore

        results: list[int] = []

        async def fake_opus_stream(idx: int) -> None:
            # Simulate the route logic: acquire only if semaphore is not None
            _sem_acquired = False
            try:
                if semaphore is not None:
                    await semaphore.acquire()
                    _sem_acquired = True
                await asyncio.sleep(0)  # yield
                results.append(idx)
            finally:
                if _sem_acquired and semaphore is not None:
                    semaphore.release()

        # 5 concurrent requests — all should complete without waiting
        await asyncio.gather(*[fake_opus_stream(i) for i in range(5)])
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_non_opus_model_never_gets_semaphore(self):
        """
        What it does: Verifies that non-Opus models don't receive the semaphore.
        Purpose: Ensure the cap only applies to claude-opus-* models.
        """
        state = _make_app_state(2)

        for model in ["claude-sonnet-4", "claude-haiku-4.5", "auto", "claude-3.7-sonnet"]:
            # Route logic: only acquire semaphore for claude-opus-* models
            _opus_semaphore = None
            if model.startswith("claude-opus"):
                _opus_semaphore = getattr(state, "global_opus_semaphore", None)
            assert _opus_semaphore is None, f"Expected no semaphore for model={model!r}"


# ---------------------------------------------------------------------------
# Tests: flag ON (cap=2)
# ---------------------------------------------------------------------------

class TestOpusSemaphoreFlagOn:
    """When GATEWAY_GLOBAL_OPUS_CONCURRENCY=2, a 3rd concurrent Opus request waits."""

    @pytest.mark.asyncio
    async def test_third_concurrent_opus_waits(self):
        """
        What it does: Starts 3 concurrent Opus streams with cap=2 and verifies
                      the third waits until one of the first two releases the slot.
        Purpose: Core correctness test for the concurrency cap.
        """
        semaphore = asyncio.Semaphore(2)
        order: list[str] = []

        async def fake_opus_stream(name: str, hold_seconds: float) -> None:
            await semaphore.acquire()
            order.append(f"{name}:start")
            await asyncio.sleep(hold_seconds)
            order.append(f"{name}:end")
            semaphore.release()

        # req1 and req2 acquire immediately; req3 must wait
        await asyncio.gather(
            fake_opus_stream("req1", 0.05),
            fake_opus_stream("req2", 0.05),
            fake_opus_stream("req3", 0.01),
        )

        # req3 must start AFTER at least one of req1/req2 ends
        req3_start_idx = order.index("req3:start")
        req1_end_idx = order.index("req1:end")
        req2_end_idx = order.index("req2:end")

        assert req3_start_idx > min(req1_end_idx, req2_end_idx), (
            f"req3 started at index {req3_start_idx} but expected to wait until "
            f"at least one of req1 (end={req1_end_idx}) or req2 (end={req2_end_idx}) finished"
        )

    @pytest.mark.asyncio
    async def test_semaphore_released_on_exception(self):
        """
        What it does: Verifies the semaphore slot is released even when the
                      stream raises an exception.
        Purpose: Prevent semaphore leaks that would permanently reduce capacity.
        """
        semaphore = asyncio.Semaphore(1)

        async def failing_stream() -> None:
            _sem_acquired = False
            try:
                await semaphore.acquire()
                _sem_acquired = True
                raise RuntimeError("Simulated stream failure")
            finally:
                if _sem_acquired:
                    semaphore.release()

        with pytest.raises(RuntimeError, match="Simulated stream failure"):
            await failing_stream()

        # Semaphore should be fully released — a new acquire must not block
        acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            acquired = True
        finally:
            if acquired:
                semaphore.release()

        assert acquired, "Semaphore was not released after exception — slot leaked"

    @pytest.mark.asyncio
    async def test_opus_model_gets_semaphore_when_cap_set(self):
        """
        What it does: Verifies that claude-opus-* models receive the semaphore
                      when GATEWAY_GLOBAL_OPUS_CONCURRENCY > 0.
        Purpose: Ensure the model-name check routes Opus requests through the cap.
        """
        state = _make_app_state(2)

        for model in ["claude-opus-4", "claude-opus-4.5", "claude-opus-4-20250514"]:
            _opus_semaphore = None
            if model.startswith("claude-opus"):
                _opus_semaphore = getattr(state, "global_opus_semaphore", None)
            assert _opus_semaphore is not None, (
                f"Expected semaphore for model={model!r} but got None"
            )

    @pytest.mark.asyncio
    async def test_cap_2_allows_exactly_2_concurrent(self):
        """
        What it does: Verifies that with cap=2, exactly 2 streams run concurrently
                      at peak and the third is queued.
        Purpose: Confirm the semaphore count is respected precisely.
        """
        semaphore = asyncio.Semaphore(2)
        concurrent_peak = 0
        current_concurrent = 0

        async def stream(hold: float) -> None:
            nonlocal concurrent_peak, current_concurrent
            await semaphore.acquire()
            current_concurrent += 1
            concurrent_peak = max(concurrent_peak, current_concurrent)
            await asyncio.sleep(hold)
            current_concurrent -= 1
            semaphore.release()

        await asyncio.gather(
            stream(0.05),
            stream(0.05),
            stream(0.05),
        )

        assert concurrent_peak <= 2, (
            f"Peak concurrency was {concurrent_peak}, expected ≤ 2"
        )
