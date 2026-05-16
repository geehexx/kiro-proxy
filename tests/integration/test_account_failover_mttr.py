"""
E2 — Account failover MTTR chaos test.

Verifies that after injecting RECOVERABLE failures into one account,
the circuit breaker transitions open → half-open → closed within the
expected wall-clock window (MTTR_p95 < 90s).

Uses time manipulation (backdating last_failure_time) to keep tests fast.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro.account_errors import ErrorType
from kiro.account_manager import AccountManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path, num_accounts: int = 2) -> AccountManager:
    creds_file = tmp_path / "credentials.json"
    state_file = tmp_path / "state.json"
    credentials = [
        {"type": "json", "path": str(tmp_path / f"account{i}.json"), "enabled": True}
        for i in range(num_accounts)
    ]
    creds_file.write_text(json.dumps(credentials))
    for i in range(num_accounts):
        (tmp_path / f"account{i}.json").write_text(
            json.dumps({"token": f"tok-{i}", "refreshToken": f"ref-{i}"})
        )
    return AccountManager(str(creds_file), str(state_file))


async def _init_manager(manager: AccountManager) -> None:
    await manager.load_credentials()
    await manager.load_state()
    for account_id in list(manager._accounts.keys()):
        account = manager._accounts[account_id]
        account.auth_manager = MagicMock()
        account.auth_manager.get_token = AsyncMock(return_value="tok")
        account.model_cache = MagicMock()
        account.model_resolver = MagicMock()
        account.model_resolver.get_available_models = MagicMock(
            return_value=["claude-sonnet-4.6", "claude-opus-4.7", "claude-haiku-4.5"]
        )
        account.initialized = True


def _get_stat(stats: list, account_id: str) -> dict:
    """Find the stat entry for an account_id by suffix match."""
    for s in stats:
        sid = s["id"].replace("…", "")
        if account_id.endswith(sid) or sid in account_id:
            return s
    raise KeyError(f"No stat found for {account_id} in {[s['id'] for s in stats]}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAccountFailoverMTTR:
    """E2 — MTTR chaos tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_single_failure_opens_circuit(self, tmp_path):
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        await manager.report_failure(
            account_id, model="claude-sonnet-4.6",
            error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
        )

        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "open", f"Expected open, got {stat['cb_state']}"
        assert stat["failures"] == 1

    @pytest.mark.asyncio
    async def test_circuit_transitions_half_open_after_timeout(self, tmp_path):
        from kiro import config as cfg
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        await manager.report_failure(
            account_id, model="claude-sonnet-4.6",
            error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
        )
        # Fast-forward past recovery timeout
        async with manager._lock:
            manager._accounts[account_id].last_failure_time = (
                time.time() - (cfg.ACCOUNT_RECOVERY_TIMEOUT + 1)
            )

        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "half-open", f"Expected half-open, got {stat['cb_state']}"

    @pytest.mark.asyncio
    async def test_circuit_closes_after_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        await manager.report_failure(
            account_id, model="claude-sonnet-4.6",
            error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
        )
        await manager.report_success(account_id, model="claude-sonnet-4.6")

        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "closed", f"Expected closed, got {stat['cb_state']}"
        assert stat["failures"] == 0

    @pytest.mark.asyncio
    async def test_mttr_p95_under_90s(self, tmp_path):
        """Full open→half-open→closed cycle; simulated MTTR must be < 90s."""
        from kiro import config as cfg
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        await manager.report_failure(
            account_id, model="claude-sonnet-4.6",
            error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
        )
        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "open"

        # Simulate recovery timeout passing
        async with manager._lock:
            manager._accounts[account_id].last_failure_time = (
                time.time() - cfg.ACCOUNT_RECOVERY_TIMEOUT
            )
        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "half-open"

        await manager.report_success(account_id, model="claude-sonnet-4.6")
        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "closed"

        # MTTR = ACCOUNT_RECOVERY_TIMEOUT (minimum possible with 1 failure)
        assert cfg.ACCOUNT_RECOVERY_TIMEOUT < 90, (
            f"MTTR_p95 threshold violated: ACCOUNT_RECOVERY_TIMEOUT={cfg.ACCOUNT_RECOVERY_TIMEOUT}s >= 90s"
        )

    @pytest.mark.asyncio
    async def test_exponential_backoff_increases_mttr(self, tmp_path):
        from kiro import config as cfg
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        for _ in range(3):
            await manager.report_failure(
                account_id, model="claude-sonnet-4.6",
                error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
            )

        expected_multiplier = min(2 ** (3 - 1), cfg.ACCOUNT_MAX_BACKOFF_MULTIPLIER)
        expected_timeout = cfg.ACCOUNT_RECOVERY_TIMEOUT * expected_multiplier

        # Just before extended timeout → still open
        async with manager._lock:
            manager._accounts[account_id].last_failure_time = (
                time.time() - (expected_timeout - 1)
            )
        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "open", f"Expected open before extended timeout, got {stat['cb_state']}"

        # Past extended timeout → half-open
        async with manager._lock:
            manager._accounts[account_id].last_failure_time = (
                time.time() - (expected_timeout + 1)
            )
        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "half-open", f"Expected half-open after extended timeout, got {stat['cb_state']}"

    @pytest.mark.asyncio
    async def test_fatal_error_does_not_open_circuit(self, tmp_path):
        manager = _make_manager(tmp_path)
        await _init_manager(manager)
        account_id = list(manager._accounts.keys())[0]

        await manager.report_failure(
            account_id, model="claude-sonnet-4.6",
            error_type=ErrorType.FATAL, status_code=400, reason="INVALID_REQUEST",
        )

        stat = _get_stat(await manager.get_account_stats(), account_id)
        assert stat["cb_state"] == "closed", f"FATAL should not open circuit, got {stat['cb_state']}"
        assert stat["failures"] == 0

    @pytest.mark.asyncio
    async def test_failover_to_second_account_when_first_open(self, tmp_path):
        manager = _make_manager(tmp_path, num_accounts=2)
        await _init_manager(manager)
        account_ids = list(manager._accounts.keys())
        account0 = account_ids[0]

        await manager.report_failure(
            account0, model="claude-sonnet-4.6",
            error_type=ErrorType.RECOVERABLE, status_code=429, reason="THROTTLING",
        )

        account = await manager.get_next_account(
            model="claude-sonnet-4.6",
            exclude_accounts={account0},
        )
        assert account is not None, "Expected failover to account 1"
        assert account.id != account0, f"Expected failover away from {account0}, got {account.id}"
