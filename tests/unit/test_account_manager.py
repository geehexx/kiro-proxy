
"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- TTL-based model cache refresh
- State persistence
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kiro.account_errors import ErrorType
from kiro.account_manager import Account, AccountManager, AccountStats, _format_duration


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """

    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.

        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")

        # Act
        account = Account(id="/test/path.json")

        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")

        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)

    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.

        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")

        # Act
        stats = AccountStats()

        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """

    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.

        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")

        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts

    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.

        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts

    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.

        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")

        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")

    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.

        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")

        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()

        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 2

    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.

        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")

        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()

        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")

        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")

        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 1  # Only valid file loaded

    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.

        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 0

    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.

        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 0

    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.

        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 0

    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.

        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")

        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 0

    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.

        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")

        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )

        # Act
        await manager.load_credentials()

        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")

        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """

    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.

        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")

        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))

        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )

        await manager.load_credentials()

        # Act
        await manager.load_state()

        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")

        assert len(manager._model_to_accounts) > 0

    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test that sticky_map starts empty after load_state (global index removed).

        What it does: Verifies sticky_map is empty dict after load
        Purpose: Verify per-key sticky behavior replaces global index
        """
        print("\n=== Test: load_state sticky_map starts empty ===")

        # Arrange — old state format with current_account_index (should be ignored gracefully)
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))

        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )

        # Act
        await manager.load_state()

        # Assert
        print(f"Sticky map: {manager._sticky_map}")

        assert isinstance(manager._sticky_map, dict)
        assert len(manager._sticky_map) == 0

    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.

        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")

        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))

        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )

        # Act
        await manager.load_state()

        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")

        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2

    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).

        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")

        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())

        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )

        await manager.load_credentials()

        # Act
        await manager.load_state()

        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")

        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
        assert account.stats.successful_requests == 97
        assert account.stats.failed_requests == 3

    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).

        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")

        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )

        # Act
        await manager.load_state()

        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Sticky map: {manager._sticky_map}")

        assert len(manager._model_to_accounts) == 0
        assert isinstance(manager._sticky_map, dict)

    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.

        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")

        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")

        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )

        # Act
        await manager.load_state()

        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")

        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """

    @pytest.mark.asyncio
    async def test_initialize_account_json_success(self, tmp_path, mock_list_models_response):
        """
        Test successful account initialization with type=json.

        What it does: Initializes account with JSON credentials
        Purpose: Verify complete initialization flow
        """
        print("\n=== Test: initialize_account with JSON ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Mock HTTP client for ListAvailableModels
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            # Act
            success = await manager._initialize_account(account_id)

        # Assert
        print(f"Initialization success: {success}")
        assert success is True
        assert manager._accounts[account_id].auth_manager is not None
        assert manager._accounts[account_id].model_cache is not None
        assert manager._accounts[account_id].model_resolver is not None

    @pytest.mark.asyncio
    async def test_initialize_account_fetch_models_fallback(self, tmp_path):
        """
        Test fallback to FALLBACK_MODELS when API fails.

        What it does: Initializes account when ListAvailableModels fails
        Purpose: Verify fallback mechanism
        """
        print("\n=== Test: initialize_account with fallback models ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Mock HTTP client to fail
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_client.request_with_retry = AsyncMock(side_effect=Exception("Network error"))
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            # Act
            success = await manager._initialize_account(account_id)

        # Assert
        print(f"Initialization success: {success}")
        assert success is True  # Should succeed with fallback
        assert manager._accounts[account_id].model_cache is not None


class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """

    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.

        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()

        # Act
        account = await manager.get_next_account("claude-opus-4.5")

        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """

    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.

        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Set failures
        manager._accounts[account_id].failures = 5

        # Act
        await manager.report_success(account_id, "claude-opus-4.5")

        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0

    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.

        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Act
        await manager.report_success(account_id, "claude-opus-4.5")

        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """

    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.

        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )

        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1

    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.

        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )

        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """

    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.

        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")

        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )

        # Act
        await manager._save_state()

        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()

        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """

    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.

        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Act
        account = manager.get_first_account()

        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None

    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.

        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")

        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """

    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.

        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")

        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )

        await manager.load_credentials()
        account_id = str(test_json.resolve())

        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        # Act
        models = manager.get_all_available_models()

        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """

    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"

    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"

    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"

    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"


def _make_initialized_account(account_id: str) -> Account:
    """Helper: create a fully initialized Account mock."""
    account = Account(id=account_id)
    account.auth_manager = Mock()
    account.model_cache = Mock()
    account.model_resolver = Mock()
    account.model_resolver.get_available_models.return_value = ["claude-opus-4.5", "claude-sonnet-4-6"]
    account.models_cached_at = time.time()
    return account


class TestStickyByKey:
    """Tests for per-(session_id, model_family) sticky behavior."""

    @pytest.mark.asyncio
    async def test_sticky_key_stored_on_success(self, tmp_path):
        """report_success with sticky_key stores account in _sticky_map."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {
            "/acc/a.json": _make_initialized_account("/acc/a.json"),
            "/acc/b.json": _make_initialized_account("/acc/b.json"),
        }

        await manager.report_success("/acc/b.json", "claude-opus-4.5", sticky_key="sess1:claude-opus")

        assert manager._sticky_map["sess1:claude-opus"] == "/acc/b.json"

    @pytest.mark.asyncio
    async def test_sticky_key_prefers_stored_account(self, tmp_path):
        """get_next_account starts from sticky account when key matches."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        acc_b = _make_initialized_account("/acc/b.json")
        manager._accounts = {"/acc/a.json": acc_a, "/acc/b.json": acc_b}
        manager._model_to_accounts = {}
        manager._sticky_map = {"sess1:claude-opus": "/acc/b.json"}

        account = await manager.get_next_account("claude-opus-4.5", sticky_key="sess1:claude-opus")

        assert account is not None
        assert account.id == "/acc/b.json"

    @pytest.mark.asyncio
    async def test_no_sticky_key_falls_back_to_first(self, tmp_path):
        """get_next_account without sticky_key starts from index 0."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        acc_b = _make_initialized_account("/acc/b.json")
        manager._accounts = {"/acc/a.json": acc_a, "/acc/b.json": acc_b}

        account = await manager.get_next_account("claude-opus-4.5", sticky_key=None)

        assert account is not None
        assert account.id == "/acc/a.json"

    @pytest.mark.asyncio
    async def test_sticky_key_unknown_account_falls_back(self, tmp_path):
        """Sticky map pointing to removed account falls back to first available."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        manager._accounts = {"/acc/a.json": acc_a}
        manager._sticky_map = {"sess1:claude-opus": "/acc/gone.json"}

        account = await manager.get_next_account("claude-opus-4.5", sticky_key="sess1:claude-opus")

        assert account is not None
        assert account.id == "/acc/a.json"

    @pytest.mark.asyncio
    async def test_different_keys_independent(self, tmp_path):
        """Different sticky keys can point to different accounts independently."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        acc_b = _make_initialized_account("/acc/b.json")
        manager._accounts = {"/acc/a.json": acc_a, "/acc/b.json": acc_b}

        await manager.report_success("/acc/a.json", "claude-opus-4.5", sticky_key="sess1:claude-opus")
        await manager.report_success("/acc/b.json", "claude-sonnet-4-6", sticky_key="sess1:claude-sonnet")

        assert manager._sticky_map["sess1:claude-opus"] == "/acc/a.json"
        assert manager._sticky_map["sess1:claude-sonnet"] == "/acc/b.json"


class TestDemotionCooldown:
    """Tests for 5-minute fast demotion triggered by 60s+2-retry."""

    @pytest.mark.asyncio
    async def test_demote_account_sets_cooldown(self, tmp_path):
        """demote_account sets _demoted_until ~5min in the future."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}

        before = time.time()
        await manager.demote_account("/acc/a.json")
        after = time.time()

        demoted_until = manager._demoted_until["/acc/a.json"]
        assert demoted_until >= before + 299  # ~5min
        assert demoted_until <= after + 301

    @pytest.mark.asyncio
    async def test_demote_increments_stats(self, tmp_path):
        """demote_account increments stats.demotions."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}

        await manager.demote_account("/acc/a.json")
        await manager.demote_account("/acc/a.json")

        assert manager._accounts["/acc/a.json"].stats.demotions == 2

    @pytest.mark.asyncio
    async def test_demoted_account_skipped_in_selection(self, tmp_path):
        """get_next_account skips demoted account and returns next available."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        acc_b = _make_initialized_account("/acc/b.json")
        manager._accounts = {"/acc/a.json": acc_a, "/acc/b.json": acc_b}

        # Demote account a
        manager._demoted_until["/acc/a.json"] = time.time() + 300

        account = await manager.get_next_account("claude-opus-4.5")

        assert account is not None
        assert account.id == "/acc/b.json"

    @pytest.mark.asyncio
    async def test_expired_demotion_allows_selection(self, tmp_path):
        """Account with expired demotion is selectable again."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc_a = _make_initialized_account("/acc/a.json")
        manager._accounts = {"/acc/a.json": acc_a}

        # Set demotion in the past
        manager._demoted_until["/acc/a.json"] = time.time() - 1

        account = await manager.get_next_account("claude-opus-4.5")

        assert account is not None
        assert account.id == "/acc/a.json"

    @pytest.mark.asyncio
    async def test_demote_unknown_account_no_crash(self, tmp_path):
        """demote_account on unknown account_id is a no-op."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))

        # Should not raise
        await manager.demote_account("/acc/nonexistent.json")


class TestGetAccountStats:
    """Tests for get_account_stats() telemetry method."""

    @pytest.mark.asyncio
    async def test_returns_list(self, tmp_path):
        """get_account_stats returns a list."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}

        result = await manager.get_account_stats()

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_stat_fields_present(self, tmp_path):
        """Each stat entry has required fields."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}

        result = await manager.get_account_stats()
        entry = result[0]

        assert "id" in entry
        assert "cb_state" in entry
        assert "failures" in entry
        assert "demoted_remaining_s" in entry
        assert "demotions" in entry
        assert "total_requests" in entry
        assert "successful_requests" in entry
        assert "failed_requests" in entry

    @pytest.mark.asyncio
    async def test_cb_state_closed_when_no_failures(self, tmp_path):
        """cb_state is 'closed' when account has no failures."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}

        result = await manager.get_account_stats()

        assert result[0]["cb_state"] == "closed"
        assert result[0]["failures"] == 0

    @pytest.mark.asyncio
    async def test_cb_state_open_during_cooldown(self, tmp_path):
        """cb_state is 'open' when account is in backoff window."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        acc = _make_initialized_account("/acc/a.json")
        acc.failures = 1
        acc.last_failure_time = time.time()  # just failed
        manager._accounts = {"/acc/a.json": acc}

        result = await manager.get_account_stats()

        assert result[0]["cb_state"] == "open"

    @pytest.mark.asyncio
    async def test_demoted_remaining_nonzero_when_demoted(self, tmp_path):
        """demoted_remaining_s > 0 when account is currently demoted."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/a.json": _make_initialized_account("/acc/a.json")}
        manager._demoted_until["/acc/a.json"] = time.time() + 200

        result = await manager.get_account_stats()

        assert result[0]["demoted_remaining_s"] > 0

    @pytest.mark.asyncio
    async def test_id_is_last_16_chars(self, tmp_path):
        """id field is last 16 chars of account_id."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))
        manager._accounts = {"/acc/some_long_path.json": _make_initialized_account("/acc/some_long_path.json")}

        result = await manager.get_account_stats()

        assert result[0]["id"] == "/acc/some_long_path.json"[-16:]

    @pytest.mark.asyncio
    async def test_empty_accounts_returns_empty_list(self, tmp_path):
        """get_account_stats returns [] when no accounts loaded."""
        manager = AccountManager(str(tmp_path / "c.json"), str(tmp_path / "s.json"))

        result = await manager.get_account_stats()

        assert result == []
