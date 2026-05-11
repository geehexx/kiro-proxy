
"""
Unit tests for DebugLogger.
Verifies the buffering and write logic for debug logs in each mode.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDebugLoggerModeOff:
    """Tests for DEBUG_MODE=off."""

    def test_prepare_new_request_does_nothing(self, tmp_path):
        """
        What it does: Verifies that prepare_new_request does nothing in off mode.
        Purpose: Confirm no directory is created in off mode.
        """
        print("Setup: Mode off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            with patch('kiro.debug_logger.DEBUG_DIR', str(tmp_path / "debug_logs")):
                # Recreate instance with the new settings
                from kiro.debug_logger import DebugLogger
                logger = DebugLogger.__new__(DebugLogger)
                logger._initialized = False
                logger.__init__()
                logger.debug_dir = tmp_path / "debug_logs"

                print("Action: Calling prepare_new_request...")
                logger.prepare_new_request()

                print("Check that the directory was not created...")
                assert not (tmp_path / "debug_logs").exists()

    def test_log_request_body_does_nothing(self, tmp_path):
        """
        What it does: Verifies that log_request_body does nothing in off mode.
        Purpose: Confirm data is not written.
        """
        print("Setup: Mode off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = tmp_path / "debug_logs"

            print("Action: Calling log_request_body...")
            logger.log_request_body(b'{"test": "data"}')

            print("Check that the file was not created...")
            assert not (tmp_path / "debug_logs" / "request_body.json").exists()


class TestDebugLoggerModeAll:
    """Tests for DEBUG_MODE=all."""

    def test_prepare_new_request_clears_directory(self, tmp_path):
        """
        What it does: Verifies that prepare_new_request clears the directory in all mode.
        Purpose: Confirm old logs are removed.
        """
        print("Setup: Mode all, creating an old file...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        old_file = debug_dir / "old_file.txt"
        old_file.write_text("old content")

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling prepare_new_request...")
            logger.prepare_new_request()

            print("Check that the old file was removed...")
            assert not old_file.exists()
            print("Check that the directory exists...")
            assert debug_dir.exists()

    def test_log_request_body_writes_immediately(self, tmp_path):
        """
        What it does: Verifies that log_request_body writes to file immediately in all mode.
        Purpose: Confirm data is written straight away.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_request_body...")
            test_data = b'{"model": "test", "messages": []}'
            logger.log_request_body(test_data)

            print("Check that the file was created...")
            file_path = debug_dir / "request_body.json"
            assert file_path.exists()

            print("Check the file contents...")
            content = json.loads(file_path.read_text())
            assert content["model"] == "test"

    def test_log_kiro_request_body_writes_immediately(self, tmp_path):
        """
        What it does: Verifies that log_kiro_request_body writes to file immediately in all mode.
        Purpose: Confirm the Kiro payload is written straight away.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_kiro_request_body...")
            test_data = b'{"conversationState": {}}'
            logger.log_kiro_request_body(test_data)

            print("Check that the file was created...")
            file_path = debug_dir / "kiro_request_body.json"
            assert file_path.exists()

    def test_log_raw_chunk_appends_to_file(self, tmp_path):
        """
        What it does: Verifies that log_raw_chunk appends to the file in all mode.
        Purpose: Confirm chunks accumulate.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_raw_chunk twice...")
            logger.log_raw_chunk(b'chunk1')
            logger.log_raw_chunk(b'chunk2')

            print("Check the file contents...")
            file_path = debug_dir / "response_stream_raw.txt"
            content = file_path.read_bytes()
            assert content == b'chunk1chunk2'


class TestDebugLoggerModeErrors:
    """Tests for DEBUG_MODE=errors."""

    def test_log_request_body_buffers_data(self, tmp_path):
        """
        What it does: Verifies that log_request_body buffers data in errors mode.
        Purpose: Confirm data is not written straight away.
        """
        print("Setup: Mode errors...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_request_body...")
            test_data = b'{"test": "buffered"}'
            logger.log_request_body(test_data)

            print("Check that the file was NOT created...")
            assert not debug_dir.exists()

            print("Check that the data is in the buffer...")
            assert logger._request_body_buffer == test_data

    def test_flush_on_error_writes_buffers(self, tmp_path):
        """
        What it does: Verifies that flush_on_error writes the buffers to files.
        Purpose: Confirm data is persisted on error.
        """
        print("Setup: Mode errors, filling buffers...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            # Fill the buffers
            logger.log_request_body(b'{"request": "body"}')
            logger.log_kiro_request_body(b'{"kiro": "request"}')
            logger.log_raw_chunk(b'raw_chunk')
            logger.log_modified_chunk(b'modified_chunk')

            print("Action: Calling flush_on_error...")
            logger.flush_on_error(400, "Bad Request")

            print("Check that all files were created...")
            assert (debug_dir / "request_body.json").exists()
            assert (debug_dir / "kiro_request_body.json").exists()
            assert (debug_dir / "response_stream_raw.txt").exists()
            assert (debug_dir / "response_stream_modified.txt").exists()
            assert (debug_dir / "error_info.json").exists()

            print("Check error_info.json...")
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 400
            assert error_info["error_message"] == "Bad Request"

    def test_flush_on_error_clears_buffers(self, tmp_path):
        """
        What it does: Verifies that flush_on_error clears the buffers after writing.
        Purpose: Confirm buffers do not accumulate between requests.
        """
        print("Setup: Mode errors...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            logger.log_request_body(b'{"test": "data"}')

            print("Action: Calling flush_on_error...")
            logger.flush_on_error(500, "Error")

            print("Check that the buffers were cleared...")
            assert logger._request_body_buffer is None
            assert logger._kiro_request_body_buffer is None
            assert len(logger._raw_chunks_buffer) == 0
            assert len(logger._modified_chunks_buffer) == 0

    def test_discard_buffers_clears_without_writing(self, tmp_path):
        """
        What it does: Verifies that discard_buffers clears the buffers without writing.
        Purpose: Confirm successful requests leave no logs.
        """
        print("Setup: Mode errors, filling buffers...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            logger.log_request_body(b'{"test": "data"}')
            logger.log_raw_chunk(b'chunk')

            print("Action: Calling discard_buffers...")
            logger.discard_buffers()

            print("Check that the directory was NOT created...")
            assert not debug_dir.exists()

            print("Check that the buffers were cleared...")
            assert logger._request_body_buffer is None
            assert len(logger._raw_chunks_buffer) == 0

    def test_flush_on_error_writes_error_info_in_mode_all(self, tmp_path):
        """
        What it does: Verifies that flush_on_error writes error_info.json in all mode.
        Purpose: Confirm error information is persisted in both modes.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling flush_on_error...")
            logger.flush_on_error(400, "Bad Request")

            print("Check that error_info.json was created...")
            assert (debug_dir / "error_info.json").exists()

            print("Check error_info.json contents...")
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 400
            assert error_info["error_message"] == "Bad Request"


class TestDebugLoggerLogErrorInfo:
    """Tests for the log_error_info() method."""

    def test_log_error_info_writes_in_mode_all(self, tmp_path):
        """
        What it does: Verifies that log_error_info writes the file in all mode.
        Purpose: Confirm error_info.json is created on errors.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_error_info...")
            logger.log_error_info(500, "Internal Server Error")

            print("Check that error_info.json was created...")
            error_file = debug_dir / "error_info.json"
            assert error_file.exists()

            print("Check the contents...")
            error_info = json.loads(error_file.read_text())
            assert error_info["status_code"] == 500
            assert error_info["error_message"] == "Internal Server Error"

    def test_log_error_info_writes_in_mode_errors(self, tmp_path):
        """
        What it does: Verifies that log_error_info writes the file in errors mode.
        Purpose: Confirm the method works in both modes.
        """
        print("Setup: Mode errors...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_error_info...")
            logger.log_error_info(404, "Not Found")

            print("Check that error_info.json was created...")
            error_file = debug_dir / "error_info.json"
            assert error_file.exists()

    def test_log_error_info_does_nothing_in_mode_off(self, tmp_path):
        """
        What it does: Verifies that log_error_info does nothing in off mode.
        Purpose: Confirm no files are created in off mode.
        """
        print("Setup: Mode off...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_error_info...")
            logger.log_error_info(500, "Error")

            print("Check that the directory was NOT created...")
            assert not debug_dir.exists()


class TestDebugLoggerHelperMethods:
    """Tests for DebugLogger helper methods."""

    def test_is_enabled_returns_true_for_errors(self):
        """
        What it does: Verifies _is_enabled() for errors mode.
        Purpose: Confirm errors mode is considered enabled.
        """
        print("Setup: Mode errors...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()

            print("Check _is_enabled()...")
            assert logger._is_enabled() is True

    def test_is_enabled_returns_true_for_all(self):
        """
        What it does: Verifies _is_enabled() for all mode.
        Purpose: Confirm all mode is considered enabled.
        """
        print("Setup: Mode all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()

            print("Check _is_enabled()...")
            assert logger._is_enabled() is True

    def test_is_enabled_returns_false_for_off(self):
        """
        What it does: Verifies _is_enabled() for off mode.
        Purpose: Confirm off mode is considered disabled.
        """
        print("Setup: Mode off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()

            print("Check _is_enabled()...")
            assert logger._is_enabled() is False

    def test_is_immediate_write_returns_true_for_all(self):
        """
        What it does: Verifies _is_immediate_write() for all mode.
        Purpose: Confirm all mode writes straight away.
        """
        print("Setup: Mode all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()

            print("Check _is_immediate_write()...")
            assert logger._is_immediate_write() is True

    def test_is_immediate_write_returns_false_for_errors(self):
        """
        What it does: Verifies _is_immediate_write() for errors mode.
        Purpose: Confirm errors mode buffers.
        """
        print("Setup: Mode errors...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()

            print("Check _is_immediate_write()...")
            assert logger._is_immediate_write() is False


class TestDebugLoggerJsonHandling:
    """Tests for JSON handling in DebugLogger."""

    def test_log_request_body_formats_json_pretty(self, tmp_path):
        """
        What it does: Verifies that JSON is pretty-formatted.
        Purpose: Confirm JSON is readable in the file.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_request_body with JSON...")
            logger.log_request_body(b'{"key":"value"}')

            print("Check the formatting...")
            content = (debug_dir / "request_body.json").read_text()
            # Should be formatted with indentation
            assert "  " in content or "\n" in content

    def test_log_request_body_handles_invalid_json(self, tmp_path):
        """
        What it does: Verifies handling of invalid JSON.
        Purpose: Confirm invalid JSON is written as-is.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir

            print("Action: Calling log_request_body with invalid JSON...")
            invalid_data = b'not a json {{'
            logger.log_request_body(invalid_data)

            print("Check that the data was written as-is...")
            content = (debug_dir / "request_body.json").read_bytes()
            assert content == invalid_data


class TestDebugLoggerAppLogsCapture:
    """Tests for application log capture (app_logs.txt)."""

    def test_prepare_new_request_sets_up_log_capture(self, tmp_path):
        """
        What it does: Verifies that prepare_new_request sets up log capture.
        Purpose: Confirm the log sink is created.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir

            print("Action: Calling prepare_new_request...")
            dbg_logger.prepare_new_request()

            print("Check that the sink was created...")
            assert dbg_logger._loguru_sink_id is not None

            # Cleanup
            dbg_logger._clear_app_logs_buffer()

    def test_flush_on_error_writes_app_logs_in_mode_errors(self, tmp_path):
        """
        What it does: Verifies that flush_on_error writes app_logs.txt in errors mode.
        Purpose: Confirm application logs are persisted on error.
        """
        print("Setup: Mode errors...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from loguru import logger as loguru_logger

            from kiro.debug_logger import DebugLogger

            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir

            # Set up log capture
            dbg_logger.prepare_new_request()

            # Add data to the buffer so that flush is triggered
            dbg_logger.log_request_body(b'{"test": "data"}')

            # Write a test log directly into the buffer (simulation)
            dbg_logger._app_logs_buffer.write("Test log message\n")

            print("Action: Calling flush_on_error...")
            dbg_logger.flush_on_error(500, "Test Error")

            print("Check that app_logs.txt was created...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert app_logs_file.exists()

            print("Check the contents...")
            content = app_logs_file.read_text()
            assert "Test log message" in content

    def test_discard_buffers_saves_logs_in_mode_all(self, tmp_path):
        """
        What it does: Verifies that discard_buffers saves logs in all mode.
        Purpose: Confirm that even successful requests persist logs in all mode.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger

            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir

            # Set up log capture
            dbg_logger.prepare_new_request()

            # Write a test log directly into the buffer
            dbg_logger._app_logs_buffer.write("Success log message\n")

            print("Action: Calling discard_buffers...")
            dbg_logger.discard_buffers()

            print("Check that app_logs.txt was created...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert app_logs_file.exists()

            print("Check the contents...")
            content = app_logs_file.read_text()
            assert "Success log message" in content

    def test_discard_buffers_does_not_save_logs_in_mode_errors(self, tmp_path):
        """
        What it does: Verifies that discard_buffers does NOT save logs in errors mode.
        Purpose: Confirm successful requests leave no logs in errors mode.
        """
        print("Setup: Mode errors...")
        debug_dir = tmp_path / "debug_logs"

        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger

            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir

            # Set up log capture
            dbg_logger.prepare_new_request()

            # Write a test log directly into the buffer
            dbg_logger._app_logs_buffer.write("Should not be saved\n")

            print("Action: Calling discard_buffers...")
            dbg_logger.discard_buffers()

            print("Check that the directory was NOT created...")
            assert not debug_dir.exists()

    def test_clear_app_logs_buffer_removes_sink(self, tmp_path):
        """
        What it does: Verifies that _clear_app_logs_buffer removes the sink.
        Purpose: Confirm the sink is removed correctly.
        """
        print("Setup: Mode all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger

            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = tmp_path / "debug_logs"

            # Set up log capture
            dbg_logger.prepare_new_request()
            sink_id = dbg_logger._loguru_sink_id
            assert sink_id is not None

            print("Action: Calling _clear_app_logs_buffer...")
            dbg_logger._clear_app_logs_buffer()

            print("Check that sink_id was reset...")
            assert dbg_logger._loguru_sink_id is None

    def test_app_logs_not_saved_when_empty(self, tmp_path):
        """
        What it does: Verifies that empty logs do not create a file.
        Purpose: Confirm app_logs.txt is not created when there are no logs.
        """
        print("Setup: Mode all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()

        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger

            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir

            # Do NOT write anything to the buffer

            print("Action: Calling _write_app_logs_to_file...")
            dbg_logger._write_app_logs_to_file()

            print("Check that app_logs.txt was NOT created...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert not app_logs_file.exists()
