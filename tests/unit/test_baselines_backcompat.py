"""
Backward-compatibility tests for baselines JSONL format.

Verifies that:
- Old-format JSONL lines (without §2 telemetry fields) parse without crash.
- New-format lines (with error_reason, retry_count, retry_after_applied_ms)
  are written and read back correctly.
- New readers tolerate old lines (missing fields → None/absent).
- Old readers tolerate new lines (extra fields are ignored by json.loads).

This ensures the append-only JSONL file can be read by both old and new
code without schema migration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro.baselines import BaselinesWriter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OLD_FORMAT_LINE = json.dumps({
    "ts": 1_700_000_000.0,
    "source": "gateway-requests",
    "message_id": "msg_01abc",
    "session_id_gw": "sess_xyz",
    "cache_key": "abcdef0123456789",
    "model": "claude-sonnet-4",
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": None,
    "cache_creation_input_tokens": None,
    "upstream_ms_total": 1234,
    "gateway_cache": "miss",
    "stream": False,
    "status": 200,
    # NOTE: no error_reason, retry_count, retry_after_applied_ms
})

NEW_FORMAT_LINE = json.dumps({
    "ts": 1_700_000_001.0,
    "source": "gateway-requests",
    "message_id": "msg_02def",
    "session_id_gw": "sess_xyz",
    "cache_key": "abcdef0123456789",
    "model": "claude-opus-4",
    "input_tokens": 200,
    "output_tokens": 80,
    "cache_read_input_tokens": None,
    "cache_creation_input_tokens": None,
    "upstream_ms_total": 2500,
    "gateway_cache": "bypass",
    "stream": True,
    "status": 429,
    "error_reason": "INSUFFICIENT_MODEL_CAPACITY",
    "retry_count": 1,
    "retry_after_applied_ms": 30000,
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOldFormatBackwardCompat:
    """Old-format lines (no §2 fields) must parse without crash."""

    def test_old_line_parses_as_valid_json(self):
        """
        What it does: Verifies an old-format JSONL line is valid JSON.
        Purpose: Ensure old lines written before §2 fields were added
                 can still be loaded by json.loads without error.
        """
        parsed = json.loads(OLD_FORMAT_LINE)
        assert parsed["model"] == "claude-sonnet-4"
        assert parsed["status"] == 200

    def test_old_line_missing_new_fields_returns_none_via_get(self):
        """
        What it does: Verifies .get() on missing §2 fields returns None.
        Purpose: New readers that call record.get("error_reason") on old
                 lines must not raise KeyError.
        """
        parsed = json.loads(OLD_FORMAT_LINE)
        assert parsed.get("error_reason") is None
        assert parsed.get("retry_count") is None
        assert parsed.get("retry_after_applied_ms") is None

    def test_old_line_core_fields_intact(self):
        """
        What it does: Verifies all original fields survive the round-trip.
        Purpose: Ensure backward compat doesn't corrupt existing fields.
        """
        parsed = json.loads(OLD_FORMAT_LINE)
        assert parsed["ts"] == 1_700_000_000.0
        assert parsed["message_id"] == "msg_01abc"
        assert parsed["input_tokens"] == 100
        assert parsed["output_tokens"] == 50
        assert parsed["gateway_cache"] == "miss"


class TestNewFormatForwardCompat:
    """New-format lines (with §2 fields) must be readable by old code."""

    def test_new_line_parses_as_valid_json(self):
        """
        What it does: Verifies a new-format JSONL line is valid JSON.
        Purpose: Ensure new lines can be loaded without error.
        """
        parsed = json.loads(NEW_FORMAT_LINE)
        assert parsed["model"] == "claude-opus-4"
        assert parsed["status"] == 429

    def test_new_line_extra_fields_ignored_by_old_reader(self):
        """
        What it does: Verifies old readers can ignore new §2 fields.
        Purpose: json.loads returns a dict; old code that only reads
                 known keys is unaffected by extra keys.
        """
        parsed = json.loads(NEW_FORMAT_LINE)
        # Old reader only accesses known fields — extra fields are just ignored
        old_reader_fields = {
            "ts", "source", "message_id", "session_id_gw", "cache_key",
            "model", "input_tokens", "output_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens", "upstream_ms_total", "gateway_cache",
            "stream", "status",
        }
        for field in old_reader_fields:
            assert field in parsed  # All old fields still present

    def test_new_line_new_fields_readable(self):
        """
        What it does: Verifies §2 fields are present and correct in new lines.
        Purpose: Ensure telemetry fields are written and readable.
        """
        parsed = json.loads(NEW_FORMAT_LINE)
        assert parsed["error_reason"] == "INSUFFICIENT_MODEL_CAPACITY"
        assert parsed["retry_count"] == 1
        assert parsed["retry_after_applied_ms"] == 30000


class TestMixedFormatFile:
    """A JSONL file with both old and new lines must be fully readable."""

    @pytest.mark.asyncio
    async def test_mixed_file_all_lines_parse(self, tmp_path: Path) -> None:
        """
        What it does: Writes old and new format lines to the same JSONL file
                      and verifies all lines parse without error.
        Purpose: Simulate a real baseline file that accumulated records before
                 and after the §2 schema addition.
        """
        jsonl_path = tmp_path / "baselines-gateway-requests.jsonl"
        # Write old-format line directly (simulating pre-§2 records)
        jsonl_path.write_text(OLD_FORMAT_LINE + "\n", encoding="utf-8")

        # Append a new-format line via BaselinesWriter
        writer = BaselinesWriter(state_dir=tmp_path)
        await writer.write("gateway-requests", {
            "ts": 1_700_000_002.0,
            "source": "gateway-requests",
            "message_id": "msg_03ghi",
            "session_id_gw": None,
            "cache_key": None,
            "model": "claude-opus-4",
            "input_tokens": 300,
            "output_tokens": 120,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "upstream_ms_total": 3000,
            "gateway_cache": "bypass",
            "stream": True,
            "status": 429,
            "error_reason": "INSUFFICIENT_MODEL_CAPACITY",
            "retry_count": 2,
            "retry_after_applied_ms": 60000,
        })

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        # Both lines must parse without error
        records = [json.loads(line) for line in lines]

        # Old line: no §2 fields
        assert records[0].get("error_reason") is None
        assert records[0]["status"] == 200

        # New line: §2 fields present
        assert records[1]["error_reason"] == "INSUFFICIENT_MODEL_CAPACITY"
        assert records[1]["retry_count"] == 2
        assert records[1]["retry_after_applied_ms"] == 60000

    @pytest.mark.asyncio
    async def test_null_new_fields_written_correctly(self, tmp_path: Path) -> None:
        """
        What it does: Verifies that None values for §2 fields are written as JSON null.
        Purpose: Ensure success-path records (no error) write null for §2 fields,
                 which is the correct sentinel for "no error occurred".
        """
        writer = BaselinesWriter(state_dir=tmp_path)
        await writer.write("gateway-requests", {
            "ts": 1.0,
            "status": 200,
            "error_reason": None,
            "retry_count": None,
            "retry_after_applied_ms": None,
        })

        path = tmp_path / "baselines-gateway-requests.jsonl"
        parsed = json.loads(path.read_text(encoding="utf-8").strip())
        assert parsed["error_reason"] is None
        assert parsed["retry_count"] is None
        assert parsed["retry_after_applied_ms"] is None


class TestComplexityLabelBackcompat:
    """Regression: complexity_label field added in 2026-05-14 session."""

    def test_new_line_has_complexity_label(self, tmp_path):
        """New records include complexity_label field."""
        import asyncio

        from kiro.baselines import BaselinesWriter as BaselineWriter

        writer = BaselineWriter(tmp_path)
        record = {
            "ts": 1.0, "source": "gateway-requests", "model": "claude-sonnet-4.6",
            "re2_applied": True, "complexity_label": "medium",
        }
        asyncio.run(writer.write("gateway-requests", record))
        path = tmp_path / "baselines-gateway-requests.jsonl"
        parsed = json.loads(path.read_text(encoding="utf-8").strip())
        assert parsed["complexity_label"] == "medium"

    def test_old_line_missing_complexity_label_returns_none(self, tmp_path):
        """Old records without complexity_label return None via .get()."""
        import asyncio

        from kiro.baselines import BaselinesWriter as BaselineWriter

        writer = BaselineWriter(tmp_path)
        record = {
            "ts": 1.0, "source": "gateway-requests", "model": "claude-sonnet-4.6",
            "re2_applied": False,
            # No complexity_label — simulates old record
        }
        asyncio.run(writer.write("gateway-requests", record))
        path = tmp_path / "baselines-gateway-requests.jsonl"
        parsed = json.loads(path.read_text(encoding="utf-8").strip())
        assert parsed.get("complexity_label") is None
