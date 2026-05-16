"""Tests for kiro.debug_capture — upstream error capture to disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kiro import debug_capture
from kiro.debug_capture import (
    _coerce_to_text,
    _filter_headers,
    _preview_user_message,
    _prune_old_files,
    _truncate_body,
    capture_upstream_error,
    set_request_context,
)


@pytest.fixture
def tmp_capture_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch DEBUG_CAPTURE_DIR + _enabled() to True for the test."""
    monkeypatch.setattr(debug_capture, "_enabled", lambda: True)

    class _FakeConfig:
        DEBUG_CAPTURE_UPSTREAM_ERRORS = True
        DEBUG_CAPTURE_DIR = str(tmp_path)
        DEBUG_CAPTURE_MAX_FILES = 200

    # Replace the config module attribute the function imports.
    monkeypatch.setattr("kiro.config.DEBUG_CAPTURE_UPSTREAM_ERRORS", True, raising=False)
    monkeypatch.setattr("kiro.config.DEBUG_CAPTURE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr("kiro.config.DEBUG_CAPTURE_MAX_FILES", 200, raising=False)
    return tmp_path


def test_disabled_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When DEBUG_CAPTURE_UPSTREAM_ERRORS=False, no files are written."""
    monkeypatch.setattr(debug_capture, "_enabled", lambda: False)
    capture_upstream_error(status_code=500, body="boom")
    assert list(tmp_path.glob("**/*.json")) == []


def test_capture_writes_file_with_redacted_headers(tmp_capture_dir: Path) -> None:
    capture_upstream_error(
        status_code=500,
        body='{"error": "internal"}',
        headers={
            "Authorization": "Bearer secret-token",
            "X-API-Key": "leak-me",
            "Content-Type": "application/json",
            "x-amz-security-token": "AWS-redacted",
            "Server": "nginx",
        },
        source="anthropic-route",
    )
    files = list(tmp_capture_dir.glob("**/*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["upstream_status"] == 500
    assert payload["source"] == "anthropic-route"
    assert payload["upstream_body"] == '{"error": "internal"}'
    headers = payload["upstream_headers"]
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers
    assert "x-amz-security-token" not in headers
    assert headers.get("Content-Type") == "application/json"
    assert headers.get("Server") == "nginx"


def test_body_truncation_at_64kb(tmp_capture_dir: Path) -> None:
    huge = "A" * 100_000  # 100 KB > 64 KiB cutoff
    capture_upstream_error(status_code=500, body=huge)
    files = list(tmp_capture_dir.glob("**/*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())["upstream_body"]
    assert body.endswith("...[truncated]")
    assert len(body.encode("utf-8")) < 100_000


def test_truncate_body_handles_none() -> None:
    assert _truncate_body(None) == ""


def test_truncate_body_handles_non_string() -> None:
    out = _truncate_body(12345)
    assert out == "12345"


def test_truncate_body_passthrough_short() -> None:
    assert _truncate_body("hello") == "hello"


def test_filter_headers_handles_none() -> None:
    assert _filter_headers(None) == {}


def test_filter_headers_handles_empty() -> None:
    assert _filter_headers({}) == {}


def test_filter_headers_redacts_case_insensitive() -> None:
    out = _filter_headers({
        "AUTHORIZATION": "Bearer x",
        "Cookie": "abc",
        "API-KEY": "k",
        "X-Trace-Id": "t1",
    })
    assert out == {"X-Trace-Id": "t1"}


def test_set_request_context_then_capture_includes_model(tmp_capture_dir: Path) -> None:
    set_request_context(
        model="claude-opus-4.7",
        messages=[{"role": "user", "content": "hello world"}],
    )
    capture_upstream_error(status_code=503, body="overload")
    files = list(tmp_capture_dir.glob("**/*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["client_model_request"]["model"] == "claude-opus-4.7"
    assert "hello world" in payload["client_model_request"]["user_preview"]


def test_set_request_context_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """When capture is disabled, set_request_context returns silently."""
    monkeypatch.setattr(debug_capture, "_enabled", lambda: False)
    set_request_context(model="x", messages=[{"role": "user", "content": "y"}])


def test_preview_user_message_handles_empty_iterable() -> None:
    assert _preview_user_message(None) == ""
    assert _preview_user_message([]) == ""


def test_preview_user_message_picks_last_user() -> None:
    msgs = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ignored2"},
        {"role": "user", "content": "second"},
    ]
    assert _preview_user_message(msgs) == "second"


def test_preview_user_message_truncates_to_200_chars() -> None:
    long_content = "x" * 500
    msgs = [{"role": "user", "content": long_content}]
    out = _preview_user_message(msgs)
    assert len(out) == 200


def test_preview_user_message_with_object_attrs() -> None:
    class _Msg:
        def __init__(self, role: str, content: Any) -> None:
            self.role = role
            self.content = content

    msgs = [_Msg("system", "x"), _Msg("user", "from-attr")]
    assert _preview_user_message(msgs) == "from-attr"


def test_coerce_to_text_string() -> None:
    assert _coerce_to_text("hi") == "hi"


def test_coerce_to_text_none() -> None:
    assert _coerce_to_text(None) == ""


def test_coerce_to_text_list_of_strings() -> None:
    assert _coerce_to_text(["a", "b"]) == "a b"


def test_coerce_to_text_list_of_text_blocks() -> None:
    blocks = [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}]
    assert _coerce_to_text(blocks) == "alpha beta"


def test_coerce_to_text_list_of_objects_with_text_attr() -> None:
    class _Block:
        def __init__(self, text: str) -> None:
            self.text = text

    blocks = [_Block("a"), _Block("b")]
    assert _coerce_to_text(blocks) == "a b"


def test_coerce_to_text_arbitrary_value() -> None:
    assert _coerce_to_text(42) == "42"


def test_prune_old_files_keeps_newest_n(tmp_path: Path) -> None:
    date_dir = tmp_path / "2026-05-15"
    date_dir.mkdir()
    # Create 10 files with monotonically increasing mtime
    for i in range(10):
        f = date_dir / f"{i:02d}.json"
        f.write_text("{}")
        # Bump mtime so sort is deterministic
        import os as _os
        _os.utime(f, (1700000000 + i, 1700000000 + i))

    _prune_old_files(tmp_path, keep=3)
    remaining = sorted(p.name for p in date_dir.glob("*.json"))
    assert len(remaining) == 3
    # Newest 3 are 07, 08, 09
    assert remaining == ["07.json", "08.json", "09.json"]


def test_prune_old_files_zero_is_noop(tmp_path: Path) -> None:
    date_dir = tmp_path / "2026-05-15"
    date_dir.mkdir()
    (date_dir / "a.json").write_text("{}")
    _prune_old_files(tmp_path, keep=0)
    # zero/negative is noop
    assert (date_dir / "a.json").exists()


def test_prune_old_files_skips_tmp_files(tmp_path: Path) -> None:
    date_dir = tmp_path / "2026-05-15"
    date_dir.mkdir()
    (date_dir / "real.json").write_text("{}")
    (date_dir / ".tmp-x.json").write_text("{}")
    _prune_old_files(tmp_path, keep=0)  # noop, but verifies it doesn't crash on .tmp-
    # Now test with keep=10 — both files survive
    _prune_old_files(tmp_path, keep=10)
    assert (date_dir / "real.json").exists()
    assert (date_dir / ".tmp-x.json").exists()


def test_prune_old_files_missing_root_is_safe(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope"
    _prune_old_files(nonexistent, keep=10)  # must not raise


def test_capture_swallows_exceptions(tmp_capture_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture-time exception must not propagate to the caller."""
    def boom(*_a, **_kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(debug_capture, "_atomic_write_json", boom)
    # Must not raise
    capture_upstream_error(status_code=500, body="oops")


def test_atomic_write_creates_no_partial_file_on_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If json.dump fails mid-write, no partial target file exists."""
    target = tmp_path / "out.json"

    def fail_dump(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("kiro.debug_capture.json.dump", fail_dump)
    with pytest.raises(OSError):
        debug_capture._atomic_write_json(target, {"k": "v"})
    assert not target.exists()
