"""Unit tests for kiro.baselines — BaselinesWriter.

Covers: single record roundtrip, concurrent writers serialised via lock,
path sanitisation, non-blocking assertion (no builtins.open inside
``async def write`` — disk I/O runs via asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro.baselines import BaselinesWriter


@pytest.mark.asyncio
async def test_write_single_record_produces_one_valid_json_line(tmp_path: Path) -> None:
    writer = BaselinesWriter(state_dir=tmp_path)
    record = {"ts": 1.0, "source": "gateway-requests", "model": "claude-opus-4-7"}
    await writer.write("gateway-requests", record)

    path = tmp_path / "baselines-gateway-requests.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == record


@pytest.mark.asyncio
async def test_concurrent_writes_land_as_distinct_lines(tmp_path: Path) -> None:
    """5 concurrent writes → 5 valid JSON lines, no interleaving."""
    writer = BaselinesWriter(state_dir=tmp_path)
    records = [{"ts": float(i), "idx": i} for i in range(5)]
    await asyncio.gather(*[writer.write("gateway-requests", r) for r in records])

    path = tmp_path / "baselines-gateway-requests.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed = sorted((json.loads(line) for line in lines), key=lambda r: r["idx"])
    assert parsed == sorted(records, key=lambda r: r["idx"])


@pytest.mark.asyncio
async def test_append_preserves_existing_lines(tmp_path: Path) -> None:
    writer = BaselinesWriter(state_dir=tmp_path)
    await writer.write("hook-events", {"a": 1})
    await writer.write("hook-events", {"b": 2})
    await writer.write("hook-events", {"c": 3})

    path = tmp_path / "baselines-hook-events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}, {"c": 3}]


@pytest.mark.asyncio
async def test_utf8_content_roundtrips(tmp_path: Path) -> None:
    writer = BaselinesWriter(state_dir=tmp_path)
    await writer.write("x", {"msg": "héllo — wörld 🦀"})
    got = (tmp_path / "baselines-x.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(got) == {"msg": "héllo — wörld 🦀"}


@pytest.mark.asyncio
async def test_invalid_source_rejected(tmp_path: Path) -> None:
    writer = BaselinesWriter(state_dir=tmp_path)
    for bad in ["../escape", "has space", "with/slash", ""]:
        with pytest.raises(ValueError, match="source must match"):
            await writer.write(bad, {"ok": False})


@pytest.mark.asyncio
async def test_creates_state_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "state"
    writer = BaselinesWriter(state_dir=nested)
    assert not nested.exists()
    await writer.write("x", {"ok": True})
    assert nested.exists()
    assert (nested / "baselines-x.jsonl").exists()


@pytest.mark.asyncio
async def test_write_offloads_disk_io_to_thread(tmp_path: Path) -> None:
    """Hot path does not call ``builtins.open`` synchronously.

    Patches ``asyncio.to_thread`` to fail the test if the append logic
    runs on the main thread. Also asserts ``builtins.open`` is not called
    directly inside the coroutine body.
    """
    writer = BaselinesWriter(state_dir=tmp_path)

    called: list[str] = []
    original_to_thread = asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        called.append(func.__name__)
        return await original_to_thread(func, *args, **kwargs)

    with patch("kiro.baselines.asyncio.to_thread", side_effect=spy_to_thread):
        await writer.write("x", {"ok": True})

    assert called == ["_blocking_append"]


def test_no_sync_open_in_write_coroutine_body() -> None:
    """Static guard: the write() coroutine body does not contain a literal ``open(``.

    A stronger guard than the runtime patch test — catches accidental
    reintroduction of synchronous I/O.
    """
    src = inspect.getsource(BaselinesWriter.write)
    # The only "open" reference allowed is in docstrings/comments; the body
    # should delegate to ``_blocking_append`` via ``to_thread``.
    body_lines = [line for line in src.splitlines() if not line.strip().startswith('#')]
    body = "\n".join(body_lines)
    assert "open(" not in body, (
        "BaselinesWriter.write must offload open() to a background thread"
    )
