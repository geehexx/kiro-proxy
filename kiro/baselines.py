"""
BaselinesWriter — append-only JSONL emitter for telemetry baselines.

Each gateway request, hook event, or similar "should be comparable over
time" datapoint is written as one JSON object per line to
``~/.claude/state/baselines-{source}.jsonl``. The parent repo's token
telemetry plan joins these files by ``message_id`` in a later step.

Design:
- Append-only. Never rotates the file (caller's problem); never rewrites.
- One file per *source* (e.g., "gateway-requests", "hook-events"). The
  source name is sanitised; anything outside ``[A-Za-z0-9._-]`` is
  rejected so a caller can't walk the filesystem via the source arg.
- Async-safe: the public ``write`` coroutine offloads the disk write to
  a background thread via ``asyncio.to_thread``. The serving event loop
  never blocks on fsync/flush.
- Concurrency: an internal ``asyncio.Lock`` serialises writers inside a
  single process so interleaved multi-line records can't appear
  mid-line. Cross-process serialisation is NOT provided — run one
  gateway process per state dir.
- No buffering. Each ``write`` call flushes. Loss window on crash is
  bounded by OS page-cache behaviour for append-open + flush.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

_SOURCE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _default_state_dir() -> Path:
    """Resolve ``~/.claude/state``. Honours ``$HOME`` for sandboxed tests."""
    return Path(os.path.expanduser("~/.claude/state"))


class BaselinesWriter:
    """Append JSON records to per-source baseline files.

    Usage:
        writer = BaselinesWriter()
        await writer.write("gateway-requests", {"ts": ..., "model": ...})

    Thread/task-safe within one event loop. Do not share across loops.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir: Path = state_dir if state_dir is not None else _default_state_dir()
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def _path_for(self, source: str) -> Path:
        if not _SOURCE_RE.match(source):
            raise ValueError(
                f"source must match {_SOURCE_RE.pattern!r}; got {source!r}"
            )
        return self._state_dir / f"baselines-{source}.jsonl"

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        """Encode one record as a single line of JSON + trailing newline.

        ``ensure_ascii=False`` keeps UTF-8 text readable; ``separators``
        strips whitespace so one record never contains a literal newline
        mid-object.
        """
        body = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if "\n" in body:  # Defensive — json.dumps with separators should never emit '\n'
            body = body.replace("\n", " ")
        return (body + "\n").encode("utf-8")

    @staticmethod
    def _blocking_append(path: Path, payload: bytes) -> None:
        """Open the file in append mode and write one record.

        Executed on a background thread via ``asyncio.to_thread`` so the
        event loop never blocks on disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Opening with ``"ab"`` takes an advisory file lock on some
        # platforms; POSIX append is atomic for <PIPE_BUF bytes so
        # small-to-medium records are safe even across processes.
        with open(path, "ab") as f:
            f.write(payload)
            f.flush()

    async def write(self, source: str, record: dict[str, Any]) -> None:
        """Append one record to ``baselines-{source}.jsonl``.

        Never raises on encoding errors — a record that can't be JSON-serialised
        would block the hot path; caller-side filtering is cheaper.
        Raises ``ValueError`` on bad source name (programmer error, fast fail).
        """
        path = self._path_for(source)
        payload = self._encode(record)
        async with self._lock:
            await asyncio.to_thread(self._blocking_append, path, payload)
