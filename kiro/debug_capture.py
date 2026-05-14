# -*- coding: utf-8 -*-
"""
Upstream error capture (follow-up #1 from 4.7 error surfacing plan).

When DEBUG_CAPTURE_UPSTREAM_ERRORS is enabled, persists non-2xx upstream
response bodies to disk so the next incident produces evidence rather
than guesses.

Zero behaviour change when disabled. All writes wrapped in broad
try/except so capture failures never break the hot path. Standard
library only.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# Per-request context set by instrumented route handlers. Async-safe.
_request_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "kiro_debug_capture_request_context", default=None
)

# Case-insensitive header keys redacted before persistence.
_REDACT_HEADER_PATTERN = re.compile(
    r"^(authorization|cookie|api-key|x-api-key|x-amz-security-token)$",
    re.IGNORECASE,
)

_BODY_TRUNCATE_BYTES = 64 * 1024  # 64 KiB
_USER_PREVIEW_CHARS = 200


def _enabled() -> bool:
    try:
        from kiro import config
        return bool(getattr(config, "DEBUG_CAPTURE_UPSTREAM_ERRORS", False))
    except Exception:
        return False


def set_request_context(
    *,
    model: Optional[str] = None,
    messages: Optional[Iterable[Any]] = None,
) -> None:
    """Record light-weight per-request context. No-op when capture disabled."""
    try:
        if not _enabled():
            return
        _request_context.set({
            "model": model,
            "user_preview": _preview_user_message(messages),
        })
    except Exception:
        pass  # Never raise from instrumentation.


def _preview_user_message(messages: Optional[Iterable[Any]]) -> str:
    if not messages:
        return ""
    try:
        for msg in reversed(list(messages)):
            role = getattr(msg, "role", None)
            if role is None and isinstance(msg, dict):
                role = msg.get("role")
            if role != "user":
                continue
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            text = _coerce_to_text(content)
            if text:
                return text[:_USER_PREVIEW_CHARS]
        return ""
    except Exception:
        return ""


def _coerce_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts)
    return str(content)


def capture_upstream_error(
    *,
    status_code: int,
    body: str,
    headers: Optional[Mapping[str, str]] = None,
    source: str = "unknown",
) -> None:
    """Persist one upstream error response to disk. No-op if disabled."""
    try:
        if not _enabled():
            return
        from kiro import config

        now = datetime.now(timezone.utc)
        root = Path(config.DEBUG_CAPTURE_DIR).expanduser()
        date_dir = root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        short_id = uuid.uuid4().hex[:8]
        target = date_dir / f"{now.strftime('%H%M%S')}-{short_id}.json"

        ctx = _request_context.get() or {}

        payload = {
            "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source": source,
            "upstream_status": int(status_code),
            "upstream_headers": _filter_headers(headers),
            "upstream_body": _truncate_body(body),
            "client_model_request": {
                "model": ctx.get("model"),
                "user_preview": ctx.get("user_preview", ""),
            },
        }

        _atomic_write_json(target, payload)
        _prune_old_files(root, keep=config.DEBUG_CAPTURE_MAX_FILES)
    except Exception as e:
        logger.debug(f"debug_capture: capture_upstream_error failed: {e!r}")


def _filter_headers(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if not headers:
        return {}
    out: Dict[str, str] = {}
    try:
        for k, v in headers.items():
            if _REDACT_HEADER_PATTERN.match(str(k)):
                continue
            out[str(k)] = str(v)
    except Exception:
        return {}
    return out


def _truncate_body(body: Any) -> str:
    if body is None:
        return ""
    if not isinstance(body, str):
        try:
            body = str(body)
        except Exception:
            return ""
    b = body.encode("utf-8", errors="replace")
    if len(b) <= _BODY_TRUNCATE_BYTES:
        return body
    return b[:_BODY_TRUNCATE_BYTES].decode("utf-8", errors="replace") + "...[truncated]"


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=".tmp-",
        suffix=target.suffix,
        delete=False,
    )
    try:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, target)


def _prune_old_files(root: Path, *, keep: int) -> None:  # noqa: C901
    """Keep at most `keep` most-recent capture files across all date subdirs."""
    if keep <= 0:
        return
    try:
        if not root.exists():
            return
        files = []
        for date_dir in root.iterdir():
            if not date_dir.is_dir():
                continue
            for f in date_dir.iterdir():
                if not (f.is_file() and f.suffix == ".json" and not f.name.startswith(".tmp-")):
                    continue
                try:
                    files.append((f.stat().st_mtime, f))
                except FileNotFoundError:
                    continue
        if len(files) <= keep:
            return
        files.sort(reverse=True)  # newest first
        for _, f in files[keep:]:
            try:
                f.unlink()
            except FileNotFoundError:
                pass
    except Exception as e:
        logger.debug(f"debug_capture: prune failed: {e!r}")
