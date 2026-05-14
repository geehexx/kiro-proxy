"""
Logfire / OpenTelemetry telemetry for kiro-gateway.

Design principles:
- Hierarchy: user_request → gateway.request → (cache.hit | upstream.call)
- Truncation: prompts/responses capped at 500 chars to control Logfire data volume
- Model names: always use normalized form (claude-sonnet-4.6, not raw client names)
- No test noise: min_level=INFO, no debug spans
- Cost tracking: Kiro Power plan ($0.02/invocation flat, $0.04 overage)
- OTel semantic conventions: gen_ai.* namespace for LLM attributes
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from loguru import logger

_MAX_PROMPT_CHARS = 500   # truncate prompts/responses in spans
_MAX_ATTR_CHARS = 200     # truncate other string attributes

try:
    import logfire as _logfire
    _LOGFIRE_AVAILABLE = True
except ImportError:
    _logfire = None  # type: ignore[assignment]
    _LOGFIRE_AVAILABLE = False

_MONTHLY_COST_USD = float(os.getenv("KIRO_PLAN_MONTHLY_COST_USD", "200.0"))
_MONTHLY_INVOCATIONS = int(os.getenv("KIRO_PLAN_MONTHLY_INVOCATIONS", "10000"))
_OVERAGE_COST_USD = float(os.getenv("KIRO_PLAN_OVERAGE_COST_USD", "0.04"))
_COST_PER_INVOCATION_FLAT = _MONTHLY_COST_USD / _MONTHLY_INVOCATIONS  # $0.02

_configured = False


def _trunc(s: Any, max_chars: int = _MAX_ATTR_CHARS) -> str:
    """Truncate a value to max_chars for Logfire attribute storage."""
    if s is None:
        return ""
    text = str(s)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"…[+{len(text)-max_chars}]"


def _cost_usd(is_overage: bool = False) -> float:
    return _OVERAGE_COST_USD if is_overage else _COST_PER_INVOCATION_FLAT


def setup_logfire() -> bool:
    """Configure logfire. Returns True on success. Non-fatal on failure."""
    global _configured
    if _configured:
        return True
    if not _LOGFIRE_AVAILABLE:
        logger.info("logfire not installed — telemetry disabled")
        return False

    # Skip telemetry in test environments to avoid polluting Logfire with test spans.
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI"):
        logger.info("test/CI environment detected — Logfire telemetry disabled")
        return False

    token = os.getenv("LOGFIRE_TOKEN", "")
    if not token:
        logger.info("LOGFIRE_TOKEN not set — telemetry disabled")
        return False

    try:
        _logfire.configure(
            token=token,
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "kiro-gateway"),
            service_version=os.getenv("APP_VERSION", "2.4-dev"),
            environment=os.getenv("LOGFIRE_ENVIRONMENT", "production"),
            send_to_logfire=True,
            console=False,
            # Only INFO+ — suppress debug spans and test noise
            min_level="info",
            # Scrub common secret patterns from attributes
            scrubbing=_logfire.ScrubbingOptions(
                extra_patterns=["token", "api.key", "auth", "bearer", "secret"]
            ) if hasattr(_logfire, "ScrubbingOptions") else None,
        )
        _configured = True
        logger.info("Logfire telemetry configured (project: kiro-gateway)")
        return True
    except Exception as e:
        logger.warning(f"Logfire setup failed (non-fatal): {e}")
        return False


def instrument_fastapi(app: object) -> None:
    """Instrument FastAPI + httpx. Excludes health/metrics endpoints."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        _logfire.instrument_fastapi(
            app,  # type: ignore[arg-type]
            capture_headers=False,
            excluded_urls="/health,/metrics,/v1/models",
        )
        _logfire.instrument_httpx()
        logger.info("Logfire FastAPI + httpx instrumentation active")
    except Exception as e:
        logger.warning(f"Logfire instrumentation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Span context managers — use these to create the hierarchy
# ---------------------------------------------------------------------------

@contextmanager
def user_request_span(
    *,
    model: str,
    stream: bool,
    message_count: int,
    session_id: Optional[str] = None,
    last_user_message_preview: Optional[str] = None,
) -> Generator[None, None, None]:
    """Top-level span for a user request. Wraps the entire gateway handling.

    Hierarchy: user_request → gateway.request → cache.hit | upstream.call

    Usage:
        with user_request_span(model=..., stream=..., ...):
            # all child spans nest under this
    """
    if not _configured or not _LOGFIRE_AVAILABLE:
        yield
        return
    try:
        attrs: dict[str, Any] = {
            "gen_ai.system": "kiro-gateway",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "chat",
            "kiro.request.stream": stream,
            "kiro.request.message_count": message_count,
        }
        if session_id:
            attrs["kiro.session.id"] = session_id[:16]
        if last_user_message_preview:
            attrs["kiro.request.prompt_preview"] = _trunc(last_user_message_preview, _MAX_PROMPT_CHARS)

        with _logfire.span("user_request", **attrs):
            yield
    except Exception:
        yield


@contextmanager
def gateway_request_span(  # noqa: PLR0913
    *,
    model: str,
    stream: bool,
    re2_applied: bool,
    cache_result: str,
    upstream_ms: Optional[int] = None,
    status: int = 200,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error_reason: Optional[str] = None,
    retry_count: int = 0,
    is_overage: bool = False,
) -> Generator[None, None, None]:
    """Span for gateway processing of one request. Nests under user_request_span."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        yield
        return
    try:
        cost = _cost_usd(is_overage)
        attrs: dict[str, Any] = {
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens or 0,
            "gen_ai.usage.output_tokens": output_tokens or 0,
            "kiro.gateway.cache": cache_result,
            "kiro.gateway.re2_applied": re2_applied,
            "kiro.gateway.stream": stream,
            "kiro.gateway.status": status,
            "kiro.gateway.upstream_ms": upstream_ms or 0,
            "kiro.gateway.retry_count": retry_count,
            "kiro.cost.invocation_usd": cost,
            "kiro.cost.plan": "kiro-power",
        }
        if error_reason:
            attrs["kiro.gateway.error_reason"] = _trunc(error_reason)
        if is_overage:
            attrs["kiro.cost.is_overage"] = True

        span_name = f"gateway {'stream' if stream else 'sync'} [{cache_result}]"
        with _logfire.span(span_name, **attrs):
            yield
    except Exception:
        yield


# ---------------------------------------------------------------------------
# Lightweight event emitters (no span overhead)
# ---------------------------------------------------------------------------

def emit_cache_event(
    *,
    event: str,
    model: str,
    cache_key_prefix: Optional[str] = None,
) -> None:
    """Emit cache.hit / cache.miss / cache.bypass as a log event."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        _logfire.info(
            f"cache.{event}",
            **{
                "kiro.cache.event": event,
                "kiro.cache.model": model,
                "kiro.cache.key": cache_key_prefix or "",
            }
        )
    except Exception:
        pass


def emit_upstream_call(
    *,
    model: str,
    upstream_ms: int,
    status: int,
    error_reason: Optional[str] = None,
) -> None:
    """Emit upstream API call timing as a log event."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        attrs: dict[str, Any] = {
            "kiro.upstream.model": model,
            "kiro.upstream.latency_ms": upstream_ms,
            "kiro.upstream.status": status,
        }
        if error_reason:
            attrs["kiro.upstream.error"] = _trunc(error_reason)
        level = "warning" if status >= 400 else "info"
        getattr(_logfire, level)(f"upstream.call [{status}]", **attrs)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Legacy flat emitter — used by _emit_gateway_baseline until routes are refactored
# ---------------------------------------------------------------------------

def record_request(  # noqa: PLR0913
    *,
    model: str,
    stream: bool,
    gateway_cache: str,
    re2_applied: bool,
    upstream_ms: Optional[int],
    status: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error_reason: Optional[str] = None,
    retry_count: Optional[int] = None,
    session_id: Optional[str] = None,
    is_overage: bool = False,
    complexity_label: Optional[str] = None,
    dedup_hit: bool = False,
) -> None:
    """Flat span emitter. Used by _emit_gateway_baseline.

    Prefer gateway_request_span() for new code — it creates proper hierarchy.
    Skips low-token requests (sub-agent tool calls, episodic memory) to reduce noise.
    """
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    # Skip low-value spans: sub-agent tool calls and episodic memory lookups
    # are high-volume but low-signal. Only log errors and substantial requests.
    from kiro.config import LOGFIRE_MIN_INPUT_TOKENS
    if (
        status == 200
        and input_tokens is not None
        and input_tokens < LOGFIRE_MIN_INPUT_TOKENS
    ):
        return
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        cost = _cost_usd(is_overage)
        attrs: dict[str, Any] = {
            "gen_ai.system": "aws_bedrock",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens or 0,
            "gen_ai.usage.output_tokens": output_tokens or 0,
            "kiro.gateway.cache": gateway_cache,
            "kiro.gateway.re2_applied": re2_applied,
            "kiro.gateway.stream": stream,
            "kiro.gateway.status": status,
            "kiro.gateway.upstream_ms": upstream_ms or 0,
            "kiro.gateway.retry_count": retry_count or 0,
            "kiro.cost.invocation_usd": cost,
            "kiro.cost.plan": "kiro-power",
        }
        if error_reason:
            attrs["kiro.gateway.error_reason"] = _trunc(error_reason)
        if session_id:
            # Use conversation.id to avoid Logfire PII scrubbing on "session" substring
            attrs["kiro.conversation.id"] = session_id[:16]
        if complexity_label:
            attrs["kiro.gateway.complexity_label"] = complexity_label
        if dedup_hit:
            attrs["kiro.gateway.dedup_hit"] = True

        span_name = f"gateway.request [{'stream' if stream else 'sync'}] [{gateway_cache}]"
        with _logfire.span(span_name, **attrs):
            pass
    except Exception as e:
        logger.debug(f"Logfire record_request failed (non-fatal): {e}")


def record_model_resolution(
    *,
    raw_model: str,
    resolved_model: str,
    resolution_source: str,
) -> None:
    """Log model name resolution when raw != resolved."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        if raw_model != resolved_model:
            _logfire.info(
                "model.resolved",
                **{
                    "kiro.model.raw": raw_model,
                    "kiro.model.resolved": resolved_model,
                    "kiro.model.source": resolution_source,
                }
            )
    except Exception:
        pass
