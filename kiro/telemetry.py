"""
Logfire / OpenTelemetry telemetry for kiro-gateway.

Provides:
- Gateway request spans with full metadata
- Cache hit/miss/bypass counters
- re2 injection tracking
- Upstream latency histograms
- Kiro plan cost estimation (invocations → $ overage)
- Model routing events

Usage:
    from kiro.telemetry import setup_logfire, record_request, record_cache_event

    # In main.py lifespan:
    setup_logfire()

    # In routes_anthropic.py after each request:
    record_request(model=..., stream=..., cache=..., re2=..., upstream_ms=..., status=...)
"""

from __future__ import annotations

import os
import time
from typing import Optional

from loguru import logger

# Lazy import logfire so the gateway still starts if logfire is not installed
try:
    import logfire as _logfire
    _LOGFIRE_AVAILABLE = True
except ImportError:
    _logfire = None  # type: ignore[assignment]
    _LOGFIRE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kiro plan cost constants (from config — imported lazily to avoid circular)
# ---------------------------------------------------------------------------
_MONTHLY_COST_USD = float(os.getenv("KIRO_PLAN_MONTHLY_COST_USD", "200.0"))
_MONTHLY_INVOCATIONS = int(os.getenv("KIRO_PLAN_MONTHLY_INVOCATIONS", "10000"))
_OVERAGE_COST_USD = float(os.getenv("KIRO_PLAN_OVERAGE_COST_USD", "0.04"))

# Per-invocation cost within the flat plan (amortised)
_COST_PER_INVOCATION_FLAT = _MONTHLY_COST_USD / _MONTHLY_INVOCATIONS  # $0.02


def _invocation_cost_usd(is_overage: bool = False) -> float:
    """Return the $ cost of one gateway invocation."""
    return _OVERAGE_COST_USD if is_overage else _COST_PER_INVOCATION_FLAT


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_configured = False


def setup_logfire() -> bool:
    """Configure logfire for the gateway process.

    Returns True if logfire was successfully configured, False otherwise.
    Failures are non-fatal — the gateway runs without telemetry.
    """
    global _configured
    if _configured:
        return True
    if not _LOGFIRE_AVAILABLE:
        logger.info("logfire not installed — telemetry disabled")
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
            # Suppress noisy internal spans
            console=False,
        )
        _configured = True
        logger.info("Logfire telemetry configured (project: kiro-gateway)")
        return True
    except Exception as e:
        logger.warning(f"Logfire setup failed (non-fatal): {e}")
        return False


def instrument_fastapi(app: object) -> None:
    """Instrument a FastAPI app with logfire auto-instrumentation."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        _logfire.instrument_fastapi(app, capture_headers=False)  # type: ignore[arg-type]
        _logfire.instrument_httpx()
        logger.info("Logfire FastAPI + httpx instrumentation active")
    except Exception as e:
        logger.warning(f"Logfire instrumentation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Per-request telemetry
# ---------------------------------------------------------------------------

def record_request(
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
) -> None:
    """Emit a logfire span for one gateway request.

    All fields map to OpenTelemetry semantic conventions where possible.
    Custom attributes use the `kiro.` namespace.
    """
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        cost_usd = _invocation_cost_usd(is_overage)
        attrs = {
            # OTel LLM semantic conventions (draft)
            "gen_ai.system": "aws_bedrock",
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.operation.name": "chat",
            # Token counts
            "gen_ai.usage.input_tokens": input_tokens or 0,
            "gen_ai.usage.output_tokens": output_tokens or 0,
            # Gateway-specific
            "kiro.gateway.cache": gateway_cache,          # hit / miss / bypass
            "kiro.gateway.re2_applied": re2_applied,
            "kiro.gateway.stream": stream,
            "kiro.gateway.status": status,
            "kiro.gateway.upstream_ms": upstream_ms or 0,
            "kiro.gateway.retry_count": retry_count or 0,
            "kiro.gateway.is_overage": is_overage,
            # Cost tracking
            "kiro.cost.invocation_usd": cost_usd,
            "kiro.cost.plan": "kiro-power",
        }
        if error_reason:
            attrs["kiro.gateway.error_reason"] = error_reason
        if session_id:
            attrs["kiro.session.id"] = session_id[:16]

        span_name = f"gateway.request {'(stream)' if stream else '(sync)'}"
        with _logfire.span(span_name, **attrs):
            pass
    except Exception as e:
        logger.debug(f"Logfire record_request failed (non-fatal): {e}")


def record_cache_event(
    *,
    event: str,  # "hit" | "miss" | "bypass" | "store" | "evict"
    model: str,
    cache_key_prefix: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> None:
    """Emit a logfire event for cache operations."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        _logfire.info(
            f"cache.{event}",
            **{
                "kiro.cache.event": event,
                "kiro.cache.model": model,
                "kiro.cache.key_prefix": cache_key_prefix or "",
                "kiro.cache.size_bytes": size_bytes or 0,
            }
        )
    except Exception:
        pass


def record_re2_decision(
    *,
    applied: bool,
    reason: str,
    model: str,
    message_count: int,
) -> None:
    """Emit a logfire event for re2 injection decisions."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        _logfire.info(
            "re2.decision",
            **{
                "kiro.re2.applied": applied,
                "kiro.re2.reason": reason,
                "kiro.re2.model": model,
                "kiro.re2.message_count": message_count,
            }
        )
    except Exception:
        pass


def record_model_resolution(
    *,
    raw_model: str,
    resolved_model: str,
    resolution_source: str,  # "alias" | "normalize" | "passthrough"
) -> None:
    """Emit a logfire event for model name resolution."""
    if not _configured or not _LOGFIRE_AVAILABLE:
        return
    try:
        if raw_model != resolved_model:
            _logfire.info(
                "model.resolved",
                **{
                    "kiro.model.raw": raw_model,
                    "kiro.model.resolved": resolved_model,
                    "kiro.model.resolution_source": resolution_source,
                }
            )
    except Exception:
        pass
