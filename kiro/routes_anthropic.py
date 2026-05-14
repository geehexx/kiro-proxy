# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
FastAPI routes for Anthropic Messages API.

Contains the /v1/messages endpoint compatible with Anthropic's Messages API.

Reference: https://docs.anthropic.com/en/api/messages
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.auth import AuthType
from kiro.config import GATEWAY_SUBAGENT_STRIP_WEB_SEARCH, PROXY_API_KEY, RE2_ENABLED, RE2_INJECTION, RE2_MIN_CHARS, RE2_MIN_MESSAGES, RE2_SKIP_EXTENDED_THINKING, STREAM_CACHE_ENABLED, WEB_SEARCH_ENABLED
from kiro.converters_anthropic import anthropic_to_kiro
from kiro.http_client import KiroHttpClient
from kiro.mcp_tools import handle_native_web_search
from kiro.models_anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from kiro.streaming_anthropic import (
    collect_anthropic_response,
    stream_kiro_to_anthropic,  # noqa: F401 — patched by tests via patch('kiro.routes_anthropic.stream_kiro_to_anthropic', ...)
    stream_with_first_token_retry_anthropic,
)
from kiro.tokenizer import estimate_request_tokens
from kiro.utils import generate_conversation_id


async def _emit_gateway_baseline(
    request: Request,
    *,
    response_body: dict,
    request_model: str,
    session_id_gw: Optional[str],
    cache_key: Optional[str],
    upstream_ms: Optional[int],
    gateway_cache: str,
    status: int,
    stream: bool = False,
    error_reason: Optional[str] = None,
    retry_count: Optional[int] = None,
    retry_after_applied_ms: Optional[int] = None,
    re2_applied: bool = False,
) -> None:
    """Append one record to baselines-gateway-requests.jsonl and emit logfire span.

    Runs AFTER the response body is fully collected so `usage` is populated.
    Failures are swallowed — telemetry must never break the hot path.
    Plan reference: plans/2026-05-11-token-telemetry.md §Step 1.
    """
    writer = getattr(request.app.state, "baselines_writer", None)
    if writer is None:
        return
    try:
        usage = response_body.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        record = {
            "ts": time.time(),
            "source": "gateway-requests",
            "message_id": response_body.get("id"),
            "session_id_gw": session_id_gw,
            "cache_key": cache_key[:16] if cache_key else None,
            "model": request_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "upstream_ms_total": upstream_ms,
            "gateway_cache": gateway_cache,
            "stream": stream,
            "status": status,
            "error_reason": error_reason,
            "retry_count": retry_count,
            "retry_after_applied_ms": retry_after_applied_ms,
            "re2_applied": re2_applied,
        }
        await writer.write("gateway-requests", record)

        # Emit logfire span (non-blocking, failures swallowed)
        try:
            from kiro.telemetry import record_request
            record_request(
                model=request_model,
                stream=stream,
                gateway_cache=gateway_cache,
                re2_applied=re2_applied,
                upstream_ms=upstream_ms,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error_reason=error_reason,
                retry_count=retry_count,
                session_id=session_id_gw,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"baseline emit failed: {exc}")


# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security scheme ---
# Anthropic uses x-api-key header instead of Authorization: Bearer
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
# Also support Authorization: Bearer for compatibility
auth_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_anthropic_api_key(
    x_api_key: Optional[str] = Security(anthropic_api_key_header), authorization: Optional[str] = Security(auth_header)
) -> bool:
    """
    Verify API key for Anthropic API.

    Supports two authentication methods:
    1. x-api-key header (Anthropic native)
    2. Authorization: Bearer header (for compatibility)

    Args:
        x_api_key: Value from x-api-key header
        authorization: Value from Authorization header

    Returns:
        True if key is valid

    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    # Check x-api-key first (Anthropic native)
    if x_api_key and x_api_key == PROXY_API_KEY:
        return True

    # Fall back to Authorization: Bearer
    if authorization and authorization == f"Bearer {PROXY_API_KEY}":
        return True

    logger.warning("Access attempt with invalid API key (Anthropic endpoint)")
    raise HTTPException(
        status_code=401,
        detail={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key. Use x-api-key header or Authorization: Bearer.",
            },
        },
    )


# --- Router ---
router = APIRouter(tags=["Anthropic API"])


@router.post("/v1/messages", dependencies=[Depends(verify_anthropic_api_key)])
async def messages(
    request: Request,
    request_data: AnthropicMessagesRequest,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
):
    """
    Anthropic Messages API endpoint.

    Compatible with Anthropic's /v1/messages endpoint.
    Accepts requests in Anthropic format and translates them to Kiro API.

    Required headers:
    - x-api-key: Your API key (or Authorization: Bearer)
    - anthropic-version: API version (optional, for compatibility)
    - Content-Type: application/json

    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in Anthropic MessagesRequest format
        anthropic_version: Anthropic API version header (optional)

    Returns:
        StreamingResponse for streaming mode (SSE)
        JSONResponse for non-streaming mode

    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/messages (model={request_data.model}, stream={request_data.stream})")

    if anthropic_version:
        logger.debug(f"Anthropic-Version header: {anthropic_version}")

    # Normalize model name early so all downstream code (baselines, cache keys,
    # logs) uses the canonical form. Aliases (sonnet[1m] → claude-sonnet-4.6)
    # and bracket suffixes ([1m]) are resolved here via the model resolver.
    from kiro.model_resolver import normalize_model_name
    from kiro.config import MODEL_ALIASES
    _raw_model = request_data.model
    _resolved = MODEL_ALIASES.get(_raw_model, _raw_model)
    _normalized = normalize_model_name(_resolved)
    if _normalized != _raw_model:
        logger.debug(f"Model normalized: {_raw_model!r} → {_normalized!r}")
        request_data = request_data.model_copy(update={"model": _normalized})

    # ==============================================================================
    # Beta feature stripping — remove unsupported Anthropic betas before processing.
    # AWS Q/CodeWhisperer does not support Anthropic beta features. When Claude Code
    # sends betas like "advanced-tool-use-2025-11-20" (Tool Search), the gateway
    # must strip them so Claude Code falls back to sending full tool definitions.
    # Without stripping, tool_reference blocks arrive with no input_schema → broken.
    # ==============================================================================
    _unsupported_betas = {"advanced-tool-use-2025-11-20", "computer-use-2024-10-22", "files-api-2025-04-14"}
    _request_betas = getattr(request_data, "betas", None) or []
    if _request_betas:
        _stripped_betas = [b for b in _request_betas if b not in _unsupported_betas]
        if len(_stripped_betas) != len(_request_betas):
            _removed = set(_request_betas) - set(_stripped_betas)
            logger.debug(f"Stripped unsupported betas: {_removed}")
            request_data = request_data.model_copy(update={"betas": _stripped_betas or None})

    # ==============================================================================
    # Response cache setup (non-streaming + streaming)
    # ==============================================================================
    # The cache is a singleton on app.state; if disabled it is None.
    # NOTE: Key computation is deferred until after truncation recovery so the
    # key reflects the actual messages sent upstream, not the pre-mutation form.
    from kiro.cache_integration import (
        compute_cache_key,
        derive_session_id,
        entry_to_response_body,
        store_cache,
        store_stream_cache,
        try_cache_lookup,
        try_stream_cache_lookup,
    )

    response_cache = getattr(request.app.state, "response_cache", None)
    cache_key: Optional[str] = None
    cache_eligible = response_cache is not None and not request_data.stream
    stream_cache_eligible = (
        response_cache is not None
        and request_data.stream
        and STREAM_CACHE_ENABLED
    )
    session_id: Optional[str] = None
    if cache_eligible or stream_cache_eligible:
        client_session_id = request.headers.get("x-kiro-session-id")
        api_key_for_scope = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or None
        )
        session_id = derive_session_id(api_key_for_scope, client_session_id)

    # Per-session concurrency limiter — default OFF via
    # GATEWAY_SESSION_LIMITER_ENABLED. When enabled, caps concurrent upstream
    # calls per caller so one chatty session can't starve other sessions.
    # Acquired AFTER in-flight dedup (so dedup collapses dupes first) and
    # only when session_id is known (i.e., cache_eligible path today).
    # Streaming path is NOT wired: correct slot-release across the
    # StreamingResponse lifetime requires careful handoff that hasn't
    # landed yet. Non-streaming is sufficient coverage for the threat
    # model because streaming sessions are inherently bounded by the
    # upstream's own rate limits.
    from kiro.config import GATEWAY_SESSION_LIMITER_ENABLED

    session_limiter = getattr(request.app.state, "session_limiter", None)
    limiter_active = (
        GATEWAY_SESSION_LIMITER_ENABLED
        and session_limiter is not None
        and session_id is not None
        and not request_data.stream
    )
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)

    # Check for truncation recovery opportunities
    from kiro.models_anthropic import AnthropicMessage
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.truncation_state import get_content_truncation, get_tool_truncation

    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0

    for msg in request_data.messages:
        # Check if this is a user message with tool_result blocks
        if msg.role == "user" and msg.content and isinstance(msg.content, list):
            modified_content_blocks = []
            has_modifications = False

            for block in msg.content:
                # Handle both dict and Pydantic objects (ToolResultContentBlock)
                if isinstance(block, dict):
                    block_type = block.get("type")
                    tool_use_id = block.get("tool_use_id")
                    original_content = block.get("content", "")
                elif hasattr(block, "type"):
                    block_type = block.type
                    tool_use_id = getattr(block, "tool_use_id", None)
                    original_content = getattr(block, "content", "")
                else:
                    modified_content_blocks.append(block)
                    continue

                if block_type == "tool_result" and tool_use_id:
                    truncation_info = get_tool_truncation(tool_use_id)
                    if truncation_info:
                        # Modify tool_result content to include truncation notice
                        synthetic = generate_truncation_tool_result(
                            tool_name=truncation_info.tool_name,
                            tool_use_id=tool_use_id,
                            truncation_info=truncation_info.truncation_info,
                        )
                        # Prepend truncation notice to original content
                        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"

                        # Create modified block (handle both dict and Pydantic)
                        if isinstance(block, dict):
                            modified_block = block.copy()
                            modified_block["content"] = modified_content
                        else:
                            # Pydantic object - use model_copy
                            modified_block = block.model_copy(update={"content": modified_content})

                        modified_content_blocks.append(modified_block)
                        tool_results_modified += 1
                        has_modifications = True
                        logger.debug(f"Modified tool_result for {tool_use_id} to include truncation notice")
                        continue

                modified_content_blocks.append(block)

            # Create NEW AnthropicMessage object if modifications were made (Pydantic immutability)
            if has_modifications:
                modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                modified_messages.append(modified_msg)
                continue  # Skip normal append since we already added modified version

        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content:
            # Extract text content for hash check
            text_content = ""
            if isinstance(msg.content, str):
                text_content = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")

            if text_content:
                truncation_info = get_content_truncation(text_content)
                if truncation_info:
                    # Add this message first
                    modified_messages.append(msg)
                    # Then add synthetic user message about truncation
                    synthetic_user_msg = AnthropicMessage(
                        role="user", content=[{"type": "text", "text": generate_truncation_user_message()}]
                    )
                    modified_messages.append(synthetic_user_msg)
                    content_notices_added += 1
                    logger.debug(
                        f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})"
                    )
                    continue  # Skip normal append since we already added it

        modified_messages.append(msg)

    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(
            f"Truncation recovery: modified {tool_results_modified} tool_result(s), "
            f"added {content_notices_added} content notice(s)"
        )

    # Resolve model name once for use in baselines (normalized form, e.g. claude-sonnet-4.6).
    from kiro.model_resolver import normalize_model_name as _normalize_model
    from kiro.config import MODEL_ALIASES as _MODEL_ALIASES
    _resolved_model = _normalize_model(_MODEL_ALIASES.get(request_data.model, request_data.model))

    # Compute re2 flag early so cache-hit paths can record it correctly.
    # The actual injection happens below, after cache lookup.
    # re2 is only useful for genuine reasoning turns — skip for haiku (fast tool calls),
    # tool_result-only last messages, and sub-agent requests.
    def _re2_eligible() -> bool:
        if 'haiku' in request_data.model.lower():
            return False
        if request.headers.get("x-claude-subagent", "").lower() in ("true", "1", "yes"):
            return False
        # Skip if extended thinking is active — re2 is neutral-to-negative when reasoning is on
        if RE2_SKIP_EXTENDED_THINKING and request_data.thinking is not None:
            return False
        # Skip if last user message has no text block (only tool_result blocks)
        last_user_text = ""
        for _m in reversed(request_data.messages):
            if _m.role == "user":
                _c = _m.content
                if isinstance(_c, list):
                    has_text = any(
                        (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "text"
                        for b in _c
                    )
                    if not has_text:
                        return False
                    # Extract text for length check
                    for b in _c:
                        if isinstance(b, dict) and b.get("type") == "text":
                            last_user_text = b.get("text", "")
                            break
                elif isinstance(_c, str):
                    last_user_text = _c
                break
        # Skip short reactive turns — unlikely to benefit from re-reading
        if len(last_user_text.strip()) < RE2_MIN_CHARS:
            return False
        return True

    _re2_active = (RE2_ENABLED or request.headers.get("x-kiro-re2", "").lower() == "true") and _re2_eligible()

    # ==============================================================================
    # Response cache lookup — runs AFTER truncation recovery but BEFORE re2
    # injection. Cache key must reflect the canonical conversation, not the
    # re2-mutated form, so that re2-on and re2-off requests for the same
    # conversation share the same cache bucket.
    # ==============================================================================
    if (cache_eligible or stream_cache_eligible) and session_id is not None:
        messages_for_cache = [msg.model_dump() for msg in request_data.messages]
        tools_for_cache = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
        if isinstance(request_data.system, list):
            system_for_cache = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
        else:
            system_for_cache = request_data.system

        cache_key = compute_cache_key(
            session_id=session_id,
            system=system_for_cache,
            messages=messages_for_cache,
            model=request_data.model,
            max_tokens=request_data.max_tokens,
            tools=tools_for_cache,
            thinking=request_data.thinking,
        )

        if cache_eligible:
            hit = try_cache_lookup(response_cache, cache_key)
            if hit is not None:
                hit_body = entry_to_response_body(hit)
                logger.info(
                    f"HTTP 200 - POST /v1/messages (non-streaming, cache hit) "
                    f"key={cache_key[:8]} "
                    f"msg_id={hit_body.get('id', 'unknown')} "
                    f"response_model={hit_body.get('model', 'unknown')}"
                )
                if debug_logger:
                    debug_logger.discard_buffers()
                await _emit_gateway_baseline(
                    request,
                    response_body=hit_body,
                    request_model=_resolved_model,
                    session_id_gw=session_id,
                    cache_key=cache_key,
                    upstream_ms=None,
                    gateway_cache="hit",
                    status=200,
                    re2_applied=_re2_active,
                )
                return JSONResponse(
                    content=hit_body,
                    headers={"x-kiro-cache": "hit"},
                )

        if stream_cache_eligible:
            stream_hit = try_stream_cache_lookup(response_cache, cache_key)
            if stream_hit is not None:
                logger.info(
                    f"HTTP 200 - POST /v1/messages (streaming, cache hit) "
                    f"key={cache_key[:8]}"
                )
                if debug_logger:
                    debug_logger.discard_buffers()
                await _emit_gateway_baseline(
                    request,
                    response_body={},
                    request_model=_resolved_model,
                    session_id_gw=session_id,
                    cache_key=cache_key,
                    upstream_ms=None,
                    gateway_cache="hit",
                    status=200,
                    stream=True,
                    re2_applied=_re2_active,
                )
                cached_bytes = stream_hit

                async def replay_stream():
                    yield cached_bytes

                return StreamingResponse(
                    replay_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "x-kiro-cache": "hit",
                    },
                )

    # ==============================================================================
    # Sub-agent web_search Strip (SDK 422 re-serialisation bug prevention)
    # ==============================================================================

    # When x-claude-subagent header is present and the feature flag is on,
    # remove any web_search tool entries from the request before forwarding.
    # This prevents the SDK 422 re-serialisation bug that fires on turn 2 of
    # sub-agent sessions when server_tool_use / web_search_tool_result pairs
    # are present (sdk_422_server_tool_bug, 2026-05-10).
    #
    # Both tool shapes are stripped:
    #   - Path A (server-side): type starts with "web_search" (e.g. "web_search_20250305")
    #   - Path B (MCP emulation): name == "web_search" with no type
    is_subagent_request = (
        GATEWAY_SUBAGENT_STRIP_WEB_SEARCH
        and request.headers.get("x-claude-subagent", "").lower() in ("true", "1", "yes")
    )
    if is_subagent_request and request_data.tools:
        before = len(request_data.tools)
        request_data.tools = [
            t for t in request_data.tools
            if not (
                (getattr(t, "type", None) or "").startswith("web_search")
                or getattr(t, "name", "") == "web_search"
            )
        ]
        stripped = before - len(request_data.tools)
        if stripped:
            logger.info(
                f"Stripped {stripped} web_search tool(s) from sub-agent request "
                f"(x-claude-subagent header present, sdk_422_server_tool_bug prevention)"
            )

    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================

    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    # Skip injection for sub-agent requests — they cannot use web_search anyway.
    if WEB_SEARCH_ENABLED and not is_subagent_request:
        if request_data.tools is None:
            request_data.tools = []

        # Check if web_search already exists (by name)
        has_ws = any(getattr(tool, "name", "") == "web_search" for tool in request_data.tools)

        if not has_ws:
            from kiro.models_anthropic import AnthropicTool

            web_search_tool = AnthropicTool(
                name="web_search",
                description=(
                    "Search the web for current information. "
                    "Use when you need up-to-date data from the internet."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")

    # ==============================================================================
    # WebSearch Support - Path A: Native Anthropic (Early Return)
    # ==============================================================================

    # Check for native Anthropic server-side tool (Path A)
    # This works ALWAYS, regardless of WEB_SEARCH_ENABLED setting
    if request_data.tools:
        for tool in request_data.tools:
            tool_type = getattr(tool, "type", None)
            if tool_type and tool_type.startswith("web_search"):
                # Path A: Early return, direct MCP call
                # Get auth_manager from first available account (no failover needed for early return)
                account = request.app.state.account_manager.get_first_account()
                if not account.auth_manager:
                    logger.error("No initialized accounts available for native web_search")
                    return JSONResponse(
                        status_code=503,
                        content={
                            "type": "error",
                            "error": {"type": "api_error", "message": "No initialized accounts available"},
                        },
                    )
                auth_manager = account.auth_manager

                logger.info("Detected native Anthropic web_search (Path A), routing to MCP API")
                return await handle_native_web_search(request, request_data, auth_manager, api_format="anthropic")

    # ==============================================================================
    # Re2 (ReRead) injection — OptiLLM technique for improved reasoning accuracy.
    # Appends "Read the question again carefully" to the last user message.
    # Runs AFTER cache lookup so the cache key reflects the canonical conversation
    # (re2-on and re2-off requests for the same conversation share one cache bucket).
    # _re2_active was computed above (before cache lookup) so cache-hit paths can
    # record it. Opt-in via X-Kiro-Re2: true header or RE2_ENABLED=true env var.
    # Zero cost — no extra API calls. Based on: github.com/codelion/optillm
    #
    # Eligibility rules (re2 is skipped when):
    # - Fewer than RE2_MIN_MESSAGES messages (default 2) — single-message polling/tool calls
    # - Last user message contains ONLY tool_result blocks (no text to re-read)
    # ==============================================================================
    if _re2_active and len(request_data.messages) >= RE2_MIN_MESSAGES:
        # Check if last user message has any text content (skip pure tool_result messages)
        _last_user_has_text = False
        for _i in range(len(request_data.messages) - 1, -1, -1):
            _msg = request_data.messages[_i]
            if _msg.role == "user":
                if isinstance(_msg.content, str):
                    _last_user_has_text = bool(_msg.content.strip())
                elif isinstance(_msg.content, list):
                    _last_user_has_text = any(
                        (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "text"
                        for b in _msg.content
                    )
                break

        if _last_user_has_text:
            for _i in range(len(request_data.messages) - 1, -1, -1):
                _msg = request_data.messages[_i]
                if _msg.role == "user":
                    if isinstance(_msg.content, str):
                        request_data.messages[_i] = _msg.model_copy(
                            update={"content": _msg.content + RE2_INJECTION}
                        )
                    elif isinstance(_msg.content, list):
                        _new_content = list(_msg.content)
                        for _j in range(len(_new_content) - 1, -1, -1):
                            _block = _new_content[_j]
                            _btype = _block.get("type") if isinstance(_block, dict) else getattr(_block, "type", None)
                            if _btype == "text":
                                _btext = _block.get("text") if isinstance(_block, dict) else getattr(_block, "text", "")
                                if isinstance(_block, dict):
                                    _new_content[_j] = {**_block, "text": _btext + RE2_INJECTION}
                                else:
                                    _new_content[_j] = _block.model_copy(update={"text": _btext + RE2_INJECTION})
                                break
                        request_data.messages[_i] = _msg.model_copy(update={"content": _new_content})
                    logger.debug("Re2 injection applied to last user message")
                    break
        else:
            _re2_active = False
            logger.debug("Re2 skipped: last user message has no text content (tool_result only)")

    # ==============================================================================
    # Account System: Account System Failover or Legacy Mode
    # ==============================================================================

    if request.app.state.account_system:
        # ==============================================================================
        # ACCOUNT SYSTEM ENABLED: Failover Loop
        # ==============================================================================
        from kiro.account_errors import ErrorType, classify_error

        account_manager = request.app.state.account_manager
        all_accounts = list(account_manager._accounts.keys())
        MAX_ATTEMPTS = len(all_accounts) * 2  # Full circle with margin

        last_error_message = None
        last_error_status = None
        tried_accounts = set()  # Track tried accounts in current failover loop

        for attempt in range(MAX_ATTEMPTS):
            # Get next available account (excluding already tried)
            account = await account_manager.get_next_account(request_data.model, exclude_accounts=tried_accounts)

            if account is None:
                # All accounts unavailable
                if len(all_accounts) == 1:
                    # Single account - return original error with original status code
                    return JSONResponse(
                        status_code=last_error_status or 503,
                        content={
                            "type": "error",
                            "error": {"type": "api_error", "message": last_error_message or "Account unavailable"},
                        },
                    )
                else:
                    # Multiple accounts - generic error with context
                    detail = "No available accounts for this model."
                    if last_error_message:
                        detail += f" Last error: {last_error_message}"
                    return JSONResponse(
                        status_code=503, content={"type": "error", "error": {"type": "api_error", "message": detail}}
                    )

            # Mark account as tried in current failover loop
            tried_accounts.add(account.id)

            # Use objects from account
            auth_manager = account.auth_manager
            model_cache = account.model_cache

            # Generate conversation ID
            conversation_id = generate_conversation_id()

            # Build payload for Kiro
            profile_arn_for_payload = ""
            if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                profile_arn_for_payload = auth_manager.profile_arn

            try:
                kiro_payload = anthropic_to_kiro(request_data, conversation_id, profile_arn_for_payload)
            except ValueError as e:
                logger.error(f"Conversion error: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
                )

            # Log Kiro payload
            try:
                kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode("utf-8")
                if debug_logger:
                    debug_logger.log_kiro_request_body(kiro_request_body)
            except Exception as e:
                logger.warning(f"Failed to log Kiro request: {e}")

            # Create HTTP client
            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.debug(f"Kiro API URL: {url} (account: {account.id})")

            if request_data.stream:
                http_client = KiroHttpClient(auth_manager, shared_client=None)
            else:
                shared_client = request.app.state.http_client
                http_client = KiroHttpClient(auth_manager, shared_client=shared_client)

            # Prepare data for token counting
            messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
            tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
            if isinstance(request_data.system, list):
                system_for_tokenizer = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
            else:
                system_for_tokenizer = request_data.system

            try:
                # Make request to Kiro API
                _upstream_start = time.monotonic()
                response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)
                _upstream_ms_total = int((time.monotonic() - _upstream_start) * 1000)

                if response.status_code == 200:
                    if request_data.stream:
                        # Streaming mode
                        async def stream_wrapper():
                            streaming_error = None
                            client_disconnected = False
                            # Capture usage from message_delta event for baseline
                            _stream_usage: dict = {}
                            # Buffer chunks for streaming cache (only when stream_cache_eligible)
                            _stream_chunks: list[bytes] = []
                            try:

                                async def make_retry_request():
                                    return await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

                                async for chunk in stream_with_first_token_retry_anthropic(
                                    make_request=make_retry_request,
                                    model=request_data.model,
                                    model_cache=model_cache,
                                    auth_manager=auth_manager,
                                    initial_response=response,
                                    request_messages=messages_for_tokenizer,
                                    request_tools=tools_for_tokenizer,
                                    request_system=system_for_tokenizer,
                                ):
                                    # Extract usage from message_delta SSE event.
                                    # SSE format: "event: message_delta\ndata: {...}\n\n"
                                    # chunk starts with "event:", not "data:", so search for "data:" anywhere.
                                    if not _stream_usage and "data:" in chunk:
                                        try:
                                            data_str = chunk.split("data:", 1)[1].strip()
                                            if data_str and data_str != "[DONE]":
                                                evt = json.loads(data_str)
                                                if evt.get("type") == "message_delta":
                                                    usage = evt.get("usage", {})
                                                    if usage.get("input_tokens") or usage.get("output_tokens"):
                                                        _stream_usage = usage
                                        except Exception:
                                            pass
                                    # Buffer for streaming cache
                                    if stream_cache_eligible and cache_key:
                                        _stream_chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
                                    yield chunk
                            except GeneratorExit:
                                client_disconnected = True
                                logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                            except Exception as e:
                                streaming_error = e
                                try:
                                    _err_payload = json.dumps(
                                        {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                                    )
                                    error_event = f"event: error\ndata: {_err_payload}\n\n"
                                    yield error_event
                                except Exception:
                                    pass
                            finally:
                                await http_client.close()
                                if streaming_error:
                                    error_type = type(streaming_error).__name__
                                    error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                                    logger.error(
                                        f"HTTP 500 - POST /v1/messages (streaming) - [{error_type}] {error_msg[:100]}"
                                    )
                                elif client_disconnected:
                                    logger.info("HTTP 200 - POST /v1/messages (streaming) - client disconnected")
                                else:
                                    # Body finished cleanly — safe to record success
                                    await account_manager.report_success(account.id, request_data.model)
                                    logger.info("HTTP 200 - POST /v1/messages (streaming) - completed")
                                    # Store in streaming cache on clean completion
                                    if stream_cache_eligible and cache_key and _stream_chunks:
                                        stored = store_stream_cache(response_cache, cache_key, _stream_chunks)
                                        if stored:
                                            logger.debug(
                                                f"stream-cache stored key={cache_key[:8]} "
                                                f"entries={response_cache.stats()['entries']}"
                                            )

                                # Emit streaming baseline with token data extracted from message_delta
                                await _emit_gateway_baseline(
                                    request,
                                    response_body={"usage": _stream_usage} if _stream_usage else {},
                                    request_model=_resolved_model,
                                    session_id_gw=session_id,
                                    cache_key=cache_key,
                                    upstream_ms=_upstream_ms_total,
                                    gateway_cache="bypass" if not stream_cache_eligible else "miss",
                                    status=200 if not streaming_error else 500,
                                    stream=True,
                                    re2_applied=_re2_active,
                                )

                                if debug_logger:
                                    if streaming_error:
                                        debug_logger.flush_on_error(500, str(streaming_error))
                                    else:
                                        debug_logger.discard_buffers()

                        return StreamingResponse(
                            stream_wrapper(),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )

                    else:
                        # Non-streaming mode
                        if limiter_active:
                            async with session_limiter.acquire(session_id):
                                anthropic_response = await collect_anthropic_response(
                                    response,
                                    request_data.model,
                                    model_cache,
                                    auth_manager,
                                    request_messages=messages_for_tokenizer,
                                    request_tools=tools_for_tokenizer,
                                    request_system=system_for_tokenizer,
                                )
                        else:
                            anthropic_response = await collect_anthropic_response(
                                response,
                                request_data.model,
                                model_cache,
                                auth_manager,
                                request_messages=messages_for_tokenizer,
                                request_tools=tools_for_tokenizer,
                                request_system=system_for_tokenizer,
                            )

                        await http_client.close()
                        # Body collected cleanly — safe to record success
                        await account_manager.report_success(account.id, request_data.model)
                        logger.info(
                            "HTTP 200 - POST /v1/messages (non-streaming) - completed "
                            f"msg_id={anthropic_response.get('id', 'unknown')} "
                            f"response_model={anthropic_response.get('model', 'unknown')}"
                        )

                        if debug_logger:
                            debug_logger.discard_buffers()

                        # Persist into the response cache if enabled.
                        if cache_eligible and cache_key is not None and response_cache is not None:
                            stored = store_cache(response_cache, cache_key, anthropic_response)
                            if stored:
                                logger.debug(
                                    f"response-cache stored key={cache_key[:8]} "
                                    f"entries={response_cache.stats()['entries']}"
                                )

                        await _emit_gateway_baseline(
                            request,
                            response_body=anthropic_response,
                            request_model=_resolved_model,
                            session_id_gw=session_id,
                            cache_key=cache_key,
                            upstream_ms=_upstream_ms_total,
                            gateway_cache="miss" if cache_eligible else "bypass",
                            status=200,
                            re2_applied=_re2_active,
                        )

                        return JSONResponse(
                            content=anthropic_response,
                            headers={"x-kiro-cache": "miss"} if cache_eligible else None,
                        )

                else:
                    # ERROR - classify and decide
                    try:
                        error_content = await response.aread()
                    except Exception:
                        error_content = b"Unknown error"

                    await http_client.close()
                    error_text = error_content.decode("utf-8", errors="replace")

                    # Extract error reason and save for final return
                    error_reason = None
                    try:
                        error_json = json.loads(error_text)
                        from kiro.kiro_errors import enhance_kiro_error

                        error_info = enhance_kiro_error(error_json)
                        error_reason = error_info.reason
                        last_error_message = error_info.user_message
                        last_error_status = response.status_code
                        logger.debug(
                            f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})"
                        )
                    except (json.JSONDecodeError, KeyError):
                        last_error_message = error_text
                        last_error_status = response.status_code

                    # Classify error
                    error_type = classify_error(response.status_code, error_reason)

                    if error_type == ErrorType.FATAL:
                        # FATAL - return to client immediately
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type, response.status_code, error_reason
                        )

                        logger.warning(f"HTTP {response.status_code} - POST /v1/messages - {last_error_message[:100]}")

                        if debug_logger:
                            debug_logger.flush_on_error(response.status_code, last_error_message)

                        return JSONResponse(
                            status_code=response.status_code,
                            content={"type": "error", "error": {"type": "api_error", "message": last_error_message}},
                        )

                    else:  # ErrorType.RECOVERABLE
                        # RECOVERABLE - try next account
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type, response.status_code, error_reason
                        )

                        # Single account - no point in failover, break immediately
                        if len(all_accounts) == 1:
                            break

                        continue  # Next iteration

            except HTTPException as e:
                await http_client.close()

                # Network errors (502/504 from request_with_retry) = RECOVERABLE
                # These are thrown ONLY for network-level issues (timeouts, connection errors)
                # NOT for HTTP-level errors (which are returned as response objects)
                if e.status_code in (502, 504):
                    # Network error → try next account
                    await account_manager.report_failure(
                        account.id, request_data.model, ErrorType.RECOVERABLE, e.status_code, None
                    )

                    last_error_message = str(e.detail)
                    last_error_status = e.status_code

                    # Single account - no point in failover, break immediately
                    if len(all_accounts) == 1:
                        break

                    logger.warning(f"Network error on account {account.id}, trying next account")
                    continue  # Try next account

                # All other HTTPException (400, 500, etc.) = application errors
                # These come from build_kiro_payload() or other places → re-raise immediately
                logger.error(f"HTTP {e.status_code} - POST /v1/messages - {e.detail}")
                if debug_logger:
                    debug_logger.flush_on_error(e.status_code, str(e.detail))
                raise
            except Exception as e:
                await http_client.close()
                logger.error(f"Internal error: {e}", exc_info=True)
                logger.error(f"HTTP 500 - POST /v1/messages - {str(e)[:100]}")
                if debug_logger:
                    debug_logger.flush_on_error(500, str(e))

                return JSONResponse(
                    status_code=500,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Internal Server Error: {str(e)}"},
                    },
                )

        # All attempts exhausted
        if len(all_accounts) == 1:
            # Single account - return its original error
            # last_error_status and last_error_message are guaranteed to be set
            return JSONResponse(
                status_code=last_error_status,
                content={"type": "error", "error": {"type": "api_error", "message": last_error_message}},
            )
        else:
            # Multiple accounts - generic error with context
            detail = "All accounts failed after full circle."
            if last_error_message:
                detail += f" Last error: {last_error_message}"
            return JSONResponse(
                status_code=503, content={"type": "error", "error": {"type": "api_error", "message": detail}}
            )

    else:
        # ==============================================================================
        # LEGACY MODE: Single Account (no failover)
        # ==============================================================================
        account = request.app.state.account_manager.get_first_account()
        if not account.auth_manager:
            logger.error("No initialized accounts available (legacy mode)")
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": "No initialized accounts available"},
                },
            )
        auth_manager = account.auth_manager
        model_cache = account.model_cache
        _ = account.model_resolver

    # ==============================================================================
    # Normal Flow (Path B will be intercepted in streaming, or no web_search)
    # ==============================================================================

    # Generate conversation ID for Kiro API (random UUID, not used for tracking)
    conversation_id = generate_conversation_id()

    # Build payload for Kiro
    # profileArn is only needed for Kiro Desktop auth
    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn

    try:
        kiro_payload = anthropic_to_kiro(request_data, conversation_id, profile_arn_for_payload)
    except ValueError as e:
        logger.error(f"Conversion error: {e}")
        return JSONResponse(
            status_code=400, content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}
        )

    # Log Kiro payload
    try:
        kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")

    # Create HTTP client with retry logic
    # For streaming: use per-request client to avoid CLOSE_WAIT leak on VPN disconnect (issue #54)
    # For non-streaming: use shared client for connection pooling
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")

    if request_data.stream:
        # Streaming mode: per-request client prevents orphaned connections
        # when network interface changes (VPN disconnect/reconnect)
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        # Non-streaming mode: shared client for efficient connection reuse
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)

    # Prepare data for token counting
    # Convert Pydantic models to dicts for tokenizer
    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    # Serialize system prompt (may be a list of Pydantic objects)
    if isinstance(request_data.system, list):
        system_for_tokenizer = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
    else:
        system_for_tokenizer = request_data.system

    try:
        # Make request to Kiro API (for both streaming and non-streaming modes)
        # Important: we wait for Kiro response BEFORE returning StreamingResponse,
        # so that we can return proper HTTP error codes if Kiro fails
        _upstream_start = time.monotonic()
        response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)
        _upstream_ms_total = int((time.monotonic() - _upstream_start) * 1000)

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            # Try to parse JSON response from Kiro to extract error message
            error_message = error_text
            _error_reason: Optional[str] = None
            _retry_after_hint_ms: Optional[int] = None
            try:
                error_json = json.loads(error_text)
                # Enhance Kiro API errors with user-friendly messages
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                _error_reason = error_info.reason if error_info.reason != "UNKNOWN" else None
                # Log original error for debugging
                logger.debug(f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})")
            except (json.JSONDecodeError, KeyError):
                pass

            # Log access log for error (before flush, so it gets into app_logs)
            logger.warning(f"HTTP {response.status_code} - POST /v1/messages - {error_message[:100]}")

            # Flush debug logs on error
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            # Build response headers — add Retry-After for capacity-exhaustion 429s
            # so upstream clients (and the http_client retry loop) can honour it.
            response_headers: dict = {}
            if response.status_code == 429 and _error_reason == "INSUFFICIENT_MODEL_CAPACITY":
                try:
                    from kiro.kiro_errors import enhance_kiro_error as _eke
                    _err_info = _eke(json.loads(error_text))
                    if _err_info.retry_after_hint is not None:
                        _retry_after_hint_ms = int(_err_info.retry_after_hint * 1000)
                        response_headers["Retry-After"] = str(int(_err_info.retry_after_hint))
                except Exception:
                    pass

            # Emit baseline record for error responses (§2 telemetry fields).
            await _emit_gateway_baseline(
                request,
                response_body={},
                request_model=_resolved_model,
                session_id_gw=session_id,
                cache_key=cache_key,
                upstream_ms=_upstream_ms_total,
                gateway_cache="bypass",
                status=response.status_code,
                error_reason=_error_reason,
                retry_after_applied_ms=_retry_after_hint_ms,
                re2_applied=_re2_active,
            )

            # Return error in Anthropic format
            return JSONResponse(
                status_code=response.status_code,
                content={"type": "error", "error": {"type": "api_error", "message": error_message}},
                headers=response_headers if response_headers else None,
            )

        if request_data.stream:
            # Streaming mode with first token retry
            # Resolve global Opus semaphore (§3 — feature-flagged OFF by default).
            # When GATEWAY_GLOBAL_OPUS_CONCURRENCY > 0 and the model is claude-opus-*,
            # the semaphore is acquired for the lifetime of the stream so that at most
            # N Opus streams run concurrently.  Default: semaphore is None → no-op.
            _opus_semaphore = None
            if request_data.model.startswith("claude-opus"):
                _opus_semaphore = getattr(request.app.state, "global_opus_semaphore", None)

            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                _sem_acquired = False
                # Buffer for streaming cache
                _stream_chunks: list[bytes] = []
                try:
                    # Acquire Opus concurrency slot if the cap is active.
                    if _opus_semaphore is not None:
                        await _opus_semaphore.acquire()
                        _sem_acquired = True

                    # Create retry request function for retries
                    async def make_retry_request():
                        return await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

                    # Use retry wrapper with initial response
                    _stream_usage: dict = {}
                    async for chunk in stream_with_first_token_retry_anthropic(
                        make_request=make_retry_request,
                        model=request_data.model,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                        initial_response=response,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                        request_system=system_for_tokenizer,
                    ):
                        # Extract usage from message_delta SSE event.
                        # SSE format: "event: message_delta\ndata: {...}\n\n"
                        # chunk starts with "event:", not "data:", so search for "data:" anywhere.
                        if not _stream_usage and "data:" in chunk:
                            try:
                                data_str = chunk.split("data:", 1)[1].strip()
                                if data_str and data_str != "[DONE]":
                                    evt = json.loads(data_str)
                                    if evt.get("type") == "message_delta":
                                        usage = evt.get("usage", {})
                                        if usage.get("input_tokens") or usage.get("output_tokens"):
                                            _stream_usage = usage
                            except Exception:
                                pass
                        # Buffer for streaming cache
                        if stream_cache_eligible and cache_key:
                            _stream_chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
                        yield chunk
                except GeneratorExit:
                    client_disconnected = True
                    logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                except Exception as e:
                    streaming_error = e
                    # Send error event to client, then gracefully end the stream
                    try:
                        _err_payload = json.dumps(
                            {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                        )
                        error_event = f"event: error\ndata: {_err_payload}\n\n"
                        yield error_event
                    except Exception:
                        pass
                finally:
                    # Release Opus concurrency slot before closing the HTTP client.
                    if _sem_acquired and _opus_semaphore is not None:
                        _opus_semaphore.release()
                    await http_client.close()
                    if streaming_error:
                        error_type = type(streaming_error).__name__
                        error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                        logger.error(f"HTTP 500 - POST /v1/messages (streaming) - [{error_type}] {error_msg[:100]}")
                    elif client_disconnected:
                        logger.info("HTTP 200 - POST /v1/messages (streaming) - client disconnected")
                    else:
                        logger.info("HTTP 200 - POST /v1/messages (streaming) - completed")
                        # Store in streaming cache on clean completion
                        if stream_cache_eligible and cache_key and _stream_chunks:
                            stored = store_stream_cache(response_cache, cache_key, _stream_chunks)
                            if stored:
                                logger.debug(
                                    f"stream-cache stored key={cache_key[:8]} "
                                    f"entries={response_cache.stats()['entries']}"
                                )

                    # Emit streaming baseline with token data extracted from message_delta
                    await _emit_gateway_baseline(
                        request,
                        response_body={"usage": _stream_usage} if _stream_usage else {},
                        request_model=_resolved_model,
                        session_id_gw=session_id,
                        cache_key=cache_key,
                        upstream_ms=_upstream_ms_total,
                        gateway_cache="bypass" if not stream_cache_eligible else "miss",
                        status=200 if not streaming_error else 500,
                        stream=True,
                        re2_applied=_re2_active,
                    )

                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(500, str(streaming_error))
                        else:
                            debug_logger.discard_buffers()

            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        else:
            # Non-streaming mode - collect entire response
            if limiter_active:
                async with session_limiter.acquire(session_id):
                    anthropic_response = await collect_anthropic_response(
                        response,
                        request_data.model,
                        model_cache,
                        auth_manager,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                        request_system=system_for_tokenizer,
                    )
            else:
                anthropic_response = await collect_anthropic_response(
                    response,
                    request_data.model,
                    model_cache,
                    auth_manager,
                    request_messages=messages_for_tokenizer,
                    request_tools=tools_for_tokenizer,
                    request_system=system_for_tokenizer,
                )

            await http_client.close()

            logger.info(
                "HTTP 200 - POST /v1/messages (non-streaming) - completed "
                f"msg_id={anthropic_response.get('id', 'unknown')} "
                f"response_model={anthropic_response.get('model', 'unknown')}"
            )

            if debug_logger:
                debug_logger.discard_buffers()

            # Persist into the response cache if enabled. We store only
            # on clean 200 responses — errors and partial replies must
            # not be cached.
            if cache_eligible and cache_key is not None and response_cache is not None:
                stored = store_cache(response_cache, cache_key, anthropic_response)
                if stored:
                    logger.debug(
                        f"response-cache stored key={cache_key[:8]} entries={response_cache.stats()['entries']}"
                    )

            await _emit_gateway_baseline(
                request,
                response_body=anthropic_response,
                request_model=_resolved_model,
                session_id_gw=session_id,
                cache_key=cache_key,
                upstream_ms=_upstream_ms_total,
                gateway_cache="miss" if cache_eligible else "bypass",
                status=200,
                re2_applied=_re2_active,
            )

            return JSONResponse(
                content=anthropic_response,
                headers={"x-kiro-cache": "miss"} if cache_eligible else None,
            )

    except HTTPException as e:
        await http_client.close()

        # Network errors (502/504 from request_with_retry) = RECOVERABLE
        # In legacy mode, we still log them but re-raise (no failover available)
        if e.status_code in (502, 504):
            logger.warning("Network error (legacy mode, no failover available)")

        logger.error(f"HTTP {e.status_code} - POST /v1/messages - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST /v1/messages - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))

        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": f"Internal Server Error: {str(e)}"}},
        )


@router.post("/v1/messages/count_tokens", dependencies=[Depends(verify_anthropic_api_key)])
async def count_tokens_endpoint(
    request: Request,
    request_data: AnthropicCountTokensRequest,
):
    """
    Anthropic Count Tokens API endpoint.

    Returns estimated token count for the given request payload.
    Used by Claude Code to decide when to trigger conversation compaction.

    Uses the same fallback estimation as Anthropic streaming (message_start event),
    since Kiro API only provides accurate token counts after request completion.
    This endpoint is called BEFORE the actual request, so we cannot use Kiro's
    contextUsagePercentage (which is only available after generation completes).

    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in Anthropic MessagesRequest format

    Returns:
        JSONResponse with {"input_tokens": int}

    Raises:
        HTTPException: 401 if authentication fails (handled by dependency)
    """
    logger.info(
        f"Request to /v1/messages/count_tokens (model={request_data.model}, messages={len(request_data.messages)})"
    )

    # Prepare data for tokenizer (same format as streaming message_start)
    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None

    # Handle system prompt (can be string or list of content blocks)
    if isinstance(request_data.system, list):
        system_for_tokenizer = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
    else:
        system_for_tokenizer = request_data.system

    # Use the SAME estimation logic as Anthropic streaming message_start
    request_token_stats = estimate_request_tokens(
        messages=messages_for_tokenizer,
        tools=tools_for_tokenizer,
        system_prompt=system_for_tokenizer,
        apply_claude_correction=True,  # CRITICAL: Enable correction for Claude models
    )

    input_tokens = request_token_stats["total_tokens"]

    logger.info(f"Token count estimate: {input_tokens} tokens")

    return JSONResponse(content={"input_tokens": input_tokens})
