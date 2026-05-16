
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
HTTP client for Kiro API with retry logic support.

Handles:
- 403: automatic token refresh and retry
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
import random
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.auth import KiroAuthManager
from kiro.config import (
    ADAPTIVE_RETRY_ENABLED,
    BASE_RETRY_DELAY,
    CAPACITY_BACKOFF_BASE,
    CAPACITY_MAX_RETRIES,
    FIRST_TOKEN_MAX_RETRIES,
    MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
)
from kiro.network_errors import NetworkErrorInfo, classify_network_error, get_short_error_message
from kiro.retry_bucket import get_singleton as _retry_bucket
from kiro.retry_classifier import RetryKind
from kiro.retry_classifier import classify as classify_retry
from kiro.utils import get_kiro_headers


class KiroHttpClient:
    """
    HTTP client for Kiro API with retry logic support.

    Automatically handles errors and retries requests:
    - 403: refreshes token and retries
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - Timeouts: waits with exponential backoff

    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)

    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.

    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)

    Example:
        >>> # Per-request client (legacy mode)
        >>> client = KiroHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)

        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = KiroHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """

    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.

        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.

        If a shared client was provided at initialization, it is returned as-is.
        Otherwise, creates a new client with appropriate timeout configuration.

        httpx timeouts:
        - connect: TCP handshake (DNS + TCP SYN/ACK)
        - read: waiting for data from server between chunks
        - write: sending data to server
        - pool: waiting for free connection from pool

        IMPORTANT: FIRST_TOKEN_TIMEOUT is NOT used here!
        It is applied in streaming_openai.py via asyncio.wait_for() to control
        the wait time for the first token from the model (retry business logic).

        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)

        Returns:
            Active HTTP client
        """
        # If using shared client, return it directly
        # Shared client should be pre-configured with appropriate timeouts
        if self._shared_client is not None:
            return self._shared_client

        # Create new client if needed (per-request mode)
        if self.client is None or self.client.is_closed:
            if stream:
                # For streaming:
                # - connect: 30 sec (TCP connection, usually < 1 sec)
                # - read: STREAMING_READ_TIMEOUT (300 sec) - model may "think" between chunks
                # - write/pool: standard values
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                )
                logger.debug(f"Creating streaming HTTP client (read_timeout={STREAMING_READ_TIMEOUT}s)")
            else:
                # For regular requests: single timeout of 300 sec
                timeout_config = httpx.Timeout(timeout=300.0)
                logger.debug("Creating non-streaming HTTP client (timeout=300s)")

            self.client = httpx.AsyncClient(timeout=timeout_config, follow_redirects=True)
        return self.client

    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.

        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.

        Uses graceful exception handling to prevent errors during cleanup
        from masking the original exception in finally blocks.
        """
        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return

        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup code
                # Propagating here could mask the original exception
                logger.warning(f"Error closing HTTP client: {e}")

    async def request_with_retry(  # noqa: C901, PLR0912, PLR0915
        self,
        method: str,
        url: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.

        Automatically handles various error types:
        - 403: refreshes token via auth_manager.force_refresh() and retries
        - 429: waits with exponential backoff (1s, 2s, 4s)
        - 5xx: waits with exponential backoff
        - Timeouts: waits with exponential backoff

        For streaming, STREAMING_READ_TIMEOUT is used for waiting between chunks.
        First token timeout is controlled separately in streaming_openai.py via asyncio.wait_for().

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Optional JSON body (for POST/PUT/PATCH)
            params: Optional query parameters (for GET)
            stream: Use streaming (default False)

        Returns:
            httpx.Response with successful response

        Raises:
            HTTPException: On failure after all attempts (502/504)
        """
        # Determine the number of retry attempts
        # FIRST_TOKEN_TIMEOUT is used in streaming_openai.py, not here
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES

        client = await self._get_client(stream=stream)
        last_error_info: Optional[NetworkErrorInfo] = None
        last_response: Optional[httpx.Response] = None  # Retained to return the last 429/5xx after exhaustion

        for attempt in range(max_retries):
            try:
                # Get current token
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(self.auth_manager, token)

                # Build request kwargs based on parameters
                request_kwargs = {"headers": headers}

                if json_data is not None:
                    request_kwargs["json"] = json_data

                if params is not None:
                    request_kwargs["params"] = params

                if stream:
                    # HTTP/2 handles connection lifecycle via stream multiplexing.
                    # Connection: close was a workaround for CLOSE_WAIT leaks (issue #38)
                    # with HTTP/1.1 — not needed with HTTP/2 and actively harmful
                    # (forces TLS re-handshake on every streaming request, ~900ms overhead).
                    req = client.build_request(method, url, **request_kwargs)  # type: ignore[arg-type]
                    logger.debug("Sending request to Kiro API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Kiro API...")
                    response = await client.request(method, url, **request_kwargs)  # type: ignore[arg-type]

                # Check status
                if response.status_code == 200:
                    if ADAPTIVE_RETRY_ENABLED:
                        await _retry_bucket().record_success()
                    return response

                # 403 - token expired, refresh and retry
                if response.status_code == 403:
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{MAX_RETRIES})")
                    if stream:
                        await response.aclose()
                    await self.auth_manager.force_refresh()
                    continue

                # 429 - rate limit, wait and retry.  Body-content classifier
                # (Pattern 3 backport) decides between NO_RETRY (hard quota
                # markers like MONTHLY_REQUEST_COUNT — never retry, just
                # surface the response), THROTTLE (capacity / rate-limit —
                # longer backoff for sustained capacity issues), and STANDARD
                # (no body marker found, fall back to status-only logic).
                if response.status_code == 429:
                    last_response = response  # Retain to return after retry exhaustion
                    if attempt < max_retries - 1:
                        # Read body BEFORE closing the stream — closing first
                        # empties the body for streamed responses, which would
                        # silently miss NO_RETRY markers (MONTHLY_REQUEST_COUNT)
                        # and burn the entire retry budget. Mirrors the 5xx
                        # pattern at L334.  See SR#1 finding 2026-05-16.
                        body_bytes = b""
                        try:
                            body_bytes = await response.aread()
                        except Exception:
                            pass
                        if stream:
                            await response.aclose()
                        decision = classify_retry(429, body_bytes)
                        if decision.kind is RetryKind.NO_RETRY:
                            logger.warning(
                                f"Received 429 with hard-quota marker ({decision.reason}); "
                                "not retrying"
                            )
                            break  # Surface the response to the caller without further retries.

                        # Capacity-aware backoff: longer base delay, capped retries.
                        # SR#3 — use the structured is_capacity flag instead of
                        # grep'ing the human-readable reason.  The 429 path
                        # restricts capacity backoff to INSUFFICIENT_MODEL_CAPACITY
                        # (the only marker that's a sustained capacity issue);
                        # the 5xx path is broader (see SR#4 — keeping the existing
                        # asymmetry; promoting all THROTTLE markers on 429 would
                        # make `ThrottlingException`-flagged 429s wait 15s instead
                        # of ~1s, regressing happy-path).
                        if decision.kind is RetryKind.THROTTLE and decision.is_capacity and CAPACITY_BACKOFF_BASE > 0:
                            if attempt >= CAPACITY_MAX_RETRIES:
                                logger.warning(
                                    f"Capacity 429 retry budget exhausted "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                                continue
                            base_delay = CAPACITY_BACKOFF_BASE * (2 ** attempt)
                            delay = base_delay * (1.0 + random.uniform(-0.25, 0.25))
                            logger.warning(
                                f"Received 429 ({decision.reason}), "
                                f"capacity backoff {delay:.1f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                        else:
                            # Standard rate-limit backoff: respect Retry-After header.
                            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
                            try:
                                base_delay = float(retry_after) if retry_after else BASE_RETRY_DELAY * (2 ** attempt)
                            except (ValueError, TypeError):
                                base_delay = BASE_RETRY_DELAY * (2 ** attempt)
                            delay = base_delay * (1.0 + random.uniform(-0.25, 0.25))
                            logger.warning(f"Received 429, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        if ADAPTIVE_RETRY_ENABLED:
                            bucket = _retry_bucket()
                            await bucket.record_throttle()
                            extra = await bucket.acquire(jitter=True)
                            if extra > 0:
                                logger.debug(f"adaptive bucket added {extra:.2f}s to retry pacing")
                    continue

                # 409 - conflict / request already in progress, retry with backoff
                if response.status_code == 409:
                    last_response = response
                    if attempt < max_retries - 1:
                        if stream:
                            await response.aclose()
                        delay = BASE_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"Received 409, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    continue

                # 5xx - server error.  Body-content classifier decides
                # whether this is NO_RETRY (hard quota markers — never retry),
                # THROTTLE (capacity masquerading as 5xx — longer backoff
                # matters), or STANDARD (generic transient).
                if 500 <= response.status_code < 600:
                    last_response = response  # Retain to return after retry exhaustion
                    if attempt < max_retries - 1:
                        body_bytes = b""
                        try:
                            body_bytes = await response.aread()
                        except Exception:
                            pass
                        if stream:
                            await response.aclose()
                        decision = classify_retry(response.status_code, body_bytes)
                        # SR#2: hard-quota markers can land in 5xx bodies —
                        # break before retrying.  Today no upstream returns
                        # 5xx + MONTHLY_REQUEST_COUNT, but this closes the
                        # gap surfaced by the silent-failure audit.
                        if decision.kind is RetryKind.NO_RETRY:
                            logger.warning(
                                f"Received {response.status_code} with hard-quota marker "
                                f"({decision.reason}); not retrying"
                            )
                            break
                        if decision.kind is RetryKind.THROTTLE and CAPACITY_BACKOFF_BASE > 0:
                            if attempt >= CAPACITY_MAX_RETRIES:
                                logger.warning(
                                    f"Capacity 5xx retry budget exhausted "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                                continue
                            base_delay = CAPACITY_BACKOFF_BASE * (2 ** attempt)
                            delay = base_delay * (1.0 + random.uniform(-0.25, 0.25))
                            logger.warning(
                                f"Received {response.status_code} ({decision.reason}), "
                                f"capacity backoff {delay:.1f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                        else:
                            delay = BASE_RETRY_DELAY * (2 ** attempt)
                            logger.warning(
                                f"Received {response.status_code}, waiting {delay}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                        await asyncio.sleep(delay)
                    continue

                # Other errors - return as is
                return response

            except httpx.TimeoutException as e:

                # Classify timeout error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info

                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)

                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{short_msg} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors

            except httpx.RequestError as e:

                # Classify the error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info

                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)

                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{short_msg} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors

        # If we have a last_response (429/5xx retry exhausted), return it
        # This allows the caller to see the real status code and error body
        if last_response is not None:
            logger.warning(
                f"Retries exhausted for HTTP {last_response.status_code}, "
                f"returning response to caller for classification"
            )
            return last_response

        # All attempts exhausted - provide detailed, user-friendly error message
        if last_error_info:
            # Use classified error information
            error_message = last_error_info.user_message

            # Add troubleshooting steps
            if last_error_info.troubleshooting_steps:
                error_message += "\n\nTroubleshooting:\n"
                for i, step in enumerate(last_error_info.troubleshooting_steps, 1):
                    error_message += f"{i}. {step}\n"

            # Add technical details for debugging
            error_message += f"\nTechnical details: {last_error_info.technical_details}"

            raise HTTPException(
                status_code=last_error_info.suggested_http_code,
                detail=error_message.strip()
            )
        else:
            # Fallback if no error was captured (shouldn't happen)
            if stream:
                raise HTTPException(
                    status_code=504,
                    detail=f"Streaming failed after {max_retries} attempts. Unknown error."
                )
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Request failed after {max_retries} attempts. Unknown error."
                )

    async def __aenter__(self) -> "KiroHttpClient":
        """Async context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()
