# -*- coding: utf-8 -*-

"""
Cache-integration helpers for the Anthropic /v1/messages route.

This module owns the contract between the ResponseCache singleton and
the request-handling path. Route code should call only the three helpers
defined here:

- ``derive_session_id(request, api_key)`` — stable per-client scope
- ``compute_cache_key(request_data, session_id)`` — request fingerprint
- ``try_cache_lookup(cache, key)`` / ``store_cache(cache, key, body)``

Keeping the glue in one module makes the streaming cache extension (a
future follow-up) drop in without re-threading the route handler.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from kiro.response_cache import CacheEntry, ResponseCache, make_key


def derive_session_id(api_key: Optional[str], client_header: Optional[str]) -> str:
    """Return a stable session scope for cache keys.

    Preference order:

    1. Explicit ``X-Kiro-Session-Id`` client header — when clients want
       to scope cache per-conversation rather than per-user.
    2. Hash of the proxy API key — scopes cache to a single proxy user.
    3. Literal ``"anonymous"`` — only reached when auth is disabled.

    The return value is an opaque ~16-char hex prefix; callers should
    not rely on its shape.
    """
    if client_header:
        # Include api_key in the scope so two clients with different API keys
        # but the same X-Kiro-Session-Id header cannot share a cache bucket.
        scope = f"{api_key or ''}|{client_header}"
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    if api_key:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return "anonymous"


def compute_cache_key(
    *,
    session_id: str,
    system: Any,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    thinking: Optional[Any] = None,
) -> str:
    """Thin wrapper over :func:`kiro.response_cache.make_key`.

    Centralised so a hash-scheme change touches one file.
    ``thinking`` is included because extended-thinking config affects output;
    a thinking-on vs thinking-off request with identical messages must not
    share a cache bucket.
    """
    return make_key(
        session_id=session_id,
        system=system,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        thinking=thinking,
    )


def try_cache_lookup(
    cache: Optional[ResponseCache], key: str
) -> Optional[CacheEntry]:
    """Return a cache entry if present, otherwise ``None``.

    Safe to call when the cache is disabled — returns ``None`` immediately
    so the route handler does not need to guard the call.
    """
    if cache is None:
        return None
    return cache.get(key)


def store_cache(
    cache: Optional[ResponseCache],
    key: str,
    body: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """Serialise a JSON response body and persist it in the cache.

    Returns ``False`` when the cache is disabled or the body is too large
    for the entry size budget.
    """
    if cache is None:
        return False
    try:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return False
    return cache.put(key, raw, headers=headers)


def entry_to_response_body(entry: CacheEntry) -> dict[str, Any]:
    """Deserialise a cache entry back into the JSON dict the route returns."""
    return json.loads(entry.body.decode("utf-8"))
