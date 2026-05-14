# Architecture

## Overview

Kiro Gateway is a proxy server that translates between the Anthropic Messages API (and OpenAI Chat Completions API) and the AWS Q Developer / CodeWhisperer API. It allows Claude Code, Cursor, Cline, and other AI tools to use Claude models via a Kiro subscription.

## Request Flow

```
Client (Claude Code / Cursor / etc.)
    │
    │  POST /v1/messages  (Anthropic format)
    │  POST /v1/chat/completions  (OpenAI format)
    ▼
┌─────────────────────────────────────────┐
│           Kiro Gateway (port 8765)       │
│                                         │
│  1. Auth verification (PROXY_API_KEY)   │
│  2. Beta stripping (unsupported betas)  │
│  3. Complexity classification           │
│  4. RE2 injection (if eligible)         │
│  5. Response cache lookup               │
│  6. Format conversion (Anthropic→Kiro)  │
│  7. Account selection (failover)        │
│  8. Token refresh (if needed)           │
└─────────────────────────────────────────┘
    │
    │  POST /generateAssistantResponse
    │  (AWS CodeWhisperer streaming format)
    ▼
AWS Q Developer API (q.{region}.amazonaws.com)
    │
    ▼
Claude models (Sonnet, Opus, Haiku, etc.)
    │
    ▼
┌─────────────────────────────────────────┐
│           Kiro Gateway                   │
│                                         │
│  1. Stream parsing (AWS→Anthropic SSE)  │
│  2. Thinking block extraction           │
│  3. Token counting                      │
│  4. Cache storage                       │
│  5. Telemetry emission                  │
└─────────────────────────────────────────┘
    │
    ▼
Client (SSE stream or JSON response)
```

## Token Refresh

The gateway automatically refreshes expired tokens:

1. Before each request, `get_access_token()` checks if the token expires within `TOKEN_REFRESH_THRESHOLD` seconds (default: 600s / 10 minutes)
2. If expiring soon, it calls `_refresh_token_request()` which uses the stored `refreshToken`
3. The refresh is protected by `asyncio.Lock()` — concurrent requests wait for the first refresh to complete
4. On success, the new token is written back to the credentials file
5. On failure (e.g., stale refresh token), the gateway falls back to using the existing access token until it actually expires

## Multi-Account Failover

When `ACCOUNT_SYSTEM=true`, the gateway manages multiple Kiro accounts:

1. Accounts are loaded from `credentials.json` (list of credential paths)
2. Each account has a circuit breaker with states: CLOSED → OPEN → HALF_OPEN
3. On 429 (capacity exhaustion) or 403 (auth failure), the account's circuit breaker opens
4. The next request is routed to the next healthy account
5. Circuit breakers reset after `CIRCUIT_BREAKER_RESET_TIMEOUT` seconds

## Cache Layers

### Response Cache (non-streaming)
- LRU cache keyed by `hash(session_id + system + messages + model + max_tokens + tools + thinking)`
- Persisted to disk on shutdown, loaded on startup
- TTL: `RESPONSE_CACHE_TTL` seconds (default: 3600)
- Max size: `RESPONSE_CACHE_MAX_BYTES` bytes (default: 500MB)

### Stream Cache (streaming)
- Same key as response cache
- Stores concatenated SSE bytes
- Replayed as a single StreamingResponse on cache hit

### In-Flight Dedup (non-streaming)
- When two identical non-streaming requests arrive simultaneously, only one hits upstream
- The second awaits the first's `asyncio.Future` and gets the same result
- Reduces duplicate upstream calls during parallel sub-agent fan-outs

## Complexity Classifier

`kiro/complexity_classifier.py` classifies each request into `SKIP | SIMPLE | MEDIUM | COMPLEX`:

**Layer 1 (deterministic):**
- SKIP: haiku model, sub-agent requests, slash commands, `tool_choice=any/tool`
- SIMPLE: single-message requests, lookup patterns

**Layer 2 (structural heuristics):**
- Token count (n1n.ai 200-Token Rule)
- Conversation length (>=20 messages → MEDIUM floor)
- Code blocks, question density, reasoning keywords
- Lookup keywords (reduce complexity)

The classifier output drives:
- `re2_eligible`: whether RE2 injection should be applied
- `thinking_budget`: suggested thinking budget (future use)
- `complexity_label`: recorded in telemetry baselines

## RE2 Injection

Based on the EMNLP 2024 paper (arxiv 2309.06275), RE2 appends "Read the question again carefully before answering." to the last user message with text content.

**Eligibility:**
- `RE2_ENABLED=true` in `.env`
- Request is `re2_eligible` per complexity classifier
- `len(messages) >= RE2_MIN_MESSAGES` (default: 2)
- Skipped when `thinking.type == "enabled"` (explicit extended thinking)

**Implementation:**
- Scans backwards through messages to find the last user message with text content
- Appends `RE2_INJECTION` string to that message's text block
- Records `re2_applied=True` and `complexity_label` in telemetry

## Telemetry

Each request emits a record to `~/.claude/state/baselines-gateway-requests.jsonl`:

```json
{
  "ts": 1234567890.0,
  "model": "claude-sonnet-4.6",
  "input_tokens": 50000,
  "output_tokens": 200,
  "gateway_cache": "miss",
  "stream": true,
  "re2_applied": true,
  "complexity_label": "medium",
  "upstream_ms_total": 1500,
  "status": 200
}
```

Optionally emits spans to Logfire if `LOGFIRE_TOKEN` is set.
