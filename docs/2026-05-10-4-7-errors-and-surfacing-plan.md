# 4.7 errors and Claude Code surfacing plan

**Status:** Working draft — 2026-05-10. Not committed.
**Author:** Investigation fork, coordinator-supervised.
**Scope:** Read-only diagnosis + proposed improvements. No gateway code modified.

## 1. Diagnosis

### 1.1 What was observed

Andrew reported AWS-side errors for Claude 4.7 (`claude-opus-4-7` default) and mitigated by switching, via Claude Code's `/model`, to "a different version of 4.7" (a specific 4.7 snapshot listed by the client).

### 1.2 What the evidence shows

The gateway is a FastAPI proxy to the **Kiro API** (not direct to Bedrock). Kiro brokers to Bedrock; the gateway never sees Bedrock errors directly.

7-day journal audit (`journalctl --user -u kiro-gateway.service`) shows:

- **Zero** `ThrottlingException`, `AccessDeniedException`, `ModelNotReadyException`, `ResourceNotFoundException`, `ValidationException` occurrences from upstream.
- **Zero** 5xx responses from the gateway.
- Non-2xx tally (7d): 54× 405 (HEAD /), 6× 401 on `/v1/models` (auth probe), 4× 422 (validation — unrelated `server_tool_use` bug, see MEMORY → `sdk_422_server_tool_bug.md`), plus a scatter of 404 on endpoints Claude Code probes but the gateway doesn't implement (`/usage`, `/stats`, `/api/organizations`, `/metrics`, `/healthz`).
- Current traffic (last hour) alternates between `model=claude-opus-4-7` and `model=claude-opus-4.7` — **both succeed with HTTP 200**.
- `state.json` registers `claude-opus-4.7` in `model_to_accounts` — it is a known Kiro-side model, not pure passthrough.

### 1.3 Root-cause assessment (confidence: MEDIUM)

The most consistent interpretation of Andrew's report + the evidence:

- Whatever 4.7 failure occurred is **not currently reproducing**, and **no error trace survived in the journal** — which is itself a finding: the error was either (a) swallowed by streaming-path error handling that emits an SSE `event: error` but does not log WARNING/ERROR, or (b) intermittent and already mitigated by Andrew's `/model` switch.
- Kiro's upstream response for unknown/deprecated 4.7 snapshots is a 400/403 with a structured `reason` (`kiro_errors.py` enumerates `CONTENT_LENGTH_EXCEEDS_THRESHOLD`, `MONTHLY_REQUEST_COUNT` — but no `INVALID_MODEL` / `MODEL_DEPRECATED` handler exists). When a model passthrough fails, the enhancement falls back to `original_message`, which is what Claude Code renders generically.
- Andrew's `/model` switch bypassing the failure is consistent with: the previous default `claude-opus-4-7` snapshot being pointed (by Kiro) to a Bedrock inference profile that's intermittently throttled or regionally degraded for his account; a different 4.7 variant picks a different snapshot-to-profile mapping on Kiro's side.

### 1.4 Why the previous 4.7 failed but the alternate works

Not determinable from gateway logs alone — the gateway has no visibility into Kiro's internal snapshot-to-Bedrock-profile routing. Two plausible mechanisms:

1. **Snapshot routing asymmetry on Kiro side.** Different 4.7 aliases (e.g. `claude-opus-4-7` vs `claude-opus-4.7` vs a specific dated snapshot) resolve to different Bedrock model IDs / cross-region inference profiles. One may lack entitlement or be throttled in Andrew's account region while another is healthy.
2. **Deprecation lag.** A specific snapshot was removed from the user's Bedrock entitlement (AWS-side entitlement drift). Kiro still lists it; passthrough fails at the Bedrock call inside Kiro.

Both are consistent with the "switch to a named variant fixes it" workaround. Definitive diagnosis requires either Kiro API error-body capture (`debug_logger.py` can do this if `DEBUG_MODE=1`) or a replay of the failing request while watching the journal.

## 2. Claude Code extensibility surface (research)

What mechanisms exist for a proxy to actionably influence Claude Code's UX? Findings from the local docs cache + Anthropic's published Claude Code extensibility (hooks, slash commands, statusline, notifications):

| Channel | What it does | Does Claude Code honor it when coming from a proxy? | Verdict |
|---|---|---|---|
| Structured error in API response (`type: "error"`, `error.type: "..."`, `error.message: "..."`) | Rendered in transcript as a red error block. No action UI. | Yes — the gateway already emits this shape. | **Works for human-readable remediation text, not for triggering UI.** |
| Special error types (`overloaded_error`, `rate_limit_error`) | Triggers Claude Code's built-in retry UX with backoff. | Yes, for the canonical Anthropic error types. | **Leverage this for transient Bedrock throttles.** |
| Injected slash-command hint in error `message` (e.g., "Run /model to pick a different one") | Text-only. Claude Code does NOT execute slash commands from server content. | No — content is display-only. | **Advisory text works; auto-invocation does not.** |
| `SessionStart` / `UserPromptSubmit` hook (client-side settings.json) | Client runs a local script at lifecycle points; script prints to stderr/stdout, can inject system context. | Yes, this is client-side, not from proxy. | **Viable path for the client to poll a gateway health file.** |
| Notifications hook | Fires when Claude Code shows a notification. Can invoke local tools. | Yes. | **Usable for desktop toasts via `notify-send`.** |
| Statusline | Custom script prints a string rendered at the bottom of the UI. | Yes. | **Perfect surface for "degraded routing" indicator.** |
| MCP server | Full tool/resource protocol. Could expose a `switch_model` tool. | Yes. But tool execution still needs user approval, and Claude Code does not have an API to self-change its model from a tool call. | **Cosmetic — can surface info, but cannot actually switch models.** |

**Net:** There is no supported way for a proxy to force Claude Code to open its model selector. The practical envelope is:

1. Emit a canonical `overloaded_error` for transient upstream failures → trigger retry.
2. Emit a human-readable `api_error` with inline remediation text ("model `X` unavailable — try `/model` and select `Y`") for non-retryable ones.
3. Use client-side hooks (statusline + notifications) to surface gateway-reported health so the user knows without the agent having to say so.

## 3. Design

### 3.1 Gateway-side error classification (new)

Create `kiro/upstream_errors.py` analogous to `kiro_errors.py` / `network_errors.py`. When the Kiro response (non-stream and SSE) indicates model-specific failure, classify into:

| Gateway class | Upstream signal | Claude Code error payload | Notification |
|---|---|---|---|
| `TRANSIENT_BEDROCK_THROTTLE` | 429, `ThrottlingException`, `ServiceUnavailable`, SSE `event: error` with `type: overloaded` | `{"type": "error", "error": {"type": "overloaded_error", "message": "…"}}` (HTTP 529) | `warn` level, debounced 60s |
| `MODEL_UNAVAILABLE` | Kiro 400/404 with model-name substring in message, or AWS `ResourceNotFoundException`, `ModelNotReadyException`, `AccessDeniedException` | `{"type": "error", "error": {"type": "api_error", "message": "Model '<orig>' is unavailable on the current account.\n\nTry `/model` and select one of: <top 3 peers from same family>.\n\nAuto-fallback to '<best peer>' is available — rerun with header `X-Kiro-Fallback: auto`."}}` (HTTP 400) | `error` level, not debounced |
| `MODEL_DEPRECATED` | Kiro message contains `deprecated`, `retired`, `sunset` | As above + mark the model in `state.json` with `deprecated_at` so next `/v1/models` response filters it | `error` level, one-shot |
| `ACCOUNT_QUOTA` | `MONTHLY_REQUEST_COUNT` (already handled), `AccessDenied` with entitlement body | Preserve existing user message + append account-switch hint if multi-account enabled | `error` level |

Reuse existing `kiro_errors.KiroErrorInfo` / `enhance_kiro_error()` plumbing — add new `reason` branches, don't replace the architecture.

### 3.2 Auto-fallback (opt-in, header-driven)

Do **not** silently substitute models by default — that's confusing and hides real capacity issues. Instead:

- Add `X-Kiro-Fallback: auto | none | <explicit-model>` request header (Claude Code doesn't send it; Andrew can set it for specific runs, or a client-side hook can inject it via the gateway's existing proxy layer). Default `none`.
- When `auto` and the upstream returns `MODEL_UNAVAILABLE`, pick the next-best peer via a fixed preference ladder (below), transparently retry, and:
  - Tag the response with `X-Kiro-Fallback-Used: <model>` (header).
  - Append a single-line system note to the SSE stream via an existing `message_delta` text block: `[kiro-gateway: served by <model> — original <orig> unavailable]`. This is visible in the transcript.
- Emit a notification so Andrew knows he's on a fallback.

**Fallback ladder (by requested family):**

| Requested | Ladder (first healthy wins) |
|---|---|
| `claude-opus-4.7` | → `claude-opus-4.6` → `claude-opus-4.5` → `claude-sonnet-4.6` |
| `claude-opus-4.6` | → `claude-opus-4.7` → `claude-opus-4.5` → `claude-sonnet-4.6` |
| `claude-opus-4.5` | → `claude-opus-4.6` → `claude-sonnet-4.6` |
| `claude-sonnet-4.6` | → `claude-sonnet-4.5` → `claude-opus-4.5` |
| `claude-sonnet-4.5` | → `claude-sonnet-4.6` → `claude-sonnet-4` |
| `claude-haiku-4.5` | → `claude-haiku-4` → `claude-sonnet-4` |

Rationale: never cross family unless there's no peer within the family; Opus↔Sonnet crossing is acceptable as last-resort since both support tool use and extended context.

### 3.3 Notification channel

**Primary: a health sentinel file + statusline script (client-side).**

- Gateway writes `~/.claude/kiro-gateway-health.json` atomically on state change:
  ```json
  {
    "updated_at": "2026-05-10T19:02:31Z",
    "status": "degraded | healthy | down",
    "last_error": {
      "reason": "MODEL_UNAVAILABLE",
      "model": "claude-opus-4-7",
      "message": "…"
    },
    "fallback_active": {"requested": "claude-opus-4.7", "serving": "claude-opus-4.6"}
  }
  ```
- A tiny statusline script (`~/.claude/bin/kiro-statusline.sh`, registered via `settings.json` `statusLine`) reads this file and renders one of:
  - `kiro: healthy`
  - `kiro: degraded (opus-4.7 → opus-4.6)`
  - `kiro: DOWN — /model to switch`
- Because the health file is read only by a local script, the gateway stays fully decoupled from Claude Code internals.

**Backup: desktop toast via WSLg `notify-send`.**

- On state-change write, the gateway shells out: `notify-send "Kiro gateway" "opus-4.7 unavailable — serving opus-4.6"`.
- Gated behind `DESKTOP_NOTIFICATIONS=1` in `.env` so headless users don't get errors.
- WSLg 1.0.73 + current WSL 2.7.3 already route `notify-send` to Windows notifications.

**Transcript-visible (always on):** the inline `[kiro-gateway: …]` tag appended to the SSE stream in §3.2 above.

### 3.4 Claude Code-side reaction (optional, client-side)

Zero-coupling, but nice-to-have once §3.1–3.3 land:

1. **Statusline** — covered above.
2. **`SessionStart` hook** — reads `kiro-gateway-health.json` and, if status is `degraded` or `down`, echoes a single line to stderr: `⚠️ kiro-gateway reports degraded routing (<details>). Consider running /model.`. Anthropic's hooks spec allows this; stderr surfaces as a banner before the first turn.
3. **`Notification` hook** — optional; can invoke `notify-send` itself when Claude Code fires notifications, augmenting the gateway's own toasts.

All of these are **opt-in per-user settings.json** changes — the gateway does not depend on them being configured. If they're missing, the only visible signal is the inline transcript tag, which is still sufficient.

## 4. Fallback priority: is a server-side default fallback "best/most appropriate"?

**Recommendation: do not default-enable auto-fallback.** Reasons:

- Silent model substitution masks capacity issues the user should know about — they are paying for Opus and want to notice if they're being served Sonnet.
- Fallback only on explicit opt-in (`X-Kiro-Fallback: auto`) preserves the principle "gateway is a gateway, not a gatekeeper" (see `model_resolver.py:29`).
- For the `/model`-triggering UX Andrew actually wants, the best achievable surface is the inline remediation message in §3.1 — it tells the user exactly what to type next.

For users who want auto-fallback, a single env var (`KIRO_AUTO_FALLBACK=1`) flips the default from `none` to `auto` system-wide.

## 5. Implementation ordering (for coordinator to plan)

1. **Add structured error classification** (`upstream_errors.py` + hooks in `routes_anthropic.py` + `streaming_anthropic.py`). Tests: replay saved upstream payloads, assert output matches canonical Anthropic error schema. Smallest unit of work, biggest immediate UX win.
2. **Health sentinel file** — add writer in `main.py` lifespan + state-change hooks in error classifier. File-only, no dependents.
3. **Statusline script** — one bash script in `scripts/`, documented in README. User wires it into their own `settings.json`.
4. **Auto-fallback behind header** — requires new retry loop in the route handler, more invasive. Ship only after 1–3 prove out.
5. **Desktop notifications + SessionStart hook** — optional polish.
6. **Debug capture** — add `DEBUG_CAPTURE_UPSTREAM_ERRORS=1` mode that persists the first N error bodies to `~/tools/kiro-gateway/debug/` so the next 4.7-class incident produces concrete evidence rather than "no error trace survived".

Each step is independently shippable and testable.

## 6. Risks

- **Fallback hides real issues.** Mitigated by (a) opt-in default, (b) always-visible transcript tag, (c) statusline indicator.
- **Model family ladder becomes stale.** `state.json` already lists available models; ladder should filter by that list at request time, not hardcode.
- **WSLg `notify-send` brittle on corporate Windows builds with notification policy restrictions.** Mitigated by gating behind `.env` flag and logging the toast too.
- **Health-file writes race with reads.** Use atomic rename (`tempfile.NamedTemporaryFile(dir=...)` + `os.replace`) — standard pattern.
- **Kiro upstream schema drift.** Error reasons are strings; add unknown reasons to the passthrough branch rather than breaking.

## 7. Counter

**Strongest argument against this plan:** the observed problem already self-resolved via Andrew's `/model` switch, and there is zero error trace to confirm the proposed taxonomy is actually the one that happened. Building classification for hypothetical failure modes risks designing the wrong net. A cheaper first move is §5.6 alone — turn on upstream-error debug capture, wait for the next incident to produce real evidence, and size the classification to what actually shows up. Everything else in §3–§4 is defensible but deferrable until we have a concrete failure in hand.

## 8. Open questions (for Andrew)

1. Which specific 4.7 variant name did you switch *from* and *to*? That pair will tell us whether it was snapshot-level (entitlement), region-level (cross-region profile), or a Kiro-side routing change.
2. Do you want fallback opt-out (default-auto) or opt-in (default-none)? Plan assumes opt-in.
3. Is `notify-send` acceptable on your Windows host, or should desktop toasts be disabled by default?
