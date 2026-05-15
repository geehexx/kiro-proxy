# Changelog

All notable changes to kiro-proxy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
## [Unreleased]

### Bug Fixes

- **routes**: Fast circuit-breaker for `INSUFFICIENT_MODEL_CAPACITY` 429s — fail
  immediately with 503 + `Retry-After: 60` + `X-Kiro-Capacity-Exhausted: <model>`
  header instead of cycling accounts. Cycling accounts doesn't help when the model
  capacity pool is exhausted across all accounts. Silent model substitution is also
  wrong because CC doesn't update session state on proxy substitution (CC Issues
  #23497, #44385, #54448). User decides to switch models via `/model claude-sonnet-4-6`.

## [2.3] - 2026-05-14

### Bug Fixes

- **models**: Add tool_reference content block support (#90)
- Disable AUTO_TRIM_PAYLOAD by default (#73)
- **anthropic**: Accurate token estimation for Anthropic API path (#135)
- **streaming**: Enable retry mechanism and return correct finish_reason on truncation (#113)
- **auth**: Preserve unknown fields in SQLite write-back (#131)
- **docker**: Mount kiro-cli volume as rw and exclude .cache from image (#97)
- **auth**: Truncate nanoseconds in SQLite expires_at parsing (#78)
- **auth**: Remove duplicate re import
- **streaming**: Import stream_with_first_token_retry_anthropic in routes_anthropic (#138)
- **docker**: Exclude credentials.json and state.json from image
- **ci**: Remove test artifacts before Docker build
- **docker**: Remove runtime files from image build
- **docker**: Remove read-only credentials.json before writing
- **docker**: Grant write permissions to /app for kiro user
- **main**: Correct indentation for credentials.json save block
- **cli**: Parse arguments before config validation for --version support
- **cache**: Correct make_key to include trailing turn; rename PrefixCache -> ResponseCache
- **cache**: Rename PREFIX_CACHE_* imports to RESPONSE_CACHE_* (INT-73)
- **cache,retry,dedup**: Correctness and robustness fixes
- **config**: Revert pyrightconfig.json venvPath to relative path
- **lint,docs**: Address remaining bot review comments
- **retry**: Add Retry-After respect and jitter to 429 backoff
- **retry**: Add 409 handling with exponential backoff
- **correctness**: Address new CodeRabbit Major findings
- **types,tests**: PEP 585 generics + correct flaky backoff test
- **async**: Offload _save_state disk IO to thread executor
- **accounts**: Guard Optional.attr deref in account_manager
- **config**: Update FALLBACK_MODELS with correct context windows for Opus 4.7 + Sonnet 4.6
- **config**: Add Opus 4.6 to FALLBACK_MODELS with 1M context window
- Apply upstream PR #158/#163 — tool write failures + thinking injection
- **model-resolver**: Strip [1m] bracket suffixes + add shorthand aliases
- **routes**: Move _re2_active before cache lookup, fix NameError on cache hits
- **openai**: Remove placeholder reference from mock organizations endpoint
- **telemetry**: Add logfire[fastapi,httpx] extras to requirements.txt
- **baselines**: Extract token usage from streaming SSE in both dispatch paths
- **telemetry**: Count tool input JSON in output_tokens for tool_use responses
- **models**: Remove deprecated claude-sonnet-4 (4.0) from fallback list
- **models**: Hide deprecated claude-sonnet-4 from /v1/models endpoint
- **re2**: Only skip RE2 when thinking.type='enabled', not 'adaptive'
- **re2**: Apply RE2 to long conversations even with short last message
- **re2**: Inject into last user message with text, not just last user message
- **re2**: Keep _re2_active=True when injection applied to earlier message
- **oss**: Make User-Agent configurable, redact personal Logfire URL
- **oss**: Fix ruff lint issues in complexity_classifier and in_flight_dedup
- **telemetry**: Suppress test spans, add complexity_label+dedup_hit to Logfire

### CI/CD

- **docker**: Replace runtime tests with structure tests

### Documentation

- **models**: Add DeepSeek-V3.2, MiniMax M2.1, Qwen3-Coder-Next
- Add payload size guard settings to .env.example (#73)
- Update funding links
- **agents**: Add feature parity and reverse engineering context
- Enforce consistency and quality standards
- **limiter**: Explain why streaming path is unwired
- **cache**: Add STREAM_CACHE_ENABLED to .env.example + streaming cache tests
- **re2**: Document RE2_ENABLED and RE2_INJECTION in .env.example
- **re2**: Document proxy-level tradeoff vs canonical optillm formulation
- Add ARCHITECTURE.md, CONTRIBUTING.md, TROUBLESHOOTING.md for OSS readiness
- **env**: Add KIRO_USER_AGENT to .env.example

### Features

- Payload size guard with pre-flight check and auto-trim (#73)
- **thinking**: Add client thinking budget support for OpenAI and Anthropic APIs (#111)
- **websearch**: Add MCP tool emulation support (#101)
- **auth**: Auto-detect API region from credentials (#132, #133)
- **account-system**: Add multi-account support with failover (#93)
- **anthropic**: Add /v1/messages/count_tokens endpoint
- **cache**: Add in-memory prefix cache module + 31 unit tests
- **cache**: Wire prefix cache singleton into app.state
- **cache**: In-flight request deduplication (Layer A)
- **gateway**: Humanised model naming + usage parity with Anthropic spec
- **cache**: Wire response cache into /v1/messages non-streaming path (INT-68)
- **dedup**: Instantiate InFlightDedup singleton on app.state
- **gateway**: Log msg_id + response_model on /v1/messages completion
- **limiter**: Add SessionLimiter — per-session concurrency cap (unwired)
- **limiter**: Add acquire_slot() for streaming lifetimes
- **limiter**: Instantiate SessionLimiter on app.state (unwired)
- **limiter**: Wire SessionLimiter into non-streaming dispatch (flag-gated)
- **baselines**: Add BaselinesWriter — async JSONL telemetry emitter (#2)
- **gateway**: Emit per-request baselines to gateway-requests.jsonl (#4)
- **gateway**: Emit streaming baselines — 100% of /v1/messages coverage (#6)
- **model_display**: Per-version descriptions differentiate Opus tiers
- **429**: Capacity-aware backoff, telemetry fields, global Opus cap
- **429**: Capacity-aware backoff, telemetry fields, global Opus cap
- **gateway**: Strip web_search from sub-agent requests to prevent SDK 422
- **gateway**: Strip web_search from sub-agent requests (SDK 422 prevention)
- **baselines**: Extract token usage from streaming message_delta events
- **streaming-cache**: Add streaming response cache infrastructure
- **re2**: Add ReRead injection — OptiLLM technique for improved reasoning
- **openai**: Add mock /api/organizations endpoint for Claude Code CLI
- **cache**: Add disk persistence + fix re2 cache key ordering
- **re2**: Add eligibility rules — skip for tool_result-only + short requests
- **re2**: Add eligibility filter — skip haiku, tool_result-only, sub-agents
- **telemetry**: Add logfire instrumentation with cost tracking
- **debug**: Add debug_capture module + fix pre-existing test failures
- **debug**: Add rotate mode for per-request log corpus collection
- **telemetry**: Filter low-token Logfire spans to reduce sub-agent noise
- **re2**: Add RE2_MIN_CHARS and RE2_SKIP_EXTENDED_THINKING guards
- **betas**: Strip unsupported Anthropic betas before upstream dispatch
- **dedup**: Wire in_flight_dedup for non-streaming requests
- **classifier**: Add complexity classifier for adaptive RE2 + thinking budget
- **telemetry**: Add complexity_label to baseline records
- **telemetry**: Add dedup_hit to baselines, cache key normalization, health stats
- **ci**: Add GitHub Actions CI workflow + property-based tests

### Maintenance

- **cla**: Update contributors
- **cla**: Update contributors
- **cla**: Update contributors
- **contributors**: Recognize core contributors
- **cla**: Update contributors
- Add license headers and update contributors (#73)
- **cla**: Update contributors
- **cla**: Update contributors
- Improve code documentation and remove unnecessary type counting (#135)
- **cla**: Update contributors
- **cla**: Update contributors
- **contributors**: Update list
- **cla**: Update list
- Bump version to 2.4-dev.10
- **contributors**: Update list
- **cla**: Update list
- **cla**: Update list
- **i18n**: Translate Russian comments and docstrings to English
- **lsp**: Add pyrightconfig.json to bind .venv for editor diagnostics
- **tooling**: Add lefthook + ruff config + dev deps (fix INT-73 lane)
- **pyright**: Bind .venv via root pyrightconfig — 0 missing-imports
- **ruff**: Remove pre-existing F401 + I001 in tests/ (#3)
- **baselines**: Rename upstream_ms_first_token → upstream_ms_total (#5)
- Add HANDOFF.md to .gitignore
- Fix HANDOFF.md gitignore pattern (was only matching htmlcov/)
- **oss**: Bandit nosec annotations, ruff complexity config, import cleanup

### Performance

- **http**: Enable HTTP/2 + extend keepalive to 120s
- **http**: Remove Connection: close header — HTTP/2 handles connection lifecycle

### Refactoring

- **payload-guard**: Improve trim logging message
- **telemetry**: Redesign logfire spans with hierarchy + truncation
- **re2**: Replace _re2_eligible() with complexity classifier

### Styling

- Strip trailing whitespace, fix pre-existing lint violations
- **types**: Modernize deprecated typing imports (partial)

### Testing

- **account-system**: Add comprehensive test suite and fix critical bugs (#93)
- **cache**: Property tests for cache correctness (5 hypothesis tests)
- **gateway**: Add PBT coverage for tokenizer + live smoke suite
- **dedup**: Add cancellation test for in_flight_dedup coalesce()
- Add thinking cache key regression + fix account stats assertions
- **regression**: Add stream cache format and capacity 429 regression tests
- **backcompat**: Add complexity_label forward/backward compat tests
- **emit**: Add complexity_label emit tests

## [2.3] - 2026-05-14

### Bug Fixes

- **thinking**: Ensure response language matches user preference
- **converters**: Ensure first message is user to prevent Improperly formed request (#60)
- **anthropic**: Extract images from tool_result content blocks (#57)
- **openai**: Extract images from tool messages for MCP screenshot support
- **converters**: Add support for unknown roles (#64)

### Documentation

- **contributing**: Add project philosophy and focused changes guideline
- Add Codex App to supported clients list (#64)

### Features

- **errors**: Add centralized Kiro API error enhancement system (#10, #63)
- **errors**: Improve MONTHLY_REQUEST_COUNT error message (#62)

### Maintenance

- **contributors**: Update contributors
- Bump version to 2.3

### Refactoring

- **converters**: Complete fix for unknown roles with alternating support (#64)
- **converters**: Use "(empty)" instead of "." for synthetic user message

## [2.2] - 2026-01-30

### Bug Fixes

- **models**: Add fallback list for DNS failure recovery (#25)
- **converters**: Handle Pydantic models in extract_text_content (#46, #50)
- **routes**: Use per-request clients for streaming to prevent CLOSE_WAIT leak (#54)
- Update CLA contributors
- **config**: Use universal q.{region}.amazonaws.com endpoint for all regions (#58)
- **docker**: Improve Docker configuration and CI/CD pipeline
- **tests**: Read PROXY_API_KEY from config instead of hardcoded value
- **config**: Restore timeout configuration warning

### Documentation

- **contributors**: Add @saaj for regional endpoint fix (#58)
- **i18n**: Add docker deployment section to all translated READMEs

### Features

- **openai**: Support Cursor flat format, inverted model names, and improve tool_results handling (#49)
- **errors**: Add network error classification with user-friendly messages (#53)
- **model-resolver**: Add alias system to resolve Cursor IDE conflict (#59)
- Add truncation recovery system (#34, #42, #56)
- **docker**: Add Docker containerization with CI/CD (#55)

### Maintenance

- **cla**: Update contributors
- **config**: Remove legacy debug settings and startup warnings
- Bump version to 2.2
- **contributors**: Update contributors

### Refactoring

- **deps**: Migrate manual_api_test.py from requests to httpx

### Testing

- **auth**: Update api_host test for new q.{region}.amazonaws.com endpoint

## [2.1] - 2026-01-20

### Bug Fixes

- **docs**: Update feature descriptions
- Update CLA contributors
- Add Connection: close header for streaming requests (#38)
- Validate tool names against 64-char Kiro API limit (#41)
- **auth**: Persist refreshed AWS SSO OIDC tokens back to SQLite (#43)
- **auth**: Use correct AWS SSO OIDC CreateToken API format (#43)
- Update CLA contributors

### Documentation

- **i18n**: Add README translations (ru, zh, es, id, pt, ja, vi, tr, ko)
- **i18n**: Fix badge anchor links (ru, zh, es, id, pt, ja, vi, tr, ko)
- Update model list and add tier-based availability notice (#39)
- Clarify Enterprise/Builder ID support and add Amazon Q Developer branding
- Clarify AWS SSO credentials configuration (#43)
- **template**: Update placeholders in bug report

### Features

- **parsers**: Add diagnostics for truncated tool call arguments (#34)
- **startup**: Add GitHub issues link to startup banner
- **auth**: Add support for social login SQLite credentials
- Add Enterprise Kiro IDE support with unified AWS SSO OIDC format (#43, #45, #48)
- **proxy**: Add HTTP/SOCKS5 proxy support for restricted networks

### Maintenance

- Add debug log before sending request to Kiro API

## [2.0] - 2026-01-11

### Bug Fixes

- **auth**: Add detailed AWS SSO OIDC error logging (#14)
- **auth**: Don't send scope in AWS SSO OIDC refresh (#14)
- **auth**: Separate SSO region from API region for AWS SSO OIDC (#16)
- Update contributors list in CLA
- Update CLA message for clarity
- **converters**: Skip thinking tag injection when toolResults present (#23)
- **auth**: Reload SQLite credentials before AWS SSO OIDC token refresh (#22)
- **auth**: Retry SQLite reload on 400 for container token refresh (#14)
- **http**: Add shared HTTP client with connection pooling (#24)
- **anthropic**: Support system as content blocks for prompt caching
- Convert tool_results to Kiro API format (toolUseId)
- **converters**: Handle orphaned tool_results and strip tool content when no tools defined
- **logging**: Suppress noisy shutdown tracebacks on Ctrl+C
- **config**: Update application version to 2.0-rc.1
- Update CLA contributors
- **converters**: Add placeholders for empty content after tool stripping (#20)
- **converters**: Convert tool content to text when tools not defined (#20)
- Standardize bug report title format and emphasize log requirement
- **auth**: Add graceful degradation for SQLite mode (#14)
- **models**: Add image content block support (#30)
- **config**: Update application version to 2.0 (#15)
- **anthropic**: Add ThinkingContentBlock to ContentBlock union (#31)
- **exceptions**: Comment out request body logging in validation error handler
- **vision**: Move images to userInputMessage.images for proper Kiro API handling (#32)

### Documentation

- Architecture for Anthropic API support (#15)
- **tests**: Add documentation for issue #20 fix tests
- Update CONTRIBUTORS.md to include @kilhyeonjun's contributions
- Add donation section and GitHub funding config
- Clarify git is optional, add ZIP download alternative (#27)
- **readme**: Improve README UX

### Features

- Add configurable server port/host (#19)
- **api**: Anthropic Messages API support (#15)
- **anthropic**: Add thinking content blocks for extended thinking
- Add dynamic model resolution with client format normalization
- Add DebugLoggerMiddleware to capture validation errors in debug logs (#31)

### Maintenance

- **i18n**: Translate Russian comments and docstrings to English
- Rename project to kiro-gateway
- Update contributors list in CLA
- **log**: Use INFO level for Kiro Desktop auth type

### Refactoring

- Rename kiro_gateway to kiro
- Update kiro_gateway naming to kiro
- Rename OpenAI-specific modules with _openai suffix
- Unify first token retry logic in core layer
- **logging**: Reduce merge_adjacent_messages log spam
- **tests**: Remove test for logging request body at debug level

### Testing

- **core**: Add coverage for streaming retry and tool stripping
- **models**: Add comprehensive Pydantic model validation tests

## [1.0.8] - 2026-01-04

### Bug Fixes

- Use original KiroIDE User-Agent format
- Remove duplicate log, reduce thinking buffer to 20 chars
- **reasoning**: Add system prompt legitimization for thinking tags
- **auth**: Don't send profileArn for AWS SSO OIDC (causes 403) (#12)

### Documentation

- Clarify profileArn not needed for AWS SSO OIDC (#12)
- Update prerequisites and credentials section

### Features

- **auth**: Add AWS SSO OIDC support for kiro-cli credentials (#12)
- Implement fake reasoning with extended thinking support (#11)

### Maintenance

- Bump version to 1.0.8

## [1.0.7] - 2025-12-18

### Bug Fixes

- Improve streaming error handling and prevent silent failures
- Update CLA message for clarity and conciseness
- Update CLA label and message
- Update contributors list in CLA

### Features

- Add CLA message for PR contributions
- Add configurable streaming read timeout (#9)
- Enhance credential loading and logging in manual_api_test.py

### Maintenance

- Bump application version to 1.0.6
- Add CLA signature for Kartvya69

### Refactoring

- Improve timeout handling and logging in http_client and streaming

## [1.0.6] - 2025-12-17

### Bug Fixes

- Add Cline support - sanitize tool schemas and handle empty descriptions

## [1.0.5] - 2025-12-17

### Documentation

- Update debugging section with DEBUG_MODE configuration

### Features

- **debug_logger**: Implement logs capture and storage
- Add CORS middleware for OPTIONS preflight support

### Maintenance

- Bump application version to 1.0.5

## [1.0.4] - 2025-12-17

### Bug Fixes

- Preserve tool_calls when merging assistant messages; add DEBUG_MODE with errors/all modes

### Features

- **tokenizer**: Add tiktoken fallback, fix multiplier for prompt_tokens, add tests

## [1.0.3] - 2025-12-16

### Bug Fixes

- Improve error handling in chat completions endpoint to return structured JSON response
- Normalize KIRO_CREDS_FILE path for cross-platform compatibility
- Read KIRO_CREDS_FILE without escape sequence processing for Windows paths
- Reduce default FIRST_TOKEN_TIMEOUT to 15 seconds

### Features

- **tokenizer**: Add token counting with tiktoken for usage tracking

### Maintenance

- Bump version to 1.0.3, centralize version constant

## [1.0.2] - 2025-12-13

### Bug Fixes

- Add index to streaming tool_calls, handle tool messages, improve deduplication
- Update application version to 1.0.2

### Documentation

- Update title in ARCHITECTURE.md to remove version number
- Add English translation of ARCHITECTURE.md
- Update architecture documentation to match codebase

### Features

- Add first token timeout retry for streaming requests

## [1.0.1] - 2025-12-13

### Bug Fixes

- **tests**: Correct error message assertion in test_raises_for_empty_messages
- Handle Kiro API 400 "Improperly formed request" for long tool descriptions
- Update version to 1.0.1 and modify author attribution

### Documentation

- Reorder configuration options in README for clarity
- Add Kiro IDE link to prerequisites

## [1.0.0] - 2025-12-13

### Bug Fixes

- Translate error messages to English

### Documentation

- Add initial architecture and .gitignore
- Clarify PROXY_API_KEY is user-defined password

### Features

- Complete Kiro OpenAI Gateway implementation
- Validation for .env file


