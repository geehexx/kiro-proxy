# Project-scoped delta for kiro-proxy

Loaded by Claude Code when launched from `~/projects/kiro-proxy/`. Sits on top
of `~/.claude/CLAUDE.md` (user-global) and the project's root `CLAUDE.md`.

## Identity

Suggested agent name: **agent-kiro**. Unique per project — collision-safe at
N>=3 because the project shortname is unique. If two sessions launch from this
repo simultaneously, the second one should request a name suffix from the
leader (proposed: `agent-kiro.2`).

## Boot checklist (run on session start)

```bash
# 1. Register with the coord layer
python3 ~/projects/claude-config/scripts/claude_coord.py register agent-kiro

# 2. Survey live peers
python3 ~/projects/claude-config/scripts/claude_coord.py list-agents

# 3. Drain any inbox messages
python3 ~/projects/claude-config/scripts/claude_coord.py recv agent-kiro --max=10
```

## Scope protocol for this session

- Acquire `tools-kiro-gateway` before editing source under `kiro/`, `tests/`,
  or `main.py`. TTL 30 min default.
- Acquire NEITHER `claude-skills`, `claude-rules`, `claude-agents`,
  `claude-commands`, `claude-md-files`, `mcp-config`, `data-basic-memory`,
  `services-claude-hooks`, `scripts` — those belong to claude-config-side
  peers. Use `scope_request` mailbox messages instead.

## Common pitfalls (lessons from prior sessions)

- **systemctl stop** without the watchdog maintenance flag has bitten us. See
  `feedback_systemctl_stop_rollback` memory + `~/projects/claude-config/coord/watchdog-design.md`.
- **OIDC token refresh on IdC accounts** is broken (the `Public client can't be used for PresignedUrl refresh grant` error). Re-auth via Kiro IDE if you see 500s on `/v1/messages` mentioning `oidc.us-east-1.amazonaws.com/token`.
- **Cache hit rate** is currently low (~5%). The cache is hot in-memory; restart
  loses it. Backup `response_cache.pkl` before any service stop.

## Coordination references

- Live HANDOFF: `~/projects/claude-config/coord/HANDOFF.md`
- Coord script: `~/projects/claude-config/scripts/claude_coord.py`
- Protocol research (in flight): `basic-memory://research/2026-05-15-a2a-protocol-overhaul/`
- Migration log (most recent): `~/projects/claude-config/coord/restructure-premortem.md`
- Premortem template: same file (use as a model for any other restructure)
