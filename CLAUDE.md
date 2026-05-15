# kiro-proxy

Local AWS Q/CodeWhisperer proxy that exposes Anthropic Messages API for Claude Code.

Service: `systemctl --user status kiro-proxy` (port 8765)
Health: `curl http://127.0.0.1:8765/health`
Tests: `uv run pytest tests/` (2030+ passing as of 2026-05-15)
Migration log: `~/projects/claude-config/coord/restructure-premortem.md`

## What this repo does

Receives Anthropic-format `/v1/messages` requests from Claude Code, forwards
them to the user's Kiro/CodeWhisperer account via AWS SDK, and translates the
SSE response back into Anthropic streaming format. Adds a response cache, RE2
prompting, complexity-based model routing, in-flight dedup, and rich Logfire
telemetry.

The "kiro-gateway" name still appears in legacy docs and the back-compat
symlink at `~/tools/kiro-gateway` — the canonical name is `kiro-proxy` (matches
the GitHub remote `geehexx/kiro-proxy`).

## Quick start

```bash
# launch the service
systemctl --user start kiro-proxy

# Claude Code points at it automatically (settings.json: ANTHROPIC_BASE_URL=http://127.0.0.1:8765)

curl -s http://127.0.0.1:8765/health | jq .
```

## Repo siblings

| Repo | Path | Purpose |
|------|------|---------|
| this repo | `~/projects/kiro-proxy` | Kiro proxy implementation |
| **claude-config** | `~/projects/claude-config` | Claude Code config — primary workspace |
| cv-builder | `~/projects/cv` | CV builder product |

## Coordination protocol

If a peer Claude Code session is running in `~/projects/claude-config` (check
`python3 ~/projects/claude-config/scripts/claude_coord.py list-agents`), you
must register and coordinate before any cross-repo edit. See
`~/projects/claude-config/coord/HANDOFF.md`.

Suggested agent name from this repo: **agent-kiro** (`agent-<project-shortname>`).

Bootstrap:

```bash
python3 ~/projects/claude-config/scripts/claude_coord.py register agent-kiro
python3 ~/projects/claude-config/scripts/claude_coord.py list-agents
python3 ~/projects/claude-config/scripts/claude_coord.py recv agent-kiro --max=10
```

Scope you would typically acquire here:
- `tools-kiro-gateway` — covers the kiro-proxy work tree

Scopes you must NOT touch (owned by claude-config-side agents):
- `claude-skills`, `claude-rules`, `claude-agents`, `claude-commands`, `claude-md-files`, `mcp-config`, `data-basic-memory`, `services-claude-hooks`, `scripts`

If a task needs an edit in those scopes, send a `scope_request` mailbox message
to the holder and wait for a `scope_grant` reply.

## Service watchdog (CRITICAL — read before any systemctl stop)

The kiro-proxy service has a watchdog timer that auto-restarts it if inactive
>2min, suppressed by `/tmp/kiro-proxy-maintenance.flag`. The pattern:

```bash
touch /tmp/kiro-proxy-maintenance.flag       # suppress watchdog
systemctl --user stop kiro-proxy             # do work
# … your work …
systemctl --user start kiro-proxy
rm /tmp/kiro-proxy-maintenance.flag          # remove suppression
```

If you forget to start, the watchdog will auto-recover within 2 minutes. The
flag self-clears after 30 minutes regardless. See
`~/projects/claude-config/coord/watchdog-design.md`.

Better: combine stop+work+start atomically with `trap 'systemctl start kiro-proxy' EXIT`.

## What lives outside this repo (but is relevant)

- Logfire dashboards — env vars set in `~/projects/claude-config/CLAUDE.md`
- Baseline JSONL — `~/.claude/state/baselines-gateway-requests.jsonl`
- Analyzer — `~/projects/claude-config/scripts/analyze_gateway_baselines.py`
- Telemetry rollup — `~/projects/claude-config/scripts/telemetry-rollup.py`

## Coding standards

- Python 3.12 + uv. Never `pip install`.
- Test suite stays green; mutation testing on hot paths (cache, dedup, classifier).
- No AI attribution in commit messages or docstrings.
- Shared rules at `~/projects/claude-config/.claude/rules/` are read-only from this repo.
