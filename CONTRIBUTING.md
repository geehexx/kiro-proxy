# Contributing to kiro-proxy

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/geehexx/kiro-proxy.git
cd kiro-proxy
uv sync
cp .env.example .env
# Edit .env with your Kiro credentials
uv run python main.py
```

## Running Tests

```bash
uv run pytest tests/ -q
uv run pytest tests/ -v --tb=short
uv run pytest tests/unit/test_cache_integration.py -v
```

## Code Style

```bash
uv run ruff check kiro/
uv run ruff check --fix kiro/
uv run ruff format kiro/
uv run pyright kiro/
```

Pre-commit hooks run on `git commit`. Manual run: `uv run lefthook run pre-commit`

## Project Structure

```
kiro/
├── routes_anthropic.py      # Anthropic /v1/messages endpoint
├── routes_openai.py         # OpenAI /v1/chat/completions endpoint
├── converters_anthropic.py  # Anthropic → Kiro format conversion
├── streaming_anthropic.py   # Streaming response handling
├── auth.py                  # Token management and refresh
├── account_manager.py       # Multi-account failover
├── response_cache.py        # LRU response cache
├── complexity_classifier.py # Request complexity classification
├── config.py                # All configuration constants
└── telemetry.py             # Logfire/OpenTelemetry integration
```

## Pull Request Guidelines

1. One feature per PR
2. Add tests for new behavior
3. Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
4. Update `.env.example` for new config options
5. Update README for user-visible features

## Commit Format

```
type(scope): short description

Longer description if needed.
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## About this fork

`kiro-proxy` is a personal fork of [`Jwadow/kiro-gateway`](https://github.com/Jwadow/kiro-gateway), used as a development workspace for proxy features and integrations. It is **not** an actively-maintained downstream and we are **not** currently submitting changes back to upstream on a routine cadence.

The upstream project is licensed under **AGPL v3** and that licence applies to this fork. All code in this repository — including any contribution accepted into it — is and remains AGPL v3. Credit for the original work belongs to the upstream maintainer ([@jwadow](https://github.com/jwadow)) and contributors.

## Contribution policy (current)

We are **not currently soliciting external contributions.** Maintainer bandwidth is limited and we do not have the testing, review, or release infrastructure in place to absorb drive-by patches responsibly.

If you open a pull request anyway, the terms below apply. We may close it without merging if it is out of scope or beyond our bandwidth — that is not a judgement on the work.

## Contribution terms

By submitting a contribution (pull request, patch, suggested change, issue-attached diff) to this repository, **you agree to the following.** "Contribution" means any work of authorship you intentionally submit for inclusion in this project.

### Inbound = outbound

Your contribution is licensed to the project and to downstream recipients under the **GNU Affero General Public License v3 or later (AGPL-3.0-or-later)** — the same licence as the rest of the project. You are not asked to assign copyright. You retain authorship of your contribution; you are granting a licence, not transferring ownership. (This follows the FSF's recommended inbound-licensing posture for AGPL projects.)

### Right to submit

You represent that you have the right to grant this licence — the contribution is your original work, or you have permission from the rights-holder (e.g. employer) to submit it under AGPL v3.

### Patent grant

You grant the project and its downstream recipients a perpetual, worldwide, non-exclusive, royalty-free patent licence covering any patent claims you own that are necessarily infringed by your contribution.

### Permission to forward upstream on your behalf

If the maintainer later forwards work that includes your contribution to `Jwadow/kiro-gateway` (or any successor upstream), you authorise the maintainer to **review, edit, rebase, squash, retest, and submit** that work upstream on your behalf, subject to upstream's CLA, DCO, or contribution policy at the time of submission. The maintainer will preserve authorship attribution where reasonably possible (`Co-authored-by:` trailers, `Signed-off-by:` lines, or upstream's preferred attribution form).

You are not promised that any contribution **will** be forwarded upstream. The realistic expectation today is that most won't be — testing, alignment, and review work would have to be done first, and that is not on the near-term roadmap.

### As-is

Contributions are provided **as-is**, with no warranty of any kind. You are not obligated to provide support.

## How to sign

We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO). Sign each commit with `git commit -s`, which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

That sign-off plus this contribution policy is sufficient. There is no separate signing ceremony, comment-on-PR ritual, or CLA bot at this time.
