# Contributing to Kiro Gateway

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/jwadow/kiro-gateway.git
cd kiro-gateway
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
