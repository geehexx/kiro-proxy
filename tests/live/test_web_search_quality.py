"""W3-5 web search quality regression test.

Smoke test: 5 representative queries run against the Kiro proxy (Path A —
native Anthropic server-side tool) and scored by Groq llama-3.3-70b.
Gate: avg_score >= 4.0.

Path A is used here because it works end-to-end in non-streaming mode.
Path B (MCP emulation) requires streaming interception and is tested via
the eval harness in scripts/eval_web_search.py.

Skipped automatically when GROQ_API_KEY or PROXY_API_KEY are not set.

Run manually:
    uv run pytest tests/live/test_web_search_quality.py -v -m live
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.live

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
if not PROXY_API_KEY:
    _env = Path(__file__).parents[2] / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            if _line.startswith("PROXY_API_KEY="):
                PROXY_API_KEY = _line.split("=", 1)[1].strip()
                break

KIRO_PROXY_URL = os.environ.get("KIRO_PROXY_URL", "http://localhost:8765")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

SMOKE_QUERIES = [
    {"id": "q01", "query": "What is the current stable version of Python?"},
    {"id": "q02", "query": "Who is the current CEO of Anthropic?"},
    {"id": "q03", "query": "FastAPI lifespan context manager example"},
    {"id": "q04", "query": "SQLite WAL mode checkpoint pragma syntax"},
    {"id": "q05", "query": "Redis vs Valkey performance comparison 2025 2026"},
]

QUALITY_GATE = 4.0


def _groq_score(query: str, result_text: str) -> float:
    """Score a result 1-5 using Groq. Returns -1 on error."""
    if not GROQ_API_KEY:
        return -1.0
    prompt = (
        f"Query: {query}\n\nResult:\n{result_text[:1500]}\n\n"
        "Score relevance 1-5 (5=best). Respond with JSON only: "
        '{"overall": N, "reason": "one sentence"}'
    )
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 80,
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}") + 1
        if start >= 0 and end > start:
            return float(json.loads(content[start:end]).get("overall", -1))
    except Exception:
        pass
    return -1.0


def _kiro_search_path_a(query: str) -> str:
    """Run a web search via Kiro proxy Path A (native server-side tool).

    Path A uses the native Anthropic web_search_20250305 type. The proxy
    intercepts this and routes to the Kiro MCP API, returning results
    directly in the response content.
    """
    headers = {
        "x-api-key": PROXY_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "web-search-2025-03-05",
        "content-type": "application/json",
    }
    resp = httpx.post(
        f"{KIRO_PROXY_URL}/v1/messages",
        headers=headers,
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": f"Search for: {query}"}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    parts = []
    for block in resp.json().get("content", []):
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "tool_result":
            # Path A may return tool_result blocks with search content
            content = block.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
    return "\n".join(parts) or "No content"


@pytest.mark.skipif(
    not PROXY_API_KEY,
    reason="PROXY_API_KEY not set — skipping web search quality test",
)
@pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY not set — skipping web search quality test",
)
class TestWebSearchQuality:
    """W3-5 quality regression gate: avg score >= 4.0 across 5 smoke queries."""

    def test_path_a_quality_gate(self):
        """Kiro proxy Path A (native server-side tool) must score >= 4.0 avg."""
        scores = []
        for q in SMOKE_QUERIES:
            result = _kiro_search_path_a(q["query"])
            score = _groq_score(q["query"], result)
            if score > 0:
                scores.append(score)

        assert scores, "No queries returned scoreable results"
        avg = sum(scores) / len(scores)
        assert avg >= QUALITY_GATE, (
            f"Web search quality regression: avg={avg:.2f} < gate={QUALITY_GATE}. "
            f"Scores: {scores}"
        )

    def test_path_a_latency(self):
        """Kiro proxy Path A must respond within 30s per query."""
        q = SMOKE_QUERIES[0]
        t0 = time.monotonic()
        _kiro_search_path_a(q["query"])
        latency = time.monotonic() - t0
        assert latency < 30.0, f"Path A latency {latency:.1f}s exceeds 30s gate"
