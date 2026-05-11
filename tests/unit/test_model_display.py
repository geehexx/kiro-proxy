"""
Unit tests for kiro/model_display.py.

The presentation helpers are pure, so tests just assert input → output
correctness for the canonical formats we route and the edge cases a
client might send. Includes Hypothesis properties for idempotence.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kiro.model_display import (
    canonical_model_id,
    describe_model,
    display_name,
)

CANONICAL_CASES = [
    # Dashed minor-version form
    ("claude-opus-4-7", "claude-opus-4.7"),
    ("claude-opus-4-6", "claude-opus-4.6"),
    ("claude-sonnet-4-5", "claude-sonnet-4.5"),
    ("claude-haiku-4-5", "claude-haiku-4.5"),
    # Dated suffixes
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4.5"),
    ("claude-opus-4-5-20251101", "claude-opus-4.5"),
    ("claude-haiku-4-5-20251001", "claude-haiku-4.5"),
    # Legacy three-digit shape
    ("claude-3-7-sonnet", "claude-3.7-sonnet"),
    ("claude-3-5-haiku", "claude-3.5-haiku"),
    # Already-canonical passthrough
    ("claude-opus-4.7", "claude-opus-4.7"),
    ("claude-sonnet-4", "claude-sonnet-4"),
    # Latest marker
    ("claude-haiku-4-5-latest", "claude-haiku-4.5"),
    # Non-Claude passthrough
    ("auto-kiro", "auto-kiro"),
    ("deepseek-3.2", "deepseek-3.2"),
    ("glm-5", "glm-5"),
    # Empty edge
    ("", ""),
]


DISPLAY_CASES = [
    ("claude-opus-4.7", "Claude Opus 4.7"),
    ("claude-opus-4-7", "Claude Opus 4.7"),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ("claude-sonnet-4", "Claude Sonnet 4"),
    ("claude-3-7-sonnet", "Claude 3.7 Sonnet"),
    ("claude-3-5-haiku", "Claude 3.5 Haiku"),
    ("auto-kiro", "Auto (Kiro)"),
    ("deepseek-3.2", "DeepSeek 3.2"),
    ("qwen3-coder-next", "Qwen3 Coder (Next)"),
    ("glm-5", "GLM 5"),
    ("minimax-m2.5", "MiniMax M2.5"),
    ("", ""),
]


class TestCanonicalModelId:
    """The canonicaliser takes any alias we accept and returns the dotted form."""

    @pytest.mark.parametrize("raw, canonical", CANONICAL_CASES)
    def test_known_aliases(self, raw: str, canonical: str) -> None:
        assert canonical_model_id(raw) == canonical

    @pytest.mark.parametrize("raw, _canonical", CANONICAL_CASES)
    def test_idempotent(self, raw: str, _canonical: str) -> None:
        """canonical(canonical(x)) == canonical(x)."""
        once = canonical_model_id(raw)
        twice = canonical_model_id(once)
        assert once == twice, f"Not idempotent for {raw!r}: {once!r} vs {twice!r}"


class TestDisplayName:
    """display_name is suitable for a picker UI."""

    @pytest.mark.parametrize("raw, label", DISPLAY_CASES)
    def test_known_cases(self, raw: str, label: str) -> None:
        assert display_name(raw) == label

    def test_unknown_non_claude_returns_canonical(self) -> None:
        """Graceful fallback for a model we don't curate."""
        assert display_name("mystery-model-1") == "mystery-model-1"


class TestDescribeModel:
    def test_opus_family(self) -> None:
        assert "reasoning" in (describe_model("claude-opus-4.7") or "").lower()

    def test_sonnet_family(self) -> None:
        description = describe_model("claude-sonnet-4.5") or ""
        assert "balanced" in description.lower()

    def test_haiku_family(self) -> None:
        description = describe_model("claude-haiku-4.5") or ""
        assert "fastest" in description.lower()

    def test_auto_kiro(self) -> None:
        assert describe_model("auto-kiro") == "Automatic model selection via Kiro."

    def test_unknown_returns_none(self) -> None:
        assert describe_model("nebula-9000") is None


class TestPropertyInvariants:
    """Hypothesis properties — the shape invariants we care about."""

    @given(st.text(max_size=80))
    def test_canonical_is_idempotent(self, raw: str) -> None:
        """canonical(canonical(x)) == canonical(x) for all input."""
        once = canonical_model_id(raw)
        twice = canonical_model_id(once)
        assert once == twice

    @given(st.text(max_size=80))
    def test_display_name_never_raises(self, raw: str) -> None:
        """display_name is total over str — never raises."""
        result = display_name(raw)
        assert isinstance(result, str)

    @given(st.text(max_size=80))
    def test_canonical_preserves_empty(self, raw: str) -> None:
        """canonical(x) == '' if and only if x.strip() == ''."""
        result = canonical_model_id(raw)
        if raw.strip() == "":
            assert result == ""
        else:
            assert result != ""
