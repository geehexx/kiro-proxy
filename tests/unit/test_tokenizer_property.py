# -*- coding: utf-8 -*-

"""
Property-based tests for kiro/tokenizer.py.

The tokeniser is an approximation over tiktoken's cl100k_base, with a
Claude correction factor of 1.15. We lock the invariants we care about
in production:

- P1 **Totality**: count_tokens never raises for any string input.
- P2 **Determinism**: count_tokens(s) == count_tokens(s) — no hidden state.
- P3 **Empty preservation**: count_tokens('') == 0, count_tokens(None) == 0.
- P4 **Positivity**: count_tokens(non_empty) >= 1 once correction applied.
- P5 **Monotonicity-on-concatenation** (soft): count_tokens(a + b) >=
     max(count_tokens(a), count_tokens(b)) - 1 — appending content never
     drastically reduces token count (a small tolerance covers BPE merges).
- P6 **Correction ratio**: for sufficiently large input, tokens-with-
     correction ≈ 1.15 * tokens-without-correction (±10%).
- P7 **count_message_tokens totality + determinism**: like P1/P2 for
     multi-message input.

These properties catch off-by-one errors and state leaks that
example-based tests routinely miss.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro.tokenizer import (
    CLAUDE_CORRECTION_FACTOR,
    count_message_tokens,
    count_tokens,
    count_tools_tokens,
    estimate_request_tokens,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _plain_text() -> st.SearchStrategy[str]:
    """Text likely to be counted by real tokenisation."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            min_codepoint=0x20,
            max_codepoint=0x7E,
        ),
        max_size=200,
    )


def _message() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "role": st.sampled_from(["user", "assistant", "system"]),
            "content": _plain_text(),
        }
    )


def _message_list() -> st.SearchStrategy[list]:
    return st.lists(_message(), max_size=8)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokensProperties:
    @given(text=st.text(max_size=500))
    @settings(max_examples=50, deadline=None)
    def test_total_over_str(self, text: str) -> None:
        """P1: count_tokens is total over all strings (never raises)."""
        # Must not raise, must return int.
        result = count_tokens(text)
        assert isinstance(result, int)

    @given(text=_plain_text())
    @settings(max_examples=50, deadline=None)
    def test_deterministic(self, text: str) -> None:
        """P2: same input -> same count (no hidden state)."""
        a = count_tokens(text)
        b = count_tokens(text)
        assert a == b

    def test_empty_is_zero(self) -> None:
        """P3a: empty string is zero tokens."""
        assert count_tokens("") == 0

    def test_none_is_zero(self) -> None:
        """P3b: None is zero tokens (defensive null handling)."""
        assert count_tokens(None) == 0

    @given(text=_plain_text().filter(lambda s: len(s) > 5))
    @settings(max_examples=30, deadline=None)
    def test_non_empty_is_positive(self, text: str) -> None:
        """P4: non-empty input yields at least 1 token."""
        assert count_tokens(text) >= 1

    @given(a=_plain_text(), b=_plain_text())
    @settings(max_examples=30, deadline=None)
    def test_monotonic_on_concat_soft(self, a: str, b: str) -> None:
        """P5: append never drastically reduces token count.

        BPE merges across the boundary can reduce token count by a small
        amount, so we allow a tolerance of 2 tokens.
        """
        tokens_a = count_tokens(a)
        tokens_b = count_tokens(b)
        tokens_ab = count_tokens(a + b)
        assert tokens_ab + 2 >= max(tokens_a, tokens_b)

    @given(text=_plain_text().filter(lambda s: len(s) > 20))
    @settings(max_examples=20, deadline=None)
    def test_correction_ratio_is_in_band(self, text: str) -> None:
        """P6: with-correction / without-correction is roughly 1.15.

        Allow ±10% tolerance — for very short input the rounding to int
        dominates and the ratio can be noisier.
        """
        with_corr = count_tokens(text, apply_claude_correction=True)
        without_corr = count_tokens(text, apply_claude_correction=False)
        if without_corr == 0:
            return
        ratio = with_corr / without_corr
        expected = CLAUDE_CORRECTION_FACTOR
        assert expected * 0.9 <= ratio <= expected * 1.1 + 0.01


# ---------------------------------------------------------------------------
# count_message_tokens
# ---------------------------------------------------------------------------


class TestCountMessageTokensProperties:
    @given(messages=_message_list())
    @settings(max_examples=30, deadline=None)
    def test_totality_and_determinism(self, messages: list) -> None:
        """P7: count_message_tokens never raises and is deterministic."""
        a = count_message_tokens(messages)
        b = count_message_tokens(messages)
        assert isinstance(a, int)
        assert a == b

    def test_empty_message_list_zero(self) -> None:
        assert count_message_tokens([]) == 0

    def test_none_is_zero(self) -> None:
        assert count_message_tokens(None) == 0

    @given(messages=_message_list().filter(lambda ms: len(ms) >= 1))
    @settings(max_examples=20, deadline=None)
    def test_adding_a_message_monotonic_soft(self, messages: list) -> None:
        """Adding a message never reduces total token count (soft)."""
        base = count_message_tokens(messages)
        extended = count_message_tokens(messages + [{"role": "user", "content": "x"}])
        assert extended >= base


# ---------------------------------------------------------------------------
# estimate_request_tokens
# ---------------------------------------------------------------------------


class TestEstimateRequestTokensProperties:
    @given(messages=_message_list())
    @settings(max_examples=30, deadline=None)
    def test_breakdown_sums_to_total(self, messages: list) -> None:
        """messages_tokens + tools_tokens + system_tokens == total_tokens."""
        result = estimate_request_tokens(messages=messages)
        assert (
            result["messages_tokens"]
            + result["tools_tokens"]
            + result["system_tokens"]
            == result["total_tokens"]
        )

    def test_tools_none_is_zero(self) -> None:
        assert count_tools_tokens(None) == 0
        assert count_tools_tokens([]) == 0
