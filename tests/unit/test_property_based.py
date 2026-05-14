"""Property-based tests for response_cache and complexity_classifier."""
from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiro.complexity_classifier import ComplexityLabel, classify_request
from kiro.response_cache import _normalize_text, make_key

_text = st.text(alphabet=string.printable, min_size=0, max_size=200)
_model = st.sampled_from(["claude-sonnet-4.6", "claude-opus-4.7", "claude-haiku-4.5-20251001", "claude-sonnet-4-5"])
_role = st.sampled_from(["user", "assistant"])
_simple_message = st.fixed_dictionaries({"role": _role, "content": _text})
_messages = st.lists(_simple_message, min_size=1, max_size=10)
_message_content = st.one_of(_text, st.lists(st.fixed_dictionaries({"type": st.just("text"), "text": _text}), min_size=1, max_size=3))
_classify_message = st.fixed_dictionaries({"role": _role, "content": _message_content})
_classify_messages = st.lists(_classify_message, min_size=1, max_size=20)


class TestMakeKeyProperties:
    @given(session_id=st.text(min_size=1, max_size=50), system=st.one_of(st.none(), _text), messages=_messages, model=_model, max_tokens=st.integers(min_value=1, max_value=8192))
    @settings(max_examples=100)
    def test_deterministic(self, session_id, system, messages, model, max_tokens):
        k1 = make_key(session_id=session_id, system=system, messages=messages, model=model, max_tokens=max_tokens)
        k2 = make_key(session_id=session_id, system=system, messages=messages, model=model, max_tokens=max_tokens)
        assert k1 == k2

    @given(system=_text)
    @settings(max_examples=200)
    def test_normalize_text_idempotent(self, system):
        once = _normalize_text(system)
        twice = _normalize_text(once)
        assert once == twice

    @given(system=_text)
    @settings(max_examples=200)
    def test_normalize_text_no_leading_trailing_whitespace(self, system):
        result = _normalize_text(system)
        if isinstance(result, str):
            assert result == result.strip()

    @given(system=_text)
    @settings(max_examples=200)
    def test_normalize_text_no_consecutive_spaces(self, system):
        result = _normalize_text(system)
        if isinstance(result, str):
            assert "  " not in result


class TestClassifyRequestProperties:
    @given(model=_model, messages=_classify_messages)
    @settings(max_examples=100)
    def test_never_raises(self, model, messages):
        try:
            result = classify_request(model=model, messages=messages)
            assert result is not None
        except Exception as e:
            pytest.fail(f"classify_request raised {type(e).__name__}: {e}")

    @given(model=_model, messages=_classify_messages)
    @settings(max_examples=100)
    def test_score_in_range(self, model, messages):
        result = classify_request(model=model, messages=messages)
        assert 0.0 <= result.score <= 1.0

    @given(model=_model, messages=_classify_messages)
    @settings(max_examples=100)
    def test_label_is_valid_enum(self, model, messages):
        result = classify_request(model=model, messages=messages)
        assert result.label in ComplexityLabel

    @given(model=_model, messages=_classify_messages)
    @settings(max_examples=100)
    def test_thinking_budget_non_negative(self, model, messages):
        result = classify_request(model=model, messages=messages)
        assert result.thinking_budget >= 0

    @given(model=st.sampled_from(["claude-haiku-4.5-20251001"]), messages=_classify_messages)
    @settings(max_examples=50)
    def test_haiku_always_skip(self, model, messages):
        result = classify_request(model=model, messages=messages)
        assert result.label == ComplexityLabel.SKIP

    @given(model=_model, messages=_classify_messages)
    @settings(max_examples=100)
    def test_deterministic(self, model, messages):
        r1 = classify_request(model=model, messages=messages)
        r2 = classify_request(model=model, messages=messages)
        assert r1.label == r2.label
        assert r1.score == r2.score
        assert r1.re2_eligible == r2.re2_eligible
