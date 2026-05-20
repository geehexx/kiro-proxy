"""Property-based tests for kiro.parsers and kiro.model_resolver.

Covers pure functions that are ideal hypothesis targets:
- find_matching_brace: structural parser — round-trip + invariants
- parse_bracket_tool_calls: never-raises + output shape
- normalize_model_name: idempotency + output invariants
"""
from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiro.model_resolver import normalize_model_name
from kiro.parsers import find_matching_brace, parse_bracket_tool_calls

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_chars = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"),
    whitelist_characters=" _-.",
)
_safe_text = st.text(alphabet=_safe_chars, min_size=0, max_size=80)

_json_value = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        _safe_text,
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_safe_text.filter(bool), children, max_size=4),
    ),
    max_leaves=10,
)

_family = st.sampled_from(["haiku", "sonnet", "opus"])
_major = st.integers(min_value=1, max_value=9)
_minor = st.integers(min_value=0, max_value=9)


# ---------------------------------------------------------------------------
# find_matching_brace — structural invariants
# ---------------------------------------------------------------------------

class TestFindMatchingBraceProperties:

    @given(value=st.dictionaries(_safe_text.filter(bool), _safe_text, min_size=1, max_size=5))
    @settings(max_examples=200)
    def test_round_trip_valid_json_dict(self, value):
        """For any dict, find_matching_brace must return the index of the final '}'."""
        text = json.dumps(value)
        end = find_matching_brace(text, 0)
        assert end == len(text) - 1, (
            f"Expected end={len(text)-1}, got {end} for text={text!r}"
        )

    @given(value=st.dictionaries(_safe_text.filter(bool), _safe_text, min_size=1, max_size=5))
    @settings(max_examples=200)
    def test_result_is_closing_brace(self, value):
        """The character at the returned index is always '}'."""
        text = json.dumps(value)
        end = find_matching_brace(text, 0)
        assert end >= 0
        assert text[end] == "}"

    @given(
        prefix=_safe_text,
        value=st.dictionaries(_safe_text.filter(bool), _safe_text, min_size=1, max_size=3),
        suffix=_safe_text,
    )
    @settings(max_examples=200)
    def test_embedded_object_extracted_correctly(self, prefix, value, suffix):
        """find_matching_brace works when the object is embedded in surrounding text."""
        inner = json.dumps(value)
        text = prefix + inner + suffix
        start = len(prefix)
        if start >= len(text) or text[start] != "{":
            return
        end = find_matching_brace(text, start)
        assert end >= start
        assert text[end] == "}"
        extracted = text[start:end + 1]
        assert json.loads(extracted) == value

    @given(text=st.text(min_size=1, max_size=200))
    @settings(max_examples=200)
    def test_never_raises(self, text):
        """find_matching_brace must never raise regardless of input."""
        try:
            result = find_matching_brace(text, 0)
            assert isinstance(result, int)
        except Exception as exc:
            pytest.fail(f"find_matching_brace raised {type(exc).__name__}: {exc}")

    @given(value=st.dictionaries(_safe_text.filter(bool), _safe_text, min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_nested_braces_in_string_values_not_confused(self, value):
        """String values containing braces must not confuse the brace counter."""
        braced_value = {k: "{" + v + "}" for k, v in value.items()}
        text = json.dumps(braced_value)
        end = find_matching_brace(text, 0)
        assert end == len(text) - 1

    def test_non_brace_start_returns_minus_one(self):
        assert find_matching_brace("hello", 0) == -1

    def test_empty_string_returns_minus_one(self):
        assert find_matching_brace("", 0) == -1

    def test_start_beyond_length_returns_minus_one(self):
        assert find_matching_brace("{}", 99) == -1


# ---------------------------------------------------------------------------
# parse_bracket_tool_calls — never-raises + output shape
# ---------------------------------------------------------------------------

class TestParseBracketToolCallsProperties:

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_never_raises(self, text):
        """parse_bracket_tool_calls must never raise on arbitrary text."""
        try:
            result = parse_bracket_tool_calls(text)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"parse_bracket_tool_calls raised {type(exc).__name__}: {exc}")

    @given(text=st.text(min_size=0, max_size=500).filter(lambda t: "[Called" not in t))
    @settings(max_examples=100)
    def test_no_called_marker_returns_empty(self, text):
        """Text without '[Called' must always return an empty list."""
        assert parse_bracket_tool_calls(text) == []

    @given(
        func_name=st.from_regex(r"[a-z][a-z_]{0,19}", fullmatch=True),
        args=st.dictionaries(
            st.from_regex(r"[a-z][a-z_]{0,9}", fullmatch=True),
            st.one_of(st.integers(), _safe_text),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100)
    def test_well_formed_call_parsed_correctly(self, func_name, args):
        """A well-formed [Called func with args: {...}] string is parsed."""
        text = f"[Called {func_name} with args: {json.dumps(args)}]"
        result = parse_bracket_tool_calls(text)
        assert len(result) == 1
        call = result[0]
        assert call["type"] == "function"
        assert call["function"]["name"] == func_name
        assert json.loads(call["function"]["arguments"]) == args

    @given(
        calls=st.lists(
            st.tuples(
                st.from_regex(r"[a-z][a-z_]{0,9}", fullmatch=True),
                st.dictionaries(
                    st.from_regex(r"[a-z][a-z_]{0,9}", fullmatch=True),
                    st.integers(),
                    min_size=1,
                    max_size=3,
                ),
            ),
            min_size=2,
            max_size=4,
        )
    )
    @settings(max_examples=50)
    def test_multiple_calls_all_parsed(self, calls):
        """Multiple [Called ...] blocks in one string are all extracted."""
        text = " ".join(
            f"[Called {name} with args: {json.dumps(args)}]"
            for name, args in calls
        )
        result = parse_bracket_tool_calls(text)
        assert len(result) == len(calls)
        for call, (name, args) in zip(result, calls, strict=True):
            assert call["function"]["name"] == name
            assert json.loads(call["function"]["arguments"]) == args

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_output_items_have_required_keys(self, text):
        """Every item in the output list has the required OpenAI tool-call keys."""
        result = parse_bracket_tool_calls(text)
        for item in result:
            assert "id" in item
            assert "type" in item
            assert "function" in item
            assert "name" in item["function"]
            assert "arguments" in item["function"]


# ---------------------------------------------------------------------------
# normalize_model_name — idempotency + output invariants
# ---------------------------------------------------------------------------

class TestNormalizeModelNameProperties:

    @given(name=st.text(min_size=1, max_size=80))
    @settings(max_examples=300)
    def test_never_raises(self, name):
        """normalize_model_name must never raise on arbitrary non-empty input."""
        try:
            result = normalize_model_name(name)
            assert isinstance(result, str)
        except Exception as exc:
            pytest.fail(f"normalize_model_name raised {type(exc).__name__}: {exc}")

    @given(name=st.text(min_size=1, max_size=80))
    @settings(max_examples=300)
    def test_idempotent(self, name):
        """Normalizing twice produces the same result as normalizing once."""
        once = normalize_model_name(name)
        twice = normalize_model_name(once)
        assert once == twice, (
            f"Not idempotent: normalize({name!r})={once!r}, "
            f"normalize({once!r})={twice!r}"
        )

    @given(
        base=st.from_regex(r"claude-[a-z]+-[0-9]+\.[0-9]+", fullmatch=True),
        suffix=st.sampled_from(["[1m]", "[200k]", "[thinking]"]),
    )
    @settings(max_examples=100)
    def test_complete_bracket_suffix_stripped(self, base, suffix):
        """Complete [x] bracket suffixes are stripped from known claude names."""
        name = base + suffix
        result = normalize_model_name(name)
        assert "[" not in result and "]" not in result, (
            f"Bracket suffix in output: normalize({name!r})={result!r}"
        )

    @given(family=_family, major=_major, minor=_minor)
    @settings(max_examples=100)
    def test_standard_dash_format_normalizes_to_dot(self, family, major, minor):
        """claude-{family}-{major}-{minor} -> claude-{family}-{major}.{minor}"""
        name = f"claude-{family}-{major}-{minor}"
        result = normalize_model_name(name)
        assert "." in result, f"Expected dot in result for {name!r}, got {result!r}"
        assert f"{major}.{minor}" in result

    @given(family=_family, major=_major, minor=_minor)
    @settings(max_examples=100)
    def test_date_suffix_stripped(self, family, major, minor):
        """Date suffix (8 digits) is always stripped from the output."""
        name = f"claude-{family}-{major}-{minor}-20251001"
        result = normalize_model_name(name)
        assert "20251001" not in result

    @given(
        family=_family,
        major=_major,
        minor=_minor,
        suffix=st.sampled_from(["[1m]", "[200k]", "[thinking]"]),
    )
    @settings(max_examples=100)
    def test_bracket_suffix_stripped(self, family, major, minor, suffix):
        """Bracket suffixes like [1m] are stripped before other processing."""
        name = f"claude-{family}-{major}.{minor}{suffix}"
        result = normalize_model_name(name)
        assert "[" not in result
        assert "]" not in result

    @given(family=_family, major=_major, minor=_minor)
    @settings(max_examples=100)
    def test_known_claude_patterns_are_lowercase(self, family, major, minor):
        """Matched claude patterns are always returned in lowercase."""
        name = f"claude-{family}-{major}-{minor}"
        result = normalize_model_name(name)
        assert result == result.lower(), (
            f"Output not lowercase: normalize({name!r})={result!r}"
        )

    def test_empty_string_returns_empty(self):
        assert normalize_model_name("") == ""

    def test_passthrough_for_unknown_names(self):
        """Unknown model names pass through unchanged (lowercased)."""
        assert normalize_model_name("gpt-4o") == "gpt-4o"
        assert normalize_model_name("auto") == "auto"
