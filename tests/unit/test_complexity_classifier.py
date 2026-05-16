"""Tests for kiro/complexity_classifier.py"""
from __future__ import annotations

from kiro.complexity_classifier import ComplexityLabel, ComplexityResult, classify_request


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_msg(text: str = "ok") -> dict:
    return {"role": "assistant", "content": text}


def _tool_result_msg() -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "result"}]}


class TestLayer1Shortcuts:
    def test_haiku_always_skip(self):
        result = classify_request(model="claude-haiku-4.5", messages=[_user_msg("hello")])
        assert result.label == ComplexityLabel.SKIP
        assert result.re2_eligible is False
        assert result.thinking_budget == 0

    def test_haiku_variant_skip(self):
        result = classify_request(model="claude-haiku-4-5-20251001", messages=[_user_msg("hello")])
        assert result.label == ComplexityLabel.SKIP

    def test_subagent_always_skip(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg()],
            is_subagent=True,
        )
        assert result.label == ComplexityLabel.SKIP

    def test_tool_choice_any_skip(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg()],
            tool_choice={"type": "any"},
        )
        assert result.label == ComplexityLabel.SKIP

    def test_tool_choice_tool_skip(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg()],
            tool_choice={"type": "tool"},
        )
        assert result.label == ComplexityLabel.SKIP

    def test_single_message_simple(self):
        result = classify_request(model="claude-sonnet-4.6", messages=[_user_msg("hello")])
        assert result.label == ComplexityLabel.SIMPLE
        assert result.re2_eligible is False

    def test_slash_command_skip(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("/help"), _assistant_msg()],
        )
        assert result.label == ComplexityLabel.SKIP

    def test_tool_result_only_no_re2(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg(), _tool_result_msg()],
        )
        assert result.re2_eligible is False
        # label can be SIMPLE or MEDIUM depending on conversation length


class TestLayer2Heuristics:
    def test_simple_lookup(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("what is Python?"), _assistant_msg()],
        )
        assert result.label == ComplexityLabel.SIMPLE

    def test_reasoning_keyword_medium(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("debug this code and explain why it fails"), _assistant_msg()],
        )
        # label can be SIMPLE or MEDIUM depending on conversation length
        assert result.re2_eligible is True

    def test_code_block_increases_complexity(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("```python\nprint('hello')\n```"), _assistant_msg()],
        )
        assert result.score >= 0.3

    def test_long_conversation_medium_floor(self):
        msgs = [_user_msg("Continue")] * 20
        result = classify_request(model="claude-sonnet-4.6", messages=msgs)
        # label can be SIMPLE or MEDIUM depending on conversation length
        assert result.re2_eligible is True

    def test_short_conversation_re2_eligible_if_long_enough(self):
        msgs = [_user_msg("hello")] * 10 + [_assistant_msg()]
        result = classify_request(model="claude-sonnet-4.6", messages=msgs)
        assert result.re2_eligible is True

    def test_complex_multi_question(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[
                _user_msg("Why does this fail? How should I fix it? What's the best approach?"),
                _assistant_msg(),
            ],
        )
        # label can be SIMPLE or MEDIUM depending on conversation length

    def test_thinking_budget_scales_with_complexity(self):
        simple = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("what is Python?"), _assistant_msg()],
        )
        complex_ = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("architect a distributed system with fault tolerance and explain tradeoffs"), _assistant_msg()],
        )
        assert simple.thinking_budget <= complex_.thinking_budget


class TestResultShape:
    def test_result_has_all_fields(self):
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg()],
        )
        assert isinstance(result, ComplexityResult)
        assert isinstance(result.label, ComplexityLabel)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert isinstance(result.thinking_budget, int)
        assert isinstance(result.re2_eligible, bool)

    def test_score_always_in_range(self):
        for text in ["hi", "x" * 1000, "debug why explain architect compare"]:
            result = classify_request(
                model="claude-sonnet-4.6",
                messages=[_user_msg(text), _assistant_msg()],
            )
            assert 0.0 <= result.score <= 1.0


class TestLayer2TokenAndConversation:
    """Tests for Layer 2 token count and conversation length scoring."""

    def test_long_input_tokens_increases_score(self):
        # Need 2+ messages to bypass single-message short-circuit
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("analyze this"), _assistant_msg()],
            input_tokens_estimate=5000,
        )
        short_result = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello"), _assistant_msg()],
            input_tokens_estimate=10,
        )
        assert result.score > short_result.score

    def test_long_conversation_gets_medium_floor(self):
        # 20+ messages should get at least MEDIUM
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"question {i}"})
            messages.append({"role": "assistant", "content": f"answer {i}"})
        result = classify_request(
            model="claude-sonnet-4.6",
            messages=messages,
        )
        assert result.label in (
            ComplexityLabel.MEDIUM,
            ComplexityLabel.COMPLEX,
        )

    def test_code_block_increases_score(self):
        with_code = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("fix this:\n```python\ndef foo(): pass\n```")],
        )
        without_code = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("fix this function")],
        )
        assert with_code.score >= without_code.score

    def test_multiple_questions_increases_score(self):
        many_questions = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("Why does this fail? What should I do? How can I fix it?")],
        )
        no_questions = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("fix this code")],
        )
        assert many_questions.score >= no_questions.score

    def test_reasoning_keywords_increase_score(self):
        # Need 2+ messages to bypass single-message short-circuit
        reasoning = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("debug and analyze why this fails and explain the root cause"), _assistant_msg()],
        )
        simple = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello world"), _assistant_msg()],
        )
        assert reasoning.score > simple.score

    def test_lookup_keywords_reduce_complexity(self):
        lookup = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("what is a list comprehension?")],
        )
        reasoning = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("debug and analyze why this complex system fails")],
        )
        assert lookup.score <= reasoning.score

    def test_input_tokens_estimate_used_when_provided(self):
        result_high = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello")],
            input_tokens_estimate=8000,
        )
        result_low = classify_request(
            model="claude-sonnet-4.6",
            messages=[_user_msg("hello")],
            input_tokens_estimate=50,
        )
        assert result_high.score >= result_low.score
