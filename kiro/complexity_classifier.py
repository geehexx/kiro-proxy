"""Request complexity classifier for adaptive RE2 + thinking budget injection.

3-layer architecture:
- Layer 1: Deterministic short-circuits (haiku, sub-agent, tool_result-only, slash commands)
- Layer 2: Structural heuristics (token count, code blocks, question density, keywords)
- Layer 3: ONNX MiniLM-L6 embedding similarity (future, not yet implemented)

Returns a ComplexityLabel: SKIP | SIMPLE | MEDIUM | COMPLEX
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ComplexityLabel(str, Enum):
    """Classification of request complexity for RE2 and thinking budget injection."""

    SKIP = "skip"       # Don't inject RE2 or thinking
    SIMPLE = "simple"   # No thinking, maybe RE2
    MEDIUM = "medium"   # Low thinking budget, RE2
    COMPLEX = "complex" # High thinking budget, RE2


@dataclass
class ComplexityResult:
    """Output of classify_request() — label plus supporting metadata for debugging."""

    label: ComplexityLabel
    score: float          # 0.0-1.0 normalized score
    reason: str           # Human-readable reason for debugging
    thinking_budget: int  # Suggested thinking budget in tokens (0 = disabled)
    re2_eligible: bool    # Whether RE2 should be applied


# Keyword lists for Layer 2
_REASONING_KEYWORDS = frozenset([
    "debug", "why", "explain", "analyse", "analyze", "architect", "design",
    "compare", "tradeoff", "trade-off", "implement", "refactor", "optimize",
    "investigate", "diagnose", "root cause", "how does", "how would",
    "what if", "should i", "best way", "approach", "strategy",
])

_LOOKUP_KEYWORDS = frozenset([
    "what is", "what are", "define", "list", "show me", "how do i",
    "where is", "when did", "who is", "which", "example of",
])

_SLASH_COMMAND_RE = re.compile(r"^/\w")
_CODE_BLOCK_RE = re.compile(r"```")
_QUESTION_RE = re.compile(r"\?")


def classify_request(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    model: str,
    messages: list[dict[str, Any]],
    thinking: Optional[dict[str, Any]] = None,
    tool_choice: Optional[Any] = None,
    is_subagent: bool = False,
    input_tokens_estimate: Optional[int] = None,
) -> ComplexityResult:
    """Classify request complexity for adaptive RE2 + thinking budget injection.

    Args:
        model: Model ID (e.g. "claude-sonnet-4.6")
        messages: Full message list
        thinking: Thinking config from request
        tool_choice: Tool choice config
        is_subagent: Whether this is a sub-agent request
        input_tokens_estimate: Estimated input token count (optional)

    Returns:
        ComplexityResult with label, score, reason, thinking_budget, re2_eligible
    """
    # =========================================================================
    # Layer 1: Deterministic short-circuits
    # =========================================================================

    # Haiku: never inject (used for tool calls, not reasoning)
    if "haiku" in model.lower():
        return ComplexityResult(
            label=ComplexityLabel.SKIP,
            score=0.0,
            reason="haiku model — skip all injection",
            thinking_budget=0,
            re2_eligible=False,
        )

    # Sub-agent: never inject (bounded scope)
    if is_subagent:
        return ComplexityResult(
            label=ComplexityLabel.SKIP,
            score=0.0,
            reason="sub-agent request — skip all injection",
            thinking_budget=0,
            re2_eligible=False,
        )

    # Forced tool call: skip thinking (model must call a tool)
    if tool_choice and isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type in ("any", "tool"):
            return ComplexityResult(
                label=ComplexityLabel.SKIP,
                score=0.0,
                reason=f"tool_choice={tc_type} — skip thinking injection",
                thinking_budget=0,
                re2_eligible=False,
            )

    # Too few messages: single-turn polling
    if len(messages) < 2:
        return ComplexityResult(
            label=ComplexityLabel.SIMPLE,
            score=0.1,
            reason="single-message request — simple",
            thinking_budget=0,
            re2_eligible=False,
        )

    # Find last user message with text content
    last_user_text = ""
    last_user_has_text = False
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_text = content
                last_user_has_text = bool(content.strip())
                break
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_user_text = block.get("text", "")
                        last_user_has_text = bool(last_user_text.strip())
                        break
                if last_user_has_text:
                    break

    # Tool-result-only last message: skip RE2 (no text to re-read)
    if not last_user_has_text:
        # Still might need thinking for complex tool results
        # but RE2 is not applicable
        return ComplexityResult(
            label=ComplexityLabel.MEDIUM,
            score=0.4,
            reason="last user message is tool_result-only — RE2 skipped, thinking allowed",
            thinking_budget=2000,
            re2_eligible=False,
        )

    # Slash command: skip injection
    if _SLASH_COMMAND_RE.match(last_user_text.strip()):
        return ComplexityResult(
            label=ComplexityLabel.SKIP,
            score=0.0,
            reason="slash command — skip all injection",
            thinking_budget=0,
            re2_eligible=False,
        )

    # =========================================================================
    # Layer 2: Structural heuristics
    # =========================================================================

    score = 0.0
    reasons = []

    # Token count heuristic (n1n.ai 200-Token Rule)
    tokens = input_tokens_estimate or (len(last_user_text.split()) * 1.3)
    if tokens < 200:
        score -= 0.2
        reasons.append(f"short input ({tokens:.0f} tokens)")
    elif tokens > 4000:
        score += 0.3
        reasons.append(f"long input ({tokens:.0f} tokens)")
    elif tokens > 2000:
        score += 0.15
        reasons.append(f"medium-long input ({tokens:.0f} tokens)")

    # Long conversation = more context = more complex
    if len(messages) >= 20:
        score += 0.2
        reasons.append(f"long conversation ({len(messages)} messages)")
    elif len(messages) >= 10:
        score += 0.1
        reasons.append(f"medium conversation ({len(messages)} messages)")

    # Code blocks
    if _CODE_BLOCK_RE.search(last_user_text):
        score += 0.2
        reasons.append("code block present")

    # Question density
    question_count = len(_QUESTION_RE.findall(last_user_text))
    if question_count >= 3:
        score += 0.2
        reasons.append(f"high question density ({question_count} ?)")
    elif question_count >= 2:
        score += 0.1
        reasons.append(f"multiple questions ({question_count} ?)")

    # Reasoning keywords
    lower_text = last_user_text.lower()
    reasoning_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower_text)
    if reasoning_hits >= 2:
        score += 0.3
        reasons.append(f"reasoning keywords ({reasoning_hits} hits)")
    elif reasoning_hits == 1:
        score += 0.15
        reasons.append("reasoning keyword (1 hit)")

    # Lookup keywords (reduce complexity)
    lookup_hits = sum(1 for kw in _LOOKUP_KEYWORDS if lower_text.startswith(kw))
    if lookup_hits >= 1:
        score -= 0.2
        reasons.append("lookup pattern detected")

    # Clamp score
    score = max(0.0, min(1.0, score + 0.3))  # base offset of 0.3

    # Long conversations are inherently complex regardless of last message length
    if len(messages) >= 20 and score < 0.35:
        score = 0.35  # bump to MEDIUM floor for long conversations

    # Map score to label + thinking budget
    if score < 0.35:
        label = ComplexityLabel.SIMPLE
        thinking_budget = 0
        re2_eligible = len(messages) >= 10  # RE2 for long conversations even if simple
    elif score < 0.65:
        label = ComplexityLabel.MEDIUM
        thinking_budget = 2000
        re2_eligible = True
    else:
        label = ComplexityLabel.COMPLEX
        thinking_budget = 6000
        re2_eligible = True

    reason = "; ".join(reasons) if reasons else "default"
    return ComplexityResult(
        label=label,
        score=score,
        reason=reason,
        thinking_budget=thinking_budget,
        re2_eligible=re2_eligible,
    )
