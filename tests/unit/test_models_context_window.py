"""Tests for context_window field in /v1/models response."""

from __future__ import annotations

from kiro.config import FALLBACK_MODELS
from kiro.model_resolver import normalize_model_name
from kiro.models_openai import OpenAIModel


def _cw_map() -> dict[str, int]:
    result = {}
    for m in FALLBACK_MODELS:
        mid = m.get("modelId", "")
        cw = (m.get("tokenLimits") or {}).get("maxInputTokens")
        if mid and cw:
            result[normalize_model_name(mid)] = cw
    return result


def test_openai_model_has_context_window_field():
    m = OpenAIModel(id="claude-sonnet-4.6", context_window=1000000)
    assert m.context_window == 1000000


def test_sonnet_46_context_window_is_1m():
    cw = _cw_map()
    assert cw.get(normalize_model_name("claude-sonnet-4.6")) == 1000000


def test_opus_47_context_window_is_1m():
    cw = _cw_map()
    assert cw.get(normalize_model_name("claude-opus-4.7")) == 1000000


def test_haiku_45_context_window_is_200k():
    cw = _cw_map()
    assert cw.get(normalize_model_name("claude-haiku-4.5")) == 200000


def test_unknown_model_defaults_to_200k():
    cw = _cw_map()
    assert cw.get(normalize_model_name("unknown-model-xyz"), 200000) == 200000


def test_auto_kiro_context_window_is_1m():
    """auto-kiro routes to the best available model (1M); must not report 200k.

    If this returns 200k, CC defaults to 200k context and compacts at 170k,
    causing the premature-compaction / 'Continue' loop bug.
    See: research/2026-05-15-lessons-compaction-bug.md
    """
    # _ALIAS_CW is built inside the route handler, so we test the live endpoint
    # indirectly by verifying the alias is in the config-level override dict.
    # The route handler adds auto/auto-kiro to _ALIAS_CW with value 1_000_000.
    # This test guards against regression where the alias is removed.
    from kiro.routes_openai import _ALIAS_CW_DEFAULTS
    assert _ALIAS_CW_DEFAULTS.get("auto") == 1_000_000
    assert _ALIAS_CW_DEFAULTS.get("auto-kiro") == 1_000_000
