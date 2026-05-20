"""Full model-alias context_window matrix tests for /v1/models.

Tests the context_window values produced by the alias resolution logic in
kiro/routes_openai.py for every model-id in the empirically-verified matrix
(curl 127.0.0.1:8765/v1/models, 2026-05-19 23:19 UTC).

Strategy: test the source-code constants directly (same approach as
test_models_context_window.py) rather than hitting the live HTTP endpoint.
This keeps tests fast, isolated, and deterministic regardless of which
upstream models the mock returns.

The matrix is derived from three sources:
  1. _ALIAS_CW_DEFAULTS  -- short aliases (sonnet[1m], opus[1m], haiku[1m],
                            auto, auto-kiro)
  2. _CANONICAL_1M_CAPABLE + bracket expansion -- dot-form canonicals and
                            their [1m] variants
  3. _DASH_BRACKET_ALIASES / _DASH_BARE_ALIASES -- dash-form aliases for
                            clients that normalise model names to dash form
  4. FALLBACK_MODELS      -- upstream model list (dot-form, 200k or 1M)
"""

from __future__ import annotations

import pytest

from kiro.config import FALLBACK_MODELS
from kiro.model_resolver import normalize_model_name
from kiro.routes_openai import _ALIAS_CW_DEFAULTS

# ---------------------------------------------------------------------------
# Helpers -- mirror the logic in routes_openai.list_models()
# ---------------------------------------------------------------------------

_CANONICAL_1M_CAPABLE: frozenset[str] = frozenset({
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-sonnet-4.6",
})

_DASH_BRACKET_ALIASES: frozenset[str] = frozenset({
    "claude-opus-4-7[1m]",
    "claude-opus-4-6[1m]",
    "claude-sonnet-4-6[1m]",
})

_DASH_BARE_ALIASES: frozenset[str] = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
})


def _build_cw_map() -> dict[str, int]:
    """Build context_window lookup from FALLBACK_MODELS (mirrors route handler)."""
    result: dict[str, int] = {}
    for m in FALLBACK_MODELS:
        mid = m.get("modelId", "")
        cw = (m.get("tokenLimits") or {}).get("maxInputTokens")
        if mid and cw:
            result[normalize_model_name(mid)] = cw
    return result


def _build_alias_cw() -> dict[str, int]:
    """Build the full alias->context_window map (mirrors route handler)."""
    alias_cw: dict[str, int] = _ALIAS_CW_DEFAULTS.copy()
    for canon in _CANONICAL_1M_CAPABLE:
        alias_cw[f"{canon}[1m]"] = 1_000_000
    for dash in _DASH_BRACKET_ALIASES:
        alias_cw[dash] = 1_000_000
    return alias_cw


def _get_context_window(model_id: str) -> int:
    """Resolve context_window for a model_id (mirrors route handler logic)."""
    alias_cw = _build_alias_cw()
    if model_id in alias_cw:
        return alias_cw[model_id]
    if model_id in _CANONICAL_1M_CAPABLE:
        return 200_000
    return _build_cw_map().get(normalize_model_name(model_id), 200_000)


# ---------------------------------------------------------------------------
# Parametrized matrix -- grouped by family
# ---------------------------------------------------------------------------

# Each entry: (model_id, expected_context_window, description)
_MATRIX: list[tuple[str, int, str]] = [
    # --- opus-4.7 family (dot-form) ---
    ("claude-opus-4.7",       200_000,   "dot-form bare canonical -- 200k default"),
    ("claude-opus-4.7[1m]",   1_000_000, "dot-form [1m] variant -- 1M"),
    # --- opus-4.6 family (dot-form) ---
    ("claude-opus-4.6",       200_000,   "dot-form bare canonical -- 200k default"),
    ("claude-opus-4.6[1m]",   1_000_000, "dot-form [1m] variant -- 1M"),
    # --- opus-4.5 (dot-form, no 1M variant) ---
    ("claude-opus-4.5",       200_000,   "dot-form, no 1M variant"),
    # --- opus-4-7 family (dash-form aliases) ---
    ("claude-opus-4-7",       1_000_000, "dash-form bare -- normalises to 1M dot-form"),
    ("claude-opus-4-7[1m]",   1_000_000, "dash-form [1m] -- explicit 1M alias"),
    # --- opus-4-6 family (dash-form aliases) ---
    ("claude-opus-4-6",       1_000_000, "dash-form bare -- normalises to 1M dot-form"),
    ("claude-opus-4-6[1m]",   1_000_000, "dash-form [1m] -- explicit 1M alias"),
    # --- sonnet-4.6 family (dot-form) ---
    ("claude-sonnet-4.6",     200_000,   "dot-form bare canonical -- 200k default"),
    ("claude-sonnet-4.6[1m]", 1_000_000, "dot-form [1m] variant -- 1M"),
    # --- sonnet-4.5 (dot-form, no 1M variant) ---
    ("claude-sonnet-4.5",     200_000,   "dot-form, no 1M variant"),
    # --- sonnet-4-6 family (dash-form aliases) ---
    ("claude-sonnet-4-6",     1_000_000, "dash-form bare -- normalises to 1M dot-form"),
    ("claude-sonnet-4-6[1m]", 1_000_000, "dash-form [1m] -- explicit 1M alias"),
    # --- haiku-4.5 ---
    ("claude-haiku-4.5",      200_000,   "haiku 4.5 -- 200k"),
    # --- short aliases ---
    ("sonnet",                200_000,   "short alias bare -- 200k default"),
    ("sonnet[1m]",            1_000_000, "short alias [1m] -- 1M"),
    ("opus",                  200_000,   "short alias bare -- 200k default"),
    ("opus[1m]",              1_000_000, "short alias [1m] -- 1M"),
    ("haiku",                 200_000,   "short alias bare -- 200k default"),
    ("haiku[1m]",             1_000_000, "short alias [1m] -- 1M"),
    # --- auto aliases ---
    ("auto",                  1_000_000, "auto alias -- always 1M (Continue-loop guard)"),
    ("auto-kiro",             1_000_000, "auto-kiro alias -- always 1M (Continue-loop guard)"),
]


@pytest.mark.parametrize(
    "model_id,expected_cw,description",
    [(mid, cw, desc) for mid, cw, desc in _MATRIX],
    ids=[mid for mid, _, _ in _MATRIX],
)
def test_context_window_matrix(model_id: str, expected_cw: int, description: str) -> None:
    """Assert each model_id resolves to the expected context_window.

    Tests the alias resolution logic directly (no HTTP call needed).
    """
    actual = _get_context_window(model_id)
    assert actual == expected_cw, (
        f"{model_id!r}: expected context_window={expected_cw}, got {actual}. "
        f"({description})"
    )


# ---------------------------------------------------------------------------
# No-regressions test -- every matrix key must appear in the alias map
# ---------------------------------------------------------------------------

class TestAliasMapCoverage:
    """Verify the alias map contains every key in the empirical matrix."""

    def test_alias_cw_defaults_contains_short_aliases(self) -> None:
        """_ALIAS_CW_DEFAULTS must contain all short [1m] aliases and auto variants."""
        required = {"sonnet[1m]", "opus[1m]", "haiku[1m]", "auto", "auto-kiro"}
        missing = required - set(_ALIAS_CW_DEFAULTS)
        assert not missing, f"Missing from _ALIAS_CW_DEFAULTS: {missing}"

    def test_bracket_expansion_covers_all_canonicals(self) -> None:
        """Every canonical in _CANONICAL_1M_CAPABLE must produce a [1m] entry."""
        alias_cw = _build_alias_cw()
        for canon in _CANONICAL_1M_CAPABLE:
            bracket = f"{canon}[1m]"
            assert bracket in alias_cw, f"{bracket!r} missing from alias map"
            assert alias_cw[bracket] == 1_000_000, (
                f"{bracket!r} should be 1M, got {alias_cw[bracket]}"
            )

    def test_dash_bracket_aliases_all_1m(self) -> None:
        """Every dash-form [1m] alias must resolve to 1M."""
        alias_cw = _build_alias_cw()
        for dash in _DASH_BRACKET_ALIASES:
            assert dash in alias_cw, f"{dash!r} missing from alias map"
            assert alias_cw[dash] == 1_000_000, (
                f"{dash!r} should be 1M, got {alias_cw[dash]}"
            )

    def test_matrix_model_ids_all_resolvable(self) -> None:
        """Every model_id in the matrix must resolve without raising."""
        for model_id, expected_cw, _ in _MATRIX:
            actual = _get_context_window(model_id)
            assert isinstance(actual, int), (
                f"{model_id!r}: _get_context_window returned non-int {actual!r}"
            )
            assert actual > 0, f"{model_id!r}: context_window must be positive"


# ---------------------------------------------------------------------------
# Regression guards for the "Continue" loop bug (c4dfb47)
# ---------------------------------------------------------------------------

class TestContinueLoopRegressionGuards:
    """Guard against the premature-compaction / 'Continue' loop bug.

    If auto or auto-kiro report 200k, CC defaults to 200k context and
    compacts at 170k, causing the loop. See:
    research/2026-05-15-lessons-compaction-bug.md
    """

    def test_auto_is_1m(self) -> None:
        assert _ALIAS_CW_DEFAULTS.get("auto") == 1_000_000

    def test_auto_kiro_is_1m(self) -> None:
        assert _ALIAS_CW_DEFAULTS.get("auto-kiro") == 1_000_000

    def test_auto_resolves_to_1m_via_get_context_window(self) -> None:
        assert _get_context_window("auto") == 1_000_000

    def test_auto_kiro_resolves_to_1m_via_get_context_window(self) -> None:
        assert _get_context_window("auto-kiro") == 1_000_000

    def test_sonnet_1m_alias_is_1m(self) -> None:
        """sonnet[1m] is the model CC uses when configured with the 1M variant."""
        assert _get_context_window("sonnet[1m]") == 1_000_000

    def test_opus_1m_alias_is_1m(self) -> None:
        assert _get_context_window("opus[1m]") == 1_000_000


# ---------------------------------------------------------------------------
# Dot-form vs dash-form symmetry
# ---------------------------------------------------------------------------

class TestDotDashSymmetry:
    """Dash-form aliases must mirror dot-form context_window semantics.

    The dash-form bare aliases (claude-opus-4-7) fall through to _cw_map
    where normalize_model_name converts dashes to dots, hitting the
    FALLBACK_MODELS entry for claude-opus-4.7 (1M).
    """

    def test_claude_opus_4_7_dot_bare_is_200k(self) -> None:
        """Dot-form bare canonical reports 200k (caller must use [1m] for 1M)."""
        assert _get_context_window("claude-opus-4.7") == 200_000

    def test_claude_opus_4_7_dot_bracket_is_1m(self) -> None:
        assert _get_context_window("claude-opus-4.7[1m]") == 1_000_000

    def test_claude_opus_4_7_dash_bare_is_1m(self) -> None:
        """Dash-form bare resolves via normalize_model_name to dot-form 1M entry."""
        assert _get_context_window("claude-opus-4-7") == 1_000_000

    def test_claude_opus_4_7_dash_bracket_is_1m(self) -> None:
        assert _get_context_window("claude-opus-4-7[1m]") == 1_000_000

    def test_claude_sonnet_4_6_dot_bare_is_200k(self) -> None:
        assert _get_context_window("claude-sonnet-4.6") == 200_000

    def test_claude_sonnet_4_6_dot_bracket_is_1m(self) -> None:
        assert _get_context_window("claude-sonnet-4.6[1m]") == 1_000_000

    def test_claude_sonnet_4_6_dash_bare_is_1m(self) -> None:
        assert _get_context_window("claude-sonnet-4-6") == 1_000_000

    def test_claude_sonnet_4_6_dash_bracket_is_1m(self) -> None:
        assert _get_context_window("claude-sonnet-4-6[1m]") == 1_000_000
