"""Unit tests for the Opus 4.7 effort guard.

Background: per the Phase 0 wire test 2026-05-19
(basic-memory://research/2026-05-19-amf-phase-0-result), Opus 4.7 returns
429 INSUFFICIENT_MODEL_CAPACITY when `effort` is `low`/`medium`/`disabled`/
`adaptive` or unset. Only `high`/`xhigh`/`max` actually flow.

The guard force-upgrades effort to 'high' (cheapest working level) when
the proxy detects an Opus 4.7 request with a non-working effort. Other
models pass through unchanged.

See kiro.converters_core.apply_opus_effort_guard.
"""

from __future__ import annotations

import pytest

from kiro.converters_core import apply_opus_effort_guard


class TestOpusEffortGuardOpusModels:
    """Opus 4.7 — guard fires."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4.7",
            "claude-opus-4-7",
            "CLAUDE-OPUS-4.7",  # case-insensitive
        ],
    )
    def test_opus_with_no_amf_gets_effort_high(self, model_id: str) -> None:
        """No AMF at all → guard injects {output_config: {effort: high}}."""
        result = apply_opus_effort_guard(model_id, None)
        assert result == {"output_config": {"effort": "high"}}

    def test_opus_with_empty_amf_gets_effort_high(self) -> None:
        """Empty AMF dict → guard injects output_config.effort=high."""
        result = apply_opus_effort_guard("claude-opus-4.7", {})
        assert result == {"output_config": {"effort": "high"}}

    @pytest.mark.parametrize(
        "bad_effort",
        ["low", "medium", "disabled", "adaptive", ""],
    )
    def test_opus_with_bad_effort_upgraded_to_high(self, bad_effort: str) -> None:
        """effort=low/medium/disabled/adaptive/empty → upgraded to high."""
        amf = {"output_config": {"effort": bad_effort}}
        result = apply_opus_effort_guard("claude-opus-4.7", amf)
        assert result["output_config"]["effort"] == "high"

    @pytest.mark.parametrize(
        "good_effort",
        ["high", "xhigh", "max"],
    )
    def test_opus_with_good_effort_unchanged(self, good_effort: str) -> None:
        """effort=high/xhigh/max → guard is a no-op."""
        amf = {"output_config": {"effort": good_effort}}
        result = apply_opus_effort_guard("claude-opus-4.7", amf)
        assert result["output_config"]["effort"] == good_effort

    def test_opus_preserves_other_amf_keys(self) -> None:
        """Guard preserves unrelated AMF keys (thinking, display, etc.)."""
        amf = {
            "thinking": {"type": "adaptive"},
            "display": "omitted",
            "output_config": {"effort": "low"},
        }
        result = apply_opus_effort_guard("claude-opus-4.7", amf)
        assert result["thinking"] == {"type": "adaptive"}
        assert result["display"] == "omitted"
        assert result["output_config"]["effort"] == "high"

    def test_opus_does_not_mutate_input(self) -> None:
        """Guard never mutates the caller's dict."""
        amf = {"output_config": {"effort": "low"}}
        original_amf = {"output_config": {"effort": "low"}}
        apply_opus_effort_guard("claude-opus-4.7", amf)
        assert amf == original_amf  # caller's dict unchanged


class TestOpusEffortGuardOtherModels:
    """Sonnet/Haiku/other — guard is a no-op."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-sonnet-4.6",
            "claude-haiku-4.5",
            "claude-opus-4.5",  # earlier opus — not 4.7, no capacity quirk
            "claude-opus-4.6",
            "deepseek-3.2",
        ],
    )
    def test_non_opus_47_with_low_effort_unchanged(self, model_id: str) -> None:
        """Non-Opus-4.7 models keep their effort unchanged (even if 'low')."""
        amf = {"output_config": {"effort": "low"}}
        result = apply_opus_effort_guard(model_id, amf)
        assert result == {"output_config": {"effort": "low"}}

    def test_sonnet_with_no_amf_returns_empty(self) -> None:
        """Non-Opus + no AMF → empty dict (caller may skip emitting AMF)."""
        result = apply_opus_effort_guard("claude-sonnet-4.6", None)
        assert result == {}

    def test_empty_model_id_no_op(self) -> None:
        """Empty/None model_id — guard does nothing."""
        result = apply_opus_effort_guard("", {"output_config": {"effort": "low"}})
        assert result == {"output_config": {"effort": "low"}}
