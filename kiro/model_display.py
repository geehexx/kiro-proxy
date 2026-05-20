
"""Human-readable presentation of Claude model identifiers.

Claude Code's /model picker and the ``model`` field returned to clients
benefit from a stable, readable label. Internal Kiro IDs use mixed
formats ("claude-opus-4-7", "claude-opus-4.7", "CLAUDE_3_7_SONNET_
20250219_V1_0"), so this module provides one canonical presentation
shape.

Two surfaces use the helpers here:

1. ``/v1/models`` — every entry gets a ``display_name`` suitable for a
   picker ("Claude Opus 4.7").
2. Streaming + non-streaming Anthropic responses — the ``model`` field
   is normalised to the dotted form ("claude-opus-4.7") so clients see a
   consistent identifier regardless of which alias they requested.

The functions are pure (no I/O, no imports from the rest of ``kiro/``)
so they are safe to unit-test in isolation and to call from hot paths
without worrying about side effects.
"""

from __future__ import annotations

import re
from typing import Optional

_DATE_SUFFIX_RE = re.compile(r"-\d{8}(-v\d+[_\-.]?\d*)?$")
_TRAILING_VERSION_RE = re.compile(r"-v\d+[_\-.]?\d*$", re.IGNORECASE)

# Map family → human-readable title
_FAMILY_TITLES: dict[str, str] = {
    "opus": "Opus",
    "sonnet": "Sonnet",
    "haiku": "Haiku",
}

# Non-Claude models: hand-picked display names for the subset we route.
_NON_CLAUDE_DISPLAY: dict[str, str] = {
    "auto-kiro": "Auto (Kiro)",
    "deepseek-3.2": "DeepSeek 3.2",
    "glm-5": "GLM 5",
    "minimax-m2.1": "MiniMax M2.1",
    "minimax-m2.5": "MiniMax M2.5",
    "qwen3-coder-next": "Qwen3 Coder (Next)",
}


def canonical_model_id(name: str) -> str:
    """Return the canonical dotted form of a Claude model identifier.

    Args:
        name: Raw model identifier string (any format).

    Returns:
        Canonical dotted form, e.g. ``"claude-opus-4.7"``.
        Returns ``""`` for empty or whitespace-only input.
        Non-Claude identifiers are returned unchanged.

    Examples:
        >>> canonical_model_id("claude-opus-4-7")
        'claude-opus-4.7'
        >>> canonical_model_id("claude-sonnet-4-5-20250929")
        'claude-sonnet-4.5'
        >>> canonical_model_id("claude-haiku-4-5-latest")
        'claude-haiku-4.5'
        >>> canonical_model_id("claude-opus-4.7")
        'claude-opus-4.7'
        >>> canonical_model_id("")
        ''
    """
    if not name:
        return ""

    raw = name.strip()

    # Pass through non-Claude identifiers unchanged.
    if not raw.lower().startswith("claude-"):
        return raw

    # Strip dated suffix (e.g. ``-20250929`` or ``-20250929-v1-0``).
    raw = _DATE_SUFFIX_RE.sub("", raw)

    # Strip explicit trailing ``-latest`` marker.
    if raw.endswith("-latest"):
        raw = raw[: -len("-latest")]

    # Strip trailing ``-vN`` version marker.
    raw = _TRAILING_VERSION_RE.sub("", raw)

    # Convert trailing ``-major-minor`` to ``-major.minor`` for Claude
    # family aliases (e.g. ``claude-opus-4-7`` → ``claude-opus-4.7``).
    # Match greedily from the right so we do not disturb legacy forms
    # like ``claude-3-7-sonnet`` which map to ``claude-3.7-sonnet``.
    # Pattern: "-<digit>-<digit>" at end of string.
    trailing = re.search(r"-(\d+)-(\d+)$", raw)
    if trailing:
        major, minor = trailing.group(1), trailing.group(2)
        raw = raw[: trailing.start()] + f"-{major}.{minor}"

    # Legacy shape: "claude-3-7-sonnet" → "claude-3.7-sonnet".
    legacy = re.match(r"^claude-(\d+)-(\d+)-(opus|sonnet|haiku)(.*)$", raw)
    if legacy:
        raw = f"claude-{legacy.group(1)}.{legacy.group(2)}-{legacy.group(3)}{legacy.group(4)}"

    return raw


def display_name(model_id: str) -> str:
    """Build a human-readable label for a model picker.

    Args:
        model_id: Raw or canonical model identifier.

    Returns:
        Human-readable label, e.g. ``"Claude Opus 4.7"``.
        Returns ``""`` for empty input.

    Examples:
        >>> display_name("claude-opus-4.7")
        'Claude Opus 4.7'
        >>> display_name("auto-kiro")
        'Auto (Kiro)'
        >>> display_name("")
        ''
    """
    if not model_id:
        return ""

    canonical = canonical_model_id(model_id)

    # Non-Claude models consult the curated map.
    if not canonical.lower().startswith("claude-"):
        return _NON_CLAUDE_DISPLAY.get(canonical.lower(), canonical)

    # Claude models: parse into ``claude-{family}-{version}`` or legacy
    # ``claude-{version}-{family}`` shape.
    parts = canonical.split("-")
    if len(parts) < 3:
        return canonical  # Unknown shape; return as-is.

    # Shape: claude-{family}-{version[.minor]}
    if parts[1] in _FAMILY_TITLES:
        family = _FAMILY_TITLES[parts[1]]
        version = "-".join(parts[2:])
        return f"Claude {family} {version}"

    # Legacy shape: claude-{version}-{family}...
    if parts[-1] in _FAMILY_TITLES:
        family = _FAMILY_TITLES[parts[-1]]
        version = "-".join(parts[1:-1])
        return f"Claude {version} {family}"

    return canonical


def describe_model(model_id: str) -> Optional[str]:
    """Return a short one-line description for a model, suitable for a picker tooltip.

    Args:
        model_id: Raw or canonical model identifier.

    Returns:
        One-line description string, or ``None`` when no curated description exists.
    """
    canonical = canonical_model_id(model_id).lower()

    # Per-version Claude descriptions, more specific than family defaults.
    _PER_VERSION: dict[str, str] = {
        "claude-opus-4.7": "Highest reasoning; 1M context; long-running agents and complex coding.",
        "claude-opus-4.6": "Highest reasoning; 200K context; previous Opus generation.",
        "claude-opus-4.5": "Highest reasoning; 200K context; legacy Opus.",
        "claude-sonnet-4.6": "Balanced speed and intelligence; near-Opus coding; production default.",
        "claude-sonnet-4.5": "Balanced speed and intelligence; previous Sonnet generation.",
        "claude-sonnet-4": "Balanced speed and intelligence; legacy Sonnet 4.",
        "claude-haiku-4.5": "Fastest model with near-frontier intelligence; cheap tool-heavy tasks.",
        "claude-3.7-sonnet": "Legacy Sonnet generation (Claude 3.7); use only for compat.",
        "claude-3.5-haiku": "Legacy Haiku generation (Claude 3.5); use only for compat.",
    }
    if canonical in _PER_VERSION:
        return _PER_VERSION[canonical]

    # Family-level fallback for versions we haven't curated individually.
    if canonical.startswith("claude-opus"):
        return "Highest reasoning depth; long-running agents and coding."
    if canonical.startswith("claude-sonnet"):
        return "Balanced speed and intelligence; general production use."
    if canonical.startswith("claude-haiku"):
        return "Fastest model with near-frontier intelligence."
    if canonical == "auto-kiro":
        return "Automatic model selection via Kiro."
    return None
