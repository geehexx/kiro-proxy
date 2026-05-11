"""
Regression test for INT-73: RESPONSE_CACHE_* config symbol contract.

History
-------
Commit 51cd264 renamed ``PrefixCache`` → ``ResponseCache`` and renamed
config symbols from ``PREFIX_CACHE_*`` → ``RESPONSE_CACHE_*``, but the
import block in ``main.py`` was left pointing at the old names. The
branch failed to start whenever ``RESPONSE_CACHE_ENABLED=true`` because
``from kiro.config import PREFIX_CACHE_ENABLED`` raised ``ImportError``.

This test locks the invariant that:

1. The canonical names are ``RESPONSE_CACHE_*`` (six of them).
2. The legacy names ``PREFIX_CACHE_*`` do NOT exist (so we don't drift
   back).
3. ``main.py`` imports the canonical names.

If someone renames the config symbols again, this test catches both
halves (the rename and the stale consumer in ``main.py``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiro import config

CANONICAL = (
    "RESPONSE_CACHE_ENABLED",
    "RESPONSE_CACHE_MAX_ENTRIES",
    "RESPONSE_CACHE_MAX_BYTES",
    "RESPONSE_CACHE_TTL_SECONDS",
    "RESPONSE_CACHE_MAX_ENTRY_BYTES",
    "RESPONSE_CACHE_LOG_HITS",
)

LEGACY = (
    "PREFIX_CACHE_ENABLED",
    "PREFIX_CACHE_MAX_ENTRIES",
    "PREFIX_CACHE_MAX_BYTES",
    "PREFIX_CACHE_TTL_SECONDS",
    "PREFIX_CACHE_MAX_ENTRY_BYTES",
    "PREFIX_CACHE_LOG_HITS",
)


@pytest.mark.parametrize("name", CANONICAL)
def test_config_defines_canonical_response_cache_name(name: str) -> None:
    """Every RESPONSE_CACHE_* symbol is defined in ``kiro.config``."""
    assert hasattr(config, name), (
        f"{name} missing from kiro.config — did the rename in "
        f"commit 51cd264 regress?"
    )


@pytest.mark.parametrize("name", LEGACY)
def test_config_has_no_legacy_prefix_cache_name(name: str) -> None:
    """Legacy PREFIX_CACHE_* names must not reappear."""
    assert not hasattr(config, name), (
        f"{name} is back in kiro.config — use RESPONSE_CACHE_* "
        f"instead (see INT-73)."
    )


def test_main_imports_canonical_cache_symbols() -> None:
    """``main.py``'s cache imports reference the canonical names only."""
    main_py = Path(__file__).resolve().parents[2] / "main.py"
    text = main_py.read_text(encoding="utf-8")

    # Canonical imports must be present. We accept multi-line imports,
    # so just check that each symbol appears in the file.
    for name in CANONICAL[:5]:  # exclude LOG_HITS — main.py doesn't need it
        assert name in text, (
            f"main.py no longer references {name} — cache wiring "
            f"may have regressed."
        )

    # Legacy names must NOT be referenced anywhere in main.py.
    for name in LEGACY:
        assert not re.search(rf"\b{re.escape(name)}\b", text), (
            f"main.py still mentions {name} — this is the INT-73 "
            f"regression; use {name.replace('PREFIX_', 'RESPONSE_')} "
            f"instead."
        )

    # And the state attribute should be ``response_cache``, not
    # ``prefix_cache``.
    assert "app.state.response_cache" in text
    assert "app.state.prefix_cache" not in text, (
        "main.py still writes to app.state.prefix_cache — use "
        "app.state.response_cache (INT-73 regression)."
    )
