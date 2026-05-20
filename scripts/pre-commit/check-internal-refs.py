#!/usr/bin/env python3
"""Pre-commit internal-ref scan — block commits containing internal tooling references.

Catches the gap between gitleaks (credential shapes) and check-pii.py (personal
identity): internal tooling URIs, private paths, and AI-tooling artifacts that
should never appear in public repos.

Categories blocked (HARD_BLOCK):
  - basic-memory:// URIs — internal MCP store references
  - data/basic-memory/ paths — internal MCP store paths
  - .kiro/ paths — Kiro IDE steering files
  - memory:// URIs — generic memory MCP references
  - /home/<user>/ paths — private home directory (covered by check-pii.py too, belt+suspenders)
  - sub-agent / steering file / hook references as implementation detail in prose

Categories warned (WARN_ONLY, exit 0):
  - ~/projects/ paths — private project paths
  - ~/.claude/ paths — private Claude config paths
  - Any /home/<user>/ path not matching known public CI paths

Bypass: add `# internal-ref-allow` on the offending line.
Bypass via --no-verify is forbidden per .claude/rules/secrets.md.

Output: sanitized — file:line:rule only, no private path values echoed.

Exit code: 1 on any HARD_BLOCK hit (not allowlisted). 0 on warn-only or clean.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOW_MARKER = "internal-ref-allow"

# Hard blocks — exit 1 if found (not allowlisted)
HARD_BLOCK_PATTERNS: dict[str, str] = {
    "basic-memory-uri": r"basic-memory://",
    "data-basic-memory-path": r"data/basic-memory/",
    "kiro-steering-path": r"\.kiro/",
    "memory-uri": r"(?<![a-z])memory://",
    "home-user-path-hard": r"/home/[a-z_][a-z0-9_-]*/(?!runner/|ubuntu/|actions/|kiro/|gxx/)",  # pii-allow
    "internal-tooling-prose": r"\b(mcp_mcp_python|invokeSubAgent)\b",  # only truly internal tool names
}

# Warn-only — exit 0 but print warning
WARN_PATTERNS: dict[str, str] = {
    "home-projects-path": r"~/projects/",
    "home-claude-path": r"~/\.claude/",
    "home-user-path": r"/home/[a-z_][a-z0-9_-]*/(?!runner/|ubuntu/|actions/)",
}

SKIP_DIR_PARTS = {
    "data", "node_modules", ".git", ".venv", "venv", "__pycache__",
    "lessons-inbox", "_archive", "archive",
}
SKIP_FILE_NAMES = {"CHANGELOG.md", "shellcheck-baseline.txt", "check-internal-refs.py"}
SKIP_FILE_SUBSTRINGS = ("session-handoff", "session-pickup", "PROMPT-FOR-AGENT")
SKIP_DIR_PREFIXES = ("overnight-",)
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz",
    ".whl", ".so", ".bin", ".woff", ".woff2", ".ttf", ".otf", ".pkl",
}


def should_skip(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if any(s in path.name for s in SKIP_FILE_SUBSTRINGS):
        return True
    if any(part in SKIP_DIR_PARTS for part in path.parts):
        return True
    if any(part.startswith(SKIP_DIR_PREFIXES) for part in path.parts):
        return True
    return False


def scan_file(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (hard_hits, warn_hits) as (line_no, rule_id) tuples."""
    if should_skip(path) or not path.exists():
        return [], []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return [], []

    hard_hits: list[tuple[int, str]] = []
    warn_hits: list[tuple[int, str]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for rule_id, pat in HARD_BLOCK_PATTERNS.items():
            if re.search(pat, line):
                hard_hits.append((line_no, rule_id))
        for rule_id, pat in WARN_PATTERNS.items():
            if re.search(pat, line):
                warn_hits.append((line_no, rule_id))

    return hard_hits, warn_hits


def sanitize_path(path: Path) -> str:
    """Return path relative to repo root, never absolute — avoids leaking private home paths in output."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Fall back to just the filename if relative resolution fails
        return path.name


def main(argv: list[str]) -> int:
    if not argv:
        return 0

    total_hard = 0
    total_warn = 0

    for arg in argv:
        path = Path(arg)
        safe_path = sanitize_path(path)
        hard_hits, warn_hits = scan_file(path)

        for line_no, rule_id in hard_hits:
            print(f"{safe_path}:{line_no}: [BLOCK:{rule_id}] internal tooling reference", file=sys.stderr)
            total_hard += 1

        for line_no, rule_id in warn_hits:
            print(f"{safe_path}:{line_no}: [WARN:{rule_id}] possible private path", file=sys.stderr)
            total_warn += 1

    if total_hard:
        print(
            f"\ncheck-internal-refs: {total_hard} internal tooling reference(s) blocked. "
            f"Add `# {ALLOW_MARKER}` to the line if intentional, or remove the reference.",
            file=sys.stderr,
        )
        return 1

    if total_warn:
        print(
            f"\ncheck-internal-refs: {total_warn} possible private path(s) — review before push. "
            f"Add `# {ALLOW_MARKER}` to suppress.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
