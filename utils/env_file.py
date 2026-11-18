"""
Minimal .env file manipulation for the ``run.py models`` browser.

Line-based upsert: comments and all other lines are preserved verbatim,
only the line for the target key is touched. Values are written bare
(no quotes), matching the style of ``.env.example``.
"""

from __future__ import annotations

import os


def default_env_path() -> str:
    """The ``.env`` file at the research_agent root (next to run.py)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".env")


def upsert_env_file(path: str, key: str, value: str) -> str:
    """Set ``key=value`` in the .env file at ``path``.

    Returns:
        "unchanged" — the file already contained that exact line
        "updated"   — an existing line for the key was replaced
        "added"     — the key was not present; the line was appended

    Creates the file if it does not exist. If the key occurs on several
    lines, every occurrence is replaced by the single new line. New keys
    are appended at the end, directly below the last entry.
    """
    lines: list[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    new_line = f"{key}={value}"
    out: list[str] = []
    state = "added"
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            if line.split("=", 1)[0].strip() == key:
                state = "unchanged" if line == new_line else "updated"
                if replaced:
                    continue  # drop duplicate occurrences
                replaced = True
                out.append(new_line)
                continue
        out.append(line)
    if state == "added":
        # Drop trailing blank lines so the new key sits directly below the
        # last entry (consecutive upserts then group together).
        while out and out[-1].strip() == "":
            out.pop()
        out.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return state
