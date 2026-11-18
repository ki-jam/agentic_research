"""
ANSI theming — colors, boxes, badges, progress bars.

Colors are enabled only when stdout is a TTY and NO_COLOR is not set.
When disabled, every helper returns the plain string, so tests can assert
on raw text regardless of terminal state.

Alignment rule: pad/truncate the RAW string before wrapping it in a color
helper — escape codes would break f-string alignment otherwise
(e.g. write theme.cyan(f"{cmd:<16}"), not f"{theme.cyan(cmd):<16}").
"""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    """Colors on only for a real TTY, unless NO_COLOR is set."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def c(text: str, code: str) -> str:
    """Wrap text in an ANSI SGR code; plain text when colors are off."""
    if _enabled():
        return f"\033[{code}m{text}\033[0m"
    return text


def bold(text: str) -> str:
    return c(text, "1")


def dim(text: str) -> str:
    return c(text, "2")


def red(text: str) -> str:
    return c(text, "31")


def green(text: str) -> str:
    return c(text, "32")


def yellow(text: str) -> str:
    return c(text, "33")


def cyan(text: str) -> str:
    return c(text, "36")


def prompt(text: str) -> str:
    """A colored input() prompt label."""
    return bold(cyan(text))


def header(title: str, subtitle: str = "") -> str:
    """A boxed screen header.

    ╭──────────────────────────────────╮
    │  Research Agent                  │
    │  Research iteration 3 complete.  │
    ╰──────────────────────────────────╯
    """
    lines = [title]
    if subtitle:
        lines.append(subtitle)
    width = max(len(s) for s in lines) + 4
    border = "─" * (width - 2)
    body = "\n".join(f"│  {s.ljust(width - 4)}  │" for s in lines)
    return (
        "\n"
        + bold(cyan("╭" + border + "╮"))
        + "\n"
        + body
        + "\n"
        + bold(cyan("╰" + border + "╯"))
    )


def section(title: str) -> str:
    """A mid-screen section divider: ``── Title ────────``."""
    return "\n" + dim(f"── {title} " + "─" * 30)


def progress_bar(value: int, total: int, width: int = 20) -> str:
    """A filled/empty bar with counts: ``[████░░] 4/10 (40%)``."""
    if total <= 0:
        return dim(f"[{'░' * width}] 0/0")
    frac = max(0.0, min(1.0, value / total))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    color = green if frac >= 1 else (yellow if frac >= 0.5 else cyan)
    return f"{color('[' + bar + ']')} {dim(f'{value}/{total} ({int(frac * 100)}%)')}"


def priority_badge(priority: str, width: int = 8) -> str:
    """HIGH (red) / MED (yellow) / LOW (dim), right-padded before coloring."""
    p = (priority or "?").lower()
    label = {"high": "HIGH", "low": "LOW"}.get(p, "MED")
    raw = label.rjust(width)
    if p == "high":
        return red(raw)
    if p == "low":
        return dim(raw)
    return yellow(raw)


def status_badge(status: str, width: int = 12) -> str:
    """A status word colored by meaning, left-padded before coloring."""
    s = (status or "?").lower()
    raw = s.ljust(width)
    if s in ("resolved", "curated", "curate"):
        return green(raw)
    if s in ("failed", "discard", "discarded"):
        return red(raw)
    if s == "in_progress":
        return yellow(raw)
    if s == "pending":
        return cyan(raw)
    return dim(raw)
