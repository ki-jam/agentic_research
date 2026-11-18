"""
Interactive UI helpers — stdlib only (ANSI escapes + input()).

Modules:
    theme  — color/box/badge helpers (auto-disabled on non-TTY or NO_COLOR)
    review — the human-review screen (checkpointed payload + input parsing)
    views  — browsers for plans, backlogs, results, reports, and stats

Everything interactive takes an ``input_fn=None`` parameter that resolves
to ``builtins.input`` at call time, so tests can patch ``builtins.input``
(or pass a scripted function) without touching module globals.
"""
