"""
CLI tests — run.py dispatch, the ui/ package (theme, review screen,
browsers), and full end-to-end sessions driven through main().

Hermetic by construction:
  * scripted LLM (MockLLM from test_nodes) — no model calls, no network
  * search functions patched (test_nodes._mock_search)
  * RESEARCH_DB_PATH → data/knowledge_test.db (fresh per test)
  * paths_config.base_dir → tmp_path (checkpoints.db + report .md files)

Run with:
    cd research_agent && ../venv/bin/python -m pytest tests/test_cli.py -v
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys

import pytest

# Ensure we can import from the package root and from this directory
# (test_nodes is a sibling module, not a subpackage — tests/ has __init__.py,
# so pytest loads it as tests.test_nodes and the top-level name needs the
# tests dir itself on sys.path).
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__) + "/..")

# Fake key — tests never call the real Exa API (search functions are patched)
os.environ["EXA_API_KEY"] = "test-key"

import run as run_mod  # noqa: E402
from db.engine import get_db  # noqa: E402
from db.knowledge import (  # noqa: E402
    create_finding,
    create_knowledge_entry,
    create_plan,
    create_questions,
    save_report,
    set_question_failed,
)
from test_nodes import MockLLM, _mock_search  # noqa: E402
from ui import review as ui_review  # noqa: E402
from ui import theme  # noqa: E402
from ui.review import parse_review_input  # noqa: E402


# ── Fixtures & helpers ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_test_db():
    """Use a test DB so production data is never touched.

    Same pattern as test_nodes: RESEARCH_DB_PATH → knowledge_test.db,
    dropped and re-created per test.
    """
    import db.engine as engine_mod

    db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_db = os.path.join(db_dir, "knowledge_test.db")

    os.environ["RESEARCH_DB_PATH"] = test_db
    if os.path.exists(test_db):
        os.remove(test_db)
    engine_mod.init_db()

    yield

    if os.path.exists(test_db):
        os.remove(test_db)
    os.environ.pop("RESEARCH_DB_PATH", None)


def _script_input(monkeypatch, script: list[str], record: list | None = None) -> None:
    """Replace builtins.input with a scripted one.

    Raises EOFError when the script is exhausted, so a test that reads
    more lines than it scripted fails loudly.

    If ``record`` is given, every prompt string passed to input() is
    appended to it — the prompt is echoed by the terminal driver, never
    written to stdout, so capsys can't see it.
    """
    it = iter(script)

    def fake_input(prompt=""):
        if record is not None:
            record.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)


class _FakeTTY:
    """Duck-typed stdout that claims to be a TTY (wraps the captured one)."""

    def __init__(self, inner):
        self._inner = inner

    def isatty(self):
        return True

    def write(self, s):
        return self._inner.write(s)

    def flush(self):
        self._inner.flush()


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class _FakeTask:
    """Duck-typed PregelTask: a crashed-mid-node task has interrupts=()."""

    def __init__(self, interrupts=()):
        self.interrupts = interrupts


class _FakeSnapshot:
    def __init__(self, next_, tasks, values):
        self.next = next_
        self.tasks = tasks
        self.values = values


class _FakeApp:
    """Duck-typed compiled graph: scripted get_state + stream."""

    def __init__(self, snapshot, stream_events):
        self._snapshot = snapshot
        self._stream_events = stream_events
        self.stream_calls = []

    def get_state(self, cfg):
        return self._snapshot

    def stream(self, input_obj, cfg, stream_mode=None):
        self.stream_calls.append(input_obj)
        return iter(self._stream_events)


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    """Isolated environment for E2E runs through main().

    Scripted LLM, mocked search, no citation discovery, and
    paths_config.base_dir → tmp_path so the SqliteSaver checkpoint store
    and report .md files never touch the production data/ directory.
    (test_nodes.e2e_env plus the base_dir patch main() needs.)
    """
    from config import settings
    from config.settings import paths_config, search_config, workflow_config
    import nodes.search as search_mod

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockLLM(),
    )
    monkeypatch.setattr(search_mod, "_search_exa", _mock_search)
    monkeypatch.setattr(search_mod, "_search_arxiv", lambda q: [])
    monkeypatch.setattr(search_config, "enable_citation_discovery", False)
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    yield


# ── ui/theme.py ─────────────────────────────────────────────────────


class TestTheme:
    def test_disabled_when_not_a_tty(self):
        assert not sys.stdout.isatty()  # pytest captures stdout
        assert theme.c("x", "31") == "x"
        assert theme.bold("x") == "x"
        assert theme.red("x") == "x"
        assert theme.dim("x") == "x"

    def test_enabled_on_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout))
        assert theme.c("x", "31") == "\033[31mx\033[0m"
        assert theme.green("ok") == "\033[32mok\033[0m"
        assert theme.dim("d") == "\033[2md\033[0m"
        # prompt() is bold(cyan(...)): nested SGR with two trailing resets.
        assert theme.prompt("> ") == "\033[1m\033[36m> \033[0m\033[0m"

    def test_no_color_wins_over_tty(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout))
        assert theme.c("x", "31") == "x"

    def test_header_and_section_plain(self):
        h = theme.header("Title", "Sub")
        assert "Title" in h and "Sub" in h
        assert "╭" in h and "╰" in h
        assert theme.section("Sec").strip().startswith("── Sec")

    def test_progress_bar(self):
        assert theme.progress_bar(4, 10, width=6) == "[██░░░░] 4/10 (40%)"
        assert theme.progress_bar(10, 10, width=4) == "[████] 10/10 (100%)"
        assert theme.progress_bar(0, 10, width=4) == "[░░░░] 0/10 (0%)"
        assert theme.progress_bar(0, 0) == "[" + "░" * 20 + "] 0/0"

    def test_priority_badge(self):
        # right-justified to width 8: "HIGH" → 4 spaces, "MED"/"LOW" → 5
        assert theme.priority_badge("high") == "    HIGH"
        assert theme.priority_badge("medium") == "     MED"
        assert theme.priority_badge("low") == "     LOW"
        assert theme.priority_badge("weird") == "     MED"

    def test_status_badge(self):
        assert theme.status_badge("resolved") == "resolved    "
        assert theme.status_badge("pending") == "pending     "
        assert theme.status_badge("in_progress") == "in_progress "
        assert theme.status_badge("failed") == "failed      "
        assert theme.status_badge("??") == "??          "

    def test_badges_pad_before_coloring(self, monkeypatch):
        """ANSI codes must not break the fixed-width alignment."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout))
        assert theme.priority_badge("high") == "\033[31m    HIGH\033[0m"
        assert theme.status_badge("pending") == "\033[36mpending     \033[0m"


# ── ui/review.py — input parsing ────────────────────────────────────


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", "continue"),
        ("   ", "continue"),
        ("1", "continue"),
        ("2", "stop"),
        ("3", "stats"),
        ("4", "findings"),
        ("5", "contradictions"),
        ("6", "rewind"),
        ("continue", "continue"),
        ("stop", "stop"),
        # no-arg commands pass through as typed (the node lowercases)
        ("CONTINUE", "CONTINUE"),
        ("? ", "help"),
        ("help", "help"),
        ("help me", "help"),
        # argument commands pass through only WITH an argument
        ("curate entry_abc", "curate entry_abc"),
        ("Curate entry_abc", "Curate entry_abc"),
        ("discard entry_abc", "discard entry_abc"),
        ("tag entry_abc climate", "tag entry_abc climate"),
        ("retry 3", "retry 3"),
        ("accept q_plan_1_2", "accept q_plan_1_2"),
        ("gap 2", "gap 2"),
        ("curate", None),
        ("discard", None),
        ("tag", None),
        ("retry", None),
        ("accept", None),
        ("gap", None),
        # unrecognized input must never silently continue the graph
        ("7", None),
        ("frobnicate", None),
    ],
)
def test_parse_review_input(line, expected):
    assert parse_review_input(line) == expected


# ── run.py — query prompt ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        # quit tokens must NOT spawn a plan (a 'quit;' once did: plan_756c6a4e)
        ("quit", ""),
        ("quit;", ""),
        ("QUIT", ""),
        ("exit", ""),
        ("exit!", ""),
        ("q", ""),
        ("q.", ""),
        ("   quit   ", ""),
        # real queries pass through untouched
        ("How do agents evaluate?", "How do agents evaluate?"),
        ("q learning for non-ML users", "q learning for non-ML users"),
        ("quit when a plan is abandoned", "quit when a plan is abandoned"),
        ("", ""),
    ],
)
def test_prompt_query_quit_tokens(monkeypatch, typed, expected):
    _script_input(monkeypatch, [typed])
    assert run_mod._prompt_query() == expected


def test_prompt_query_eof_returns_empty(monkeypatch):
    _script_input(monkeypatch, [])
    assert run_mod._prompt_query() == ""


# ── ui/review.py — screen + prompt loop ─────────────────────────────

REVIEW_PAYLOAD = {"prompt": "How would you like to proceed?"}


class TestAskReviewResponse:
    def test_numbered_shortcut(self, monkeypatch, capsys):
        prompts: list[str] = []
        _script_input(monkeypatch, ["3"], prompts)
        assert ui_review.ask_review_response(REVIEW_PAYLOAD) == "stats"
        out = capsys.readouterr().out
        assert "How would you like to proceed?" in out
        # the prompt is passed to input() (terminal-echoed, not in stdout)
        assert prompts == ["Your command> "]
        assert "Actions (type a number" in out

    def test_unknown_input_reprompts(self, monkeypatch, capsys):
        prompts: list[str] = []
        _script_input(monkeypatch, ["frobnicate", "2"], prompts)
        assert ui_review.ask_review_response(REVIEW_PAYLOAD) == "stop"
        out = capsys.readouterr().out
        assert "Unknown command: 'frobnicate'" in out
        assert prompts == ["Your command> ", "Your command> "]

    def test_help_then_command(self, monkeypatch, capsys):
        _script_input(monkeypatch, ["?", "6"])
        assert ui_review.ask_review_response(REVIEW_PAYLOAD) == "rewind"
        out = capsys.readouterr().out
        # help menu appears with the render and again after "?"
        assert out.count("Actions (type a number") == 2

    def test_eof_propagates(self, monkeypatch):
        _script_input(monkeypatch, [])
        with pytest.raises(EOFError):
            ui_review.ask_review_response(REVIEW_PAYLOAD)


class TestRenderReview:
    def test_rich_payload(self, capsys):
        payload = {
            "message": "Iteration 1 complete.",
            "stats": {"total": 3, "by_status": {"raw": 2, "curated": 1}},
            "question_progress": {"pending": 1, "failed": 1, "resolved": 2},
            "open_questions": [
                {
                    "question": "Pending question one?",
                    "priority": "high",
                    "status": "pending",
                },
                {
                    "question": "Failed question two?",
                    "priority": "medium",
                    "status": "failed",
                    "last_error": "boom",
                },
            ],
            "contradictions": [
                {"finding_a": "f1", "finding_b": "f2", "explanation": "x"}
            ],
            "recommendations": [
                {"id": "entry_abc", "action": "curate", "reason": "rich"}
            ],
            "raw_entries": [
                {"id": "entry_abc", "title": "Some source", "tags": ["a", "b"]}
            ],
            "prompt": "How would you like to proceed?",
        }
        ui_review.render_review(payload)
        out = capsys.readouterr().out
        assert "Review" in out
        assert "Iteration 1 complete." in out
        assert "library 3 entries" in out
        assert "2 raw" in out and "1 curated" in out
        assert "1 pending" in out and "1 failed" in out and "2 resolved" in out
        assert "Open questions (pending first, then failed)" in out
        assert "Pending question one?" in out
        assert "[FAILED] Failed question two?" in out
        assert "boom" in out
        assert "Contradictions: 1 pair(s) flagged" in out
        assert "Curation recommendations" in out
        assert "entry_abc" in out
        assert "Raw entries (curate <id> / discard <id> / tag <id> <t>)" in out
        assert "How would you like to proceed?" in out
        assert "Actions (type a number" in out

    def test_minimal_payload(self, capsys):
        ui_review.render_review({"prompt": "p"})
        out = capsys.readouterr().out
        assert "p" in out
        assert "Actions (type a number" in out

    def test_non_dict_payload(self, capsys):
        ui_review.render_review("plain string payload")
        assert "plain string payload" in capsys.readouterr().out


# ── run.py — CLI dispatch ───────────────────────────────────────────


class TestMainDispatch:
    def test_bare_invocation_shows_menu(self, monkeypatch, capsys):
        _script_input(monkeypatch, ["9", "0"])
        run_mod.main([])
        out = capsys.readouterr().out
        assert "New research" in out
        assert "Models" in out
        assert "Type 1–7 or 0 to quit." in out

    def test_menu_routes_to_plans_browser(self, monkeypatch, capsys):
        # "2" → plans (empty DB returns immediately, reading no input),
        # back to the menu, "0" quits
        _script_input(monkeypatch, ["2", "0"])
        run_mod.main([])
        out = capsys.readouterr().out
        assert "No plans yet" in out
        assert out.count("New research") == 2

    def test_menu_routes_to_models_browser(self, monkeypatch, capsys):
        # "7" → models ("q" quits the browser), back to the menu, "0" quits
        _script_input(monkeypatch, ["7", "q", "0"])
        run_mod.main([])
        out = capsys.readouterr().out
        assert "Models per step" in out
        assert out.count("New research") == 2

    def test_ask_without_query_exits(self, monkeypatch, capsys):
        _script_input(monkeypatch, [""])
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["ask"])
        assert exc.value.code == 1
        assert "No query provided" in capsys.readouterr().out

    def test_legacy_without_query_exits(self, monkeypatch, capsys):
        _script_input(monkeypatch, [""])
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["--provider", "local"])
        assert exc.value.code == 1
        assert "No query provided" in capsys.readouterr().out

    def test_plan_requires_id(self, capsys):
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["plan"])
        assert exc.value.code == 2
        assert "Usage: python run.py plan <plan_id>" in capsys.readouterr().out

    def test_plan_not_found(self, capsys):
        run_mod.main(["plan", "plan_nope"])
        assert "not found" in capsys.readouterr().out

    def test_resume_unknown_session_exits(self, tmp_path, monkeypatch, capsys):
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["resume", "plan_nope"])
        assert exc.value.code == 1
        assert "No session found" in capsys.readouterr().out

    def test_legacy_resume_unknown_session_exits(self, tmp_path, monkeypatch):
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["--resume", "plan_nope"])
        assert exc.value.code == 1

    def test_resume_mid_node_crash_does_not_show_review_screen(
        self, tmp_path, monkeypatch, capsys
    ):
        """Crashed mid-node (state.next set, no interrupt payload): continue
        from the checkpoint. The old code showed a fake "Session is
        suspended" review screen and consumed the user's reply as a resume
        value for a graph that wasn't suspended — that input was silently
        lost. Now: no review screen, no input read, resume value is None."""
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        fake = _FakeApp(
            _FakeSnapshot(
                next_=("extraction",),
                tasks=(_FakeTask(interrupts=()),),
                values={"plan_ref": "plan_x"},
            ),
            stream_events=[],
        )
        monkeypatch.setattr(run_mod, "build_app", lambda: fake)
        consumed = []
        _script_input(monkeypatch, ["stop"], record=consumed)

        run_mod.main(["resume", "thread_crashed_mid_node"])

        out = capsys.readouterr().out
        assert "crashed mid-node" in out
        # input() was never called — the scripted "stop" is untouched.
        assert consumed == []
        # Graph continues from the checkpoint with no resume value.
        assert fake.stream_calls == [None]

    def test_resume_genuine_interrupt_redisplay_payload_and_resume(
        self, tmp_path, monkeypatch, capsys
    ):
        """Suspended at a review interrupt: re-display the checkpointed
        payload and resume with the user's command. The interrupted task
        is deliberately NOT tasks[0] — resume must find the task that
        actually carries the interrupt."""
        from config.settings import paths_config
        from langgraph.types import Command

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        payload = {"message": "Research iteration 5 complete.", "prompt": "Do it?"}
        fake = _FakeApp(
            _FakeSnapshot(
                next_=("prepare_review",),
                tasks=(
                    _FakeTask(interrupts=()),
                    _FakeTask(interrupts=(_FakeInterrupt(payload),)),
                ),
                values={"plan_ref": "plan_x"},
            ),
            stream_events=[],
        )
        monkeypatch.setattr(run_mod, "build_app", lambda: fake)
        consumed = []
        _script_input(monkeypatch, ["1"], record=consumed)  # → "continue"

        run_mod.main(["resume", "thread_at_review"])

        out = capsys.readouterr().out
        assert "Research iteration 5 complete." in out
        assert "Do it?" in out
        assert len(consumed) == 1 and "Your command" in consumed[0]
        assert fake.stream_calls == [Command(resume="continue")]

    def test_config_overrides(self, monkeypatch):
        from config.settings import config, workflow_config

        monkeypatch.setattr(workflow_config, "max_iterations", 10)
        monkeypatch.setattr(config, "provider", "local")
        monkeypatch.setattr(config, "openai_model", "orig")

        run_mod._apply_config_overrides(
            argparse.Namespace(provider=None, model=None, max_iterations=7)
        )
        assert workflow_config.max_iterations == 7

        run_mod._apply_config_overrides(
            argparse.Namespace(provider="openai", model="gpt-test", max_iterations=None)
        )
        assert config.provider == "openai"
        assert config.openai_model == "gpt-test"


# ── ui/views.py — plans browser ─────────────────────────────────────


class TestPlans:
    def test_show_plans_empty(self, capsys):
        run_mod.main(["plans"])
        out = capsys.readouterr().out
        assert "No plans yet" in out
        assert 'python run.py ask "your query"' in out

    def test_show_plans_detail_via_number(self, monkeypatch, capsys):
        create_plan("Plans browser seed", [])
        _script_input(monkeypatch, ["1", "q"])
        run_mod.main(["plans"])
        out = capsys.readouterr().out
        assert "1 plan(s)" in out
        assert "Plans browser seed" in out
        assert "Query:" in out  # detail view was shown

    def test_show_plan_detail_subcommand(self, capsys):
        plan_id = create_plan("Plans by id seed", [])
        run_mod.main(["plan", plan_id])
        out = capsys.readouterr().out
        assert "Query:" in out
        assert plan_id in out


# ── ui/views.py — backlog browser ───────────────────────────────────


class TestBacklog:
    @pytest.fixture
    def seeded(self):
        """Plan with 3 pending questions (high/medium/low) + 1 failed."""
        plan_id = create_plan("Backlog seed", [])
        ids = create_questions(
            plan_id,
            [
                {"question": "High pending question", "priority": "high"},
                {"question": "Medium pending question", "priority": "medium"},
                {"question": "Low pending question", "priority": "low"},
                {"question": "Failed question", "priority": "high"},
            ],
        )
        set_question_failed(ids[3], "boom")
        return plan_id, ids

    def test_backlog_command_roundtrip(self, seeded, monkeypatch, capsys):
        plan_id, (q1, q2, q3, q4) = seeded
        _script_input(
            monkeypatch,
            [
                "up 3",
                "set 3 high",
                "down 2",
                "add What is a brand new angle?",
                "retry 5",
                "drop 2",
                "bogus",
                "q",
            ],
        )
        run_mod.main(["backlog", plan_id])
        out = capsys.readouterr().out
        assert "Backlog" in out
        assert "High pending question" in out
        assert "[FAILED] Failed question" in out
        assert "now #3 in line" in out  # up 3 → low → medium
        assert "now #2 in line" in out  # set 3 high
        assert "Added " in out  # add
        assert "now #4 in line" in out  # add → newest medium
        assert "Retried" in out  # retry 5 (the failed one)
        assert "Dropped" in out  # drop 2 (the retried one)
        assert "status → discarded" in out
        assert "Commands: up <n>" in out  # unknown-command hint

        with get_db() as conn:
            rows = {
                r["id"]: (r["status"], r["priority"])
                for r in conn.execute(
                    "SELECT id, status, priority FROM questions WHERE plan_id = ?",
                    (plan_id,),
                )
            }
        assert rows[q1] == ("pending", "high")
        assert rows[q2] == ("pending", "medium")
        assert rows[q3] == ("pending", "medium")
        assert rows[q4] == ("discarded", "low")
        new = [i for i in rows if i not in (q1, q2, q3, q4)]
        assert len(new) == 1
        assert rows[new[0]] == ("pending", "medium")

    def test_backlog_picks_plan_interactively(self, seeded, monkeypatch, capsys):
        _script_input(monkeypatch, ["1", "q"])
        run_mod.main(["backlog"])
        out = capsys.readouterr().out
        assert "Pick a plan" in out
        assert "Backlog" in out


# ── ui/views.py — results browser ───────────────────────────────────


class TestResults:
    @pytest.fixture
    def seeded(self):
        plan_id = create_plan("Results seed", [])
        qids = create_questions(
            plan_id, [{"question": "Seed question?", "priority": "high"}]
        )
        entry = create_knowledge_entry(
            "Seed entry",
            "Entry summary",
            ["tag-a"],
            "",
            "https://example.com/seed",
            "user",
        )
        create_finding(
            qids[0],
            None,
            entry,
            [{"claim": "Seed fact", "confidence": "high"}],
            "A good finding summary.",
            ["tag-a"],
        )
        return plan_id

    def test_results_detail_via_number(self, seeded, monkeypatch, capsys):
        _script_input(monkeypatch, ["1", "q"])
        run_mod.main(["results", seeded])
        out = capsys.readouterr().out
        assert "1 of 1 finding(s) valid · 1 knowledge entry" in out
        assert "A good finding summary." in out
        assert "ID:" in out
        assert "Question: Seed question?" in out
        assert "Summary:  A good finding summary." in out
        assert "- Seed fact" in out

    def test_results_invalid_number(self, seeded, monkeypatch, capsys):
        _script_input(monkeypatch, ["9", "q"])
        run_mod.main(["results", seeded])
        assert "Type a finding number, r, or q." in capsys.readouterr().out

    def test_results_empty_plan(self, monkeypatch, capsys):
        plan_id = create_plan("Empty results", [])
        _script_input(monkeypatch, ["q"])
        run_mod.main(["results", plan_id])
        assert (
            "No findings or knowledge entries for this plan yet."
            in capsys.readouterr().out
        )


# ── ui/views.py — reports browser + save_report ─────────────────────


class TestReports:
    def test_report_shows_saved_report(self, monkeypatch, tmp_path, capsys):
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        plan_id = create_plan("Report seed", [])
        save_report(plan_id, "Report seed query", "# R1\n\nBody one.")
        save_report(plan_id, "Report seed query", "# R2\n\nBody two.")
        _script_input(monkeypatch, ["1", "q"])
        run_mod.main(["report", plan_id])
        out = capsys.readouterr().out
        assert "report_" in out
        assert "Body two." in out  # newest report listed first
        assert "Body one." not in out

    def test_report_empty(self, monkeypatch, capsys):
        plan_id = create_plan("No reports", [])
        _script_input(monkeypatch, ["q"])
        run_mod.main(["report", plan_id])
        assert "No reports saved for this plan yet." in capsys.readouterr().out

    def test_report_picks_plan_interactively(self, monkeypatch, tmp_path, capsys):
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        plan_id = create_plan("Pick report", [])
        save_report(plan_id, "Pick report query", "# P\n\nPicked body.")
        _script_input(monkeypatch, ["1", "1", "q"])
        run_mod.main(["report"])
        out = capsys.readouterr().out
        assert "Pick a plan" in out
        assert "Picked body." in out

    def test_save_report_writes_file_and_row(self, tmp_path, monkeypatch):
        from config.settings import paths_config

        monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
        plan_id = create_plan("File report", [])
        rid = save_report(plan_id, "File report query", "content here")
        assert rid.startswith("report_")
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        assert files[0].read_text() == "content here"
        with get_db() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM reports WHERE plan_id = ?", (plan_id,)
            ).fetchone()[0]
        assert n == 1


# ── ui/views.py — stats ─────────────────────────────────────────────


class TestStats:
    def test_stats_shows_entries_and_plans(self, capsys):
        create_knowledge_entry("S1", "s", ["t"], "")
        create_knowledge_entry("S2", "s", ["t"], "")
        plan_id = create_plan("Stats seed", [])
        create_questions(plan_id, [{"question": "Q", "priority": "high"}])
        run_mod.main(["stats"])
        out = capsys.readouterr().out
        assert "Knowledge base" in out
        assert "Entries: 2 total" in out
        assert "raw: 2" in out
        assert "Top tags: t×2" in out
        assert plan_id in out
        assert "1 open" in out


# ── run.py — step-level progress logging (_print_event) ────────────
#
# A running session must be checkable at a glance: iteration headers,
# the question being worked on, findings deltas, and the review
# checkpoint marker. These are the lines you read when tailing the log.


class TestPrintEvent:
    def test_manage_backlog_shows_picked_question(self, capsys):
        from db.knowledge import set_question_in_progress

        plan_id = create_plan("Step lines", [])
        qids = create_questions(
            plan_id, [{"question": "The picked question?", "priority": "high"}]
        )
        set_question_in_progress(qids[0])

        # manage_backlog_node returns {} — the line must come from the DB.
        run_mod._print_event({"manage_backlog": {}}, {"plan_ref": plan_id})
        out = capsys.readouterr().out
        assert "Question: The picked question?" in out

    def test_manage_backlog_no_plan_falls_back(self, capsys):
        run_mod._print_event({"manage_backlog": {}}, {})
        out = capsys.readouterr().out
        assert "Selecting next question" in out

    def test_extract_and_verify_shows_findings_delta(self, capsys):
        from db.knowledge import set_question_in_progress, store_raw_result

        plan_id = create_plan("Step lines", [])
        qids = create_questions(
            plan_id, [{"question": "Seed question?", "priority": "high"}]
        )
        set_question_in_progress(qids[0])
        rid = store_raw_result(qids[0], plan_id, "q", "exa", "content", "Source")
        create_finding(
            qids[0],
            rid,
            None,
            [{"claim": "c1", "confidence": "high"}],
            "Summary one.",
            ["t"],
        )
        create_finding(
            qids[0],
            rid,
            None,
            [{"claim": "c2", "confidence": "high"}],
            "Summary two.",
            ["t"],
        )

        ctx = {"plan_ref": plan_id}
        run_mod._print_event({"extract_and_verify": {}}, ctx)
        out = capsys.readouterr().out
        assert "Findings: 2 total (+2 new) (this plan)" in out
        # Second pass: same count → no delta suffix
        run_mod._print_event({"extract_and_verify": {}}, ctx)
        out = capsys.readouterr().out
        assert "Findings: 2 total (this plan)" in out

    def test_consolidation_always_prints_iteration_header(self, capsys):
        run_mod._print_event(
            {"consolidation": {"_route": "manage_backlog", "iteration_count": 3}}, {}
        )
        out = capsys.readouterr().out
        assert "Iteration 3" in out
        # manage_backlog route: no extra route line
        assert "Consolidation:" not in out

    def test_consolidation_non_backlog_route_prints_route_line(self, capsys):
        run_mod._print_event(
            {"consolidation": {"_route": "prepare_review", "iteration_count": 5}}, {}
        )
        out = capsys.readouterr().out
        assert "Iteration 5" in out
        assert "Consolidation: prepare_review" in out

    def test_prepare_review_prints_checkpoint_line(self, capsys):
        run_mod._print_event({"prepare_review": {"_review_payload": {}}}, {})
        out = capsys.readouterr().out
        assert "Review checkpoint" in out


# ── E2E: full sessions through main() ───────────────────────────────


class TestEndToEnd:
    def test_ask_stops_at_first_review(self, e2e_env, tmp_path, monkeypatch, capsys):
        from config.settings import workflow_config

        monkeypatch.setattr(workflow_config, "max_iterations", 10)
        monkeypatch.setattr(workflow_config, "review_every_n_iterations", 1)
        MockLLM.prompts.clear()
        _script_input(monkeypatch, ["2"])
        run_mod.main(["ask", "what is an F1 visa"])
        out = capsys.readouterr().out
        assert 'Starting research on: "what is an F1 visa"' in out
        assert "Plan created: plan_" in out
        assert "How would you like to proceed?" in out
        assert "E2E-REPORT" in out
        assert "Session cost" in out
        assert "Research complete." in out

        with get_db() as conn:
            plan = conn.execute("SELECT id, thread_id FROM plans").fetchone()
            n_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
            n_resolved = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE status = 'resolved'"
            ).fetchone()[0]
        assert plan is not None
        assert plan["thread_id"]
        assert n_reports == 1
        assert n_resolved == 1
        assert (tmp_path / "checkpoints.db").exists()
        # short query → single question, no LLM refinement prompt
        assert not any(
            "focused, searchable sub-questions" in p for p in MockLLM.prompts
        )

    def test_ask_continue_then_stop(self, e2e_env, tmp_path, monkeypatch, capsys):
        """Iteration 1 → review → 'continue' → iteration 2 also hits the
        review cadence (2 % 1 == 0) → 'stop' writes the final report."""
        from config.settings import workflow_config

        monkeypatch.setattr(workflow_config, "max_iterations", 10)
        monkeypatch.setattr(workflow_config, "review_every_n_iterations", 1)
        MockLLM.prompts.clear()
        _script_input(monkeypatch, ["1", "2"])
        run_mod.main(["ask", "what is an F1 visa"])
        out = capsys.readouterr().out
        assert out.count("How would you like to proceed?") == 2
        assert out.count("E2E-REPORT") == 1
        assert "Research complete." in out

        with get_db() as conn:
            n_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
            n_q = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        assert n_reports == 1
        # no DoD gap questions added: both consolidations routed to
        # human_review, not manage_backlog
        assert n_q == 1
