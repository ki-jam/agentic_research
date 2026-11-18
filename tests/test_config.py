"""
Config tests — per-step model/provider resolution, the effective-config
banner, and the ``run.py models`` browser (utils/env_file + ui/views).

Hermetic: no LLM calls, no network, no DB. All assertions monkeypatch the
config singletons to known values so the real .env (which carries live keys)
never influences the outcome; the models browser is pointed at a tmp .env.

Run with:
    cd research_agent && ../venv/bin/python -m pytest tests/test_config.py -v
"""

from __future__ import annotations

import builtins
import os
import sys

import pytest

# Ensure the package root is importable (same pattern as the other test files).
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from config.settings import (  # noqa: E402
    config,
    resolve_step_model,
    update_config,
    workflow_config,
)
from ui import views as ui_views  # noqa: E402
from utils.env_file import upsert_env_file  # noqa: E402


STEPS = ("planning", "extraction", "curation", "report")


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    """Reset the config singletons to a known state for every test."""
    monkeypatch.setattr(config, "provider", "local")
    monkeypatch.setattr(config, "local_model", "local-default")
    monkeypatch.setattr(config, "openai_model", "gpt-4o")
    monkeypatch.setattr(config, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(config, "anthropic_model", "claude-default")
    monkeypatch.setattr(config, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(config, "anthropic_base_url", "")
    for step in STEPS:
        monkeypatch.setattr(workflow_config, f"{step}_model", "")
        monkeypatch.setattr(workflow_config, f"{step}_provider", "")
    yield


# ── resolve_step_model ──────────────────────────────────────────────


class TestResolveStepModel:
    def test_default_is_global_provider_and_its_model(self):
        for step in STEPS:
            assert resolve_step_model(step) == ("local", "local-default")

    def test_model_override_keeps_global_provider(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "extraction_model", "fast-7b")
        assert resolve_step_model("extraction") == ("local", "fast-7b")
        # Other steps are unaffected.
        assert resolve_step_model("curation") == ("local", "local-default")

    def test_provider_override_falls_back_to_that_providers_default_model(
        self, monkeypatch
    ):
        monkeypatch.setattr(workflow_config, "report_provider", "anthropic")
        assert resolve_step_model("report") == ("anthropic", "claude-default")
        monkeypatch.setattr(workflow_config, "report_provider", "openai")
        assert resolve_step_model("report") == ("openai", "gpt-4o")

    def test_provider_and_model_override(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "report_provider", "anthropic")
        monkeypatch.setattr(workflow_config, "report_model", "claude-sonnet-5-5")
        assert resolve_step_model("report") == ("anthropic", "claude-sonnet-5-5")

    def test_unknown_provider_raises_with_env_var_name(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "planning_provider", "gemini")
        with pytest.raises(ValueError, match="PLANNING_PROVIDER"):
            resolve_step_model("planning")

    def test_unknown_task_falls_back_to_global(self):
        # A task without override fields resolves like the global config.
        assert resolve_step_model("bogus") == ("local", "local-default")


# ── get_chat_model with per-step routing ────────────────────────────


class TestGetChatModel:
    @staticmethod
    def _model_name(m):
        return getattr(m, "model", None) or getattr(m, "model_name", None)

    def test_task_provider_selects_provider_class(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "extraction_provider", "openai")
        m = config.get_chat_model(task="extraction")
        assert type(m).__name__ == "ChatOpenAI"
        assert self._model_name(m) == "gpt-4o"

        monkeypatch.setattr(workflow_config, "extraction_provider", "anthropic")
        m = config.get_chat_model(task="extraction")
        assert type(m).__name__ == "ChatAnthropic"
        assert self._model_name(m) == "claude-default"

    def test_task_provider_with_model_override(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "report_provider", "anthropic")
        monkeypatch.setattr(workflow_config, "report_model", "claude-sonnet-5-5")
        m = config.get_chat_model(task="report")
        assert type(m).__name__ == "ChatAnthropic"
        assert self._model_name(m) == "claude-sonnet-5-5"

    def test_task_provider_with_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "report_provider", "gemini")
        with pytest.raises(ValueError, match="REPORT_PROVIDER"):
            config.get_chat_model(task="report")

    def test_explicit_model_wins_over_task_overrides(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "extraction_model", "fast-7b")
        m = config.get_chat_model(model="explicit", task="extraction")
        assert self._model_name(m) == "explicit"

    def test_no_task_ignores_task_overrides(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "extraction_provider", "openai")
        monkeypatch.setattr(workflow_config, "extraction_model", "fast-7b")
        m = config.get_chat_model()
        # The local provider also builds a ChatOpenAI (OpenAI-compatible).
        assert type(m).__name__ == "ChatOpenAI"
        assert self._model_name(m) == "local-default"

    def test_unknown_task_provider_fails(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "curation_provider", "nope")
        with pytest.raises(ValueError, match="CURATION_PROVIDER"):
            config.get_chat_model(task="curation")


# ── update_config ───────────────────────────────────────────────────


class TestUpdateConfig:
    def test_provider_fields_are_updatable(self, monkeypatch):
        monkeypatch.setattr(workflow_config, "report_provider", "")
        monkeypatch.setattr(workflow_config, "report_model", "")
        update_config(report_provider="anthropic", report_model="claude-x")
        assert workflow_config.report_provider == "anthropic"
        assert workflow_config.report_model == "claude-x"

    def test_summarize_model_is_gone(self):
        # SUMMARIZE_MODEL was a dead knob (no step ever used task="summarize")
        # and was removed — update_config must reject it now.
        with pytest.raises(ValueError, match="Unknown config field"):
            update_config(summarize_model="x")


# ── effective-config banner (run._print_effective_config) ───────────


class TestEffectiveConfigBanner:
    def test_all_defaults(self, capsys):
        import run as run_mod

        run_mod._print_effective_config()
        out = capsys.readouterr().out
        assert "Effective config" in out
        assert "local/local-default (all steps)" in out
        assert "search:" in out

    def test_overrides_are_listed(self, capsys, monkeypatch):
        import run as run_mod

        monkeypatch.setattr(workflow_config, "report_provider", "anthropic")
        monkeypatch.setattr(workflow_config, "report_model", "claude-sonnet-5-5")
        run_mod._print_effective_config()
        out = capsys.readouterr().out
        assert "(all steps, except:)" in out
        assert "report:" in out and "anthropic/claude-sonnet-5-5" in out

    def test_invalid_provider_exits_at_startup(self, capsys, monkeypatch):
        import run as run_mod

        monkeypatch.setattr(workflow_config, "planning_provider", "gemini")
        with pytest.raises(SystemExit):
            run_mod._print_effective_config()
        out = capsys.readouterr().out
        assert "PLANNING_PROVIDER" in out


# ── utils/env_file ──────────────────────────────────────────────────


class TestUpsertEnvFile:
    def test_adds_new_key_and_preserves_comments(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# comment line\nPROVIDER=local\n", encoding="utf-8")
        assert upsert_env_file(str(p), "REPORT_MODEL", "claude-x") == "added"
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# comment line"
        assert lines[1] == "PROVIDER=local"
        assert lines[-1] == "REPORT_MODEL=claude-x"

    def test_updates_existing_key(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("PROVIDER=local\nREPORT_MODEL=old\n", encoding="utf-8")
        assert upsert_env_file(str(p), "REPORT_MODEL", "new") == "updated"
        assert "REPORT_MODEL=new" in p.read_text(encoding="utf-8")
        assert "old" not in p.read_text(encoding="utf-8")

    def test_same_value_is_unchanged(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("REPORT_MODEL=same\n", encoding="utf-8")
        assert upsert_env_file(str(p), "REPORT_MODEL", "same") == "unchanged"

    def test_creates_file_when_missing(self, tmp_path):
        p = tmp_path / ".env"
        assert upsert_env_file(str(p), "EXTRACTION_PROVIDER", "openai") == "added"
        assert p.read_text(encoding="utf-8").strip() == "EXTRACTION_PROVIDER=openai"

    def test_replaces_duplicate_occurrences(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("A=1\nA=2\n", encoding="utf-8")
        upsert_env_file(str(p), "A", "3")
        assert p.read_text(encoding="utf-8").splitlines() == ["A=3"]

    def test_empty_value_means_default(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("REPORT_PROVIDER=anthropic\n", encoding="utf-8")
        upsert_env_file(str(p), "REPORT_PROVIDER", "")
        assert "REPORT_PROVIDER=" in p.read_text(encoding="utf-8")


# ── run.py models — the models browser ──────────────────────────────


def _script_input(monkeypatch, script: list[str]) -> None:
    """Replace builtins.input with a scripted one (EOFError when exhausted)."""
    it = iter(script)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)


class TestModelsBrowser:
    def test_shows_resolved_table_and_guidance(self, capsys, monkeypatch, tmp_path):
        _script_input(monkeypatch, ["q"])
        ui_views.show_models(env_path=str(tmp_path / ".env"))
        out = capsys.readouterr().out
        assert "Models per step" in out
        assert "default:" in out
        for step in ("planning", "extraction", "curation", "report"):
            assert f"{step}:" in out
        assert "set <step>" in out and "unset <step>" in out

    def test_set_oneline_writes_env_and_config(self, capsys, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("PROVIDER=local\n", encoding="utf-8")
        _script_input(monkeypatch, ["set report anthropic/claude-sonnet-5-5", "q"])
        ui_views.show_models(env_path=str(env))
        out = capsys.readouterr().out
        assert "report → anthropic/claude-sonnet-5-5" in out
        lines = env.read_text(encoding="utf-8").splitlines()
        assert "REPORT_PROVIDER=anthropic" in lines
        assert "REPORT_MODEL=claude-sonnet-5-5" in lines
        assert "PROVIDER=local" in lines  # untouched line preserved
        assert workflow_config.report_provider == "anthropic"
        assert workflow_config.report_model == "claude-sonnet-5-5"

    def test_set_bare_name_keeps_current_provider(self, capsys, monkeypatch, tmp_path):
        _script_input(monkeypatch, ["set extraction fast-7b", "q"])
        ui_views.show_models(env_path=str(tmp_path / ".env"))
        out = capsys.readouterr().out
        assert "extraction → local/fast-7b" in out
        assert workflow_config.extraction_model == "fast-7b"
        assert workflow_config.extraction_provider == ""

    def test_set_interactive_prompts(self, capsys, monkeypatch, tmp_path):
        _script_input(monkeypatch, ["set curation", "openai", "gpt-4o-mini", "q"])
        ui_views.show_models(env_path=str(tmp_path / ".env"))
        out = capsys.readouterr().out
        assert "curation → openai/gpt-4o-mini" in out
        assert workflow_config.curation_provider == "openai"
        assert workflow_config.curation_model == "gpt-4o-mini"

    def test_set_unknown_provider_rejected(self, capsys, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("PROVIDER=local\n", encoding="utf-8")
        _script_input(monkeypatch, ["set report gemini/whatever", "q"])
        ui_views.show_models(env_path=str(env))
        out = capsys.readouterr().out
        assert "Unknown provider" in out
        assert "gemini" not in env.read_text(encoding="utf-8")
        assert workflow_config.report_provider == ""

    def test_set_unknown_step_rejected(self, capsys, monkeypatch, tmp_path):
        _script_input(monkeypatch, ["set summarization x/y", "q"])
        ui_views.show_models(env_path=str(tmp_path / ".env"))
        out = capsys.readouterr().out
        assert "Usage: set" in out

    def test_unset_clears_overrides(self, capsys, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        _script_input(monkeypatch, ["set report anthropic/claude-x", "q"])
        ui_views.show_models(env_path=str(env))
        _script_input(monkeypatch, ["unset report", "q"])
        ui_views.show_models(env_path=str(env))
        out = capsys.readouterr().out
        assert "report → default" in out
        assert "REPORT_PROVIDER=" in env.read_text(encoding="utf-8")
        assert "REPORT_MODEL=" in env.read_text(encoding="utf-8")
        assert workflow_config.report_provider == ""
        assert workflow_config.report_model == ""

    def test_env_command_prints_env_lines(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr(workflow_config, "report_provider", "anthropic")
        monkeypatch.setattr(workflow_config, "report_model", "claude-x")
        _script_input(monkeypatch, ["env", "q"])
        ui_views.show_models(env_path=str(tmp_path / ".env"))
        out = capsys.readouterr().out
        assert "REPORT_PROVIDER=anthropic" in out
        assert "REPORT_MODEL=claude-x" in out
        assert "PLANNING_PROVIDER=" in out


class TestModelsSubcommand:
    def test_run_py_models_dispatch(self, capsys, monkeypatch):
        import run as run_mod

        _script_input(monkeypatch, ["q"])
        run_mod.main(["models"])
        out = capsys.readouterr().out
        assert "Models per step" in out
