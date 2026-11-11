"""
Tests for the deterministic scope guard (utils/scope.py).

Covers entity extraction, scope-block construction, persistence on the
plan row, and that the curation/extraction prompt templates actually
consume the {scope} placeholder.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure we can import from the package root
sys.path.insert(0, os.path.dirname(__file__) + "/..")

os.environ["EXA_API_KEY"] = "test-key"

from utils.scope import build_scope_block, extract_entities, scope_for_plan


@pytest.fixture(autouse=True)
def fresh_test_db():
    """Same test-DB isolation as test_nodes.py (production data untouched)."""
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


# ── Entity extraction ───────────────────────────────────────────────


def test_extract_entities_finds_named_systems():
    entities = extract_entities("Compare vLLM, SGLang, and TGI for LLM serving")
    assert entities == ["vLLM", "SGLang", "TGI"]


def test_extract_entities_keeps_acronym_systems():
    entities = extract_entities("What does GPTQ vs AWQ mean for TGI deployments?")
    assert entities == ["GPTQ", "AWQ", "TGI"]


def test_extract_entities_dedupes_case_insensitively():
    entities = extract_entities("How does vLLM compare to SGLang and vLLM?")
    assert entities == ["vLLM", "SGLang"]


def test_extract_entities_no_named_systems():
    assert extract_entities("The future of micro-frontends in 2026") == []
    assert extract_entities("") == []


def test_extract_entities_ignores_years_and_generic_words():
    entities = extract_entities("How does TGI in 2026 compare to SGLang?")
    assert entities == ["TGI", "SGLang"]


# ── Scope block construction ────────────────────────────────────────


def test_build_scope_block_with_entities():
    block = build_scope_block("Compare vLLM, SGLang, and TGI for LLM serving")
    assert "RESEARCH SCOPE (enforced):" in block
    assert 'Topic: "Compare vLLM, SGLang, and TGI for LLM serving"' in block
    assert "Named systems in scope: vLLM, SGLang, TGI" in block


def test_build_scope_block_without_entities():
    block = build_scope_block("The future of micro-frontends in 2026")
    assert "RESEARCH SCOPE (enforced):" in block
    assert "Named systems in scope" not in block
    assert "No specific systems are named" in block


# ── Persistence + lookup ────────────────────────────────────────────


def test_scope_stored_on_plan_row():
    from db.knowledge import create_plan, get_plan

    q = "Compare vLLM, SGLang, and TGI for LLM serving"
    plan_id = create_plan(q, [], scope=build_scope_block(q))
    plan = get_plan(plan_id)
    assert plan["scope"] == build_scope_block(q)
    assert scope_for_plan(plan_id) == build_scope_block(q)


def test_scope_for_plan_rebuilds_when_column_empty():
    """Plans created before the column existed (scope='') still get a
    working scope block, rebuilt from the query."""
    from db.knowledge import create_plan

    plan_id = create_plan("Compare vLLM and TGI for serving", [])  # scope=""
    block = scope_for_plan(plan_id)
    assert "vLLM" in block and "TGI" in block
    assert scope_for_plan("") == ""
    assert scope_for_plan("plan_does_not_exist") == ""


def test_make_plan_node_stores_scope():
    """The planning node computes and persists the scope for every plan."""
    from db.knowledge import get_plan
    from nodes.planning import make_plan_node

    q = "Compare vLLM, SGLang, and TGI for LLM serving"
    result = make_plan_node({"original_query": q})
    plan = get_plan(result["plan_ref"])
    assert plan["scope"]
    assert "Named systems in scope: vLLM, SGLang, TGI" in plan["scope"]


# ── Prompt wiring ───────────────────────────────────────────────────


def test_prompt_templates_consume_scope():
    """EXTRACTION_USER and CURATE_USER must inject the scope block —
    if a {scope} placeholder is removed, this fails (not silently)."""
    from config.prompts import CURATE_USER, EXTRACTION_USER

    scope = build_scope_block("Compare vLLM and TGI for serving")
    p1 = EXTRACTION_USER.format(question="Q", context="ctx", scope=scope)
    p2 = CURATE_USER.format(query="Q", scope=scope, entries_text="e", findings_text="f")
    assert "RESEARCH SCOPE (enforced):" in p1
    assert "Named systems in scope: vLLM, TGI" in p1
    assert "RESEARCH SCOPE (enforced):" in p2
    assert "Named systems in scope: vLLM, TGI" in p2


def test_prompts_no_longer_hardcode_past_query_systems():
    """The DISCARD rules must be scope-driven, not anchored to the one
    topic they were originally written for."""
    from config.prompts import CURATE_SYSTEM, EXTRACTION_SYSTEM

    for name, template in (
        ("CURATE_SYSTEM", CURATE_SYSTEM),
        ("EXTRACTION_SYSTEM", EXTRACTION_SYSTEM),
    ):
        assert "Kafka" not in template, f"{name} still hard-codes Kafka"
        assert "vLLM" not in template, f"{name} still hard-codes vLLM"
        assert "SGLang" not in template, f"{name} still hard-codes SGLang"
        assert "Quark" not in template, f"{name} still hard-codes Quark"
