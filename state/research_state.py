"""
ResearchState — minimal shared state for the LangGraph workflow.

All heavy data (plan, questions, cache, results, findings) lives in SQLite.
The state only carries control-flow information.

Every key a node returns MUST be declared here — LangGraph silently drops
undeclared keys from node output, which breaks control flow (e.g. the human
review response or the quality evaluation would never reach the next node).
"""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # The research question
    original_query: str

    # Optional user-supplied questions (bypasses LLM refinement)
    questions: list

    # Reference to the plan stored in the DB
    plan_ref: str

    # Loop counter
    iteration_count: int

    # Done flag
    is_done: bool

    # Internal routing hint (set by consolidation, read by router)
    _route: str

    # Human-in-the-loop: set by the review interrupt, consumed by curation
    _human_response: str | None

    # Review analysis payload — built once by prepare_review, consumed by
    # the interrupt node, cleared after the response
    _review_payload: dict

    # Consolidation outputs consumed by the review screen
    _quality: dict
    _hit_max_iterations: bool

    # Question-bank coverage computed by consolidation (total, closed, open,
    # high_open) — shown on the review screen and enforced by the coverage
    # floor gate on auto-stop.
    _coverage: dict

    # Stagnation tracking (set by consolidation, read by the next consolidation)
    _findings_streak: int
    _prev_finding_count: int

    # Display payloads for the CLI (transient, cleared after display)
    _stats_display: dict
    _findings_display: dict
    _contradictions_display: dict

    # Pre-flight warning: similar active plans found during plan creation
    _similar_plans: list

    # Error surfaced to the CLI
    _error: str

    # Final report text (set by the write_findings terminal node)
    final_report: str

    # Rewind flag — set by the "rewind" command on the review screen
    # (curate_knowledge_node); consumed by the router, which sends the graph
    # back to manage_backlog so the user can choose a different path.
    _rewind_from: int | None
