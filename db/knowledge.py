"""
Database CRUD operations — all table access is centralized here.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .engine import get_db, _now, _row_to_dict, _serialize

try:
    from ..config.settings import paths_config
except ImportError:
    from config.settings import paths_config


# ── Plan operations ──────────────────────────────────────────────────


def create_plan(query: str, questions: List[dict], thread_id: str = "") -> str:
    """Create a research plan and return its ID.

    IDs are uuid4-based (not hash-based) so they are stable across processes
    and cannot collide.
    """
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO plans (id, query, questions, thread_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (plan_id, query, _serialize(questions), thread_id, _now()),
        )
    return plan_id


def get_plan(plan_ref: str) -> Optional[Dict]:
    """Fetch a plan by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_ref,)).fetchone()
    return _row_to_dict(row) if row else None


# ── Question bank operations ─────────────────────────────────────────


def create_questions(plan_id: str, questions: List[dict]) -> List[str]:
    """Create questions from user input and return their IDs."""
    ids = []
    with get_db() as conn:
        for i, q in enumerate(questions):
            qid = f"q_{plan_id}_{i}"
            conn.execute(
                """INSERT INTO questions (id, plan_id, question, status, priority, discovered_by, created_at)
                   VALUES (?, ?, ?, 'pending', ?, NULL, ?)""",
                (
                    qid,
                    plan_id,
                    q.get("question", ""),
                    q.get("priority", "medium"),
                    _now(),
                ),
            )
            ids.append(qid)
    return ids


def get_pending_question(plan_id: str) -> Optional[Dict]:
    """Get the next pending question for a plan, ordered by priority."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM questions
               WHERE plan_id = ? AND status = 'pending'
               ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END ASC
               LIMIT 1""",
            (plan_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_pending_questions(plan_id: str) -> List[Dict]:
    """Get ALL pending questions for a plan, ordered by priority."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM questions
               WHERE plan_id = ? AND status = 'pending'
               ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END ASC""",
            (plan_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_question_priority(question_id: str, new_priority: str) -> None:
    """Update a question's priority."""
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET priority = ? WHERE id = ?",
            (new_priority, question_id),
        )


def get_in_progress_question(plan_id: str) -> Optional[Dict]:
    """Get the question currently marked in_progress for a plan.

    Normally there is at most one; if several exist (a crash left a stale
    row), prefer the most recently created one — the one the backlog node
    most recently dispatched.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM questions WHERE plan_id = ? AND status = 'in_progress' ORDER BY created_at DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_question(question_id: str) -> Optional[Dict]:
    """Get a single question by ID (any status)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def set_question_in_progress(question_id: str) -> None:
    """Mark a question as in_progress."""
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET status = 'in_progress' WHERE id = ?", (question_id,)
        )


def set_question_resolved(question_id: str) -> None:
    """Mark a question as resolved."""
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET status = 'resolved' WHERE id = ?", (question_id,)
        )


def set_question_failed(question_id: str, error: str = "") -> None:
    """Mark a question as failed (e.g. the LLM call failed).

    'failed' is NON-terminal: failed questions keep blocking
    all_questions_are_resolved(), so the loop cannot silently declare a
    plan complete when research actually failed. The interim analysis
    screen shows them so a human can decide what to do.
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET status = 'failed', last_error = ? WHERE id = ?",
            (error, question_id),
        )


def set_question_pending(question_id: str) -> None:
    """Reset a question's status to pending (used by retry command)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET status = 'pending', last_error = NULL WHERE id = ?",
            (question_id,),
        )


def reset_stale_in_progress(plan_id: str) -> None:
    """Reset in_progress questions back to pending (self-heal after a crash).

    A crash between set_question_in_progress() and set_question_resolved()
    orphans a question as in_progress: it never shows up on the review
    screen, it blocks all_questions_are_resolved(), and
    get_in_progress_question() keeps returning it, so every iteration
    reprocesses the same stale row while the rest of the backlog sits
    behind it. manage_backlog_node() calls this before each pick so the
    stale question re-enters the pending pool and competes normally.
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET status = 'pending', last_error = NULL WHERE plan_id = ? AND status = 'in_progress'",
            (plan_id,),
        )


def add_question(plan_id: str, question: str, discovered_by: str = "extraction") -> str:
    """Add a new question to the bank.

    Uses a uuid4 suffix so dynamically added questions can never collide
    with the index-based IDs created by create_questions().
    """
    qid = f"q_{plan_id}_{uuid.uuid4().hex[:8]}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO questions (id, plan_id, question, status, priority, discovered_by, created_at)
               VALUES (?, ?, ?, 'pending', 'medium', ?, ?)""",
            (qid, plan_id, question, discovered_by, _now()),
        )
    return qid


def get_plan_stats(plan_id: str) -> Dict:
    """Get question counts by status for a plan."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM questions WHERE plan_id = ? GROUP BY status",
            (plan_id,),
        ).fetchall()
    return {row["status"]: row["count"] for row in rows}


def deduplicate_and_prioritize(plan_id: str) -> None:
    """Deduplicate and reprioritize questions in the question bank.

    Runs during curation to keep the backlog clean.
    - Groups near-duplicate questions (word-overlap similarity)
    - Keeps the highest-priority representative from each group
    - Updates discovered_by to 'curation' for tracking
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM questions WHERE plan_id = ? AND status = 'pending'",
            (plan_id,),
        ).fetchall()

    if len(rows) < 2:
        return  # nothing to dedup

    # Extract key words from a question (remove common stop words)
    STOP_WORDS = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "how",
        "do",
        "does",
        "did",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "about",
        "that",
        "this",
        "it",
        "and",
        "or",
        "but",
        "not",
        "no",
        "can",
        "will",
        "would",
        "should",
        "could",
        "been",
        "being",
    }

    def normalize_word(w: str) -> str:
        """Normalize a word: lowercase, strip punctuation, remove hyphens, drop trailing 's'."""
        w = w.strip("?.!,;:\"'").replace("-", "").replace("_", "").lower()
        # Simple plural handling: drop trailing 's' (not for 'ss' words)
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        return w

    def keywords(text: str) -> set:
        return {
            normalize_word(w)
            for w in text.lower().split()
            if normalize_word(w) not in STOP_WORDS
        }

    def similarity(a: set, b: set) -> float:
        """Jaccard similarity between two keyword sets."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Group questions by word-overlap similarity.
    # Precompute keyword sets to avoid O(n²·m) recomputation.
    groups: list[tuple[list, list[set]]] = []  # (question_list, keyword_set_list)
    for row in rows:
        kw = keywords(row["question"])
        merged = False
        for group_qs, group_kws in groups:
            # Check similarity against all precomputed members of this group
            if any(similarity(kw, gkw) >= 0.4 for gkw in group_kws):
                group_qs.append(dict(row))
                group_kws.append(kw)
                merged = True
                break
        if not merged:
            groups.append(([dict(row)], [kw]))

    priority_order = {"high": 0, "medium": 1, "low": 2}

    # Keep the best question from each group, discard the rest
    with get_db() as conn:
        for group_qs, _group_kws in groups:
            if len(group_qs) == 1:
                continue

            # Sort by priority (high first), then by creation date
            group_qs.sort(key=lambda q: priority_order.get(q["priority"], 1))
            keep = group_qs[0]

            for q in group_qs[1:]:
                # Update the keeper to inherit any higher priority
                if priority_order.get(q["priority"], 1) < priority_order.get(
                    keep["priority"], 1
                ):
                    conn.execute(
                        "UPDATE questions SET priority = ? WHERE id = ?",
                        (q["priority"], keep["id"]),
                    )
                # Mark duplicates as discarded
                conn.execute(
                    "UPDATE questions SET status = 'discarded', priority = ?, discovered_by = ? WHERE id = ?",
                    ("low", "curation", q["id"]),
                )


def all_questions_are_resolved(plan_id: str) -> bool:
    """Check if all questions in a plan are in a terminal state.

    'discarded' (deduped duplicates) is terminal — otherwise a plan can
    never be ready because dedup runs at every review. 'failed' is NOT
    terminal: a failed question is a real gap that must be surfaced.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM questions WHERE plan_id = ? AND status NOT IN ('resolved', 'discarded')",
            (plan_id,),
        ).fetchone()
    return (row["cnt"] == 0) if row else True


# ── Search cache operations ──────────────────────────────────────────


def cache_check(query_hash: str) -> Optional[dict]:
    """Check if a query hash exists in the cache."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM search_cache WHERE query_hash = ? AND cached_at > datetime('now', '-7 days')",
            (query_hash,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def cache_store(query_hash: str, query: str, engine: str, results: list) -> None:
    """Store a search result in the cache."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO search_cache (query_hash, query, engine, results, cached_at)
               VALUES (?, ?, ?, ?, ?)""",
            (query_hash, query, engine, _serialize(results), _now()),
        )


# ── Raw results operations ───────────────────────────────────────────


def store_raw_result(
    question_id: str,
    plan_id: str,
    query: str,
    engine: str,
    content: str,
    title: str = "",
    source_url: str = "",
) -> int:
    """Store a raw search result and return its ID."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO raw_results (question_id, plan_id, query, engine, content, title, source_url, retrieved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (question_id, plan_id, query, engine, content, title, source_url, _now()),
        )
        return cur.lastrowid


def get_raw_results_for_question(question_id: str) -> List[Dict]:
    """Get all raw results for a question."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM raw_results WHERE question_id = ? ORDER BY id DESC",
            (question_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ── Knowledge entries operations ─────────────────────────────────────


def create_knowledge_entry(
    title: str,
    summary: str,
    topic_tags: list,
    full_text_path: str,
    source_url: str = "",
    discovered_by: str = "",
    full_text: str = "",
) -> str:
    """Create a knowledge entry and return its ID.

    Args:
        full_text_path: Path to full text file.
        full_text: Inline full text content (used when no file path).
    """
    entry_id = f"entry_{uuid.uuid4().hex[:8]}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO knowledge_entries (id, title, topic_tags, summary, full_text_path, source_url, status, discovered_by, added_at, curated_at, full_text)
               VALUES (?, ?, ?, ?, ?, ?, 'raw', ?, ?, NULL, ?)""",
            (
                entry_id,
                title,
                _serialize(topic_tags),
                summary,
                full_text_path,
                source_url,
                discovered_by,
                _now(),
                full_text,
            ),
        )
    return entry_id


def update_knowledge_status(entry_id: str, status: str) -> Dict:
    """Update knowledge entry status."""
    now = _now() if status == "curated" else None
    with get_db() as conn:
        conn.execute(
            "UPDATE knowledge_entries SET status = ?, curated_at = ? WHERE id = ?",
            (status, now, entry_id),
        )
        row = conn.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    return _row_to_dict(row)


def update_tags(entry_id: str, tags: list) -> Dict:
    """Convenience wrapper around update_knowledge_tags."""
    return update_knowledge_tags(entry_id, tags)


def update_knowledge_tags(entry_id: str, tags: list) -> Dict:
    """Update topic tags for an entry."""
    with get_db() as conn:
        conn.execute(
            "UPDATE knowledge_entries SET topic_tags = ? WHERE id = ?",
            (_serialize(tags), entry_id),
        )
        row = conn.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    return _row_to_dict(row)


def search_knowledge(
    query: str,
    status: str = "curated",
    limit: int = 10,
    created_after: str = "",
    plan_id: str = "",
) -> List[Dict]:
    """Search knowledge entries by keyword across title, summary, and tags.

    Args:
        query: Search terms (space-separated; ALL terms must match — each
               is matched against title, summary, and tags).
        status: Filter by status ('raw', 'curated', 'discarded'). Default: 'curated'.
        limit: Max results to return.
        created_after: ISO timestamp — only return entries created after this time.
                       Format: "2026-10-24T15:00:00" or "2026-10-24"
        plan_id: When set, only return entries linked to this plan's findings
                 (entries are linked via curated_findings → questions → plans).

    Returns list of matching entries with a relevance score.
    """
    terms = query.lower().split()
    if not terms:
        return []

    # Build dynamic WHERE clause for term matching
    conditions = []
    params = []
    for term in terms:
        conditions.append("(title LIKE ? OR summary LIKE ? OR topic_tags LIKE ?)")
        like_term = f"%{term}%"
        params.extend([like_term, like_term, like_term])

    # Add time filter if provided
    if created_after:
        conditions.append("added_at > ?")
        params.append(created_after)

    # Add plan scope: entries reachable through this plan's findings
    if plan_id:
        conditions.append(
            """EXISTS (SELECT 1 FROM curated_findings cf
                       JOIN questions q ON q.id = cf.question_id
                       WHERE cf.knowledge_entry_id = knowledge_entries.id
                         AND q.plan_id = ?)"""
        )
        params.append(plan_id)

    where_clause = " AND ".join(conditions)

    with get_db() as conn:
        sql = f"""
            SELECT id, title, summary, topic_tags, source_url, status, added_at
            FROM knowledge_entries
            WHERE {where_clause}
              AND status = ?
            ORDER BY added_at DESC
            LIMIT ?
        """
        params.extend([status, limit])
        rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        # Calculate simple relevance score (more matches = higher score)
        score = 0
        for term in terms:
            if term in (d.get("title", "") or "").lower():
                score += 10  # Title match is strongest signal
            if term in (d.get("summary", "") or "").lower():
                score += 5
            if term in (d.get("topic_tags", "") or "").lower():
                score += 3
        d["relevance_score"] = score
        results.append(d)

    return sorted(results, key=lambda x: -x["relevance_score"])


def get_plan_entries(plan_id: str, status: str = "") -> List[Dict]:
    """Get knowledge entries linked to a plan's findings.

    Entries are associated with plans through the findings that created
    them (curated_findings → questions → plans).

    Args:
        plan_id: Plan to scope to.
        status: Optional status filter ('raw', 'curated', 'discarded').
    """
    with get_db() as conn:
        sql = """SELECT DISTINCT k.*
                 FROM knowledge_entries k
                 JOIN curated_findings cf ON cf.knowledge_entry_id = k.id
                 JOIN questions q ON q.id = cf.question_id
                 WHERE q.plan_id = ?"""
        params: list = [plan_id]
        if status:
            sql += " AND k.status = ?"
            params.append(status)
        sql += " ORDER BY k.added_at DESC"
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ── Curated findings (notecards) operations ──────────────────────────


def create_finding(
    question_id: str,
    raw_result_id: Optional[int],
    knowledge_entry_id: Optional[str],
    facts: list,
    summary: str,
    tags: list,
) -> str:
    """Create a curated finding (notecard) and return its ID."""
    finding_id = f"finding_{uuid.uuid4().hex[:8]}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO curated_findings (id, question_id, raw_result_id, knowledge_entry_id, facts, summary, tags, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'extracted', ?)""",
            (
                finding_id,
                question_id,
                raw_result_id,
                knowledge_entry_id,
                _serialize(facts),
                summary,
                _serialize(tags),
                _now(),
            ),
        )
    return finding_id


def create_degraded_evidence(
    question_id: str, question_text: str, raw_results: List[Dict], error: str
) -> str:
    """Record a placeholder finding for a question whose extraction failed.

    The search succeeded (raw results exist) but the LLM could not produce
    structured findings. The placeholder is tagged 'extraction-failed' so the
    quality filter keeps it (utils/quality.py) and the report/review screen
    can show the gap instead of losing it. Returns the knowledge entry ID.
    """
    urls = [rr.get("source_url", "") for rr in raw_results if rr.get("source_url")]
    summary = (
        f"Extraction failed ({error[:120]}). {len(raw_results)} raw source(s) were "
        "found, but no structured findings could be produced."
    )
    entry_id = create_knowledge_entry(
        title=f"[no extraction] {question_text[:100]}",
        summary=summary,
        topic_tags=["extraction-failed"],
        full_text_path="",
        source_url=urls[0] if urls else "",
        discovered_by="extraction-fallback",
        full_text=summary,
    )
    create_finding(
        question_id=question_id,
        raw_result_id=raw_results[0]["id"] if raw_results else None,
        knowledge_entry_id=entry_id,
        facts=[{"claim": summary, "confidence": "low", "source": 0}],
        summary=summary,
        tags=["extraction-failed"],
    )
    return entry_id


def get_plan_findings(plan_id: str, status: Optional[str] = None) -> List[Dict]:
    """Get all findings belonging to a plan (via its questions).

    All report/DoD/review paths must be scoped this way — findings from
    other plans must never leak into a plan's report or readiness check.
    """
    with get_db() as conn:
        sql = """SELECT cf.*, q.question as source_question, k.title as source_title
                 FROM curated_findings cf
                 JOIN questions q ON cf.question_id = q.id
                 LEFT JOIN knowledge_entries k ON cf.knowledge_entry_id = k.id
                 WHERE q.plan_id = ?"""
        params: list = [plan_id]
        if status:
            sql += " AND cf.status = ?"
            params.append(status)
        sql += " ORDER BY cf.created_at DESC"
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ── Stats ────────────────────────────────────────────────────────────


def get_stats() -> Dict:
    """Return summary counts by status and top tags."""
    with get_db() as conn:
        status_counts = conn.execute(
            "SELECT status, COUNT(*) as count FROM knowledge_entries GROUP BY status"
        ).fetchall()
        tag_counts = conn.execute("SELECT topic_tags FROM knowledge_entries").fetchall()

    status_map = {row["status"]: row["count"] for row in status_counts}

    tag_freq: dict[str, int] = {}
    for row in tag_counts:
        try:
            tags = json.loads(row["topic_tags"])
            for tag in tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    top_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "by_status": status_map,
        "top_tags": top_tags,
        "total": sum(status_map.values()),
    }


# ── Pre-flight: find similar active plans ──────────────────────────

STOP_WORDS = {
    "what",
    "is",
    "are",
    "was",
    "were",
    "how",
    "do",
    "does",
    "did",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "about",
    "that",
    "this",
    "it",
    "and",
    "or",
    "but",
    "not",
    "no",
    "can",
    "will",
    "would",
    "should",
    "could",
    "been",
    "being",
    "looking",
    "forward",
    "prices",
    "capacities",
    "help",
    "like",
    "support",
    "approaching",
    "way",
    "sustainable",
    "best",
    "ideas",
}


def _normalize_words(text: str) -> set:
    """Extract normalized keywords from text."""
    words = set()
    for w in text.lower().split():
        w = w.strip("?.!,;:\"'()[]{}")
        if len(w) > 1 and w not in STOP_WORDS:
            words.add(w)
    return words


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_similar_active_plans(
    new_query: str, threshold: float = 0.25, exclude_plan_id: str = ""
) -> list[dict]:
    """Find active plans whose query is similar to the new query.

    Returns a list of dicts (sorted by similarity descending):
        {
            "plan_id": str,
            "query": str,
            "similarity": float,
            "pending_count": int,
            "resolved_count": int,
            "progress_pct": int,
        }

    Only returns plans where NOT all questions are resolved (still active).
    Optionally excludes a specific plan ID (e.g. the plan being continued).
    """
    new_kw = _normalize_words(new_query)
    matches: list[dict] = []

    with get_db() as conn:
        plans = conn.execute(
            """SELECT id, query FROM plans ORDER BY created_at"""
        ).fetchall()

    for row in plans:
        pid = row["id"]
        query = row["query"]

        # Skip the plan being continued (self-match is noise)
        if pid == exclude_plan_id:
            continue

        # Skip if already all resolved
        if all_questions_are_resolved(pid):
            continue

        similarity = _jaccard(new_kw, _normalize_words(query))
        if similarity < threshold:
            continue

        stats = get_plan_stats(pid)
        resolved = stats.get("resolved", 0)
        pending = stats.get("pending", 0) + stats.get("failed", 0)
        total = resolved + pending
        progress_pct = int(resolved / total * 100) if total > 0 else 0

        matches.append(
            {
                "plan_id": pid,
                "query": query,
                "similarity": round(similarity, 2),
                "pending_count": pending,
                "resolved_count": resolved,
                "progress_pct": progress_pct,
            }
        )

    matches.sort(key=lambda m: -m["similarity"])
    return matches


# ── Report operations ───────────────────────────────────────────────


def save_report(plan_id: str, query: str, report: str) -> str:
    """Save a research report (file first, then DB) and return its ID.

    The file is written before the DB row so a failed insert can at worst
    leave an orphan file (the user-visible artifact), never a DB row
    pointing at a missing file. Filenames get a counter suffix on collision.
    """
    report_id = f"report_{uuid.uuid4().hex[:8]}"

    base_dir = paths_config.base_dir
    os.makedirs(base_dir, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in query.lower())[:80]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(base_dir, f"{timestamp}_{slug}.md")
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(base_dir, f"{timestamp}_{slug}_{counter}.md")
        counter += 1
    with open(filepath, "w") as f:
        f.write(report)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO reports (id, plan_id, query, report, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (report_id, plan_id, query, report, _now()),
        )
    return report_id


def report_quality_gate(plan_id: str) -> dict:
    """Assess whether there's sufficient evidence for a quality report.

    Returns:
        {"ok": bool, "reason": str, "stats": dict}

    Quality checks:
    - All findings discarded as generic/filler → insufficient evidence
    - No sources found → no evidence
    - Contradictions outnumber resolved findings → conflicting evidence
    """
    if not plan_id:
        return {"ok": False, "reason": "no plan", "stats": {}}

    try:
        from nodes.curation import _filter_quality

        all_findings = get_plan_findings(plan_id)
        valid_findings, discarded = _filter_quality(all_findings)

        with get_db() as conn:
            # Count unique sources (raw results with non-citation engines)
            source_count = conn.execute(
                """SELECT COUNT(DISTINCT rr.id)
                   FROM raw_results rr
                   WHERE rr.plan_id = ? AND rr.engine NOT LIKE 'exa-answer%'""",
                (plan_id,),
            ).fetchone()[0]

            # Count resolved questions
            resolved_count = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE plan_id = ? AND status = 'resolved'",
                (plan_id,),
            ).fetchone()[0]

        stats = {
            "total_findings": len(all_findings),
            "valid_findings": len(valid_findings),
            "discarded": len(discarded),
            "sources": source_count,
            "resolved_questions": resolved_count,
        }

        # Check 1: No sources found at all (check first — covers empty plans)
        if source_count == 0:
            return {"ok": False, "reason": "no_sources", "stats": stats}

        # Check 2: All findings discarded as generic/filler
        if len(valid_findings) == 0 and len(discarded) > 0:
            return {"ok": False, "reason": "insufficient_evidence", "stats": stats}

        return {"ok": True, "reason": "sufficient", "stats": stats}
    except Exception:
        return {"ok": True, "reason": "sufficient", "stats": {}}


def get_source_coverage(plan_id: str) -> list[dict]:
    """Return per-source coverage stats for a plan.

    Inspired by deep-researcher-node's Evidence Matrix: a machine-readable
    control surface that tracks which sources are well-covered vs
    under-represented.

    Returns list of dicts with:
        source_url, source_title, fact_count, first_seen, last_seen
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT rr.source_url, rr.title,
                          COUNT(f.id) as fact_count,
                          MIN(rr.retrieved_at) as first_seen,
                          MAX(rr.retrieved_at) as last_seen
                   FROM raw_results rr
                   LEFT JOIN curated_findings f ON f.raw_result_id = rr.id
                   WHERE rr.plan_id = ?
                     AND rr.engine NOT LIKE 'exa-answer%'
                   GROUP BY rr.id
                   ORDER BY fact_count DESC""",
                (plan_id,),
            ).fetchall()
        return [
            {
                "source_url": r[0] or "",
                "source_title": r[1] or "",
                "fact_count": r[2],
                "first_seen": r[3] or "",
                "last_seen": r[4] or "",
            }
            for r in rows
        ]
    except Exception:
        return []
