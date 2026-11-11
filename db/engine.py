"""
SQLite database engine for the research agent knowledge base.

Handles connection, schema creation, and all database operations.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator

from config.settings import paths_config


# ── Connection helpers ───────────────────────────────────────────────


def _get_connection() -> sqlite3.Connection:
    """Return a new sqlite3 connection with Row factory.

    Uses RESEARCH_DB_PATH env var if set (for testing), otherwise
    falls back to the configured production path.
    """
    db_path = os.environ.get("RESEARCH_DB_PATH") or paths_config.db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables. Safe to call multiple times."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS plans (
                id         TEXT PRIMARY KEY,
                query      TEXT NOT NULL,
                questions  TEXT NOT NULL,
                thread_id  TEXT,
                scope      TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS questions (
                id             TEXT PRIMARY KEY,
                plan_id        TEXT NOT NULL,
                question       TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                priority       TEXT NOT NULL DEFAULT 'medium',
                discovered_by  TEXT,
                last_error     TEXT,
                created_at     TEXT NOT NULL,
                FOREIGN KEY (plan_id) REFERENCES plans(id)
            );
            CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);
            CREATE INDEX IF NOT EXISTS idx_questions_plan   ON questions(plan_id);

            CREATE TABLE IF NOT EXISTS search_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash  TEXT NOT NULL UNIQUE,
                query       TEXT NOT NULL,
                engine      TEXT NOT NULL,
                results     TEXT NOT NULL,
                cached_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                plan_id     TEXT NOT NULL,
                query       TEXT NOT NULL,
                engine      TEXT NOT NULL,
                content     TEXT NOT NULL,
                title       TEXT,
                source_url  TEXT,
                retrieved_at TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_raw_question ON raw_results(question_id);
            CREATE INDEX IF NOT EXISTS idx_raw_plan ON raw_results(plan_id);
            CREATE INDEX IF NOT EXISTS idx_raw_engine ON raw_results(engine);

            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id               TEXT PRIMARY KEY,
                title            TEXT NOT NULL,
                topic_tags       TEXT NOT NULL,
                summary          TEXT NOT NULL,
                full_text_path   TEXT NOT NULL,
                source_url       TEXT,
                status           TEXT NOT NULL DEFAULT 'raw',
                discovered_by    TEXT,
                added_at         TEXT NOT NULL,
                curated_at       TEXT,
                full_text        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_entries(status);
            CREATE INDEX IF NOT EXISTS idx_knowledge_tags   ON knowledge_entries(topic_tags);
            CREATE INDEX IF NOT EXISTS idx_knowledge_added  ON knowledge_entries(added_at);

            CREATE TABLE IF NOT EXISTS curated_findings (
                id                   TEXT PRIMARY KEY,
                question_id          TEXT NOT NULL,
                raw_result_id        INTEGER,
                knowledge_entry_id   TEXT,
                facts                TEXT NOT NULL,
                summary              TEXT NOT NULL,
                tags                 TEXT NOT NULL,
                status               TEXT NOT NULL DEFAULT 'extracted',
                created_at           TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (raw_result_id) REFERENCES raw_results(id),
                FOREIGN KEY (knowledge_entry_id) REFERENCES knowledge_entries(id)
            );
            CREATE INDEX IF NOT EXISTS idx_findings_question ON curated_findings(question_id);
            CREATE INDEX IF NOT EXISTS idx_findings_status   ON curated_findings(status);

            CREATE TABLE IF NOT EXISTS reports (
                id         TEXT PRIMARY KEY,
                plan_id    TEXT NOT NULL,
                query      TEXT NOT NULL,
                report     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (plan_id) REFERENCES plans(id)
            );
        """)
        # Migrations for pre-existing DBs (CREATE TABLE IF NOT EXISTS doesn't add columns)
        for migration in (
            "ALTER TABLE plans ADD COLUMN thread_id TEXT",
            "ALTER TABLE questions ADD COLUMN last_error TEXT",
            "ALTER TABLE plans ADD COLUMN scope TEXT",
        ):
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database operations."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Helpers ──────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> str:
    """Serialize a value to JSON text for storage."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _deserialize(text: str) -> Any:
    """Deserialize JSON text from the database."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _row_to_dict(row) -> Dict:
    """Convert a sqlite3.Row to a plain dict, JSON-parsing text fields."""
    if row is None:
        return {}
    d = dict(row)
    for key in ("questions", "results", "facts", "topic_tags", "tags"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
