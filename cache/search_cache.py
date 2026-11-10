"""
Search cache — SQLite-backed, avoid re-querying the same searches.
"""

from __future__ import annotations

import hashlib

from db.knowledge import cache_check, cache_store


def _normalize_query(query: str) -> str:
    return query.strip().lower()


def hash_query(query: str) -> str:
    return hashlib.sha256(_normalize_query(query).encode()).hexdigest()[:16]


def check_cache(query: str) -> list | None:
    """Check SQLite cache for a matching, non-expired result."""
    q_hash = hash_query(query)
    cached = cache_check(q_hash)
    if cached:
        return cached.get("results")
    return None


def store_cache(query: str, results: list, engine: str = "unknown") -> None:
    """Store a search result in the SQLite cache."""
    q_hash = hash_query(query)
    cache_store(q_hash, query, engine, results)
