"""Shared fixtures for the whole test suite.

Goal: hermetic — no test may depend on the developer's real .env (live
keys, LAN LLM endpoint) or reach the network.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the package root is importable (same pattern as the test files).
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from config.settings import config  # noqa: E402


@pytest.fixture(autouse=True)
def dead_local_endpoint(monkeypatch):
    """Pin the local LLM endpoint to a dead loopback port.

    The effective-config banner probes LOCAL_BASE_URL (GET /models) to
    warn before a session that would otherwise degrade silently. Port 9
    on loopback refuses instantly, so the probe fails fast and
    deterministically — no external network, no dependence on whether
    the developer's local LLM server happens to be up.
    """
    monkeypatch.setattr(config, "local_base_url", "http://127.0.0.1:9/v1")
