"""Approved stack calls: exact, single-use, short-lived."""

from __future__ import annotations

from medmcp import approvals


def test_an_approval_is_spent_by_the_one_call_it_was_for() -> None:
    """Arguments must match exactly, key order aside, and each grant covers one call."""
    ledger = approvals.ApprovalLedger()
    ledger.grant("s", "t", {"a": 1, "b": [1, 2]})
    ledger.grant("s", "t", {"a": 1, "b": [1, 2]})
    assert ledger.consume("s", "t", {"a": 2, "b": [1, 2]}) is False
    assert ledger.consume("s", "other", {"a": 1, "b": [1, 2]}) is False
    assert ledger.consume("s", "t", {"b": [1, 2], "a": 1}) is True
    assert ledger.consume("s", "t", {"a": 1, "b": [1, 2]}) is True
    assert ledger.consume("s", "t", {"a": 1, "b": [1, 2]}) is False


def test_an_unused_approval_expires() -> None:
    """An approval nobody used must not stay open."""
    now = [0.0]
    ledger = approvals.ApprovalLedger(ttl_sec=10, clock=lambda: now[0])
    ledger.grant("s", "t", {})
    now[0] = 11
    assert ledger.consume("s", "t", {}) is False
