"""Tests for the UI session registry (custom titles, archived flag)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# pyright: reportPrivateUsage=false
from medmcp import sessions


@pytest.fixture(autouse=True)
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry at a temp file for every test."""
    path = tmp_path / "ui_sessions.json"
    monkeypatch.setattr(sessions, "REGISTRY_PATH", path)
    return path


def test_missing_file_is_empty() -> None:
    """No file yet → an empty registry, not an error."""
    assert sessions.load_registry() == {}


def test_corrupt_file_is_empty(registry_path: Path) -> None:
    """A garbage file is treated as empty rather than raising."""
    registry_path.write_text("not json{", encoding="utf-8")
    assert sessions.load_registry() == {}


def test_set_title_round_trips() -> None:
    """A title set is read back, and persisted to disk."""
    sessions.set_title("s1", "  Brain MRI pipeline  ")
    assert sessions.get_entry("s1") == {"title": "Brain MRI pipeline"}


def test_blank_title_clears_and_drops_entry() -> None:
    """Clearing the only field removes the entry entirely (file stays small)."""
    sessions.set_title("s1", "x")
    sessions.set_title("s1", "   ")
    assert sessions.get_entry("s1") == {}
    assert sessions.load_registry() == {}


def test_archive_and_restore() -> None:
    """Archiving sets the flag; restoring removes the now-empty entry."""
    sessions.set_archived("s1", True)
    assert sessions.get_entry("s1") == {"archived": True}
    sessions.set_archived("s1", False)
    assert sessions.load_registry() == {}


def test_title_and_archived_coexist() -> None:
    """Independent fields don't clobber each other; clearing one keeps the other."""
    sessions.set_title("s1", "Keep me")
    sessions.set_archived("s1", True)
    assert sessions.get_entry("s1") == {"title": "Keep me", "archived": True}
    sessions.set_archived("s1", False)
    assert sessions.get_entry("s1") == {"title": "Keep me"}


def test_remove_forgets_entry() -> None:
    """remove() deletes a session's metadata."""
    sessions.set_title("s1", "x")
    sessions.set_title("s2", "y")
    sessions.remove("s1")
    assert "s1" not in sessions.load_registry()
    assert sessions.get_entry("s2") == {"title": "y"}


def test_writes_are_valid_json(registry_path: Path) -> None:
    """The persisted file is plain JSON keyed by session id."""
    sessions.set_title("s1", "Title")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    assert data == {"s1": {"title": "Title"}}
