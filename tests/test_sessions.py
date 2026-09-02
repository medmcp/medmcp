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


def test_auto_title_is_stored_and_marked() -> None:
    """A generated title lands with its source, so it can be told from a user's."""
    assert sessions.set_auto_title("s1", " Skull-strip subject 12 ") is True
    assert sessions.get_entry("s1") == {"title": "Skull-strip subject 12", "title_source": "auto"}
    assert sessions.has_manual_title(sessions.get_entry("s1")) is False


def test_auto_title_never_overwrites_a_manual_one() -> None:
    """The user's name for a chat wins over anything generated later."""
    sessions.set_title("s1", "My pipeline")
    assert sessions.set_auto_title("s1", "Generated") is False
    assert sessions.get_entry("s1") == {"title": "My pipeline"}


def test_manual_rename_replaces_auto_title_and_its_marker() -> None:
    """Renaming a generated title makes it manual; clearing drops both."""
    sessions.set_auto_title("s1", "Generated")
    sessions.set_title("s1", "Chosen")
    assert sessions.get_entry("s1") == {"title": "Chosen"}
    assert sessions.has_manual_title(sessions.get_entry("s1")) is True
    sessions.set_title("s1", "")
    assert sessions.get_entry("s1") == {}
    assert sessions.set_auto_title("s1", "Generated again") is True


def test_repeated_auto_title_is_a_no_op() -> None:
    """The same generated title twice reports no change (nothing to push)."""
    assert sessions.set_auto_title("s1", "Same") is True
    assert sessions.set_auto_title("s1", "Same") is False
    assert sessions.set_auto_title("s1", "   ") is False


def test_pre_feature_title_counts_as_manual() -> None:
    """Entries written before generated titles existed carry no source: manual."""
    assert sessions.has_manual_title({"title": "Old"}) is True
    assert sessions.has_manual_title({}) is False


# ── chat_title ───────────────────────────────────────────────────────────────


def _vibe_title(_session_id: str) -> str | None:
    return "vibe"


def _no_vibe_title(_session_id: str) -> str | None:
    return None


def test_chat_title_prefers_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry title, generated or not, is the chat's name."""
    sessions.set_auto_title("s1", "Brain MRI pipeline")
    monkeypatch.setattr(sessions.provenance, "vibe_manual_session_title", _vibe_title)
    assert sessions.chat_title("s1") == "Brain MRI pipeline"


def test_chat_title_falls_back_to_vibes_manual_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no registry entry, the user-set title vibe recorded is used."""
    monkeypatch.setattr(sessions.provenance, "vibe_manual_session_title", _vibe_title)
    assert sessions.chat_title("s1") == "vibe"


def test_chat_title_none_when_unnamed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat nobody has named yet has no title."""
    monkeypatch.setattr(sessions.provenance, "vibe_manual_session_title", _no_vibe_title)
    assert sessions.chat_title("s1") is None
