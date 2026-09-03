"""UI session registry: per-session metadata the workspace overlays on vibe-acp.

vibe-acp owns each chat's transcript and can list/load it; what it does not
store is the workspace's own view of a session — a user-set title and whether
the user archived it. That lives here, in ``.vibe/ui_sessions.json``, keyed by
vibe's session id.

Writes are atomic (mkstemp + ``os.replace``) so a second server instance or the
CLI can't corrupt the file. An entry that carries no metadata is dropped, so the
file stays small and "no entry" and "default entry" mean the same thing.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from medmcp import provenance
from medmcp.acp import VIBE_HOME

JsonDict = dict[str, Any]

# Module-level so tests can monkeypatch it to a temp path.
REGISTRY_PATH: Path = VIBE_HOME / "ui_sessions.json"


def load_registry() -> dict[str, JsonDict]:
    """Load the session-metadata map (empty if the file is absent or corrupt)."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k): cast("JsonDict", v)
        for k, v in cast("JsonDict", data).items()
        if isinstance(v, dict)
    }


def _save_registry(registry: dict[str, JsonDict]) -> None:
    """Write the registry atomically (unique temp + ``os.replace``)."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=REGISTRY_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(registry, fh)
        os.replace(tmp_name, REGISTRY_PATH)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def get_entry(session_id: str) -> JsonDict:
    """Return the stored metadata for *session_id* (empty dict if none)."""
    return load_registry().get(session_id, {})


def chat_title(session_id: str) -> str | None:
    """The chat's current name, or ``None`` if it has none yet.

    The registry title comes first, generated or user-set; failing that, the
    user-set title vibe recorded in its own session metadata (the registry lives
    in the container layer and may have been reset). Used to name what is
    derived from a chat — a distilled workflow — after the chat itself.
    """
    title = get_entry(session_id).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return provenance.vibe_manual_session_title(session_id)


def _update(session_id: str, mutate: Callable[[JsonDict], None]) -> None:
    """Apply *mutate* to a session's entry, then persist (dropping it if empty)."""
    registry = load_registry()
    entry = dict(registry.get(session_id, {}))
    mutate(entry)
    if entry:
        registry[session_id] = entry
    else:
        registry.pop(session_id, None)
    _save_registry(registry)


def set_title(session_id: str, title: str) -> None:
    """Set the user's title override, or clear it when *title* is blank.

    A user-set title carries no ``title_source`` — the same shape entries had
    before generated titles existed — so :func:`has_manual_title` treats both
    alike. Clearing drops any generated title too, so the next refresh may
    name the chat again.
    """

    def _mutate(entry: JsonDict) -> None:
        cleaned = title.strip()
        if cleaned:
            entry["title"] = cleaned
        else:
            entry.pop("title", None)
        entry.pop("title_source", None)

    _update(session_id, _mutate)


def has_manual_title(entry: JsonDict) -> bool:
    """Whether *entry* carries a title the user chose (any title not marked auto)."""
    return bool(entry.get("title")) and entry.get("title_source") != "auto"


def set_auto_title(session_id: str, title: str) -> bool:
    """Store a generated title unless the user named the chat themselves.

    Returns ``True`` when the stored title changed. A manual title always wins
    — generation never second-guesses a name the person typed — and a repeat
    of the current generated title is a no-op so callers can skip the UI push.
    """
    cleaned = title.strip()
    if not cleaned:
        return False
    changed = False

    def _mutate(entry: JsonDict) -> None:
        nonlocal changed
        if has_manual_title(entry) or entry.get("title") == cleaned:
            return
        entry["title"] = cleaned
        entry["title_source"] = "auto"
        changed = True

    _update(session_id, _mutate)
    return changed


def set_archived(session_id: str, archived: bool) -> None:
    """Mark a session archived (hidden from the default list) or restore it."""

    def _mutate(entry: JsonDict) -> None:
        if archived:
            entry["archived"] = True
        else:
            entry.pop("archived", None)

    _update(session_id, _mutate)


def remove(session_id: str) -> None:
    """Forget a session's UI metadata (used when the session is deleted)."""
    registry = load_registry()
    if registry.pop(session_id, None) is not None:
        _save_registry(registry)
