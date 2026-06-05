"""Tests for the data layer override that purges session logs on thread delete."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import medmcp.app as app
from medmcp import provenance

# pyright: reportPrivateUsage=false

JsonDict = dict[str, Any]


def _make_threads_db(path: Path, rows: list[tuple[str, str | None]]) -> None:
    """Create a minimal threads DB with (id, metadata) rows for GC tests."""
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE threads ("id" TEXT PRIMARY KEY, "metadata" TEXT)')
    con.executemany('INSERT INTO threads ("id", "metadata") VALUES (?, ?)', rows)
    con.commit()
    con.close()


def _new_data_layer() -> app._MedMcpDataLayer:
    """Build the data layer without running __init__ (which needs a real DB)."""
    return object.__new__(app._MedMcpDataLayer)


async def _run_delete(monkeypatch: pytest.MonkeyPatch, *, metadata: object) -> list[str]:
    """Drive delete_thread with a stubbed get_thread; return purged session ids."""
    purged: list[str] = []

    async def fake_get_thread(self: object, thread_id: str) -> JsonDict:
        return {"metadata": metadata}

    async def fake_super_delete(self: object, thread_id: str) -> None:
        return None

    def fake_purge(session_id: str) -> None:
        purged.append(session_id)

    monkeypatch.setattr(app._MedMcpDataLayer, "get_thread", fake_get_thread, raising=True)
    monkeypatch.setattr(app.SQLAlchemyDataLayer, "delete_thread", fake_super_delete, raising=True)
    monkeypatch.setattr(provenance, "purge_session", fake_purge)

    await _new_data_layer().delete_thread("thread-1")
    return purged


@pytest.mark.asyncio
async def test_purges_when_metadata_is_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread whose metadata dict carries vibe_session_id triggers a purge."""
    purged = await _run_delete(monkeypatch, metadata={"vibe_session_id": "S123"})
    assert purged == ["S123"]


@pytest.mark.asyncio
async def test_purges_when_metadata_is_json_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata stored as a JSON string is parsed before the purge."""
    purged = await _run_delete(monkeypatch, metadata=json.dumps({"vibe_session_id": "S999"}))
    assert purged == ["S999"]


@pytest.mark.asyncio
async def test_no_purge_without_session_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread with no vibe_session_id leaves the filesystem untouched."""
    purged = await _run_delete(monkeypatch, metadata={"other": 1})
    assert purged == []


# ── fail-safe orphan GC reference reading ─────────────────────────────────────


class TestReferencedVibeSessionIds:
    """_referenced_vibe_session_ids must never under-report (which would over-GC)."""

    def test_collects_ids_from_valid_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid metadata rows yield their vibe_session_ids; empty metadata is fine."""
        db = tmp_path / "threads.db"
        _make_threads_db(
            db,
            [
                ("t1", json.dumps({"vibe_session_id": "S1"})),
                ("t2", json.dumps({"vibe_session_id": "S2"})),
                ("t3", json.dumps({"name": "no session"})),
                ("t4", None),
            ],
        )
        monkeypatch.setattr(app, "THREADS_DB_PATH", db)
        assert app._referenced_vibe_session_ids() == {"S1", "S2"}

    def test_none_when_db_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing DB returns None (uncertain) so the GC is skipped."""
        monkeypatch.setattr(app, "THREADS_DB_PATH", tmp_path / "absent.db")
        assert app._referenced_vibe_session_ids() is None

    def test_none_on_corrupt_metadata_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single unparseable metadata row makes the whole read return None."""
        db = tmp_path / "threads.db"
        _make_threads_db(
            db,
            [
                ("t1", json.dumps({"vibe_session_id": "S1"})),
                ("t2", "not-json{{{"),
            ],
        )
        monkeypatch.setattr(app, "THREADS_DB_PATH", db)
        assert app._referenced_vibe_session_ids() is None

    def test_none_on_query_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the threads table is missing, the read returns None (not empty)."""
        db = tmp_path / "threads.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE other (x TEXT)")  # no `threads` table
        con.commit()
        con.close()
        monkeypatch.setattr(app, "THREADS_DB_PATH", db)
        assert app._referenced_vibe_session_ids() is None


class TestGcOrphanedProvenance:
    """_gc_orphaned_provenance only deletes when the reference set is certain."""

    def test_skips_when_references_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When references can't be determined (None), nothing is purged."""
        called: list[set[str]] = []

        def fake_purge(ids: set[str]) -> list[str]:
            called.append(ids)
            return []

        monkeypatch.setattr(app, "_provenance_gc_done", False)
        monkeypatch.setattr(app, "_referenced_vibe_session_ids", lambda: None)
        monkeypatch.setattr(provenance, "purge_orphans", fake_purge)
        app._gc_orphaned_provenance()
        assert called == []  # purge_orphans never invoked

    def test_purges_when_references_known(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a determined set, the GC forwards it to purge_orphans exactly once."""
        called: list[set[str]] = []

        def fake_purge(ids: set[str]) -> list[str]:
            called.append(ids)
            return []

        monkeypatch.setattr(app, "_provenance_gc_done", False)
        monkeypatch.setattr(app, "_referenced_vibe_session_ids", lambda: {"S1"})
        monkeypatch.setattr(provenance, "purge_orphans", fake_purge)
        app._gc_orphaned_provenance()
        assert called == [{"S1"}]
