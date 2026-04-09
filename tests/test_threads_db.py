"""Tests for ``_bootstrap_threads_db``: schema creation, migration, and repair.

The function runs on every Chainlit server startup, so it needs to be
idempotent, must add missing columns to pre-existing databases, and must repair
rows orphaned by the pre-fix schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from medmcp.app import _bootstrap_threads_db


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names on ``table``."""
    return {row[0] for row in conn.execute(f'SELECT name FROM pragma_table_info("{table}")')}


def _tables(conn: sqlite3.Connection) -> set[str]:
    """Return the set of user table names."""
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    """Return the set of explicitly-named index names."""
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        if row[0] is not None
    }


# ── Schema creation ───────────────────────────────────────


def test_bootstrap_creates_all_expected_tables(tmp_path: Path) -> None:
    """A fresh bootstrap should create every table the chainlit data layer needs."""
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = _tables(conn)

    assert {"users", "threads", "steps", "elements", "feedbacks"}.issubset(tables)


def test_bootstrap_steps_table_has_required_columns(tmp_path: Path) -> None:
    """The ``steps`` table must include the columns chainlit's Step.to_dict() emits.

    The bug fixed in app.py:303-311 was that missing columns silently dropped
    every Step write. Pin the full set so a future schema edit can't regress
    chat resume.
    """
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = _columns(conn, "steps")

    required = {
        "id",
        "name",
        "type",
        "threadId",
        "parentId",
        "streaming",
        "waitForAnswer",
        "isError",
        "metadata",
        "tags",
        "input",
        "output",
        "createdAt",
        "start",
        "end",
        "generation",
        "showInput",
        "language",
        # The four columns the migration loop also adds — these are the ones
        # whose absence broke tool/run step persistence.
        "defaultOpen",
        "autoCollapse",
        "command",
        "modes",
    }
    assert required.issubset(cols)


def test_bootstrap_creates_indexes(tmp_path: Path) -> None:
    """The expected per-thread indexes should exist after bootstrap."""
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        idx = _indexes(conn)

    assert {"idx_steps_threadId", "idx_elements_threadId", "idx_feedbacks_forId"}.issubset(idx)


def test_bootstrap_creates_parent_directory(tmp_path: Path) -> None:
    """A non-existent parent directory should be created, not raise."""
    db_path = tmp_path / "nested" / "subdir" / "threads.db"
    _bootstrap_threads_db(db_path)
    assert db_path.exists()


# ── Idempotency ────────────────────────────────────────────


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    """Bootstrapping the same database twice must not raise."""
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        assert "steps" in _tables(conn)


# ── Migration of pre-existing databases ────────────────────


def test_bootstrap_migrates_old_steps_table(tmp_path: Path) -> None:
    """A pre-existing ``steps`` table missing the new columns should be migrated."""
    db_path = tmp_path / "old.db"
    # Pre-create a ``steps`` table missing all four post-fix columns. The
    # CREATE TABLE IF NOT EXISTS in bootstrap will keep this table as-is, so
    # the migration loop is what has to add the columns.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE steps (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT
            )
        """)
        conn.commit()

    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = _columns(conn, "steps")

    for added in ("defaultOpen", "autoCollapse", "command", "modes"):
        assert added in cols, f"migration did not add column {added}"


def test_bootstrap_migration_is_idempotent_on_partial_old_schema(tmp_path: Path) -> None:
    """Running the migration twice on an old schema should not raise.

    sqlite's ``ALTER TABLE ADD COLUMN`` has no ``IF NOT EXISTS``, so if the
    column probe regresses we'd get a duplicate-column error on the second run.
    """
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE steps (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "command" TEXT
            )
        """)
        conn.commit()

    _bootstrap_threads_db(db_path)
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = _columns(conn, "steps")

    assert {"command", "modes", "defaultOpen", "autoCollapse"}.issubset(cols)


# ── Orphan repair ──────────────────────────────────────────


def test_bootstrap_repairs_orphan_parent_id(tmp_path: Path) -> None:
    """Rows whose ``parentId`` points at a missing row should be promoted to NULL."""
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'INSERT INTO steps ("id", "threadId", "parentId") VALUES (?, ?, ?)',
            ("orphan", "t1", "missing-parent"),
        )
        conn.commit()

    # Re-run bootstrap; the orphan repair should promote ``parentId`` to NULL.
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute('SELECT "parentId" FROM steps WHERE "id" = ?', ("orphan",)).fetchone()
    assert row[0] is None


def test_bootstrap_leaves_valid_parent_references_alone(tmp_path: Path) -> None:
    """Rows with a valid ``parentId`` must NOT be touched by the orphan repair."""
    db_path = tmp_path / "threads.db"
    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'INSERT INTO steps ("id", "threadId", "parentId") VALUES (?, ?, ?)',
            ("parent", "t1", None),
        )
        conn.execute(
            'INSERT INTO steps ("id", "threadId", "parentId") VALUES (?, ?, ?)',
            ("child", "t1", "parent"),
        )
        conn.commit()

    _bootstrap_threads_db(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute('SELECT "parentId" FROM steps WHERE "id" = ?', ("child",)).fetchone()
    assert row[0] == "parent"
