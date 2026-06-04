"""Tests for the data layer override that purges session logs on thread delete."""

from __future__ import annotations

import json
from typing import Any

import pytest

import medmcp.app as app
from medmcp import provenance

# pyright: reportPrivateUsage=false

JsonDict = dict[str, Any]


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
