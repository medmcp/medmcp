"""Tests for the pending Rename/Refine input routing.

Clicking Rename/Refine arms a pending action in the user session; the user's
next message is then consumed by ``_consume_pending_workflow_input`` instead of
being forwarded to the agent. These tests verify that routing without a live
Chainlit/vibe-acp context by faking the few ``cl`` touchpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

import medmcp.app as app

# pyright: reportPrivateUsage=false


class _FakeSession:
    """Minimal stand-in for ``cl.user_session`` backed by a dict."""

    def __init__(self) -> None:
        self._d: dict[str, object] = {}

    def get(self, key: str, default: object = None) -> object:
        return self._d.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._d[key] = value


class _FakeMessage:
    """Captures content of every message ``send()``-ed during a test."""

    sent: ClassVar[list[str]] = []

    def __init__(self, content: str = "", **_: object) -> None:
        self.content = content

    async def send(self) -> _FakeMessage:
        _FakeMessage.sent.append(self.content)
        return self

    async def remove(self) -> None:
        return None


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    """Patch the cl touchpoints and return the fake user session."""
    _FakeMessage.sent = []
    sess = _FakeSession()
    monkeypatch.setattr(app.cl, "user_session", sess, raising=False)
    monkeypatch.setattr(app.cl, "Message", _FakeMessage, raising=False)

    previewed: list[Path] = []

    async def fake_preview(draft_dir: Path) -> None:
        previewed.append(draft_dir)

    monkeypatch.setattr(app, "_send_workflow_preview", fake_preview)
    sess._d["_previewed"] = previewed  # surface for assertions
    return sess


@pytest.mark.asyncio
async def test_no_pending_returns_false(session: _FakeSession) -> None:
    """With nothing armed, the message is not consumed."""
    assert await app._consume_pending_workflow_input("hello") is False


@pytest.mark.asyncio
async def test_rename_applies_and_previews(
    session: _FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending rename routes the text to rename_draft and re-previews."""
    calls: list[tuple[str, str]] = []

    def fake_rename(name: str, new_name: str) -> Path:
        calls.append((name, new_name))
        return Path("/tmp/draft/new-name")

    monkeypatch.setattr(app.distill, "rename_draft", fake_rename)
    session.set("pending_workflow", {"action": "rename", "name": "old-name"})

    consumed = await app._consume_pending_workflow_input("New Name")

    assert consumed is True
    assert calls == [("old-name", "New Name")]
    assert session.get("pending_workflow") is None  # consumed
    assert session._d["_previewed"] == [Path("/tmp/draft/new-name")]


@pytest.mark.asyncio
async def test_dash_cancels(session: _FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sending '-' cancels without touching the draft."""
    called = False

    def fake_rename(name: str, new_name: str) -> Path:
        nonlocal called
        called = True
        return Path("/x")

    monkeypatch.setattr(app.distill, "rename_draft", fake_rename)
    session.set("pending_workflow", {"action": "rename", "name": "keep-me"})

    assert await app._consume_pending_workflow_input("-") is True
    assert called is False
    assert any("cancelled" in m.lower() for m in _FakeMessage.sent)


@pytest.mark.asyncio
async def test_refine_routes_to_refine_draft(
    session: _FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending refine routes the instruction to refine_draft."""
    calls: list[tuple[str, str]] = []

    def fake_refine(name: str, instruction: str) -> Path:
        calls.append((name, instruction))
        return Path("/tmp/draft/flow")

    monkeypatch.setattr(app.distill, "refine_draft", fake_refine)
    session.set("pending_workflow", {"action": "refine", "name": "flow"})

    assert await app._consume_pending_workflow_input("make it generic") is True
    assert calls == [("flow", "make it generic")]


@pytest.mark.asyncio
async def test_rename_failure_reports_and_consumes(
    session: _FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If rename_draft raises, the error is surfaced and the message is consumed."""

    def boom(name: str, new_name: str) -> Path:
        raise RuntimeError("nope")

    monkeypatch.setattr(app.distill, "rename_draft", boom)
    session.set("pending_workflow", {"action": "rename", "name": "x"})

    assert await app._consume_pending_workflow_input("y") is True
    assert any("could not rename" in m.lower() for m in _FakeMessage.sent)
    assert session.get("pending_workflow") is None
