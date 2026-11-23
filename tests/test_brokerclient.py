"""Ops the core answers itself over the broker, and the helpers' client for them."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from medmcp import brokerclient
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import resolver

JsonDict = dict[str, Any]


def _echo(req: JsonDict) -> JsonDict:
    return {"echo": req.get("value")}


async def test_a_core_op_is_answered_without_a_stack(tmp_path: Path) -> None:
    """The server's own op answers; nothing is started, no ``stack`` is needed."""
    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(pool, tmp_path / "b.sock", ops={"echo": _echo})
    await broker.start()
    try:
        answer = await brokerclient.ask("echo", {"value": 7}, socket=broker.socket_path)
        assert answer == {"echo": 7, "id": 1, "ok": True}
        # An unknown op is still refused, which the client reports as no answer.
        assert await brokerclient.ask("nope", {}, socket=broker.socket_path) is None
        assert pool.warm_names() == []
    finally:
        await broker.aclose()
        await pool.aclose()


def test_a_core_op_cannot_replace_the_stack_ops(tmp_path: Path) -> None:
    """``call_tool`` is approval-gated; no op may stand in for it."""
    pool = BackendPool(resolve_spec=resolver({}))
    with pytest.raises(ValueError, match="cannot replace"):
        BackendBroker(pool, tmp_path / "b.sock", ops={"call_tool": _echo})


async def test_the_socket_is_found_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``MEDMCP_BROKER_SOCK`` first, else ``$VIBE_HOME/backend.sock``."""
    monkeypatch.delenv("MEDMCP_BROKER_SOCK", raising=False)
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    assert brokerclient.default_socket() == tmp_path / "backend.sock"
    monkeypatch.setenv("MEDMCP_BROKER_SOCK", str(tmp_path / "other.sock"))
    assert brokerclient.default_socket() == tmp_path / "other.sock"

    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(pool, tmp_path / "other.sock", ops={"echo": _echo})
    await broker.start()
    try:
        # The synchronous form, as a hook process uses it.
        answer = await asyncio.to_thread(brokerclient.ask_sync, "echo", {"value": "x"})
        assert answer is not None
        assert answer["echo"] == "x"
    finally:
        await broker.aclose()
        await pool.aclose()


def test_no_core_is_no_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No socket, or something that is not a broker: ``None``, never an exception."""
    monkeypatch.setenv("MEDMCP_BROKER_SOCK", str(tmp_path / "missing.sock"))
    assert brokerclient.ask_sync("echo", {}) is None
    not_a_socket = tmp_path / "file.sock"
    not_a_socket.write_text("")
    assert brokerclient.ask_sync("echo", {}, socket=not_a_socket) is None
