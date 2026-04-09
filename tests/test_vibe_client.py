"""Tests for :class:`medmcp.app.VibeAcpClient` request/response routing.

These tests bypass the real ``vibe-acp`` subprocess by attaching a fake
``proc`` object whose ``stdin`` is an in-memory accumulator and whose ``stdout``
is a real :class:`asyncio.StreamReader`. The reader task is started directly so
the test can drive the wire protocol from both sides.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from medmcp.app import VibeAcpClient

# ── Fakes for the asyncio subprocess interface ────────────


class _FakeStdin:
    """In-memory replacement for ``proc.stdin``.

    Records every ``write`` call as a parsed JSON-RPC frame so tests can assert
    on what was sent without parsing bytes themselves.
    """

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.raw: bytes = b""

    def write(self, data: bytes) -> None:
        self.raw += data
        # Frames are newline-delimited; one write may contain one frame.
        for line in data.splitlines():
            if not line:
                continue
            self.frames.append(json.loads(line.decode()))

    async def drain(self) -> None:
        return None


class _FakeProc:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = asyncio.StreamReader()


def _make_started_client() -> tuple[VibeAcpClient, _FakeProc, asyncio.Task[None]]:
    """Build a client wired to a fake subprocess and start its read loop."""
    client = VibeAcpClient()
    proc = _FakeProc()
    # The real ``ensure_started`` would create a subprocess and run an
    # ``initialize`` handshake. We bypass it by attaching the fake proc and
    # starting the reader directly.
    client.proc = cast("Any", proc)
    client._initialized = True
    reader_task = asyncio.create_task(client._read_loop())
    return client, proc, reader_task


def _feed_response(proc: _FakeProc, req_id: int, result: dict[str, Any]) -> None:
    """Feed a JSON-RPC response frame into the fake stdout."""
    frame = {"jsonrpc": "2.0", "id": req_id, "result": result}
    proc.stdout.feed_data((json.dumps(frame) + "\n").encode())


def _feed_server_frame(proc: _FakeProc, frame: dict[str, Any]) -> None:
    """Feed a server-initiated JSON-RPC frame (notification or request)."""
    proc.stdout.feed_data((json.dumps(frame) + "\n").encode())


async def _shutdown(proc: _FakeProc, reader_task: asyncio.Task[None]) -> None:
    """Close the fake stdout and wait for the reader to exit cleanly."""
    proc.stdout.feed_eof()
    await reader_task


# ── Request / response correlation ─────────────────────────


async def test_request_response_correlation_out_of_order() -> None:
    """Two in-flight requests should resolve to their matching responses."""
    client, proc, reader_task = _make_started_client()
    try:
        fut_a = asyncio.ensure_future(client.request("foo"))
        fut_b = asyncio.ensure_future(client.request("bar"))

        # Let both requests be written to stdin and registered in _pending.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(proc.stdin.frames) == 2
        id_a = proc.stdin.frames[0]["id"]
        id_b = proc.stdin.frames[1]["id"]
        assert id_a != id_b

        # Deliver in reverse order.
        _feed_response(proc, id_b, {"who": "bar"})
        _feed_response(proc, id_a, {"who": "foo"})

        resp_a = await fut_a
        resp_b = await fut_b
        assert resp_a["result"] == {"who": "foo"}
        assert resp_b["result"] == {"who": "bar"}
        assert client._pending == {}
    finally:
        await _shutdown(proc, reader_task)


async def test_cancelled_request_clears_pending() -> None:
    """Cancelling an in-flight ``request`` must not leak ``_pending`` entries."""
    client, proc, reader_task = _make_started_client()
    try:
        task = asyncio.ensure_future(client.request("slow"))
        # Let the request reach ``await fut``.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(client._pending) == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client._pending == {}
    finally:
        await _shutdown(proc, reader_task)


# ── Notification path ─────────────────────────────────────


async def test_notify_writes_no_id_and_no_pending() -> None:
    """``notify`` should emit an id-less frame and never touch ``_pending``."""
    client, proc, reader_task = _make_started_client()
    try:
        await client.notify("session/cancel", {"session_id": "s1"})
        assert len(proc.stdin.frames) == 1
        frame = proc.stdin.frames[0]
        assert frame["method"] == "session/cancel"
        assert "id" not in frame
        assert frame["params"] == {"session_id": "s1"}
        assert client._pending == {}
    finally:
        await _shutdown(proc, reader_task)


# ── Session queue routing ─────────────────────────────────


async def test_limbo_flush_on_register_session() -> None:
    """Frames that arrive before ``register_session`` should land in limbo."""
    client, proc, reader_task = _make_started_client()
    try:
        # Frame arrives BEFORE the session is registered.
        _feed_server_frame(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "abc", "update": {"sessionUpdate": "agent_message_chunk"}},
            },
        )
        # Give the reader a chance to route it into limbo.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert "abc" in client._limbo
        assert len(client._limbo["abc"]) == 1

        queue = client.register_session("abc")
        # Limbo should be drained and the queue should now hold the frame.
        assert "abc" not in client._limbo
        assert queue.qsize() == 1
        buffered = queue.get_nowait()
        assert buffered["method"] == "session/update"
    finally:
        await _shutdown(proc, reader_task)


async def test_register_unregister_session_isolation() -> None:
    """Two sessions should receive their own frames; unregister doesn't affect peers."""
    client, proc, reader_task = _make_started_client()
    try:
        queue_a = client.register_session("a")
        queue_b = client.register_session("b")

        _feed_server_frame(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "a", "update": {"tag": "for-a"}},
            },
        )
        _feed_server_frame(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "b", "update": {"tag": "for-b"}},
            },
        )

        msg_a = await asyncio.wait_for(queue_a.get(), timeout=1)
        msg_b = await asyncio.wait_for(queue_b.get(), timeout=1)
        assert msg_a["params"]["update"]["tag"] == "for-a"
        assert msg_b["params"]["update"]["tag"] == "for-b"

        client.unregister_session("a")
        assert client.get_session_queue("a") is None
        assert client.get_session_queue("b") is queue_b
    finally:
        await _shutdown(proc, reader_task)


async def test_session_id_camel_and_snake_case() -> None:
    """The dispatcher must accept both ``sessionId`` and ``session_id``."""
    client, proc, reader_task = _make_started_client()
    try:
        queue = client.register_session("s1")

        _feed_server_frame(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "s1", "update": {"tag": "camel"}},
            },
        )
        _feed_server_frame(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"session_id": "s1", "update": {"tag": "snake"}},
            },
        )

        first = await asyncio.wait_for(queue.get(), timeout=1)
        second = await asyncio.wait_for(queue.get(), timeout=1)
        tags = {first["params"]["update"]["tag"], second["params"]["update"]["tag"]}
        assert tags == {"camel", "snake"}
    finally:
        await _shutdown(proc, reader_task)


# ── Subprocess teardown semantics ─────────────────────────


async def test_eof_fails_pending_with_connection_error() -> None:
    """When the subprocess closes, every pending request should fail cleanly."""
    client, proc, reader_task = _make_started_client()
    try:
        fut_a = asyncio.ensure_future(client.request("foo"))
        fut_b = asyncio.ensure_future(client.request("bar"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(client._pending) == 2

        proc.stdout.feed_eof()
        await reader_task  # let the read loop drain and fail pendings

        with pytest.raises(ConnectionError, match="vibe-acp subprocess closed"):
            await fut_a
        with pytest.raises(ConnectionError, match="vibe-acp subprocess closed"):
            await fut_b
        assert client._pending == {}
    finally:
        if not reader_task.done():
            reader_task.cancel()
