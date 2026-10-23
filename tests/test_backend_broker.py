"""End-to-end tests for the backend broker (unix socket → pool → fake server)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import mcp.types as mcp_types

from medmcp import approvals, replay
from medmcp.backend_broker import STREAM_LIMIT

from ._pool import broker

JsonDict = dict[str, Any]


async def _roundtrip(socket_path: Path, request: JsonDict) -> JsonDict:
    """Send one request line and return the parsed response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path), limit=STREAM_LIMIT)
    try:
        writer.write(json.dumps(request).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    return json.loads(line)


async def test_socket_is_only_reachable_by_its_owner(tmp_path: Path) -> None:
    """The broker socket's mode is set, not inherited from the process umask.

    Connecting to it means invoking any installed stack's tools on the workspace,
    below the permission flow entirely. Connect(2) needs write permission, which
    a 022 umask already withholds from others — but the umask belongs to whoever
    launched the process (a unit file with UMask=0000 would produce 0777), so the
    mode is asserted here rather than assumed.
    """
    async with broker(tmp_path) as sock:
        assert sock.stat().st_mode & 0o077 == 0


async def test_list_tools_over_socket(tmp_path: Path) -> None:
    """list_tools returns the stack's tools through the broker."""
    async with broker(tmp_path) as sock:
        resp = await _roundtrip(sock, {"op": "list_tools", "stack": "fake", "id": 1})
    assert resp["ok"] is True
    assert resp["id"] == 1
    names = {t["name"] for t in resp["tools"]}
    assert {"echo", "warmup", "crash"} <= names


async def test_call_tool_over_socket(tmp_path: Path) -> None:
    """call_tool runs the tool and relays a reconstructable CallToolResult."""
    async with broker(tmp_path) as sock:
        resp = await _roundtrip(
            sock,
            {"op": "call_tool", "stack": "fake", "tool": "echo", "args": {"text": "hi"}, "id": 2},
        )
    assert resp["ok"] is True
    result = mcp_types.CallToolResult.model_validate(resp["result"])
    assert replay.extract_structured(result) == {"text": "hi"}


async def test_unknown_stack_returns_error(tmp_path: Path) -> None:
    """A call for an uninstalled stack comes back as ok=false with a message."""
    async with broker(tmp_path) as sock:
        resp = await _roundtrip(sock, {"op": "list_tools", "stack": "ghost", "id": 3})
    assert resp["ok"] is False
    assert "not installed" in resp["error"]


async def test_bad_op_and_missing_fields(tmp_path: Path) -> None:
    """Protocol errors are reported, not dropped, and the connection survives."""
    async with broker(tmp_path) as sock:
        bad_op = await _roundtrip(sock, {"op": "nope", "stack": "fake", "id": 4})
        assert bad_op["ok"] is False and "unknown op" in bad_op["error"]

        no_stack = await _roundtrip(sock, {"op": "list_tools", "id": 5})
        assert no_stack["ok"] is False and "stack" in no_stack["error"]

        no_tool = await _roundtrip(sock, {"op": "call_tool", "stack": "fake", "id": 6})
        assert no_tool["ok"] is False and "tool" in no_tool["error"]


async def test_multiple_requests_on_one_connection(tmp_path: Path) -> None:
    """A single connection can issue several requests in sequence."""
    async with broker(tmp_path) as sock:
        reader, writer = await asyncio.open_unix_connection(str(sock), limit=STREAM_LIMIT)
        try:
            for i, text in enumerate(["a", "b", "c"]):
                writer.write(
                    json.dumps(
                        {
                            "op": "call_tool",
                            "stack": "fake",
                            "tool": "echo",
                            "args": {"text": text},
                            "id": i,
                        }
                    ).encode()
                    + b"\n"
                )
                await writer.drain()
                resp = json.loads(await reader.readline())
                assert resp["id"] == i
                result = mcp_types.CallToolResult.model_validate(resp["result"])
                assert replay.extract_structured(result) == {"text": text}
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


async def test_an_unapproved_call_never_runs(tmp_path: Path) -> None:
    """The agent's shell reaches this socket too: only an approved call runs, once."""
    ledger = approvals.ApprovalLedger()
    call = {"op": "call_tool", "stack": "fake", "tool": "echo", "args": {"text": "hi"}, "id": 1}
    async with broker(tmp_path, authorize=ledger.consume) as sock:
        refused = await _roundtrip(sock, call)
        ledger.grant("fake", "echo", {"text": "hi"})
        other_args = await _roundtrip(sock, {**call, "args": {"text": "other"}})
        approved = await _roundtrip(sock, call)
        replayed = await _roundtrip(sock, call)
        listed = await _roundtrip(sock, {"op": "list_tools", "stack": "fake", "id": 2})
    assert refused["ok"] is False and "not approved" in refused["error"]
    assert other_args["ok"] is False
    assert approved["ok"] is True
    assert replayed["ok"] is False
    assert listed["ok"] is True
