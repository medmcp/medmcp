"""Integration tests for the medmcp-neuro stack.

These tests require ``uv sync --extra neuro`` and are skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import Any, cast

import pytest

neuro_available = importlib.util.find_spec("medmcp_neuro") is not None

skip_no_neuro = pytest.mark.skipif(not neuro_available, reason="medmcp-neuro not installed")

JsonDict = dict[str, Any]


def _read_response(proc: subprocess.Popen[str], req_id: int, timeout: float = 10) -> JsonDict:
    """Read lines from the server until we find the response matching *req_id*."""
    import select

    assert proc.stdout is not None
    deadline = __import__("time").monotonic() + timeout
    while __import__("time").monotonic() < deadline:
        remaining = deadline - __import__("time").monotonic()
        ready, _, _ = select.select([proc.stdout], [], [], max(remaining, 0))
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = cast("JsonDict", parsed)
            if obj.get("id") == req_id:
                return obj
    msg = f"Timed out waiting for response id={req_id}"
    raise TimeoutError(msg)


def _write(proc: subprocess.Popen[str], msg: str) -> None:
    """Write a single JSON-RPC line to the server's stdin."""
    assert proc.stdin is not None
    proc.stdin.write(msg if msg.endswith("\n") else msg + "\n")
    proc.stdin.flush()


def _start_server() -> subprocess.Popen[str]:
    """Spawn the medmcp-neuro MCP server."""
    return subprocess.Popen(
        [sys.executable, "-m", "medmcp_neuro"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _initialize(proc: subprocess.Popen[str]) -> JsonDict:
    """Run the MCP initialize handshake and return the server capabilities."""
    _write(
        proc,
        '{"jsonrpc":"2.0","id":1,"method":"initialize",'
        '"params":{"protocolVersion":"2024-11-05",'
        '"capabilities":{},'
        '"clientInfo":{"name":"test","version":"0.1"}}}',
    )
    resp = _read_response(proc, req_id=1)
    # Send the required initialized notification.
    _write(proc, '{"jsonrpc":"2.0","method":"notifications/initialized"}')
    return resp


@skip_no_neuro
def test_neuro_server_starts() -> None:
    """The medmcp-neuro MCP server starts and responds to initialize."""
    proc = _start_server()
    try:
        resp = _initialize(proc)
        assert "result" in resp, f"initialize failed: {resp}"
    finally:
        proc.kill()
        proc.wait()


@skip_no_neuro
def test_neuro_server_exposes_tools() -> None:
    """The medmcp-neuro MCP server advertises at least one tool via tools/list."""
    proc = _start_server()
    try:
        _initialize(proc)
        _write(proc, '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
        tools_response = _read_response(proc, req_id=2)

        assert "result" in tools_response, (
            f"tools/list returned an error: {tools_response.get('error')}"
        )
        result_payload: JsonDict = cast("JsonDict", tools_response["result"])
        tools: list[Any] = cast("list[Any]", result_payload.get("tools", []))
        assert len(tools) > 0, "medmcp-neuro server registered zero tools"
    finally:
        proc.kill()
        proc.wait()
