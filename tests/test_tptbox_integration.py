"""Integration tests for the medmcp-tptbox stack.

These tests require the medmcp-tptbox stack to be installed as an isolated uv
tool (``just install-stack ../medmcp-tptbox``) and are skipped otherwise.
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pytest

STACK = "medmcp-tptbox"

#: The tools the stack's two SKILL.md workflows tell the agent to call.
EXPECTED_TOOLS = {
    "query_bids_dataset",
    "describe_image",
    "list_operations",
    "process_nifti",
    "compute_poi",
    "export_mesh",
}


def _tptbox_installed() -> bool:
    """Return True if medmcp-tptbox is present in the uv tool store."""
    result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True)
    return STACK in result.stdout


skip_no_tptbox = pytest.mark.skipif(
    not _tptbox_installed(), reason=f"{STACK} not installed via uv tool"
)

JsonDict = dict[str, Any]


def _read_response(proc: subprocess.Popen[str], req_id: int, timeout: float = 30) -> JsonDict:
    """Read lines from the server until we find the response matching *req_id*.

    A generous timeout: the server itself imports no TPTBox, but the first
    tools/call spawns the worker subprocess, which imports SimpleITK and
    scikit-image.
    """
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], max(deadline - time.monotonic(), 0))
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            # A non-JSON line on stdout means the stack leaked print output into
            # the protocol stream — the exact failure server.py guards against.
            pytest.fail(f"non-JSON line on stdout (protocol stream polluted): {line!r}")
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


def _tptbox_executable() -> str:
    """Return the absolute path to medmcp-tptbox inside its isolated uv tool env."""
    result = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True)
    candidate = Path(result.stdout.strip()) / STACK / "bin" / STACK
    return str(candidate) if candidate.exists() else STACK


def _start_server() -> subprocess.Popen[str]:
    """Spawn the medmcp-tptbox MCP server via its isolated uv tool environment."""
    return subprocess.Popen(
        [_tptbox_executable()],
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
    _write(proc, '{"jsonrpc":"2.0","method":"notifications/initialized"}')
    return resp


@skip_no_tptbox
def test_tptbox_server_starts() -> None:
    """The medmcp-tptbox MCP server starts and responds to initialize."""
    proc = _start_server()
    try:
        resp = _initialize(proc)
        assert "result" in resp, f"initialize failed: {resp}"
    finally:
        proc.kill()
        proc.wait()


@skip_no_tptbox
def test_tptbox_server_exposes_its_tools() -> None:
    """The stack advertises the tools its SKILL.md workflows tell the agent to call."""
    proc = _start_server()
    try:
        _initialize(proc)
        _write(proc, '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
        response = _read_response(proc, req_id=2)

        assert "result" in response, f"tools/list returned an error: {response.get('error')}"
        result_payload: JsonDict = cast("JsonDict", response["result"])
        names = {cast("JsonDict", t)["name"] for t in cast("list[Any]", result_payload["tools"])}
        assert names >= EXPECTED_TOOLS, f"missing tools: {EXPECTED_TOOLS - names}"
    finally:
        proc.kill()
        proc.wait()


@skip_no_tptbox
def test_tptbox_stack_reports_its_operations() -> None:
    """list_operations returns the op vocabulary the SKILL tells the agent to consult.

    The cheapest end-to-end proof that the op registry shipped with the stack:
    it touches no image data and needs no GPU.
    """
    proc = _start_server()
    try:
        _initialize(proc)
        _write(
            proc,
            '{"jsonrpc":"2.0","id":3,"method":"tools/call",'
            '"params":{"name":"list_operations","arguments":{}}}',
        )
        response = _read_response(proc, req_id=3)
        assert "result" in response, f"tools/call failed: {response.get('error')}"
        blob = json.dumps(response["result"])
        for op in ("reorient", "rescale", "dilate", "map_labels"):
            assert op in blob, f"{op} missing from list_operations output"
    finally:
        proc.kill()
        proc.wait()
