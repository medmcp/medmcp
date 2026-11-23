"""Integration tests for the medmcp-spine stack.

These tests require the medmcp-spine stack to be installed as an isolated uv
tool (``just install-stack ../medmcp-spine``) and are skipped otherwise.
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pytest

STACK = "medmcp-spine"

#: Tools the stack promises; its SKILL.md instructs the agent to call all three.
EXPECTED_TOOLS = {"segment_spine", "list_spine_models", "list_spine_labels"}


def _spine_installed() -> bool:
    """Return True if medmcp-spine is present in the uv tool store."""
    result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True)
    return STACK in result.stdout


skip_no_spine = pytest.mark.skipif(
    not _spine_installed(), reason=f"{STACK} not installed via uv tool"
)

JsonDict = dict[str, Any]


def _read_response(proc: subprocess.Popen[str], req_id: int, timeout: float = 30) -> JsonDict:
    """Read lines from the server until we find the response matching *req_id*.

    A generous timeout: the first request pays the stack's import cost, which
    for a torch-backed stack is seconds, not milliseconds.
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


def _spine_executable() -> str:
    """Return the absolute path to medmcp-spine inside its isolated uv tool env."""
    result = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True)
    candidate = Path(result.stdout.strip()) / STACK / "bin" / STACK
    return str(candidate) if candidate.exists() else STACK


def _start_server() -> subprocess.Popen[str]:
    """Spawn the medmcp-spine MCP server via its isolated uv tool environment."""
    return subprocess.Popen(
        [_spine_executable()],
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


@skip_no_spine
def test_spine_server_starts() -> None:
    """The medmcp-spine MCP server starts and responds to initialize."""
    proc = _start_server()
    try:
        resp = _initialize(proc)
        assert "result" in resp, f"initialize failed: {resp}"
    finally:
        proc.kill()
        proc.wait()


@skip_no_spine
def test_spine_server_exposes_its_tools() -> None:
    """The stack advertises the three tools its SKILL.md tells the agent to call."""
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


@skip_no_spine
def test_spine_stack_reports_its_baked_models() -> None:
    """list_spine_models runs without a GPU and names the offline-capable models.

    This is the cheapest end-to-end proof that the stack's model registry
    shipped with it — it needs no weights loaded and no CUDA device.

    Note that ``weights_present`` is deliberately not asserted: it reports
    whether the *image's* baked-weight manifest is there, and this test drives a
    uv-tool install, where it is legitimately absent.
    """
    proc = _start_server()
    try:
        _initialize(proc)
        _write(
            proc,
            '{"jsonrpc":"2.0","id":3,"method":"tools/call",'
            '"params":{"name":"list_spine_models","arguments":{}}}',
        )
        response = _read_response(proc, req_id=3)
        assert "result" in response, f"tools/call failed: {response.get('error')}"
        blob = json.dumps(response["result"])
        for model in ("t2w", "ct", "instance", "ct_instance"):
            assert model in blob, f"{model} missing from list_spine_models output"
        for modality in ("t2w", "ct"):
            assert f'"{modality}"' in blob, f"{modality} missing from the reported modalities"
    finally:
        proc.kill()
        proc.wait()
