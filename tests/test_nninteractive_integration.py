"""Integration tests for the medmcp-nninteractive stack.

These tests require the medmcp-nninteractive stack to be installed as an isolated
uv tool and are skipped otherwise. The stack depends on nninteractive-agent, which
declares mcp>=2.3 for a server the stack does not use, so the install needs the
stack's overrides file next to the core's constraints (one line)::

    uv tool install --constraints stack-constraints.txt
        --overrides ../medmcp-nninteractive/overrides.txt ../medmcp-nninteractive

None of them needs a GPU or the nnInteractive weights: the model loads with the
first image, and no test loads one.
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pytest

STACK = "medmcp-nninteractive"

#: The tools the stack's nninteractive-segmentation skill tells the agent to call.
#: ``python`` is left out: the stack can be configured without it.
EXPECTED_TOOLS = {
    "load_image",
    "image_info",
    "view",
    "montage",
    "projection",
    "render3d",
    "new_object",
    "bbox",
    "scribble",
    "point",
    "undo",
    "stats",
    "commit",
    "discard",
    "edit_object",
    "delete_object",
    "list_objects",
    "save",
}


def _stack_installed() -> bool:
    """Return True if medmcp-nninteractive is present in the uv tool store."""
    result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True)
    return STACK in result.stdout


skip_no_stack = pytest.mark.skipif(
    not _stack_installed(), reason=f"{STACK} not installed via uv tool"
)

JsonDict = dict[str, Any]


def _read_response(proc: subprocess.Popen[str], req_id: int, timeout: float = 60) -> JsonDict:
    """Read lines from the server until we find the response matching *req_id*.

    A generous timeout: the server imports numpy, scipy, scikit-image and
    matplotlib at start, which is slow on a cold network filesystem.
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
            # nnInteractive prints progress to stdout; the stack moves fd 1 to stderr
            # so that it cannot reach the protocol stream.
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


def _executable() -> str:
    """Return the absolute path to the stack inside its isolated uv tool env."""
    result = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True)
    candidate = Path(result.stdout.strip()) / STACK / "bin" / STACK
    return str(candidate) if candidate.exists() else STACK


def _start_server() -> subprocess.Popen[str]:
    """Spawn the MCP server via its isolated uv tool environment."""
    return subprocess.Popen(
        [_executable()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
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


def _call(proc: subprocess.Popen[str], req_id: int, tool: str, arguments: JsonDict) -> JsonDict:
    """Call *tool* and return the JSON-RPC result."""
    request = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    _write(proc, json.dumps(request))
    response = _read_response(proc, req_id=req_id)
    assert "result" in response, f"tools/call failed: {response.get('error')}"
    return cast("JsonDict", response["result"])


@skip_no_stack
def test_nninteractive_server_starts() -> None:
    """The medmcp-nninteractive MCP server starts and responds to initialize."""
    proc = _start_server()
    try:
        resp = _initialize(proc)
        assert "result" in resp, f"initialize failed: {resp}"
    finally:
        proc.kill()
        proc.wait()


@skip_no_stack
def test_nninteractive_server_exposes_its_tools() -> None:
    """The stack advertises the tools its skill tells the agent to call, and their effects.

    Only ``save`` writes files; every other skill tool leaves the user's files as
    they were, so the approval card must not call it anything else.
    """
    proc = _start_server()
    try:
        _initialize(proc)
        _write(proc, '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
        response = _read_response(proc, req_id=2)

        assert "result" in response, f"tools/list returned an error: {response.get('error')}"
        result_payload: JsonDict = cast("JsonDict", response["result"])
        tools = {cast("JsonDict", t)["name"]: cast("JsonDict", t) for t in result_payload["tools"]}
        assert set(tools) >= EXPECTED_TOOLS, f"missing tools: {EXPECTED_TOOLS - set(tools)}"
        effects = {
            name: cast("JsonDict", tools[name].get("_meta") or {}).get("org.medmcp/effect")
            for name in EXPECTED_TOOLS
        }
        assert effects.pop("save") == "writes_new"
        assert set(effects.values()) == {"read_only"}, effects
    finally:
        proc.kill()
        proc.wait()


@skip_no_stack
def test_nninteractive_python_is_guarded_by_default(tmp_path: Path) -> None:
    """The python tool computes, and refuses the workspace's file methods.

    The cheapest end-to-end proof that the stack's workspace comes up: no image,
    no model, no GPU. The refusal is what lets the stack declare the tool
    read-only; a stack configured with the tool switched off skips this.
    """
    proc = _start_server()
    try:
        _initialize(proc)
        _write(proc, '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
        listed = cast("JsonDict", _read_response(proc, req_id=2)["result"])
        if "python" not in {cast("JsonDict", t)["name"] for t in listed["tools"]}:
            pytest.skip("the python tool is switched off in this install")

        result = _call(proc, 3, "python", {"code": "print(6 * 7)"})
        assert not result.get("isError"), result
        assert "42" in json.dumps(result["content"])

        target = tmp_path / "escape"
        refused = _call(proc, 4, "python", {"code": f"ws.save({str(target)!r})"})
        assert refused.get("isError") is True, refused
        assert "guarded" in json.dumps(refused["content"])
        assert not target.exists()
    finally:
        proc.kill()
        proc.wait()
