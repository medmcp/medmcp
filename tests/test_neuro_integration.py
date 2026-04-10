"""Integration tests for the medmcp-neuro stack.

These tests require ``uv sync --extra neuro`` and are skipped otherwise.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

neuro_available = True
try:
    import medmcp_neuro  # noqa: F401
except ImportError:
    neuro_available = False

skip_no_neuro = pytest.mark.skipif(not neuro_available, reason="medmcp-neuro not installed")


@skip_no_neuro
def test_neuro_server_starts() -> None:
    """The medmcp-neuro MCP server starts and responds to initialize."""
    # Send a JSON-RPC initialize request and check the server responds.
    # The server runs over stdio, so we pipe a request and read the response.
    init_request = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize",'
        '"params":{"protocolVersion":"2024-11-05",'
        '"capabilities":{},'
        '"clientInfo":{"name":"test","version":"0.1"}}}\n'
    )
    result = subprocess.run(
        [sys.executable, "-m", "medmcp_neuro"],
        input=init_request,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0 or result.stdout, (
        f"Server failed to start: stderr={result.stderr}"
    )
    assert "medmcp-neuro" in result.stdout or "result" in result.stdout
