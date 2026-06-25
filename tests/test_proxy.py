"""End-to-end tests for the medmcp-mcp-proxy shim.

Each test spawns the proxy as a real MCP stdio server (``python -m medmcp.proxy
<stack>``) and drives it as an MCP client, exercising both paths: forwarding to a
live broker, and the direct-spawn fallback when no broker is reachable.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import mcp.types as mcp_types
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medmcp import replay
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool, BackendSpec

_FAKE_SERVER: Path = Path(__file__).parent / "fake_stack_server.py"


def _spec(name: str) -> BackendSpec:
    return BackendSpec(
        name=name,
        command=sys.executable,
        args=[str(_FAKE_SERVER)],
        env={},
        gpu=False,
        idle_ttl_sec=300.0,
        startup_timeout_sec=30.0,
        tool_timeout_sec=30.0,
    )


def _resolver(specs: dict[str, BackendSpec]) -> Callable[[str], BackendSpec | None]:
    def resolve(name: str) -> BackendSpec | None:
        return specs.get(name)

    return resolve


@asynccontextmanager
async def _proxy_client(env: dict[str, str]) -> AsyncGenerator[ClientSession]:
    """Spawn the proxy as an MCP server and yield a connected client session."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medmcp.proxy", "fake"],
        env={**os.environ, **env},
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.asyncio
async def test_proxy_forwards_to_broker(tmp_path: Path) -> None:
    """With a live broker, the proxy forwards list_tools and call_tool."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    broker = BackendBroker(pool, tmp_path / "broker.sock")
    await broker.start()
    try:
        async with _proxy_client({"MEDMCP_BROKER_SOCK": str(broker.socket_path)}) as session:
            names = {t.name for t in (await session.list_tools()).tools}
            assert {"echo", "warmup", "crash"} <= names
            result = await session.call_tool("echo", {"text": "hi"})
            assert replay.extract_structured(result) == {"text": "hi"}
            assert result.isError is False
    finally:
        await broker.aclose()
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_falls_back_to_direct_spawn(tmp_path: Path) -> None:
    """With no reachable broker, the proxy spawns the stack from the registry."""
    backends = tmp_path / "backends.json"
    backends.write_text(
        json.dumps(
            {
                "fake": {
                    "command": sys.executable,
                    "args": [str(_FAKE_SERVER)],
                    "env": {},
                    "startup_timeout_sec": 30.0,
                    "tool_timeout_sec": 30.0,
                }
            }
        )
    )
    env = {
        "MEDMCP_BROKER_SOCK": str(tmp_path / "does-not-exist.sock"),
        "MEDMCP_BACKENDS_FILE": str(backends),
    }
    async with _proxy_client(env) as session:
        result = await session.call_tool("echo", {"text": "via-fallback"})
        assert replay.extract_structured(result) == {"text": "via-fallback"}


@pytest.mark.asyncio
async def test_proxy_reports_broker_error_as_tool_error(tmp_path: Path) -> None:
    """An uninstalled stack comes back as an error result, not a crash."""
    pool = BackendPool(resolve_spec=_resolver({}))  # nothing installed
    broker = BackendBroker(pool, tmp_path / "broker.sock")
    await broker.start()
    try:
        async with _proxy_client({"MEDMCP_BROKER_SOCK": str(broker.socket_path)}) as session:
            result = await session.call_tool("echo", {"text": "x"})
            assert result.isError is True
            text = "".join(b.text for b in result.content if isinstance(b, mcp_types.TextContent))
            assert "not installed" in text
    finally:
        await broker.aclose()
        await pool.aclose()
