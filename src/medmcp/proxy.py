"""``medmcp-mcp-proxy`` — the thin per-call MCP shim vibe-acp spawns.

vibe-acp spawns one ``medmcp-mcp-proxy <stack>`` process per tool call. Instead of
launching the real (heavy) stack server, this shim presents as an MCP stdio server
to vibe and forwards ``list_tools`` / ``call_tool`` to the persistent backend pool
via the broker socket (``MEDMCP_BROKER_SOCK``). The expensive spawn/import/CUDA
cost is paid once in the pool, not on every call. The shim itself imports almost
nothing, so vibe's per-call spawn stays cheap.

If the broker socket is unreachable (the pool isn't running), the shim falls back
to spawning the real stack directly from the backend registry
(``MEDMCP_BACKENDS_FILE`` / ``.vibe/backends.json``) — degraded to a per-call cold
start, but a chat never breaks. See ``docs/stack-prewarm-proxy.md`` (Layer 1).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from medmcp.acp import VIBE_HOME
from medmcp.backend_broker import STREAM_LIMIT

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

_DEFAULT_STARTUP_TIMEOUT_SEC: float = 60.0
_DEFAULT_TOOL_TIMEOUT_SEC: float = 900.0


class ProxyError(Exception):
    """A broker- or registry-level error to surface to the caller as a tool error."""


class _BrokerUnavailableError(Exception):
    """The broker socket could not be reached; triggers the direct-spawn fallback."""


def _backends_path() -> Path:
    """Path to the backend registry (env override, else ``.vibe/backends.json``)."""
    override = os.environ.get("MEDMCP_BACKENDS_FILE", "").strip()
    return Path(override) if override else VIBE_HOME / "backends.json"


class _Forwarder:
    """Forwards a stack's tool calls to the broker, with a direct-spawn fallback."""

    def __init__(self, stack: str) -> None:
        self._stack = stack
        self._sock_path = os.environ.get("MEDMCP_BROKER_SOCK", "").strip()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._req_id = 0
        # No socket configured ⇒ go straight to direct-spawn mode.
        self._direct = not self._sock_path
        self._spec_cache: JsonDict | None = None

    async def list_tools(self) -> list[mcp_types.Tool]:
        if not self._direct:
            try:
                resp = await self._request({"op": "list_tools", "stack": self._stack})
                raw = cast("list[Any]", resp.get("tools", []))
                return [mcp_types.Tool.model_validate(t) for t in raw]
            except _BrokerUnavailableError as exc:
                log.warning("broker unavailable (%s); direct-spawning %s", exc, self._stack)
                self._direct = True
        return await self._direct_list_tools()

    async def call_tool(self, name: str, args: JsonDict) -> mcp_types.CallToolResult:
        if not self._direct:
            try:
                resp = await self._request(
                    {"op": "call_tool", "stack": self._stack, "tool": name, "args": args}
                )
                return mcp_types.CallToolResult.model_validate(resp["result"])
            except _BrokerUnavailableError as exc:
                log.warning("broker unavailable (%s); direct-spawning %s", exc, self._stack)
                self._direct = True
        return await self._direct_call(name, args)

    async def aclose(self) -> None:
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    # ── broker transport ─────────────────────────────────────────────────────

    async def _request(self, payload: JsonDict) -> JsonDict:
        if self._writer is None or self._reader is None:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    self._sock_path, limit=STREAM_LIMIT
                )
            except OSError as exc:
                raise _BrokerUnavailableError(str(exc)) from exc
        self._req_id += 1
        payload["id"] = self._req_id
        try:
            self._writer.write(json.dumps(payload).encode() + b"\n")
            await self._writer.drain()
            line = await self._reader.readline()
        except OSError as exc:
            raise _BrokerUnavailableError(str(exc)) from exc
        if not line:
            raise _BrokerUnavailableError("broker closed the connection")
        resp = cast("JsonDict", json.loads(line))
        if not resp.get("ok"):
            raise ProxyError(str(resp.get("error", "broker error")))
        return resp

    # ── direct-spawn fallback ────────────────────────────────────────────────

    def _spec(self) -> JsonDict:
        if self._spec_cache is not None:
            return self._spec_cache
        try:
            data = cast("JsonDict", json.loads(_backends_path().read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProxyError(f"cannot read backend registry: {exc}") from exc
        entry = data.get(self._stack)
        if not isinstance(entry, dict):
            raise ProxyError(f"stack {self._stack!r} is not configured")
        self._spec_cache = cast("JsonDict", entry)
        return self._spec_cache

    def _server_params(self) -> StdioServerParameters:
        spec = self._spec()
        env: dict[str, str] = {**os.environ}
        extra = spec.get("env")
        if isinstance(extra, dict):
            env.update({str(k): str(v) for k, v in cast("JsonDict", extra).items()})
        cwd = spec.get("cwd") or os.environ.get("MEDMCP_WORKSPACE")
        return StdioServerParameters(
            command=str(spec["command"]),
            args=[str(a) for a in cast("list[Any]", spec.get("args", []))],
            env=env,
            cwd=str(cwd) if cwd else None,
        )

    def _startup_timeout(self) -> timedelta:
        secs = self._spec().get("startup_timeout_sec") or _DEFAULT_STARTUP_TIMEOUT_SEC
        return timedelta(seconds=float(secs))

    def _tool_timeout(self) -> timedelta:
        secs = self._spec().get("tool_timeout_sec") or _DEFAULT_TOOL_TIMEOUT_SEC
        return timedelta(seconds=float(secs))

    async def _direct_list_tools(self) -> list[mcp_types.Tool]:
        async with (
            stdio_client(self._server_params()) as (read, write),
            ClientSession(read, write, read_timeout_seconds=self._startup_timeout()) as session,
        ):
            await session.initialize()
            resp = await session.list_tools()
            return list(resp.tools)

    async def _direct_call(self, name: str, args: JsonDict) -> mcp_types.CallToolResult:
        async with (
            stdio_client(self._server_params()) as (read, write),
            ClientSession(read, write, read_timeout_seconds=self._startup_timeout()) as session,
        ):
            await session.initialize()
            return await session.call_tool(name, args, read_timeout_seconds=self._tool_timeout())


async def _serve(stack: str) -> None:
    """Run the proxy MCP server for *stack* over stdio until vibe disconnects."""
    server: Server[object, object] = Server(f"medmcp-proxy-{stack}")
    forwarder = _Forwarder(stack)

    # Registered via decorator side effect; pyright can't see the access.
    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]
        return await forwarder.list_tools()

    @server.call_tool(validate_input=False)
    async def _call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str, arguments: JsonDict
    ) -> mcp_types.CallToolResult:
        try:
            return await forwarder.call_tool(name, arguments)
        except ProxyError as exc:
            # Surface a broker/registry error as a tool failure, not a crash.
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=str(exc))],
                isError=True,
            )

    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await forwarder.aclose()


def main() -> None:
    """Console-script entry point: ``medmcp-mcp-proxy <stack-name>``."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    args = sys.argv[1:]
    if len(args) != 1 or not args[0].strip():
        sys.stderr.write("usage: medmcp-mcp-proxy <stack-name>\n")
        raise SystemExit(2)
    asyncio.run(_serve(args[0].strip()))


if __name__ == "__main__":
    main()
