"""Unix-socket broker fronting the :class:`~medmcp.backend_pool.BackendPool`.

vibe-acp spawns ``medmcp-mcp-proxy <stack>`` per tool call (see ``proxy.py``); that
shim connects to this broker over a unix domain socket and forwards ``list_tools``
/ ``call_tool`` for its stack. The broker dispatches to the persistent pool —
warming the backend if needed — and relays the result back, so the expensive
spawn/import/CUDA cost is paid once in the pool instead of on every call.

This is the server half of the broker protocol (Layer 1 of
``docs/stack-prewarm-proxy.md``). Activation pre-warm / eviction are driven
in-process by the workspace server holding the pool directly; the socket carries
only the per-call forwarding ops.

Wire protocol — newline-delimited JSON, one object per line.

Request (proxy → broker)::

    {"op": "list_tools", "stack": "medmcp-neuro", "id": 1}
    {"op": "call_tool", "stack": "medmcp-neuro", "tool": "skull_strip",
     "args": {...}, "id": 2}

Response (broker → proxy)::

    {"id": 1, "ok": true, "tools": [ <Tool.model_dump> ... ]}
    {"id": 2, "ok": true, "result": <CallToolResult.model_dump>}
    {"id": 2, "ok": false, "error": "stack 'x' is not installed"}

Sampling relay is intentionally omitted in v1 (no stack uses MCP sampling).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from medmcp.backend_pool import BackendError, BackendPool

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# Per-connection stream buffer cap. Tool results can be large (mirrors the 16 MB
# vibe-acp stdout buffer); newline-delimited JSON otherwise overflows asyncio's
# default 64 KB stream limit. Proxy clients MUST use the same cap on their reader.
STREAM_LIMIT: int = 32 * 1024 * 1024


class BrokerRequestError(Exception):
    """Raised for a malformed broker request (reported back as ``ok: false``)."""


class BackendBroker:
    """Serves a :class:`BackendPool` to proxy shim processes over a unix socket."""

    def __init__(self, pool: BackendPool, socket_path: Path) -> None:
        """Create a broker; call :meth:`start` to begin listening.

        Args:
            pool: The persistent backend pool to dispatch to (owned elsewhere — the
                broker does not close it).
            socket_path: Path to the unix domain socket to bind.
        """
        self._pool = pool
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Bind the unix socket and start accepting proxy connections."""
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()  # clear a stale socket from a prior crash
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self._socket_path), limit=STREAM_LIMIT
        )
        # Whoever can connect here can invoke any installed stack's tools on the
        # workspace, below the permission flow entirely — so the mode is set
        # rather than inherited. Connecting needs write permission, which the
        # usual 022 umask already withholds from others, but the umask is a
        # property of how the process was launched (a unit file with UMask=0000
        # yields 0777) and not something this code should depend on.
        os.chmod(self._socket_path, 0o600)
        log.info("backend broker listening at %s", self._socket_path)

    async def aclose(self) -> None:
        """Stop listening and remove the socket file (does not close the pool)."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()

    @property
    def socket_path(self) -> Path:
        """The unix socket path this broker binds."""
        return self._socket_path

    # ── connection handling ──────────────────────────────────────────────────

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one proxy connection until it disconnects (EOF on stdin close)."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # proxy disconnected
                await self._dispatch_line(line, writer)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:  # a bad connection must not take down the broker
            log.warning("broker connection error: %s", exc)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch_line(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            parsed: object = json.loads(line)
        except json.JSONDecodeError as exc:
            await self._send(writer, {"id": None, "ok": False, "error": f"bad request: {exc}"})
            return
        if not isinstance(parsed, dict):
            await self._send(
                writer, {"id": None, "ok": False, "error": "request must be a JSON object"}
            )
            return

        req = cast("JsonDict", parsed)
        req_id = req.get("id")
        op = req.get("op")
        stack = req.get("stack")
        try:
            resp = await self._run_op(req_id, op, stack, req)
        except (BackendError, BrokerRequestError) as exc:
            resp = {"id": req_id, "ok": False, "error": str(exc)}
        except Exception as exc:  # surface any failure rather than dropping the call
            log.warning("broker dispatch failed (op=%s stack=%s): %s", op, stack, exc)
            resp = {"id": req_id, "ok": False, "error": str(exc)}
        await self._send(writer, resp)

    async def _run_op(self, req_id: object, op: object, stack: object, req: JsonDict) -> JsonDict:
        if not isinstance(stack, str) or not stack:
            raise BrokerRequestError("missing 'stack'")

        if op == "list_tools":
            tools = await self._pool.list_tools(stack)
            dumped = [t.model_dump(mode="json") for t in tools]
            return {"id": req_id, "ok": True, "tools": dumped}

        if op == "call_tool":
            tool = req.get("tool")
            if not isinstance(tool, str) or not tool:
                raise BrokerRequestError("missing 'tool'")
            raw_args: object = req.get("args") or {}
            if not isinstance(raw_args, dict):
                raise BrokerRequestError("'args' must be an object")
            args: JsonDict = cast("JsonDict", raw_args)
            result = await self._pool.call(stack, tool, args)
            return {"id": req_id, "ok": True, "result": result.model_dump(mode="json")}

        raise BrokerRequestError(f"unknown op {op!r}")

    async def _send(self, writer: asyncio.StreamWriter, payload: JsonDict) -> None:
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()
