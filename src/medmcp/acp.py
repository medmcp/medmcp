"""UI-agnostic client for the vibe-acp subprocess.

Owns the JSON-RPC 2.0 framing over stdin/stdout and the demultiplexing of
server-initiated frames into per-session queues. This module carries **no
Chainlit dependency** so it can back any frontend (the Chainlit app in
``app.py`` and the workspace server in ``server.py``).

The security model lives with the callers: every UI built on this client must
gate ``session/request_permission`` on an interactive user decision — there is
no auto-approval path here or anywhere downstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

# A loose alias for parsed JSON-RPC payloads. The ACP protocol is too dynamic
# to model exhaustively as TypedDicts, so we keep the wire format as
# ``dict[str, Any]`` and narrow at the read sites.
JsonDict = dict[str, Any]

# ── Audit logger ───────────────────────────────────────────

# Permission decisions are written to stderr so they show up in the server
# terminal. This is the only audit trail; do not silence it.
_audit: logging.Logger = logging.getLogger("medmcp.audit")
if not _audit.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[medmcp.audit %(asctime)s] %(message)s"))
    _audit.addHandler(_h)
    _audit.setLevel(logging.INFO)
    _audit.propagate = False

# src/medmcp → src → <root>. Callers may override per-instance via the
# ``VibeAcpClient`` constructor.
PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent.parent)
VIBE_HOME: Path = Path(PROJECT_ROOT) / ".vibe"


def _encode(msg: JsonDict) -> bytes:
    """Encode a JSON-RPC message as a single UTF-8 line."""
    return (json.dumps(msg) + "\n").encode()


def _rpc_response(req_id: int, result: JsonDict) -> bytes:
    """Build a JSON-RPC response payload for a given request id."""
    return _encode({"jsonrpc": "2.0", "id": req_id, "result": result})


async def _read_line(stdout: asyncio.StreamReader) -> JsonDict | None:
    """Read one JSON-RPC line from ``stdout``, skipping non-JSON noise.

    Returns ``None`` on EOF. Otherwise returns the parsed object as a dict.
    Lines that don't parse as JSON are silently dropped — vibe-acp occasionally
    emits log lines on its own stdout that we just want to ignore.
    """
    while True:
        line = await stdout.readline()
        if not line:
            return None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast("JsonDict", parsed)


# ── vibe-acp client (one subprocess, many sessions) ───────
class VibeAcpClient:
    """Owns a single ``vibe-acp`` subprocess and demuxes JSON-RPC frames.

    A long-running background reader task reads frames from the subprocess's
    stdout and routes them in two directions:

    - **Responses** (frames with an ``id`` and a ``result``/``error`` field) are
      delivered to the ``asyncio.Future`` registered for that request id by
      :meth:`request`.
    - **Server-initiated frames** (notifications and ``session/request_permission``
      requests) are routed to the per-session ``asyncio.Queue`` registered by
      :meth:`register_session`. Frames that arrive *before* a queue is
      registered for their session id are buffered in ``_limbo`` and flushed
      when the session is registered — this matters because vibe-acp may emit
      ``update_available_commands`` for a freshly-created session before our
      ``new_session`` response has come back to the caller.

    A single global ``_write_lock`` serializes writes to stdin so concurrent
    callers cannot interleave bytes mid-frame.
    """

    def __init__(self, cwd: str = PROJECT_ROOT, vibe_home: Path = VIBE_HOME) -> None:
        """Build an unstarted client. Call :meth:`ensure_started` before use."""
        self._cwd: str = cwd
        self._vibe_home: Path = vibe_home
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id: int = 0
        self._pending: dict[int, asyncio.Future[JsonDict]] = {}
        self._sessions: dict[str, asyncio.Queue[JsonDict]] = {}
        self._limbo: dict[str, list[JsonDict]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    async def stop(self) -> None:
        """Terminate the vibe-acp process and reset client state.

        After this call :meth:`ensure_started` will spawn a fresh process.
        Existing session queues are cleared; any in-flight requests will
        receive a ``RuntimeError`` via their futures.
        """
        async with self._init_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Tear down the process and reset state; caller holds ``_init_lock``."""
        if self.proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            self.proc = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("vibe-acp restarted"))
        self._pending.clear()
        self._sessions.clear()
        self._limbo.clear()
        self._initialized = False

    async def ensure_started(self) -> None:
        """Spawn vibe-acp on first use and run the ACP ``initialize`` handshake.

        Subsequent calls are no-ops. The init lock makes this safe under
        concurrent ``on_chat_start`` calls (e.g. multiple browser tabs).
        """
        async with self._init_lock:
            if self._initialized:
                return
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    "vibe-acp",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=None,  # inherit parent stderr so logs appear in terminal
                    limit=16 * 1024 * 1024,  # 16 MB; default 64 KB overflows with LLM responses
                    cwd=self._cwd,
                    env={**os.environ, "VIBE_HOME": str(self._vibe_home)},
                )
                self._reader_task = asyncio.create_task(self._read_loop())
                resp = await self.request(
                    "initialize",
                    {"protocol_version": 1, "client_capabilities": {}},
                )
                if "error" in resp:
                    raise RuntimeError(f"vibe-acp initialize failed: {resp['error']}")
            except BaseException:
                # A failed handshake must not leak the live process/reader:
                # the next ensure_started() would overwrite self.proc with a
                # fresh spawn while the old one keeps running.
                await self._stop_locked()
                raise
            self._initialized = True

    async def _read_loop(self) -> None:
        """Read JSON-RPC frames forever and route them.

        On EOF or unexpected exception, fail every still-pending future so the
        callers don't hang waiting for a response that will never come.
        """
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("vibe-acp is not running; call ensure_started() first")
        try:
            while True:
                msg = await _read_line(self.proc.stdout)
                if msg is None:
                    break
                # Response to one of our outgoing requests
                if "id" in msg and ("result" in msg or "error" in msg):
                    raw_id = msg.get("id")
                    if isinstance(raw_id, int):
                        fut = self._pending.pop(raw_id, None)
                        if fut is not None and not fut.done():
                            fut.set_result(msg)
                    continue
                # Server-initiated: notification (no id) OR request (with id)
                method = msg.get("method")
                if not isinstance(method, str):
                    continue
                params = cast("JsonDict", msg.get("params") or {})
                # Pydantic models in vibe-acp use camelCase aliases on the wire.
                session_id_raw = params.get("sessionId") or params.get("session_id")
                if not isinstance(session_id_raw, str):
                    continue
                session_id = session_id_raw
                if session_id in self._sessions:
                    await self._sessions[session_id].put(msg)
                else:
                    # Stash until register_session() catches up.
                    self._limbo.setdefault(session_id, []).append(msg)
        except Exception:
            _audit.exception("vibe-acp reader crashed")
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("vibe-acp subprocess closed"))
            self._pending.clear()
            # Allow ensure_started() to respawn on the next request.
            self._initialized = False
            self.proc = None

    async def request(self, method: str, params: JsonDict | None = None) -> JsonDict:
        """Send a JSON-RPC request and await its response.

        The ``try/finally`` around ``await fut`` guarantees the pending-request
        entry is removed from ``_pending`` even if the awaiting task is
        cancelled mid-flight (e.g. the user clicks Stop). Without it, cancelled
        requests would leak entries into ``_pending`` forever.
        """
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("vibe-acp is not running; call ensure_started() first")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[JsonDict] = loop.create_future()
        async with self._write_lock:
            req_id = self._next_id
            self._next_id += 1
            self._pending[req_id] = fut
            msg: JsonDict = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                msg["params"] = params
            self.proc.stdin.write(_encode(msg))
            await self.proc.stdin.drain()
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def notify(self, method: str, params: JsonDict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("vibe-acp is not running; call ensure_started() first")
        async with self._write_lock:
            msg: JsonDict = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
            self.proc.stdin.write(_encode(msg))
            await self.proc.stdin.drain()

    async def respond(self, req_id: int, result: JsonDict) -> None:
        """Send a JSON-RPC response for a server-initiated request."""
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("vibe-acp is not running; call ensure_started() first")
        async with self._write_lock:
            self.proc.stdin.write(_rpc_response(req_id, result))
            await self.proc.stdin.drain()

    def register_session(self, session_id: str) -> asyncio.Queue[JsonDict]:
        """Create the per-session inbound queue and flush any limbo messages."""
        queue: asyncio.Queue[JsonDict] = asyncio.Queue()
        self._sessions[session_id] = queue
        for buffered in self._limbo.pop(session_id, []):
            queue.put_nowait(buffered)
        return queue

    def unregister_session(self, session_id: str) -> None:
        """Drop the per-session queue. Does not affect the subprocess."""
        self._sessions.pop(session_id, None)

    def get_session_queue(self, session_id: str) -> asyncio.Queue[JsonDict] | None:
        """Look up an existing per-session queue without creating one."""
        return self._sessions.get(session_id)
