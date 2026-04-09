"""Chainlit chat UI for MedMCP, backed by vibe-acp.

Spawns a single vibe-acp subprocess and speaks JSON-RPC 2.0 over stdin/stdout.
This gives the UI full access to vibe's tool system (bash, read_file, grep, etc.)
and lets multiple chats live side-by-side as independent ACP sessions on top of
the same subprocess.

Run with:  chainlit run src/medmcp/app.py -w

SECURITY MODEL
==============
This app exposes vibe-acp's full tool surface (bash, write_file, search_replace,
web_fetch, ...) through a chat box. The threat model assumes:

1. The Chainlit server runs on localhost only and is reachable only by the
   operator. There is no real authentication: the ``header_auth_callback``
   below returns a fixed local user solely so Chainlit's data layer (which
   requires a user identifier to scope threads) is happy. Do NOT bind to
   0.0.0.0 or expose port 8000 over a network without replacing this with a
   real auth callback.
2. Every tool call is gated by an interactive ``cl.AskActionMessage`` permission
   prompt — see :func:`_ask_user_for_permission`. The user must click Approve
   before any side effect occurs. There is no auto-approval path. Do NOT change
   this without understanding that the local model may be steered by prompt
   injection (e.g. content pasted from untrusted documents) into running
   arbitrary commands.
3. vibe-acp's own bash allowlist/denylist (``.vibe/config.toml``) is the second
   line of defense. Keep it current.
4. Permission decisions are logged to stderr (the chainlit terminal) for audit.

CHAT HISTORY
============
Chats are persisted in two places:

- vibe-acp writes its own JSONL transcripts to ``.vibe/logs/session/`` (one
  directory per session, with ``messages.jsonl`` and ``meta.json``). This is
  the source of truth that vibe replays from on ``session/load``.
- Chainlit's SQLAlchemy data layer writes a thin index of threads/steps to
  ``.vibe/medmcp_threads.db`` (sqlite). This is what powers the sidebar in the
  Chainlit UI and the chainlit thread_id ↔ vibe-acp session_id mapping (stored
  in thread metadata under the ``vibe_session_id`` key).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import chainlit as cl
from chainlit.context import context as cl_context
from chainlit.data import get_data_layer as cl_get_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from chainlit.user import User

# A loose alias for parsed JSON-RPC payloads. The ACP protocol is too dynamic
# to model exhaustively as TypedDicts, so we keep the wire format as
# ``dict[str, Any]`` and narrow at the read sites.
JsonDict = dict[str, Any]

# ── Audit logger ───────────────────────────────────────────

# Permission decisions are written to stderr so they show up in the chainlit
# terminal. This is the only audit trail; do not silence it.
_audit: logging.Logger = logging.getLogger("medmcp.audit")
if not _audit.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[medmcp.audit %(asctime)s] %(message)s"))
    _audit.addHandler(_h)
    _audit.setLevel(logging.INFO)
    _audit.propagate = False

# ── Configuration ──────────────────────────────────────────

PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent.parent)
VIBE_HOME: Path = Path(PROJECT_ROOT) / ".vibe"
THREADS_DB_PATH: Path = VIBE_HOME / "medmcp_threads.db"

# Single fixed user identity used by the data layer. There is no auth: this
# exists only so chainlit's per-user thread scoping has a stable key.
LOCAL_USER_ID: str = "local"

# ── JSON-RPC wire helpers ──────────────────────────────────


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

    def __init__(self) -> None:
        """Build an unstarted client. Call :meth:`ensure_started` before use."""
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id: int = 0
        self._pending: dict[int, asyncio.Future[JsonDict]] = {}
        self._sessions: dict[str, asyncio.Queue[JsonDict]] = {}
        self._limbo: dict[str, list[JsonDict]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    async def ensure_started(self) -> None:
        """Spawn vibe-acp on first use and run the ACP ``initialize`` handshake.

        Subsequent calls are no-ops. The init lock makes this safe under
        concurrent ``on_chat_start`` calls (e.g. multiple browser tabs).
        """
        async with self._init_lock:
            if self._initialized:
                return
            self.proc = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "vibe-acp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT_ROOT,
                env={**os.environ, "VIBE_HOME": str(VIBE_HOME)},
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            resp = await self.request(
                "initialize",
                {"protocol_version": 1, "client_capabilities": {}},
            )
            if "error" in resp:
                raise RuntimeError(f"vibe-acp initialize failed: {resp['error']}")
            self._initialized = True

    async def _read_loop(self) -> None:
        """Read JSON-RPC frames forever and route them.

        On EOF or unexpected exception, fail every still-pending future so the
        callers don't hang waiting for a response that will never come.
        """
        assert self.proc is not None and self.proc.stdout is not None
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

    async def request(self, method: str, params: JsonDict | None = None) -> JsonDict:
        """Send a JSON-RPC request and await its response.

        The ``try/finally`` around ``await fut`` guarantees the pending-request
        entry is removed from ``_pending`` even if the awaiting task is
        cancelled mid-flight (e.g. the user clicks Stop). Without it, cancelled
        requests would leak entries into ``_pending`` forever.
        """
        assert self.proc is not None and self.proc.stdin is not None
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
        assert self.proc is not None and self.proc.stdin is not None
        async with self._write_lock:
            msg: JsonDict = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
            self.proc.stdin.write(_encode(msg))
            await self.proc.stdin.drain()

    async def respond(self, req_id: int, result: JsonDict) -> None:
        """Send a JSON-RPC response for a server-initiated request."""
        assert self.proc is not None and self.proc.stdin is not None
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


# Module-level singleton. Chainlit imports this file once at startup, but the
# subprocess itself is started lazily on first chat to avoid blocking import.
_client: VibeAcpClient = VibeAcpClient()


# ── Chainlit data layer (sqlite under .vibe/) ─────────────


def _bootstrap_threads_db(db_path: Path) -> None:
    """Create the chainlit data-layer schema if it doesn't exist yet.

    Chainlit's ``SQLAlchemyDataLayer`` does not auto-create tables; it just
    runs raw SQL against whatever schema exists. We bootstrap the minimum
    schema synchronously with stdlib sqlite3 (no async cost, runs once per
    process) so the data layer factory below can return immediately.

    The ``steps`` columns must cover every key that chainlit's
    ``Step.to_dict()`` emits with a non-None default — the data layer's
    ``create_step`` filters out ``None`` values but inserts everything else,
    so a missing column silently fails every ``Step`` write (``type="run"``,
    ``type="tool"``, ...). ``Message.to_dict()`` leaves ``command``/``modes``
    at ``None`` by default, which is why user/assistant messages persist even
    with a minimal schema but tool and on_message ``run`` steps do not —
    breaking chat resume because assistant messages end up as children of a
    run step that was never written.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                "id" TEXT PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" TEXT NOT NULL DEFAULT '{}',
                "createdAt" TEXT
            );

            CREATE TABLE IF NOT EXISTS threads (
                "id" TEXT PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                "tags" TEXT,
                "metadata" TEXT
            );

            CREATE TABLE IF NOT EXISTS steps (
                "id" TEXT PRIMARY KEY,
                "name" TEXT,
                "type" TEXT,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "streaming" BOOLEAN,
                "waitForAnswer" BOOLEAN,
                "isError" BOOLEAN,
                "metadata" TEXT,
                "tags" TEXT,
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" TEXT,
                "showInput" TEXT,
                "language" TEXT,
                "defaultOpen" BOOLEAN,
                "autoCollapse" BOOLEAN,
                "command" TEXT,
                "modes" TEXT
            );

            CREATE TABLE IF NOT EXISTS elements (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INTEGER,
                "language" TEXT,
                "forId" TEXT,
                "mime" TEXT,
                "props" TEXT
            );

            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT,
                "value" INTEGER NOT NULL,
                "comment" TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_steps_threadId ON steps("threadId");
            CREATE INDEX IF NOT EXISTS idx_elements_threadId ON elements("threadId");
            CREATE INDEX IF NOT EXISTS idx_feedbacks_forId ON feedbacks("forId");
        """)

        # Migrate pre-existing databases that were created before the columns
        # above were added. ``ALTER TABLE ADD COLUMN`` is idempotent-unfriendly
        # in sqlite (no IF NOT EXISTS), so probe ``pragma_table_info`` first.
        existing_cols = {
            row[0] for row in conn.execute('SELECT name FROM pragma_table_info("steps")')
        }
        for col, col_type in (
            ("defaultOpen", "BOOLEAN"),
            ("autoCollapse", "BOOLEAN"),
            ("command", "TEXT"),
            ("modes", "TEXT"),
        ):
            if col not in existing_cols:
                conn.execute(f'ALTER TABLE steps ADD COLUMN "{col}" {col_type}')

        # Repair rows orphaned by the pre-fix schema: assistant messages whose
        # ``parentId`` pointed at a ``run`` step that failed to insert. Promote
        # them to top level so they render on chat resume instead of vanishing
        # into a missing parent. Idempotent — a no-op once there are no
        # dangling parent references.
        conn.execute(
            """
            UPDATE steps
               SET "parentId" = NULL
             WHERE "parentId" IS NOT NULL
               AND "parentId" NOT IN (SELECT "id" FROM steps)
            """
        )

        conn.commit()


@cl.header_auth_callback  # pyright: ignore[reportUnknownMemberType]
async def header_auth_callback(_headers: object) -> User | None:
    """Return a fixed local user so chainlit's data layer can scope threads.

    There is no real authentication: every connection is treated as the same
    operator regardless of headers. The threat model is that this app is
    reachable only on localhost. See module docstring for the full security
    model.
    """
    return User(identifier=LOCAL_USER_ID, metadata={"role": "local"})


@cl.data_layer  # pyright: ignore[reportUnknownMemberType]
def get_data_layer() -> SQLAlchemyDataLayer:
    """Wire chainlit to a local sqlite database for thread persistence."""
    _bootstrap_threads_db(THREADS_DB_PATH)
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{THREADS_DB_PATH}")


# ── Permission UI ──────────────────────────────────────────


def _format_permission_prompt(tc: JsonDict) -> str:
    """Build the markdown body shown in the permission dialog.

    ``tc`` is a ToolCallUpdate from ACP — it has ``title`` (human label, e.g.
    ``"bash: ls -la ~/msseg"``) and ``rawInput`` (the JSON-serialized tool args).
    """
    title = tc.get("title") or "tool call"
    raw_input = tc.get("rawInput")

    body = f"**Approve tool call?**\n\n`{title}`"
    if raw_input:
        if isinstance(raw_input, str):
            input_str = raw_input
        else:
            try:
                input_str = json.dumps(raw_input, indent=2)
            except (TypeError, ValueError):
                input_str = str(raw_input)
        body += f"\n\n```json\n{input_str}\n```"
    return body


async def _ask_user_for_permission(tc: JsonDict, options: list[JsonDict]) -> JsonDict:
    """Render an interactive permission prompt and return the ACP outcome.

    Returns the value to put in ``RequestPermissionResponse.outcome``:

    - ``{"outcome": "selected", "optionId": "..."}`` when the user clicks
    - ``{"outcome": "cancelled"}`` on timeout or if no options were offered

    Every decision is logged to stderr via the ``medmcp.audit`` logger so the
    operator running ``just ui`` can see what was approved/denied.
    """
    title = tc.get("title") or tc.get("toolCallId") or "<unknown>"

    if not options:
        _audit.warning("permission request had no options; cancelling: %s", title)
        return {"outcome": "cancelled"}

    actions: list[cl.Action] = [
        cl.Action(
            name=f"perm_{opt.get('optionId', '')}",
            payload={"optionId": opt.get("optionId", "")},
            label=opt.get("name") or opt.get("optionId", ""),
        )
        for opt in options
        if opt.get("optionId")
    ]

    _audit.info("permission requested: %s", title)
    response = await cl.AskActionMessage(
        content=_format_permission_prompt(tc),
        actions=actions,
        timeout=300,
    ).send()

    if response is None:
        _audit.warning("permission timed out: %s", title)
        return {"outcome": "cancelled"}

    # AskActionResponse is a TypedDict whose ``payload`` field is typed as a
    # bare ``Dict``; pyright can't see the contents we put in it on the way out.
    payload = cast("dict[str, Any]", response["payload"] or {})
    option_id = payload.get("optionId")
    if not option_id:
        _audit.warning("permission response missing optionId: %s", title)
        return {"outcome": "cancelled"}

    _audit.info("permission decision: %s -> %s", title, option_id)
    return {"outcome": "selected", "optionId": option_id}


# ── Session helpers ────────────────────────────────────────


def _get_session_id() -> str | None:
    """Return the vibe-acp session id stashed on the current chainlit chat."""
    return cast(
        "str | None",
        cl.user_session.get("vibe_session_id"),  # pyright: ignore[reportUnknownMemberType]
    )


def _set_session_id(session_id: str) -> None:
    """Stash the vibe-acp session id on the current chainlit chat."""
    cl.user_session.set("vibe_session_id", session_id)  # pyright: ignore[reportUnknownMemberType]


# ── Update rendering ──────────────────────────────────────


def _stringify_raw(raw: object) -> str:
    """Render a tool ``rawInput``/``rawOutput`` value for display in a step."""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, indent=2)
    except (TypeError, ValueError):
        return str(raw)


def _extract_text_blocks(content: object) -> list[str]:
    """Pull text out of ACP ``content`` blocks attached to a tool result."""
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for cb in cast("list[Any]", content):
        if not isinstance(cb, dict):
            continue
        cb_dict = cast("JsonDict", cb)
        if cb_dict.get("type") != "content":
            continue
        inner = cb_dict.get("content")
        if isinstance(inner, dict):
            text = cast("JsonDict", inner).get("text")
            if isinstance(text, str):
                out.append(text)
    return out


async def _handle_tool_call(
    update: JsonDict,
    tool_steps: dict[str, cl.Step],
    tool_call_info: dict[str, JsonDict],
    parent_id: str | None,
) -> None:
    """Handle a ``tool_call`` ACP session update.

    vibe-acp emits this event twice for one tool call: first to announce the
    tool name, then again with the resolved ``rawInput``. We treat repeats with
    the same ``toolCallId`` as updates so the UI doesn't grow duplicate steps.
    """
    tc_id = cast("str", update.get("toolCallId") or "")
    info = tool_call_info.setdefault(tc_id, {})
    if (t := update.get("title")) is not None:
        info["title"] = t
    if (ri := update.get("rawInput")) is not None:
        info["rawInput"] = ri

    raw_input = update.get("rawInput")
    if tc_id in tool_steps:
        step = tool_steps[tc_id]
        new_title = update.get("title")
        if isinstance(new_title, str) and new_title:
            step.name = new_title
        if raw_input is not None:
            step.input = _stringify_raw(raw_input)
        await step.update()
    else:
        title_val = update.get("title")
        tool_title = title_val if isinstance(title_val, str) and title_val else "tool"
        step = cl.Step(name=tool_title, type="tool", parent_id=parent_id)
        if raw_input is not None:
            step.input = _stringify_raw(raw_input)
        await step.send()
        tool_steps[tc_id] = step


async def _handle_tool_call_update(update: JsonDict, tool_steps: dict[str, cl.Step]) -> None:
    """Handle a ``tool_call_update`` ACP session update (progress + final result)."""
    tc_id = cast("str", update.get("toolCallId") or "")
    status = cast("str", update.get("status") or "")
    if tc_id not in tool_steps:
        return
    step = tool_steps[tc_id]
    raw_output = update.get("rawOutput")
    if raw_output is not None:
        step.output = _stringify_raw(raw_output)
    else:
        text_parts = _extract_text_blocks(update.get("content"))
        if text_parts:
            step.output = "\n".join(text_parts)
    if status in ("completed", "failed"):
        await step.update()


# ── Chainlit hooks ─────────────────────────────────────────


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]
async def on_chat_start() -> None:
    """Create a fresh vibe-acp session for this chainlit thread.

    The vibe-acp subprocess is shared across all chats; this only allocates a
    new session id on top of it. The mapping from chainlit thread → vibe
    session is persisted in thread metadata so :func:`on_chat_resume` can pick
    it back up later.
    """
    await _client.ensure_started()

    resp = await _client.request("session/new", {"cwd": PROJECT_ROOT, "mcp_servers": []})
    if "error" in resp:
        await cl.Message(content=f"Failed to create vibe-acp session: {resp['error']}").send()
        return

    result = cast("JsonDict", resp.get("result") or {})
    session_id = cast("str", result["sessionId"])
    _set_session_id(session_id)
    _client.register_session(session_id)

    # Persist the chainlit-thread → vibe-session mapping so on_chat_resume can
    # find it. Chainlit creates the thread row on first message anyway; calling
    # update_thread here upserts an empty thread with the metadata set so the
    # mapping is recoverable even if the user never sends a message.
    thread_id = cl_context.session.thread_id
    data_layer = cast("Any", cl_get_data_layer())
    if thread_id and data_layer is not None:
        with contextlib.suppress(Exception):
            await data_layer.update_thread(
                thread_id=thread_id,
                user_id=_persisted_user_id(),
                metadata={"vibe_session_id": session_id},
            )


@cl.on_chat_resume  # pyright: ignore[reportUnknownMemberType]
async def on_chat_resume(thread: ThreadDict) -> None:
    """Reattach to a previously-persisted vibe-acp session.

    Chainlit has already loaded thread state from its own data layer and is
    rendering it from ``thread["steps"]``; we don't need to re-emit any chat
    UI here. We just need to tell vibe-acp to load the session into memory so
    the next prompt has the full context. vibe will replay the conversation
    history at us via ``session/update`` events; we drain and discard them
    because chainlit's persistence is the source of truth for the UI.
    """
    await _client.ensure_started()

    # ThreadDict's typing carries Dict[Unknown, Unknown] for the metadata field,
    # which infects any direct access. Cast to a plain dict[str, Any] once and
    # operate on that.
    thread_any = cast("dict[str, Any]", thread)
    raw_metadata: object = thread_any.get("metadata") or {}
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = cast("object", json.loads(raw_metadata))
        except json.JSONDecodeError:
            raw_metadata = {}
    metadata: dict[str, Any] = (
        cast("dict[str, Any]", raw_metadata) if isinstance(raw_metadata, dict) else {}
    )
    vibe_session_id: object = metadata.get("vibe_session_id")

    if not isinstance(vibe_session_id, str):
        # Old thread without a mapping (or one created before this code shipped).
        # Fall back to a fresh session so the UI is at least usable.
        _audit.warning(
            "resume: thread %s has no vibe_session_id; starting fresh",
            thread_any.get("id"),
        )
        await on_chat_start()
        return

    # Pre-register the queue *before* sending session/load so any replay events
    # that arrive while we're waiting for the response land in the queue
    # rather than in limbo.
    queue = _client.register_session(vibe_session_id)
    _set_session_id(vibe_session_id)

    resp = await _client.request(
        "session/load",
        {"cwd": PROJECT_ROOT, "session_id": vibe_session_id, "mcp_servers": []},
    )
    if "error" in resp:
        _audit.warning("resume: session/load failed: %s", resp["error"])
        await cl.Message(content=f"Could not reload previous session: {resp['error']}").send()
        return

    # Drain replay events. Chainlit already has the conversation in its own
    # data layer and renders it from there, so we just acknowledge and discard.
    #
    # Invariant: vibe-acp flushes all replay frames BEFORE writing the
    # session/load response, so by the time we get here the reader task has
    # already routed them into this queue. Do not change to ``await queue.get()``
    # without verifying that invariant still holds — otherwise stale replay
    # frames could leak into the next on_message.
    while not queue.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()


def _persisted_user_id() -> str | None:
    """Best-effort lookup of the persisted user row id for the current session."""
    user = cl_context.session.user
    if user is None:
        return None
    # PersistedUser has an ``id``; bare User does not. Either is acceptable to
    # update_thread (it falls back to userIdentifier-only if user_id is None).
    return getattr(user, "id", None)


@cl.on_message  # pyright: ignore[reportUnknownMemberType]
async def on_message(message: cl.Message) -> None:
    """Send a prompt to vibe-acp and stream the response back into the UI."""
    session_id = _get_session_id()
    if session_id is None:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    queue = _client.get_session_queue(session_id)
    if queue is None:
        # Defensive: a queue should always exist for an in-flight chat.
        queue = _client.register_session(session_id)

    # Chainlit wraps each on_message handler in a parent Step(type="run").
    # We attach tool steps AND the assistant message as siblings of that run
    # step, and create them in temporal order. The frontend renders children
    # in append order, so tool steps appear before the assistant text.
    run_step = cl_context.current_step
    parent_id: str | None = run_step.id if run_step else None

    assistant_msg: cl.Message | None = None

    async def _ensure_assistant_msg() -> cl.Message:
        nonlocal assistant_msg
        if assistant_msg is None:
            assistant_msg = cl.Message(content="", parent_id=parent_id)
            await assistant_msg.send()
        return assistant_msg

    tool_steps: dict[str, cl.Step] = {}
    # Cache the tool-call metadata from each `tool_call` event so we can show
    # it later in the permission dialog. vibe-acp's `session/request_permission`
    # payload only carries `toolCallId`, not the title or raw_input.
    tool_call_info: dict[str, JsonDict] = {}

    # Send the prompt and get a future for its response. We then race the
    # future against queue reads so session_update notifications and
    # request_permission requests are interleaved with the agent loop.
    prompt_fut = asyncio.ensure_future(
        _client.request(
            "session/prompt",
            {
                "session_id": session_id,
                "prompt": [{"type": "text", "text": message.content}],
            },
        )
    )

    async def _cancel_and_drain() -> None:
        """Tell vibe-acp to abort its agent loop on this session.

        Used when our own asyncio task gets cancelled (Chainlit stop button or
        the user sending a new message). Without this, vibe-acp keeps running
        the previous agent loop and the next ``session/prompt`` gets rejected
        with "Concurrent prompts are not supported yet".
        """
        with contextlib.suppress(Exception):
            await _client.notify("session/cancel", {"session_id": session_id})

    try:
        while True:
            # Wait for either the next inbound frame for this session or the
            # prompt response. We can't just `await queue.get()` because the
            # response future may resolve while the queue is empty.
            get_task: asyncio.Task[JsonDict] = asyncio.ensure_future(queue.get())
            done, _pending = await asyncio.wait(
                {get_task, prompt_fut},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if get_task in done:
                msg = get_task.result()
            else:
                # Prompt finished. Cancel the queue read and drain anything
                # already buffered (e.g. a final usage_update).
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await get_task
                while not queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        leftover = queue.get_nowait()
                        await _process_session_frame(
                            leftover,
                            assistant_msg_getter=_ensure_assistant_msg,
                            tool_steps=tool_steps,
                            tool_call_info=tool_call_info,
                            parent_id=parent_id,
                        )
                # Surface any error from the prompt response itself.
                resp = prompt_fut.result()
                if "error" in resp:
                    err = cast("JsonDict", resp["error"])
                    target = await _ensure_assistant_msg()
                    err_msg = err.get("message", str(err))
                    await target.stream_token(f"\n\nError: {err_msg}")
                break

            await _process_session_frame(
                msg,
                assistant_msg_getter=_ensure_assistant_msg,
                tool_steps=tool_steps,
                tool_call_info=tool_call_info,
                parent_id=parent_id,
            )

    except asyncio.CancelledError:
        # Chainlit cancels our task when the user clicks Stop or sends a new
        # message mid-stream. Tell vibe-acp to abort its agent loop too,
        # otherwise the next session/prompt is rejected as concurrent.
        #
        # Also cancel the in-flight prompt request so the ``try/finally`` in
        # ``VibeAcpClient.request`` runs and pops its entry from ``_pending``.
        if not prompt_fut.done():
            prompt_fut.cancel()
        await _cancel_and_drain()
        raise

    if assistant_msg is not None:
        await assistant_msg.update()


async def _process_session_frame(
    msg: JsonDict,
    *,
    assistant_msg_getter: Callable[[], Awaitable[cl.Message]],
    tool_steps: dict[str, cl.Step],
    tool_call_info: dict[str, JsonDict],
    parent_id: str | None,
) -> None:
    """Dispatch one inbound JSON-RPC frame from a session queue.

    Handles both ``session/update`` notifications (text chunks, tool calls,
    tool results) and ``session/request_permission`` server requests, which
    must be answered with a JSON-RPC response carrying the original request id.
    """
    method = msg.get("method")

    if method == "session/update":
        params = cast("JsonDict", msg.get("params") or {})
        update = cast("JsonDict", params.get("update") or {})
        update_type = update.get("sessionUpdate")

        if update_type == "agent_message_chunk":
            content = cast("JsonDict", update.get("content") or {})
            if content.get("type") == "text":
                target = await assistant_msg_getter()
                text = cast("str", content.get("text") or "")
                await target.stream_token(text)

        elif update_type == "tool_call":
            await _handle_tool_call(update, tool_steps, tool_call_info, parent_id)

        elif update_type == "tool_call_update":
            await _handle_tool_call_update(update, tool_steps)

    elif method == "session/request_permission":
        req_id_raw = msg.get("id")
        if not isinstance(req_id_raw, int):
            return  # cannot respond without a request id
        req_id: int = req_id_raw
        params = cast("JsonDict", msg.get("params") or {})
        tc: JsonDict = dict(cast("JsonDict", params.get("toolCall") or {}))
        options = cast("list[JsonDict]", params.get("options") or [])
        # Backfill title/rawInput from the cached tool_call event, because
        # request_permission only ships the toolCallId.
        cached = tool_call_info.get(cast("str", tc.get("toolCallId") or ""), {})
        for key in ("title", "rawInput"):
            if tc.get(key) is None and cached.get(key) is not None:
                tc[key] = cached[key]

        outcome = await _ask_user_for_permission(tc, options)
        await _client.respond(req_id, {"outcome": outcome})


@cl.on_stop  # pyright: ignore[reportUnknownMemberType]
async def on_stop() -> None:
    """Forward Chainlit's stop button to vibe-acp's ``session/cancel``.

    Without this, Chainlit cancels its own task but vibe-acp keeps running its
    agent loop — and the next user prompt fails with
    "Concurrent prompts are not supported yet".
    """
    session_id = _get_session_id()
    if session_id is None:
        return
    with contextlib.suppress(Exception):
        await _client.notify("session/cancel", {"session_id": session_id})


@cl.on_chat_end  # pyright: ignore[reportUnknownMemberType]
async def on_chat_end() -> None:
    """Detach the chat from its vibe-acp session queue.

    The subprocess is shared across chats and stays alive; we only release the
    inbound queue so its memory can be reclaimed. The vibe-acp session itself
    remains in vibe's in-memory session table (and on disk under
    ``.vibe/logs/session/``) so it can be resumed later via ``session/load``.
    """
    session_id = _get_session_id()
    if session_id is not None:
        _client.unregister_session(session_id)
