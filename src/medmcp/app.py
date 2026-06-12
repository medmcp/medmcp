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
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import chainlit as cl
import httpx
from chainlit.context import context as cl_context
from chainlit.data import get_data_layer as cl_get_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.server import app as _chainlit_app
from chainlit.types import CommandDict, ThreadDict
from chainlit.user import User
from chainlit.utils import utc_now as _utc_now
from fastapi.routing import APIRoute

from medmcp import distill, provenance, replay
from medmcp.acp import JsonDict, VibeAcpClient
from medmcp.explain import RISK_CATEGORIES
from medmcp.explain import SEVERITY_ICON as _SEVERITY_ICON
from medmcp.explain import generate_explanation as _generate_explanation
from medmcp.explain import raw_input_to_str as _raw_input_to_str
from medmcp.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from medmcp.settings import active_servers as _active_servers
from medmcp.settings import cached_context_window as _cached_context_window
from medmcp.settings import discover_workflows as _discover_workflows
from medmcp.settings import fetch_context_window as _fetch_context_window
from medmcp.settings import load_active_server_names as _load_active_server_names
from medmcp.settings import load_active_workflow_names as _load_active_workflow_names
from medmcp.settings import load_explain_enabled as _load_explain_enabled
from medmcp.settings import load_mcp_servers as _load_mcp_servers
from medmcp.settings import load_provenance_enabled as _load_provenance_enabled
from medmcp.settings import load_workflows_enabled as _load_workflows_enabled
from medmcp.settings import save_active_server_names as _save_active_server_names
from medmcp.settings import save_active_workflow_names as _save_active_workflow_names
from medmcp.settings import save_explain_enabled as _save_explain_enabled
from medmcp.settings import save_provenance_enabled as _save_provenance_enabled
from medmcp.settings import save_workflows_enabled as _save_workflows_enabled
from medmcp.settings import sync_servers_to_vibe_config as _sync_servers_to_vibe_config
from medmcp.workflow import Recipe

# ── Audit logger ───────────────────────────────────────────

# Permission decisions are written to stderr so they show up in the chainlit
# terminal (handler configured in medmcp.acp). This is the only audit trail;
# do not silence it.
_audit: logging.Logger = logging.getLogger("medmcp.audit")

# ── Configuration ──────────────────────────────────────────

PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent.parent)
VIBE_HOME: Path = Path(PROJECT_ROOT) / ".vibe"
THREADS_DB_PATH: Path = VIBE_HOME / "medmcp_threads.db"

# Single fixed user identity used by the data layer. There is no auth: this
# exists only so chainlit's per-user thread scoping has a stable key.
LOCAL_USER_ID: str = "local"

# Composer command id (a button in the message box) and the review action names
# for the in-chat workflow-distillation flow. See _handle_save_workflow_command.
SAVE_WORKFLOW_COMMAND: str = "save-workflow"
MANAGE_WORKFLOWS_COMMAND: str = "manage-workflows"
PROMOTE_WORKFLOW_ACTION: str = "promote_workflow"
REFINE_WORKFLOW_ACTION: str = "refine_workflow"
RENAME_WORKFLOW_ACTION: str = "rename_workflow"
DISCARD_WORKFLOW_ACTION: str = "discard_workflow"
TEST_WORKFLOW_ACTION: str = "test_workflow"
DELETE_WORKFLOW_ACTION: str = "delete_workflow"
EDIT_WORKFLOW_ACTION: str = "edit_workflow"
RUN_WORKFLOW_ACTION: str = "run_workflow"
CONFIRM_REPLAY_ACTION: str = "confirm_replay"


def _build_chat_settings_inputs() -> list[Any]:
    """Return the ChatSettings Tab list for on_chat_start / on_chat_resume."""
    general_tab = cl.input_widget.Tab(
        id="general",
        label="General",
        inputs=[
            cl.input_widget.Switch(
                id="explain_tools",
                label="Explain tool calls",
                initial=_load_explain_enabled(),
                description=(
                    "Enable this option to add a plain-language explanation to each tool call."
                ),
            ),
            cl.input_widget.Switch(
                id="record_provenance",
                label="Record provenance",
                initial=_load_provenance_enabled(),
                description=(
                    "Record a reproducible log of each session: environment, tool calls, "
                    "permission decisions, and a summary report."
                ),
            ),
            cl.input_widget.Switch(
                id="workflows_enabled",
                label="Personal workflows",
                initial=_load_workflows_enabled(),
                description=(
                    "Turn the personal-workflows feature on or off. When off, the Save/Manage "
                    "workflow buttons are hidden and no saved workflow is loaded as a skill."
                ),
            ),
        ],
    )

    servers = _load_mcp_servers()
    active_names = _load_active_server_names()
    stack_inputs: list[cl.input_widget.InputWidget] = [
        cl.input_widget.Switch(
            id=f"stack_{srv['name']}",
            label=srv["name"] + (f"  [v{srv['version']}]" if srv.get("version") else ""),
            initial=srv["name"] in active_names,
            description=(
                f"Load the {srv['name']} MCP stack. Changes take effect on the next conversation."
            ),
        )
        for srv in servers
    ]
    stacks_tab = cl.input_widget.Tab(
        id="stacks",
        label="Extensions (MCP)",
        inputs=stack_inputs,
    )

    tabs: list[Any] = [general_tab, stacks_tab]

    # The per-workflow switches only matter while the feature is on; when the
    # master toggle is off they would be inert, so hide the whole tab.
    workflows = _discover_workflows() if _load_workflows_enabled() else []
    if workflows:
        active_workflows = _load_active_workflow_names()
        workflow_inputs: list[cl.input_widget.InputWidget] = [
            cl.input_widget.Switch(
                id=f"workflow_{wf['name']}",
                label=wf["name"] + ("  (draft)" if wf["kind"] == "draft" else ""),
                initial=wf["name"] in active_workflows,
                description=(
                    (wf["description"] or "Personal workflow")
                    + " — loaded as a skill when on. Changes take effect on the next message."
                ),
            )
            for wf in workflows
        ]
        tabs.append(cl.input_widget.Tab(id="workflows", label="Workflows", inputs=workflow_inputs))

    return tabs


def _workflow_commands() -> list[CommandDict]:
    """Return the composer command list (Save / Manage workflow buttons).

    Rendered as buttons in the message box; clicking one and sending sets
    ``message.command`` so ``on_message`` can act on it (distill the chat, or
    list workflows) instead of forwarding the text to the agent.

    Returns an empty list when the personal-workflows feature is toggled off, so
    the buttons disappear from the composer.
    """
    if not _load_workflows_enabled():
        return []

    save: CommandDict = {
        "id": SAVE_WORKFLOW_COMMAND,
        "icon": "save",
        "description": "Distill this chat into a reusable workflow",
        "button": True,
        "persistent": False,
        "selected": False,
    }
    manage: CommandDict = {
        "id": MANAGE_WORKFLOWS_COMMAND,
        "icon": "list",
        "description": "List your workflows to test, promote, or delete them",
        "button": True,
        "persistent": False,
        "selected": False,
    }
    return [save, manage]


# ── Explain tool calls (opt-in) ──────────────────────────
#
# When enabled by the user, each permission prompt is preceded by a short
# LLM-generated plain-language explanation of what the tool call does.
# The explanation is produced by the same local Ollama model that powers the
# agent, via a lightweight direct API call.


# Latest token count from vibe-acp usage_update frames.  Single-user app, so
# no per-session scoping is needed — the most recent update wins.
# The window size itself comes from medmcp.settings (fetch_context_window).
_latest_context_used: int = 0


async def _context_usage_api() -> dict[str, int]:  # pyright: ignore[reportUnusedFunction]
    """Return current context token usage for the context-meter in the UI."""
    return {"used": _latest_context_used, "size": await _fetch_context_window()}


async def _warmup_ollama() -> None:
    """Load the model into Ollama's memory before the first user message.

    Sends a minimal no-op chat request with keep_alive=2h so the model stays
    resident for the typical session duration without tying up GPU memory forever.
    Errors are silently ignored — this is a best-effort optimisation.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": "2h",
                    "options": {"num_predict": 1},
                },
            )
    except Exception:
        pass


# Chainlit's router registers a /{full_path:path} SPA catch-all before our
# module runs.  Starlette matches routes in registration order, so a route
# appended after the catch-all is never reached.  Insert at position 0 to
# ensure this specific path is matched first.
_chainlit_app.router.routes.insert(  # pyright: ignore[reportUnknownMemberType]
    0, APIRoute("/api/context-usage", _context_usage_api, methods=["GET"])
)


# Module-level singleton. Chainlit imports this file once at startup, but the
# subprocess itself is started lazily on first chat to avoid blocking import.
_client: VibeAcpClient = VibeAcpClient(cwd=PROJECT_ROOT, vibe_home=VIBE_HOME)

# Set to True when the active MCP stack set changes (via on_settings_update).
# on_chat_start checks this flag and restarts the vibe-acp process before
# creating the new session, so the fresh process reads the updated config.toml.
# Existing open chats are not affected — they continue on the old process until
# the process is replaced at the next on_chat_start.
_vibe_restart_needed: bool = False

# Guards the one-shot orphaned-provenance GC (see _gc_orphaned_provenance).
_provenance_gc_done: bool = False

# Pre-warm MCP server discovery so on_chat_start doesn't block on the first
# browser connection. lru_cache is effectively thread-safe in CPython; if the
# thread hasn't finished by the time on_chat_start fires it simply re-runs.
threading.Thread(target=_load_mcp_servers, daemon=True).start()


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


class _NullStorageClient(BaseStorageClient):
    """No-op storage client used when file uploads are disabled.

    Chainlit's SQLAlchemyDataLayer logs a warning when no storage provider is
    supplied, even though file uploads are explicitly disabled in config.toml.
    Passing this stub silences the warning without changing any behaviour —
    upload/delete/read_url calls should never occur with uploads disabled.
    """

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        return {}

    async def delete_file(self, object_key: str) -> bool:
        return True

    async def get_read_url(self, object_key: str) -> str:
        return ""

    async def close(self) -> None:
        pass


class _MedMcpDataLayer(SQLAlchemyDataLayer):
    """Data layer that also purges on-disk session logs when a thread is deleted.

    Chainlit's delete only removes the thread from the SQLite index; the vibe-acp
    transcript and our provenance record live on disk and would otherwise be
    orphaned. We resolve the thread's ``vibe_session_id`` first, then purge both
    after the row is gone.
    """

    async def _vibe_session_id_for(self, thread_id: str) -> str | None:
        """Read the vibe-acp session id from a thread's metadata, if present."""
        with contextlib.suppress(Exception):
            thread = await self.get_thread(thread_id)
            if thread is None:
                return None
            raw_meta: object = cast("dict[str, Any]", thread).get("metadata") or {}
            if isinstance(raw_meta, str):
                raw_meta = cast("object", json.loads(raw_meta))
            if isinstance(raw_meta, dict):
                sid = cast("dict[str, Any]", raw_meta).get("vibe_session_id")
                return sid if isinstance(sid, str) else None
        return None

    async def delete_thread(self, thread_id: str) -> None:
        """Delete the thread, then purge its vibe transcript and provenance logs."""
        session_id = await self._vibe_session_id_for(thread_id)
        await super().delete_thread(thread_id)  # pyright: ignore[reportUnknownMemberType]
        if session_id:
            with contextlib.suppress(Exception):
                provenance.purge_session(session_id)
            _audit.info("deleted thread %s and purged session logs %s", thread_id, session_id)


def _referenced_vibe_session_ids() -> set[str] | None:
    """Collect every vibe_session_id referenced by a persisted thread.

    Returns ``None`` — meaning "could not determine the full reference set" — if
    the threads DB is missing, the query fails, or any row's metadata can't be
    parsed. Callers MUST treat ``None`` as "do not delete anything": an empty or
    partial set would otherwise make the GC wipe logs for chats that still exist.
    A successfully read DB with zero matching rows correctly returns an empty set.
    """
    if not THREADS_DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(THREADS_DB_PATH)
        try:
            rows = con.execute("SELECT metadata FROM threads").fetchall()
        finally:
            con.close()
    except Exception:
        return None
    ids: set[str] = set()
    for (raw,) in rows:
        if not raw:
            # No metadata → this thread has no provenance to reference; safe.
            continue
        try:
            meta: object = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Can't tell what this thread references → bail rather than risk
            # deleting its logs.
            return None
        if not isinstance(meta, dict):
            return None
        sid = cast("dict[str, Any]", meta).get("vibe_session_id")
        if isinstance(sid, str):
            ids.add(sid)
    return ids


def _gc_orphaned_provenance() -> None:
    """Once per process, purge provenance records no thread references anymore.

    Belt-and-suspenders to the on-delete purge: cleans up records left behind by
    older builds, crashes, or any path that created provenance without persisting
    a thread mapping, so users never have to fall back to the CLI / just recipes.

    Fail-safe: if the referenced set can't be determined with certainty
    (:func:`_referenced_vibe_session_ids` returns ``None``) the GC is skipped, so
    a transient read error can never delete logs for chats that still exist.
    """
    global _provenance_gc_done
    if _provenance_gc_done:
        return
    _provenance_gc_done = True
    referenced = _referenced_vibe_session_ids()
    if referenced is None:
        _audit.info("gc: skipped — could not determine referenced sessions")
        return
    with contextlib.suppress(Exception):
        purged = provenance.purge_orphans(referenced)
        if purged:
            _audit.info("gc: purged %d orphaned provenance record(s): %s", len(purged), purged)


@cl.data_layer  # pyright: ignore[reportUnknownMemberType]
def get_data_layer() -> SQLAlchemyDataLayer:
    """Wire chainlit to a local sqlite database for thread persistence."""
    _bootstrap_threads_db(THREADS_DB_PATH)
    _gc_orphaned_provenance()
    return _MedMcpDataLayer(
        conninfo=f"sqlite+aiosqlite:///{THREADS_DB_PATH}",
        storage_provider=_NullStorageClient(),
    )


# ── Tool-call explanation (opt-in) ────────────────────────


# ── Permission UI ──────────────────────────────────────────


def _format_risk_badges(risks: list[str]) -> str:
    """Render a list of risk keys as a compact inline string with severity icons."""
    parts: list[str] = []
    for key in risks:
        entry = RISK_CATEGORIES.get(key)
        if entry is None:
            continue
        label, severity = entry
        icon = _SEVERITY_ICON.get(severity, "⚠️")
        parts.append(f"{icon} {label}")
    return "  ·  ".join(parts)


def _format_permission_prompt(tc: JsonDict) -> str:
    """Build the markdown body shown in the permission dialog.

    ``tc`` is a ToolCallUpdate from ACP — it has ``title`` (human label, e.g.
    ``"bash: ls -la ~/msseg"``), ``rawInput`` (the JSON-serialized tool args),
    ``humanReadable`` (a plain-language explanation aimed at a non-technical
    user), and optionally ``risks`` (a list of :data:`RISK_CATEGORIES` keys).
    """
    title = tc.get("title") or "tool call"
    raw_input = tc.get("rawInput")
    human_readable = tc.get("humanReadable")
    raw_risks = tc.get("risks")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    risks: list[str] = cast("list[str]", raw_risks) if isinstance(raw_risks, list) else []

    body = f"**Approve tool call?**\n\n`{title}`"
    if isinstance(human_readable, str) and human_readable:
        body += f"\n\n> {human_readable}"
    if risks:
        body += f"\n\n{_format_risk_badges(risks)}"
    if raw_input:
        input_str = _raw_input_to_str(raw_input)
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
    ask_msg = cl.AskActionMessage(
        content=_format_permission_prompt(tc),
        actions=actions,
        timeout=300,
    )
    response = await ask_msg.send()

    if response is None:
        _audit.warning("permission timed out: %s", title)
        return {"outcome": "cancelled"}

    # Remove the permission prompt from the chat so approved/denied tool calls
    # don't pile up as stale "Selected: ..." bubbles. The tool step that wraps
    # the call already provides a persistent record of what ran.
    await ask_msg.remove()

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
    info.setdefault("_started", time.monotonic())
    if (t := update.get("title")) is not None:
        info["title"] = t
    if (ri := update.get("rawInput")) is not None:
        info["rawInput"] = ri
    if (hr := update.get("humanReadable")) is not None:
        info["humanReadable"] = hr

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
        step = cl.Step(name=tool_title, type="run", parent_id=parent_id)
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


# ── Explain toggle (ChatSettings) ─────────────────────────


def _is_explain_enabled() -> bool:
    """Check whether the user has opted in to tool-call explanations.

    Falls back to the persisted preference (shared with the workspace UI)
    when the per-chat value is unset.
    """
    val = cl.user_session.get("explain_tools")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if val is None:
        return _load_explain_enabled()
    return bool(val)  # pyright: ignore[reportUnknownArgumentType]


def _is_provenance_enabled() -> bool:
    """Check whether provenance capture is enabled for the current chat.

    Falls back to the persisted preference when the per-chat value is unset (e.g.
    a code path that runs before ``on_chat_start`` has populated the session).
    """
    val = cl.user_session.get("record_provenance")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if val is None:
        return _load_provenance_enabled()
    return bool(val)  # pyright: ignore[reportUnknownArgumentType]


@cl.on_settings_update  # pyright: ignore[reportUnknownMemberType]
async def on_settings_update(settings: dict[str, Any]) -> None:
    """Persist chat-settings changes into the user session.

    Handles four setting types:
    - ``explain_tools`` — persisted to ``.vibe/explain_enabled.json`` (shared
      with the workspace UI) and mirrored into the user session.
    - ``record_provenance`` — persisted to ``.vibe/provenance_enabled.json`` and
      mirrored into the user session; takes effect immediately.
    - ``stack_<name>`` — updates the persistent active-stack set in
      ``.vibe/active_stacks.json``; the new set takes effect on the next
      conversation (vibe-acp reads config.toml at session-creation time).
    - ``workflows_enabled`` — master on/off for personal workflows, persisted to
      ``.vibe/workflows_enabled.json``. Hides/shows the composer buttons
      immediately and toggles workflow skill loading on the next session.
    - ``workflow_<name>`` — updates the active-workflow set in
      ``.vibe/active_workflows.json``; deactivated workflows are written to
      ``disabled_skills`` on the next session so vibe-acp skips loading them.
    """
    global _vibe_restart_needed

    new_value = bool(settings.get("explain_tools", False))
    cl.user_session.set("explain_tools", new_value)  # pyright: ignore[reportUnknownMemberType]
    if new_value != _load_explain_enabled():
        _save_explain_enabled(new_value)
        _audit.info("explain_tools set to %s via settings", new_value)

    prov_value = bool(settings.get("record_provenance", True))
    cl.user_session.set("record_provenance", prov_value)  # pyright: ignore[reportUnknownMemberType]
    if prov_value != _load_provenance_enabled():
        _save_provenance_enabled(prov_value)
        _audit.info("provenance recording set to %s via settings", prov_value)

    all_servers = _load_mcp_servers()
    if all_servers:
        active_names: set[str] = {
            srv["name"] for srv in all_servers if bool(settings.get(f"stack_{srv['name']}", True))
        }
        if active_names != _load_active_server_names():
            _save_active_server_names(active_names)
            _vibe_restart_needed = True
            _audit.info(
                "active stacks updated to: %s; vibe-acp will restart on next session",
                sorted(active_names),
            )

    wf_enabled = bool(settings.get("workflows_enabled", True))
    if wf_enabled != _load_workflows_enabled():
        _save_workflows_enabled(wf_enabled)
        _vibe_restart_needed = True
        # Show/hide the Save/Manage composer buttons immediately for this chat.
        with contextlib.suppress(Exception):
            await cl.context.emitter.set_commands(_workflow_commands())  # pyright: ignore[reportUnknownMemberType]
        _audit.info("personal workflows %s via settings", "enabled" if wf_enabled else "disabled")

    # Per-workflow switches only exist while the feature is on; when off the
    # Workflows tab is hidden, so skip this to preserve the saved active set
    # instead of resetting it from absent settings keys.
    workflows = _discover_workflows() if wf_enabled else []
    if workflows:
        active_workflows: set[str] = {
            wf["name"] for wf in workflows if bool(settings.get(f"workflow_{wf['name']}", True))
        }
        if active_workflows != _load_active_workflow_names():
            _save_active_workflow_names(active_workflows)
            _vibe_restart_needed = True
            _audit.info(
                "active workflows updated to: %s; vibe-acp will reload on next message",
                sorted(active_workflows),
            )


# ── Chainlit hooks ─────────────────────────────────────────


async def _create_new_session(reload_session_id: str | None = None) -> bool:
    """Create (or reload) a vibe-acp session and store its ID in the user session.

    When *reload_session_id* is given the existing JSONL transcript is replayed
    via ``session/load`` so conversation history is preserved after an MCP-stack
    toggle.  Otherwise a fresh ``session/new`` is issued.

    Handles the vibe-acp restart if :data:`_vibe_restart_needed` is set.
    Returns ``True`` on success or ``False`` after emitting an error message.
    """
    global _vibe_restart_needed, _latest_context_used
    if _vibe_restart_needed:
        await _client.stop()
        _vibe_restart_needed = False
    _latest_context_used = 0

    active = _active_servers()
    _sync_servers_to_vibe_config(active)
    await _client.ensure_started()

    if reload_session_id is not None:
        # Reload the existing session so the model retains conversation history
        # while the new MCP servers (from the updated config.toml) become active.
        queue = _client.register_session(reload_session_id)
        _set_session_id(reload_session_id)
        resp = await _client.request(
            "session/load",
            {"cwd": PROJECT_ROOT, "session_id": reload_session_id, "mcpServers": []},
        )
        if "error" in resp:
            await cl.Message(content=f"Failed to reload vibe-acp session: {resp['error']}").send()
            return False
        # Drain replay frames before the next on_message (same invariant as on_chat_resume).
        while not queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        # Refresh the provenance manifest so the stack/model set is recorded for
        # the post-toggle continuation of this session.
        if _is_provenance_enabled():
            with contextlib.suppress(Exception):
                provenance.write_manifest(
                    reload_session_id, servers=active, model_name=OLLAMA_MODEL
                )
        return True

    resp = await _client.request(
        "session/new",
        {"cwd": PROJECT_ROOT, "mcpServers": []},
    )
    if "error" in resp:
        await cl.Message(content=f"Failed to create vibe-acp session: {resp['error']}").send()
        return False

    result = cast("JsonDict", resp.get("result") or {})
    session_id = cast("str", result.get("sessionId", ""))
    if not session_id:
        await cl.Message(content="vibe-acp session/new returned no sessionId").send()
        return False
    _set_session_id(session_id)
    _client.register_session(session_id)
    # The Tier-1 provenance manifest is intentionally NOT written here. session/new
    # runs eagerly on every page load (to warm MCP servers), but the thread→session
    # mapping that lets the UI purge provenance on delete is only saved on the first
    # message. Writing the manifest here would leak un-purgeable records for chats
    # that are opened but never used; it is written in on_message instead.
    return True


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]
async def on_chat_start() -> None:
    """Set up the UI for a new chat and eagerly create a vibe-acp session.

    The session is created here (not deferred to the first message) so that MCP
    server subprocesses are already running when the user sends their first prompt.
    If the user changes stack settings before typing, ``on_settings_update`` sets
    ``_vibe_restart_needed`` and ``on_message`` recreates the session then.
    """
    cl.user_session.set("explain_tools", _load_explain_enabled())  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set("record_provenance", _load_provenance_enabled())  # pyright: ignore[reportUnknownMemberType]
    await cl.context.emitter.set_commands(_workflow_commands())  # pyright: ignore[reportUnknownMemberType]
    # Start vibe-acp and build the settings widget concurrently — both involve
    # subprocess calls on first run, so overlapping them saves startup time.
    warmup_task = asyncio.create_task(_client.ensure_started())
    ollama_task = asyncio.create_task(_warmup_ollama())
    await cl.ChatSettings(inputs=_build_chat_settings_inputs()).send()
    with contextlib.suppress(Exception):
        await warmup_task
    # Eagerly create the session so MCP servers are warm before the first message.
    # Non-fatal: on_message will retry if this fails.
    with contextlib.suppress(Exception):
        await _create_new_session()
    with contextlib.suppress(Exception):
        await ollama_task


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
    cl.user_session.set("explain_tools", _load_explain_enabled())  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set("record_provenance", _load_provenance_enabled())  # pyright: ignore[reportUnknownMemberType]
    await cl.context.emitter.set_commands(_workflow_commands())  # pyright: ignore[reportUnknownMemberType]
    await cl.ChatSettings(inputs=_build_chat_settings_inputs()).send()

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
        # ChatSettings are already sent above; session will be created lazily on
        # the first message, identical to a fresh on_chat_start.
        _audit.warning(
            "resume: thread %s has no vibe_session_id; session will be created on first message",
            thread_any.get("id"),
        )
        return

    global _vibe_restart_needed
    if _vibe_restart_needed:
        await _client.stop()
        _vibe_restart_needed = False

    active = _active_servers()
    _sync_servers_to_vibe_config(active)
    await _client.ensure_started()

    # Pre-register the queue *before* sending session/load so any replay events
    # that arrive while we're waiting for the response land in the queue
    # rather than in limbo.
    queue = _client.register_session(vibe_session_id)
    _set_session_id(vibe_session_id)
    # Resumed threads already have the mapping in their metadata (that's how
    # we found vibe_session_id), so flag this chat as already persisted to
    # skip the lazy update_thread call in on_message.
    cl.user_session.set("vibe_session_persisted", True)  # pyright: ignore[reportUnknownMemberType]

    resp = await _client.request(
        "session/load",
        {
            "cwd": PROJECT_ROOT,
            "session_id": vibe_session_id,
            "mcpServers": [],
        },
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
    # The "Save workflow" composer command distills the current chat instead of
    # being forwarded to the agent.
    if message.command == SAVE_WORKFLOW_COMMAND:
        await _handle_save_workflow_command()
        return
    if message.command == MANAGE_WORKFLOWS_COMMAND:
        await _handle_manage_workflows_command()
        return

    # A pending Rename/Refine (set by clicking those buttons) consumes this
    # message as its input value instead of forwarding it to the agent.
    if await _consume_pending_workflow_input(message.content):
        return

    session_id = _get_session_id()
    stack_reload = _vibe_restart_needed and session_id is not None
    if session_id is None or _vibe_restart_needed:
        # First message → session/new; stack toggle mid-chat → session/load with
        # the existing session ID so conversation history is preserved.
        reload_id = session_id if stack_reload else None
        if not await _create_new_session(reload_session_id=reload_id):
            return
        session_id = _get_session_id()
        if session_id is None:
            await cl.Message(content="Internal error: session was not initialized.").send()
            return

    # Persist the chainlit-thread → vibe-session mapping on the first message of
    # this chat. Done lazily so refreshes that never produce a conversation don't
    # leave empty threads in the sidebar. This mapping is what lets the UI find
    # and purge the session's logs on delete, so the "persisted" flag is only set
    # once the mapping is actually written — otherwise a chat whose thread_id was
    # not ready yet (or whose write failed) would lose its logs forever. We retry
    # on the next message until it sticks.
    if not cl.user_session.get("vibe_session_persisted"):  # pyright: ignore[reportUnknownMemberType]
        thread_id = cl_context.session.thread_id
        data_layer = cast("Any", cl_get_data_layer())
        mapping_saved = False
        if thread_id and data_layer is not None:
            try:
                await data_layer.update_thread(
                    thread_id=thread_id,
                    user_id=_persisted_user_id(),
                    metadata={"vibe_session_id": session_id},
                )
                mapping_saved = True
            except Exception:
                _audit.warning(
                    "could not persist thread→session mapping for %s; will retry", session_id
                )
        if mapping_saved:
            # Write the Tier-1 provenance manifest only now that the mapping
            # exists, so provenance is never created for a session the UI can't
            # later find and purge on delete.
            if _is_provenance_enabled():
                with contextlib.suppress(Exception):
                    provenance.write_manifest(
                        session_id, servers=_active_servers(), model_name=OLLAMA_MODEL
                    )
            cl.user_session.set("vibe_session_persisted", True)  # pyright: ignore[reportUnknownMemberType]

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
    # Mutable holder for the single visible tool-summary Step (type="tool").
    # Created lazily on the first tool call; keyed by "step" when present.
    tool_summary_holder: dict[str, cl.Step] = {}

    # Send the prompt and get a future for its response. We then race the
    # future against queue reads so session_update notifications and
    # request_permission requests are interleaved with the agent loop.
    prompt_text = message.content
    if stack_reload:
        active_names = ", ".join(s["name"] for s in _active_servers()) or "none"
        prompt_text = (
            f"[System note: MCP stack settings were just changed. "
            f"Active MCP stacks: {active_names}. "
            f"MCP tools from disabled stacks are no longer available; "
            f"all built-in tools remain unchanged.]\n\n"
            f"{prompt_text}"
        )

    prompt_fut = asyncio.create_task(
        _client.request(
            "session/prompt",
            {
                "session_id": session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
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
            get_task: asyncio.Task[JsonDict] = asyncio.create_task(queue.get())
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
                            tool_summary_holder=tool_summary_holder,
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
                tool_summary_holder=tool_summary_holder,
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
    except Exception:
        # Any unexpected exception from _process_session_frame (e.g. a Chainlit
        # error inside _ask_user_for_permission) must still cancel the in-flight
        # prompt and drain vibe-acp, otherwise the next session/prompt is
        # rejected as concurrent and the UI becomes unresponsive.
        if not prompt_fut.done():
            prompt_fut.cancel()
        await _cancel_and_drain()
        raise

    if assistant_msg is not None:
        await assistant_msg.update()

    # Final refresh of the tool summary step — set end to stop the spinner.
    await _update_tool_summary(tool_summary_holder, tool_call_info, final=True)


def _build_tool_summary(tool_call_info: dict[str, JsonDict]) -> str:
    """Build a markdown summary of all tool calls for the summary step output."""
    lines: list[str] = []
    for info in tool_call_info.values():
        title = str(info.get("title") or "tool")
        status = info.get("status")
        status_icon = (
            "done" if status == "completed" else ("error" if status == "failed" else "...")
        )
        line = f"- **{title}** — *{status_icon}*"

        human_readable = info.get("humanReadable")
        if isinstance(human_readable, str) and human_readable:
            line += f"\n  > {human_readable}"
        raw_risks: object = info.get("risks")
        if isinstance(raw_risks, list) and raw_risks:
            badges = _format_risk_badges(cast("list[str]", raw_risks))
            if badges:
                line += f"\n  {badges}"

        ro = info.get("rawOutput")
        ot = info.get("outputText")
        output_str: str | None = None
        if ro is not None:
            output_str = _stringify_raw(ro)
        elif isinstance(ot, str):
            output_str = ot
        if output_str is not None:
            preview = output_str[:200].replace("\n", " ")
            if len(output_str) > 200:
                preview += "…"
            line += f"\n  ```\n  {preview}\n  ```"
        lines.append(line)
    return "\n".join(lines)


async def _update_tool_summary(
    tool_summary_holder: dict[str, cl.Step],
    tool_call_info: dict[str, JsonDict],
    *,
    final: bool = False,
) -> None:
    """Refresh the single visible tool-summary Step.

    While tools are running the step keeps ``start`` set and ``end`` unset so
    that Chainlit's frontend treats it as a running ``type="tool"`` step.  This
    is important because the frontend suppresses its own loader when it detects
    a running tool step; without this the user sees two spinners.

    Pass ``final=True`` once the agent turn is complete to set ``end`` and stop
    the spinner.
    """
    step = tool_summary_holder.get("step")
    if step is None:
        return
    step.output = _build_tool_summary(tool_call_info)
    if final:
        step.end = _utc_now()
    await step.update()


def _record_tool_event(session_id: str, tc_id: str, info: JsonDict) -> None:
    """Append a normalized run-log event for a completed tool call (best-effort)."""
    provenance.record_tool_event(
        session_id, tc_id, info, [str(s["name"]) for s in _active_servers()]
    )


async def _process_session_frame(
    msg: JsonDict,
    *,
    assistant_msg_getter: Callable[[], Awaitable[cl.Message]],
    tool_steps: dict[str, cl.Step],
    tool_call_info: dict[str, JsonDict],
    tool_summary_holder: dict[str, cl.Step],
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
            tc_id = cast("str", update.get("toolCallId") or "")
            is_new_tool = tc_id not in tool_steps
            await _handle_tool_call(update, tool_steps, tool_call_info, parent_id)
            # Maintain a single visible summary Step (type="tool") that the
            # user can expand. Individual tool steps use type="run" and are
            # hidden by cot="tool_call".
            if is_new_tool:
                if "step" not in tool_summary_holder:
                    summary = cl.Step(name="Tool Calls (1)", type="tool", parent_id=parent_id)
                    summary.start = _utc_now()
                    summary.output = _build_tool_summary(tool_call_info)
                    await summary.send()
                    tool_summary_holder["step"] = summary
                else:
                    tool_summary_holder["step"].name = f"Tool Calls ({len(tool_call_info)})"
                    await _update_tool_summary(tool_summary_holder, tool_call_info)

        elif update_type == "usage_update":
            global _latest_context_used
            used = update.get("used")
            if isinstance(used, int):
                _latest_context_used = used
                size = _cached_context_window()
                await cl.context.emitter.emit(  # pyright: ignore[reportUnknownMemberType]
                    "ctx_update", {"used": used, "size": size}
                )

        elif update_type == "tool_call_update":
            # Track tool output for the summary step.
            tc_id = cast("str", update.get("toolCallId") or "")
            if tc_id in tool_call_info:
                raw_output = update.get("rawOutput")
                if raw_output is not None:
                    tool_call_info[tc_id]["rawOutput"] = raw_output
                else:
                    text_parts = _extract_text_blocks(update.get("content"))
                    if text_parts:
                        tool_call_info[tc_id]["outputText"] = "\n".join(text_parts)
                status = update.get("status")
                if isinstance(status, str):
                    tool_call_info[tc_id]["status"] = status
            await _handle_tool_call_update(update, tool_steps)
            # Refresh the summary step after each tool completion.
            if update.get("status") in ("completed", "failed"):
                if "step" in tool_summary_holder:
                    await _update_tool_summary(tool_summary_holder, tool_call_info)
                # Record the normalized run-log event (Tier-1 provenance).
                sid = _get_session_id()
                if sid and tc_id in tool_call_info and _is_provenance_enabled():
                    _record_tool_event(sid, tc_id, tool_call_info[tc_id])

    elif method == "session/request_permission":
        req_id_raw = msg.get("id")
        if not isinstance(req_id_raw, int):
            return  # cannot respond without a request id
        req_id: int = req_id_raw
        params = cast("JsonDict", msg.get("params") or {})
        tc: JsonDict = dict(cast("JsonDict", params.get("toolCall") or {}))
        options = cast("list[JsonDict]", params.get("options") or [])
        # Backfill title/rawInput/humanReadable/risks from the cached tool_call
        # event, because request_permission only ships the toolCallId.
        cached = tool_call_info.get(cast("str", tc.get("toolCallId") or ""), {})
        for key in ("title", "rawInput", "humanReadable", "risks"):
            if tc.get(key) is None and cached.get(key) is not None:
                tc[key] = cached[key]

        # Generate a plain-language explanation + risk assessment if the user
        # opted in and one wasn't already provided by vibe-acp.
        if _is_explain_enabled() and tc.get("humanReadable") is None:
            with contextlib.suppress(Exception):
                result = await _generate_explanation(tc)
                if result is not None:
                    tc["humanReadable"], tc["risks"] = result

        # Persist explanation and risks back into tool_call_info so the
        # summary step can display them when the user unfolds it.
        tc_id_perm = cast("str", tc.get("toolCallId") or "")
        if tc_id_perm in tool_call_info:
            updated = False
            if isinstance(tc.get("humanReadable"), str):
                tool_call_info[tc_id_perm]["humanReadable"] = tc["humanReadable"]
                updated = True
            if isinstance(tc.get("risks"), list):
                tool_call_info[tc_id_perm]["risks"] = tc["risks"]
                updated = True
            if updated and "step" in tool_summary_holder:
                await _update_tool_summary(tool_summary_holder, tool_call_info)

        outcome = await _ask_user_for_permission(tc, options)
        # Persist the decision: stash it on tool_call_info so the run-log event
        # carries it, and mirror it to the on-disk permissions log.
        decision = (
            outcome.get("optionId")
            if outcome.get("outcome") == "selected"
            else outcome.get("outcome")
        )
        if tc_id_perm in tool_call_info:
            tool_call_info[tc_id_perm]["decision"] = decision
        sid = _get_session_id()
        if sid and _is_provenance_enabled():
            with contextlib.suppress(Exception):
                provenance.log_permission(
                    sid, title=str(tc.get("title") or tc_id_perm), decision=str(decision)
                )
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


async def _send_workflow_preview(draft_dir: Path) -> None:
    """Render a draft workflow's SKILL.md inline with the review action buttons."""
    skill_md = (draft_dir / "SKILL.md").read_text(encoding="utf-8")
    await cl.Message(
        content=(
            f"**Draft workflow `{draft_dir.name}`** — review below, then **Test** it in this "
            f"chat, **Refine**/**Rename** it, **Promote** to keep it, or **Discard** it.\n\n"
            f"```markdown\n{skill_md}\n```"
        ),
        actions=[
            cl.Action(
                name=TEST_WORKFLOW_ACTION,
                payload={"name": draft_dir.name},
                label="Test",
                tooltip="Load the draft so you can try it in this chat before promoting",
            ),
            cl.Action(
                name=PROMOTE_WORKFLOW_ACTION,
                payload={"name": draft_dir.name},
                label="Promote",
                tooltip="Keep this workflow permanently as a reusable skill",
            ),
            cl.Action(
                name=REFINE_WORKFLOW_ACTION,
                payload={"name": draft_dir.name},
                label="Refine",
                tooltip="Describe a change and regenerate the workflow",
            ),
            cl.Action(
                name=RENAME_WORKFLOW_ACTION,
                payload={"name": draft_dir.name},
                label="Rename",
                tooltip="Give the workflow a different name",
            ),
            cl.Action(
                name=DISCARD_WORKFLOW_ACTION,
                payload={"name": draft_dir.name},
                label="Discard",
                tooltip="Delete this draft",
            ),
        ],
    ).send()


def _action_name(action: cl.Action) -> str | None:
    """Extract the ``name`` from an action payload, or ``None`` if absent."""
    payload = cast("dict[str, Any]", action.payload or {})  # pyright: ignore[reportUnknownMemberType]
    name = payload.get("name")
    return name if isinstance(name, str) else None


async def _handle_save_workflow_command() -> None:
    """Distill the current chat into a draft workflow and preview it in chat.

    Triggered by the "Save workflow" composer command. Runs the (hybrid,
    LLM-assisted) distillation off the event loop, then shows the draft with
    review controls. Distillation reads vibe-acp's own transcript, so it works
    regardless of whether provenance recording is enabled.
    """
    session_id = _get_session_id()
    if session_id is None:
        await cl.Message(content="There's nothing to save yet — send a message first.").send()
        return

    progress = cl.Message(content="Distilling this chat into a workflow…")
    await progress.send()
    try:
        draft_dir = await asyncio.to_thread(distill.distill_session, session_id, use_llm=True)
    except Exception as exc:
        await progress.remove()
        await cl.Message(content=f"Could not distill a workflow: {exc}").send()
        return
    await progress.remove()
    await _send_workflow_preview(draft_dir)


def _manage_actions(name: str) -> list[cl.Action]:
    """Build the per-workflow buttons for the Manage list.

    Every workflow offers Run, Edit and Delete. Run replays the workflow
    deterministically on new inputs (no LLM); Edit opens the full editable preview
    (Test / Promote / Rename / Refine / Discard); for a promoted workflow Edit first
    unpromotes it back to a draft. Activate/deactivate stays in the settings gear.
    """
    return [
        cl.Action(
            name=RUN_WORKFLOW_ACTION,
            payload={"name": name},
            label="Run",
            tooltip="Replay this workflow on new inputs — runs the exact tools, no LLM",
        ),
        cl.Action(
            name=EDIT_WORKFLOW_ACTION,
            payload={"name": name},
            label="Edit",
            tooltip="Open editing controls (test, promote, rename, refine, discard)",
        ),
        cl.Action(
            name=DELETE_WORKFLOW_ACTION,
            payload={"name": name},
            label="Delete",
            tooltip="Delete this workflow from disk",
        ),
    ]


async def _handle_manage_workflows_command() -> None:
    """List all personal workflows with per-item Test / Promote / Delete buttons.

    Triggered by the "Manage workflows" composer command. This is the persistent
    place to act on a workflow after its Save-time preview has scrolled away —
    e.g. to promote a draft once you've tested it, or to delete any workflow.
    """
    workflows = await asyncio.to_thread(_discover_workflows)
    if not workflows:
        await cl.Message(
            content="You have no workflows yet. Use **Save workflow** to distill this chat."
        ).send()
        return
    active_names = await asyncio.to_thread(_load_active_workflow_names)
    await cl.Message(content=f"**Your workflows ({len(workflows)})**").send()
    for wf in workflows:
        name = str(wf["name"])
        kind = str(wf["kind"])
        is_active = name in active_names
        status = "draft" if kind == "draft" else ("active" if is_active else "inactive")
        description = str(wf["description"]) or "_no description_"
        await cl.Message(
            content=f"**`{name}`** · {status}\n\n{description}",
            actions=_manage_actions(name),
        ).send()


# ── Deterministic replay (Run) ────────────────────────────────────────────────


def _workflow_dir(name: str) -> Path | None:
    """Return the on-disk dir for workflow *name* (active wins over draft), or None."""
    for kind in ("active", "draft"):
        d = VIBE_HOME / "workflows" / kind / name
        if (d / "recipe.yaml").exists():
            return d
    return None


def _input_prompt(name: str, example: str, description: str) -> str:
    """Build the message asking the user for one replay input value."""
    desc = f" — {description}" if description else ""
    return f"**`{name}`**{desc}\n(e.g. `{example}`)"


def _format_replay_preview(recipe: Recipe, inputs: dict[str, str]) -> str:
    """Render the resolved steps a replay will run, for the confirm prompt."""
    lines: list[str] = [f"**Replay `{recipe.name}`** will run these steps, no LLM:\n"]
    if recipe.inputs:
        lines.append("**Inputs**")
        for wf_input in recipe.inputs:
            desc = f" — {wf_input.description}" if wf_input.description else ""
            lines.append(f"- `{wf_input.name}`{desc} = `{inputs.get(wf_input.name, '?')}`")
        lines.append("")
    lines.append(f"**Steps ({len(recipe.steps)})**")
    bindings: dict[str, Any] = dict(inputs)
    for i, step in enumerate(recipe.steps, start=1):
        # Inputs are known now; cross-step refs ({{stepM.*}}) resolve at runtime,
        # so they intentionally still show as placeholders here.
        args = replay.resolve_arguments(step.arguments, bindings)
        rendered = json.dumps(args, default=str)
        lines.append(f"{i}. `{step.server}:{step.tool}` — `{rendered}`")
    return "\n".join(lines)


async def _send_replay_preview(recipe: Recipe, inputs: dict[str, str]) -> None:
    """Show the resolved steps and a Run-now button carrying the bound inputs."""
    await cl.Message(
        content=_format_replay_preview(recipe, inputs),
        actions=[
            cl.Action(
                name=CONFIRM_REPLAY_ACTION,
                payload={"name": recipe.name, "inputs": inputs},
                label="Run now",
                tooltip="Execute the steps above on the inputs shown",
            )
        ],
    ).send()


async def _run_replay_and_report(name: str, inputs: dict[str, str]) -> None:
    """Execute a workflow replay, streaming per-step status, then report the outcome."""
    draft_dir = _workflow_dir(name)
    if draft_dir is None:
        await cl.Message(content=f"Could not find a recipe for `{name}`.").send()
        return
    recipe = await asyncio.to_thread(distill.load_recipe, draft_dir)
    servers = _active_servers()

    progress = cl.Message(content=f"▶️ Replaying **`{name}`**…")
    await progress.send()
    log_lines: list[str] = []

    async def _on_step(step_result: replay.StepResult) -> None:
        icon = "✅" if step_result.ok else "❌"
        line = f"{icon} {step_result.index}. `{step_result.server}:{step_result.tool}`"
        if step_result.produced:
            produced = ", ".join(f"`{v}`" for v in step_result.produced.values())
            line += f" → {produced}"
        if not step_result.ok and step_result.error:
            line += f"\n   ↳ {step_result.error}"
        log_lines.append(line)
        progress.content = f"▶️ Replaying **`{name}`**…\n\n" + "\n".join(log_lines)
        await progress.update()

    result = await replay.run(recipe, inputs, servers=servers, cwd=PROJECT_ROOT, on_step=_on_step)

    if result.ok:
        outputs: list[str] = []
        for step_result in result.steps:
            outputs.extend(step_result.produced.values())
        summary = f"✅ **Replay of `{name}` complete** — {len(result.steps)} step(s) ran."
        if outputs:
            summary += "\n\n**Outputs**\n" + "\n".join(f"- `{o}`" for o in outputs)
    else:
        summary = f"❌ **Replay of `{name}` failed.** {result.error or ''}".rstrip()
    progress.content = f"Replayed **`{name}`**\n\n" + "\n".join(log_lines)
    await progress.update()
    await cl.Message(content=summary).send()


@cl.action_callback(RUN_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_run_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Start a deterministic replay: collect new inputs, then preview and confirm."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to run.").send()
        return
    draft_dir = _workflow_dir(name)
    if draft_dir is None:
        await cl.Message(content=f"Could not find a recipe for `{name}`.").send()
        return
    recipe = await asyncio.to_thread(distill.load_recipe, draft_dir)

    # Surface non-input problems (built-in steps, uninstalled stacks) up front by
    # validating with the recorded examples standing in for the inputs.
    examples = {i.name: i.example for i in recipe.inputs}
    structural = replay.validate(recipe, examples, _active_servers())
    if structural is not None:
        await cl.Message(content=f"Can't replay `{name}`: {structural}").send()
        return

    if not recipe.inputs:
        await _send_replay_preview(recipe, {})
        return

    # Collect each input as a normal message (same pattern as Rename/Refine).
    descriptions = {i.name: i.description for i in recipe.inputs}
    cl.user_session.set(  # pyright: ignore[reportUnknownMemberType]
        "pending_workflow",
        {
            "action": "run",
            "name": name,
            "input_names": [i.name for i in recipe.inputs],
            "examples": examples,
            "descriptions": descriptions,
            "collected": {},
        },
    )
    first = recipe.inputs[0]
    await cl.Message(
        content=(
            f"Replaying **`{name}`** on new data. Provide a value for each input "
            f"(send `-` at any point to cancel).\n\n"
            + _input_prompt(first.name, first.example, first.description)
        )
    ).send()


@cl.action_callback(CONFIRM_REPLAY_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_confirm_replay(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Execute a previewed replay when the user clicks Run now."""
    payload = cast("dict[str, Any]", action.payload or {})  # pyright: ignore[reportUnknownMemberType]
    name = payload.get("name")
    raw_inputs: object = payload.get("inputs") or {}
    if not isinstance(name, str) or not isinstance(raw_inputs, dict):
        await cl.Message(content="Could not start the replay (missing parameters).").send()
        return
    inputs = {str(k): str(v) for k, v in cast("JsonDict", raw_inputs).items()}
    await _run_replay_and_report(name, inputs)


@cl.action_callback(TEST_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_test_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Make a draft loadable so the user can try it in this chat before promoting."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to test.").send()
        return
    # Draft dirs are already in skill_paths; flag a vibe-acp restart so the next
    # message respawns the subprocess, which re-scans and loads the draft skill.
    # The reload preserves the conversation (session/load).
    global _vibe_restart_needed
    _vibe_restart_needed = True
    await cl.Message(
        content=(
            f"Draft **`{name}`** is ready to test. On your next message, run it with "
            f"`/{name}` (optionally add your own inputs after it), e.g. "
            f"`/{name} on data/your_scan.nii.gz`. When it works, **Promote** to keep it "
            f"(or **Delete** to remove it)."
        ),
        actions=[
            cl.Action(
                name=PROMOTE_WORKFLOW_ACTION,
                payload={"name": name},
                label="Promote",
                tooltip="Keep this workflow permanently as a reusable skill",
            ),
            cl.Action(
                name=DELETE_WORKFLOW_ACTION,
                payload={"name": name},
                label="Delete",
                tooltip="Delete this workflow from disk",
            ),
        ],
    ).send()


@cl.action_callback(PROMOTE_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_promote_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Promote a reviewed draft workflow into the active (reusable) set."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to promote.").send()
        return
    try:
        dst = await asyncio.to_thread(distill.promote_draft, name)
    except Exception as exc:
        await cl.Message(content=f"Could not promote the workflow: {exc}").send()
        return
    # Make the new skill available without restarting the UI: flag a vibe-acp
    # restart so the next message respawns the subprocess, which re-reads
    # skill_paths (the active workflows dir) and re-scans for the new SKILL.md.
    # The restart is transparent and preserves the conversation (session/load).
    global _vibe_restart_needed
    _vibe_restart_needed = True
    await cl.Message(
        content=(
            f"Promoted **`{name}`**. It now lives in your active workflows "
            f"(`{dst}`) and will be loaded as a skill on your next message — "
            f"no UI restart needed."
        )
    ).send()


@cl.action_callback(DISCARD_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_discard_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Delete a draft workflow."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to discard.").send()
        return
    try:
        await asyncio.to_thread(distill.discard_draft, name)
    except Exception as exc:
        await cl.Message(content=f"Could not discard the workflow: {exc}").send()
        return
    await cl.Message(content=f"Discarded draft workflow **`{name}`**.").send()


@cl.action_callback(DELETE_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_delete_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Delete a personal workflow (draft or promoted) from disk."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to delete.").send()
        return
    try:
        await asyncio.to_thread(distill.delete_workflow, name)
    except Exception as exc:
        await cl.Message(content=f"Could not delete the workflow: {exc}").send()
        return
    # Drop it from the loaded skill set on the next message.
    global _vibe_restart_needed
    _vibe_restart_needed = True
    await cl.Message(content=f"Deleted workflow **`{name}`** from disk.").send()


@cl.action_callback(EDIT_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_edit_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Open a workflow's editing controls.

    For a draft this just re-opens its preview. For a promoted workflow it first
    unpromotes it back to a draft (and reloads so it leaves the active skill set).
    """
    global _vibe_restart_needed
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to edit.").send()
        return

    draft_dir = VIBE_HOME / "workflows" / "draft" / name
    active_dir = VIBE_HOME / "workflows" / "active" / name

    if active_dir.is_dir() and not draft_dir.is_dir():
        # Promoted → unpromote back to a draft before editing.
        try:
            draft_dir = await asyncio.to_thread(distill.unpromote_workflow, name)
        except Exception as exc:
            await cl.Message(content=f"Could not edit the workflow: {exc}").send()
            return
        _vibe_restart_needed = True
        await cl.Message(
            content=(
                f"Moved **`{name}`** back to drafts for editing. Make your changes below, "
                f"then **Promote** again to keep it."
            )
        ).send()

    if not (draft_dir / "SKILL.md").exists():
        await cl.Message(content=f"Could not find a workflow named `{name}` to edit.").send()
        return
    await _send_workflow_preview(draft_dir)


@cl.action_callback(RENAME_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_rename_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Arm a rename: the user's next message becomes the new name."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to rename.").send()
        return
    cl.user_session.set("pending_workflow", {"action": "rename", "name": name})  # pyright: ignore[reportUnknownMemberType]
    await cl.Message(
        content=f"Type the new name for **`{name}`** and send it (or send `-` to cancel)."
    ).send()


@cl.action_callback(REFINE_WORKFLOW_ACTION)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def _on_refine_workflow(action: cl.Action) -> None:  # pyright: ignore[reportUnusedFunction]
    """Arm a refine: the user's next message becomes the adjustment instruction."""
    name = _action_name(action)
    if name is None:
        await cl.Message(content="Could not determine which workflow to refine.").send()
        return
    cl.user_session.set("pending_workflow", {"action": "refine", "name": name})  # pyright: ignore[reportUnknownMemberType]
    await cl.Message(
        content=(
            f"How should I adjust **`{name}`**? Send an instruction "
            f"(e.g. 'make it generic for any MRI', 'drop the first step'), or `-` to cancel."
        )
    ).send()


async def _consume_pending_workflow_input(text: str) -> bool:
    """Apply a pending Rename/Refine to *text*; return True if one was consumed.

    Clicking Rename/Refine stores a pending action in the user session and asks
    the user to type the value as a normal message. Consuming it through the
    standard ``on_message`` path (rather than a blocking ``AskUserMessage`` inside
    an action callback) keeps the flow on Chainlit's well-tested message channel.
    """
    pending = cast(
        "JsonDict | None",
        cl.user_session.get("pending_workflow"),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    if not pending:
        return False
    cl.user_session.set("pending_workflow", None)  # pyright: ignore[reportUnknownMemberType]  # consume regardless of outcome

    name = str(pending.get("name") or "")
    action = str(pending.get("action") or "")
    value = text.strip()
    if value in ("", "-"):
        await cl.Message(content=f"{action.capitalize()} cancelled — `{name}` is unchanged.").send()
        return True

    if action == "run":
        input_names = [str(n) for n in cast("list[Any]", pending.get("input_names") or [])]
        examples = cast("JsonDict", pending.get("examples") or {})
        descriptions = cast("JsonDict", pending.get("descriptions") or {})
        collected = {
            str(k): str(v) for k, v in cast("JsonDict", pending.get("collected") or {}).items()
        }
        # Record this answer for the next unfilled input.
        remaining = [n for n in input_names if n not in collected]
        if remaining:
            collected[remaining[0]] = value
        still = [n for n in input_names if n not in collected]
        if still:
            # More inputs to gather — re-arm and prompt for the next one.
            cl.user_session.set(  # pyright: ignore[reportUnknownMemberType]
                "pending_workflow", {**pending, "collected": collected}
            )
            nxt = still[0]
            await cl.Message(
                content=_input_prompt(
                    nxt, str(examples.get(nxt, "")), str(descriptions.get(nxt, ""))
                )
            ).send()
            return True
        draft_dir = _workflow_dir(name)
        if draft_dir is None:
            await cl.Message(content=f"Could not find a recipe for `{name}`.").send()
            return True
        recipe = await asyncio.to_thread(distill.load_recipe, draft_dir)
        await _send_replay_preview(recipe, collected)
        return True

    if action == "refine":
        progress = cl.Message(content="Refining the workflow… (this can take up to a minute)")
        await progress.send()
        try:
            draft_dir = await asyncio.to_thread(distill.refine_draft, name, value)
        except Exception as exc:
            await progress.remove()
            await cl.Message(content=f"Could not refine the workflow: {exc}").send()
            return True
        await progress.remove()
        await _send_workflow_preview(draft_dir)
        return True

    # Default: rename.
    try:
        draft_dir = await asyncio.to_thread(distill.rename_draft, name, value)
    except Exception as exc:
        await cl.Message(content=f"Could not rename the workflow: {exc}").send()
        return True
    await _send_workflow_preview(draft_dir)
    return True


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
        # A session that was eagerly created (to warm MCP servers) but never
        # received a message has no thread mapping, so it can never be deleted
        # from the UI. Purge its on-disk logs here so abandoned chats — page
        # refreshes, opened-and-closed tabs — don't leak transcripts/provenance.
        if not cl.user_session.get("vibe_session_persisted"):  # pyright: ignore[reportUnknownMemberType]
            with contextlib.suppress(Exception):
                provenance.purge_session(session_id)
            _audit.info("purged logs for abandoned (unsent) session %s", session_id)
            return
        # Render the human-readable provenance report for this session.
        if _is_provenance_enabled():
            with contextlib.suppress(Exception):
                provenance.write_report(session_id)
