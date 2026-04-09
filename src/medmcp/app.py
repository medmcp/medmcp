"""Chainlit chat UI for MedMCP, backed by vibe-acp.

Spawns vibe-acp as a subprocess and speaks JSON-RPC 2.0 over stdin/stdout.
This gives the UI full access to vibe's tool system (bash, read_file, grep, etc.)

Run with:  chainlit run src/medmcp/app.py -w

SECURITY MODEL
==============
This app exposes vibe-acp's full tool surface (bash, write_file, search_replace,
web_fetch, ...) through a chat box. The threat model assumes:

1. The Chainlit server runs on localhost only and is reachable only by the
   operator. There is no authentication. Do NOT bind to 0.0.0.0 or expose
   port 8000 over a network without adding ``password_auth_callback`` first.
2. Every tool call is gated by an interactive ``cl.AskActionMessage`` permission
   prompt — see :func:`_ask_user_for_permission`. The user must click Approve
   before any side effect occurs. There is no auto-approval path. Do NOT change
   this without understanding that the local model may be steered by prompt
   injection (e.g. content pasted from untrusted documents) into running
   arbitrary commands.
3. vibe-acp's own bash allowlist/denylist (``.vibe/config.toml``) is the second
   line of defense. Keep it current.
4. Permission decisions are logged to stderr (the chainlit terminal) for audit.
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

import chainlit as cl
from chainlit.context import context as cl_context

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

# ── JSON-RPC helpers ───────────────────────────────────────

_msg_id: int = 0


def _next_id() -> int:
    """Return a fresh, monotonically increasing JSON-RPC request id."""
    global _msg_id
    _msg_id += 1
    return _msg_id


def _encode(msg: JsonDict) -> bytes:
    """Encode a JSON-RPC message as a single UTF-8 line."""
    return (json.dumps(msg) + "\n").encode()


def _rpc_response(req_id: int, result: JsonDict) -> bytes:
    """Build a JSON-RPC response payload for a given request id."""
    return _encode({"jsonrpc": "2.0", "id": req_id, "result": result})


async def _send(
    proc: asyncio.subprocess.Process,
    method: str,
    params: JsonDict | None = None,
) -> int:
    """Send a JSON-RPC request to ``proc`` and return its assigned id.

    The subprocess must have been created with ``stdin=PIPE``; this is enforced
    by an assertion to satisfy strict type checking.
    """
    assert proc.stdin is not None, "subprocess must be created with stdin=PIPE"
    req_id = _next_id()
    msg: JsonDict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        msg["params"] = params
    proc.stdin.write(_encode(msg))
    await proc.stdin.drain()
    return req_id


async def _notify(
    proc: asyncio.subprocess.Process,
    method: str,
    params: JsonDict | None = None,
) -> None:
    """Send a JSON-RPC notification (no id, no response expected)."""
    assert proc.stdin is not None, "subprocess must be created with stdin=PIPE"
    msg: JsonDict = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    proc.stdin.write(_encode(msg))
    await proc.stdin.drain()


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
        # JSON-RPC peers should only send objects at the top level; ignore arrays/scalars.


async def _read_until_response(proc: asyncio.subprocess.Process, req_id: int) -> JsonDict:
    """Read messages until we receive the response matching ``req_id``."""
    assert proc.stdout is not None, "subprocess must be created with stdout=PIPE"
    while True:
        msg = await _read_line(proc.stdout)
        if msg is None:
            return {"error": "EOF"}
        if msg.get("id") == req_id and ("result" in msg or "error" in msg):
            return msg


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


def _get_session_state() -> tuple[asyncio.subprocess.Process | None, str | None]:
    """Return the per-chat ``(vibe-acp process, ACP session id)`` pair.

    The values are stored on Chainlit's per-user session by :func:`on_chat_start`.
    Casts narrow Chainlit's untyped ``user_session.get`` return value.
    """
    proc = cast(
        "asyncio.subprocess.Process | None",
        cl.user_session.get("proc"),  # pyright: ignore[reportUnknownMemberType]
    )
    session_id = cast(
        "str | None",
        cl.user_session.get("session_id"),  # pyright: ignore[reportUnknownMemberType]
    )
    return proc, session_id


def _set_session_state(key: str, value: object) -> None:
    """Wrap ``cl.user_session.set`` so the untyped call is in one place."""
    cl.user_session.set(key, value)  # pyright: ignore[reportUnknownMemberType]


# ── Chainlit hooks ─────────────────────────────────────────


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]
async def on_chat_start() -> None:
    """Spawn vibe-acp and initialize a session with the local model."""
    proc = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "vibe-acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env={**os.environ, "VIBE_HOME": str(Path(PROJECT_ROOT) / ".vibe")},
    )
    _set_session_state("proc", proc)

    # 1. Initialize
    init_id = await _send(
        proc,
        "initialize",
        {"protocol_version": 1, "client_capabilities": {}},
    )
    resp = await _read_until_response(proc, init_id)
    if "error" in resp:
        await cl.Message(content=f"Failed to initialize vibe-acp: {resp}").send()
        return

    # 2. Create session
    session_id_req = await _send(
        proc,
        "session/new",
        {"cwd": PROJECT_ROOT, "mcp_servers": []},
    )
    resp = await _read_until_response(proc, session_id_req)
    if "error" in resp:
        await cl.Message(content=f"Failed to create session: {resp}").send()
        return

    result = cast("JsonDict", resp.get("result") or {})
    session_id = cast("str", result["sessionId"])
    _set_session_state("session_id", session_id)


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


@cl.on_message  # pyright: ignore[reportUnknownMemberType]
async def on_message(message: cl.Message) -> None:
    """Send a prompt to vibe-acp and stream the response back into the UI."""
    proc, session_id = _get_session_state()

    if proc is None or session_id is None:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    assert proc.stdout is not None, "subprocess must be created with stdout=PIPE"

    prompt_id = await _send(
        proc,
        "session/prompt",
        {
            "session_id": session_id,
            "prompt": [{"type": "text", "text": message.content}],
        },
    )

    async def _cancel_and_drain() -> None:
        """Cancel any in-flight vibe-acp task for this session.

        Used when our own asyncio task gets cancelled (Chainlit stop button or
        the user sending a new message). Without this, vibe-acp keeps running
        the previous agent loop and the next ``session/prompt`` gets rejected
        with "Concurrent prompts are not supported yet".
        """
        with contextlib.suppress(Exception):
            await _notify(proc, "session/cancel", {"session_id": session_id})

    # Chainlit wraps each on_message handler in a parent Step(type="run").
    # We attach tool steps AND the assistant message as siblings of that run
    # step, and create them in temporal order. The frontend renders children
    # in append order, so tool steps appear before the assistant text.
    run_step = cl_context.current_step
    parent_id: str | None = run_step.id if run_step else None

    # Lazily create the assistant message on the first text chunk so it gets
    # appended *after* any tool steps from this turn.
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

    try:
        while True:
            msg = await _read_line(proc.stdout)
            if msg is None:
                break

            # ── Response to our prompt (completion or error) ──
            if msg.get("id") == prompt_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    err = cast("JsonDict", msg["error"])
                    target = await _ensure_assistant_msg()
                    err_msg = err.get("message", str(err))
                    await target.stream_token(f"\n\nError: {err_msg}")
                break

            method = msg.get("method")

            # ── Notification: session/update ──
            if method == "session/update":
                params = cast("JsonDict", msg.get("params") or {})
                update = cast("JsonDict", params.get("update") or {})
                update_type = update.get("sessionUpdate")

                if update_type == "agent_message_chunk":
                    content = cast("JsonDict", update.get("content") or {})
                    if content.get("type") == "text":
                        target = await _ensure_assistant_msg()
                        text = cast("str", content.get("text") or "")
                        await target.stream_token(text)

                elif update_type == "tool_call":
                    await _handle_tool_call(update, tool_steps, tool_call_info, parent_id)

                elif update_type == "tool_call_update":
                    await _handle_tool_call_update(update, tool_steps)

            # ── Server request: permission ──
            elif method == "session/request_permission":
                req_id_raw = msg.get("id")
                if not isinstance(req_id_raw, int):
                    continue  # cannot respond without a request id
                req_id: int = req_id_raw
                params = cast("JsonDict", msg.get("params") or {})
                tc: JsonDict = dict(cast("JsonDict", params.get("toolCall") or {}))
                options = cast("list[JsonDict]", params.get("options") or [])
                # Backfill title/rawInput from the cached tool_call event,
                # because request_permission only ships the toolCallId.
                cached = tool_call_info.get(cast("str", tc.get("toolCallId") or ""), {})
                for key in ("title", "rawInput"):
                    if tc.get(key) is None and cached.get(key) is not None:
                        tc[key] = cached[key]

                outcome = await _ask_user_for_permission(tc, options)

                assert proc.stdin is not None
                proc.stdin.write(_rpc_response(req_id, {"outcome": outcome}))
                await proc.stdin.drain()

    except asyncio.CancelledError:
        # Chainlit cancels our task when the user clicks Stop or sends a new
        # message mid-stream. Tell vibe-acp to abort its agent loop too,
        # otherwise the next session/prompt is rejected as concurrent.
        await _cancel_and_drain()
        raise

    if assistant_msg is not None:
        await assistant_msg.update()


@cl.on_stop  # pyright: ignore[reportUnknownMemberType]
async def on_stop() -> None:
    """Forward Chainlit's stop button to vibe-acp's ``session/cancel``.

    Without this, Chainlit cancels its own task but vibe-acp keeps running its
    agent loop — and the next user prompt fails with
    "Concurrent prompts are not supported yet".
    """
    proc, session_id = _get_session_state()
    if proc is None or session_id is None or proc.returncode is not None:
        return
    with contextlib.suppress(Exception):
        await _notify(proc, "session/cancel", {"session_id": session_id})


@cl.on_chat_end  # pyright: ignore[reportUnknownMemberType]
async def on_chat_end() -> None:
    """Clean up the vibe-acp subprocess when the chat thread ends."""
    proc, _ = _get_session_state()
    if proc is not None and proc.returncode is None:
        proc.terminate()
        await proc.wait()
