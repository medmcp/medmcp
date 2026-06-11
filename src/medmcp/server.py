"""Workspace UI server for MedMCP.

Serves the three-panel workspace frontend (file explorer, image viewer, chat)
and exposes:

- a small filesystem API rooted at ``WORKSPACE_ROOT`` (tree listing, raw file
  content for the viewer, rename/delete/mkdir/upload),
- a WebSocket chat endpoint that relays the vibe-acp agent loop to the browser
  (text chunks, tool calls, usage updates, and interactive permission
  requests).

Run with:  medmcp-workspace  (or ``just workspace``)

SECURITY MODEL
==============
Same threat model as the Chainlit app (``app.py``):

1. The server binds to localhost only. There is no authentication — do NOT
   expose the port over a network.
2. Every tool call is gated by an interactive permission request forwarded to
   the browser; the user must click Approve before any side effect occurs.
   There is no auto-approval path. Do not add one.
3. Permission decisions are logged via the ``medmcp.audit`` logger (stderr).
4. The filesystem API refuses paths that resolve outside ``WORKSPACE_ROOT``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from medmcp.acp import PROJECT_ROOT, JsonDict, VibeAcpClient

_audit: logging.Logger = logging.getLogger("medmcp.audit")

# Directory shown in the file explorer. Defaults to the repo root (where the
# agent's own tools operate); override with MEDMCP_WORKSPACE for a data dir.
WORKSPACE_ROOT: Path = Path(os.environ.get("MEDMCP_WORKSPACE", PROJECT_ROOT)).resolve()
FRONTEND_DIST: Path = Path(PROJECT_ROOT) / "frontend" / "dist"
DEFAULT_PORT: int = 8100

# Tool/VCS internals hidden from the explorer tree.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".vibe",
        ".chainlit",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".uv_cache",
    }
)
_TREE_MAX_DEPTH: int = 12
_TREE_MAX_ENTRIES_PER_DIR: int = 500

app = FastAPI(title="MedMCP Workspace")

# One vibe-acp subprocess shared by every websocket connection, exactly like
# the Chainlit app shares one across browser tabs.
_client: VibeAcpClient = VibeAcpClient()


# ── Filesystem API ─────────────────────────────────────────


def _safe_path(rel: str) -> Path:
    """Resolve ``rel`` inside the workspace, rejecting traversal attempts."""
    if Path(rel).is_absolute():
        raise HTTPException(status_code=400, detail="absolute paths are not allowed")
    resolved = (WORKSPACE_ROOT / rel).resolve()
    if resolved != WORKSPACE_ROOT and not resolved.is_relative_to(WORKSPACE_ROOT):
        raise HTTPException(status_code=400, detail="path escapes the workspace")
    return resolved


def _tree_node(path: Path, depth: int) -> JsonDict:
    """Build one explorer-tree node (recursing into directories)."""
    rel = str(path.relative_to(WORKSPACE_ROOT))
    node: JsonDict = {"id": rel, "name": path.name}
    if path.is_dir():
        children: list[JsonDict] = []
        if depth < _TREE_MAX_DEPTH:
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                entries = []
            for child in entries[:_TREE_MAX_ENTRIES_PER_DIR]:
                if child.name in _SKIP_DIRS:
                    continue
                children.append(_tree_node(child, depth + 1))
        node["children"] = children
    else:
        with contextlib.suppress(OSError):
            node["size"] = path.stat().st_size
    return node


@app.get("/api/workspace")
async def get_workspace() -> JsonDict:
    """Return the workspace root path shown in the explorer header."""
    return {"root": str(WORKSPACE_ROOT)}


@app.get("/api/tree")
async def get_tree() -> JsonDict:
    """Return the full explorer tree for the workspace."""
    root = _tree_node(WORKSPACE_ROOT, 0)
    return {"tree": root.get("children", [])}


@app.get("/api/raw/{rel_path:path}")
async def get_raw(rel_path: str) -> FileResponse:
    """Serve a file's raw bytes (viewer content, downloads)."""
    path = _safe_path(rel_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


class MkdirPayload(BaseModel):
    """Request body for directory creation."""

    path: str


@app.post("/api/files/mkdir")
async def post_mkdir(payload: MkdirPayload) -> JsonDict:
    """Create a directory (parents included)."""
    path = _safe_path(payload.path)
    path.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


class RenamePayload(BaseModel):
    """Request body for rename/move."""

    path: str
    new_path: str


@app.post("/api/files/rename")
async def post_rename(payload: RenamePayload) -> JsonDict:
    """Rename or move a file/directory inside the workspace."""
    src = _safe_path(payload.path)
    dst = _safe_path(payload.new_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="source not found")
    if dst.exists():
        raise HTTPException(status_code=409, detail="target already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {"ok": True}


@app.delete("/api/files")
async def delete_file(path: str) -> JsonDict:
    """Delete a file or directory (recursive). The UI confirms before calling."""
    target = _safe_path(path)
    if target == WORKSPACE_ROOT:
        raise HTTPException(status_code=400, detail="refusing to delete the workspace root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="not found")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True}


@app.post("/api/files/upload")
async def post_upload(file: UploadFile, dir: str = "") -> JsonDict:
    """Store an uploaded file under ``dir`` (workspace-relative)."""
    target_dir = _safe_path(dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.bin").name
    target = target_dir / name
    with target.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)
    return {"ok": True, "path": str(target.relative_to(WORKSPACE_ROOT))}


# ── WebSocket chat ─────────────────────────────────────────
#
# Wire protocol (JSON messages):
#
#   server → client
#     {"type": "ready", "sessionId": str}
#     {"type": "chunk", "text": str}
#     {"type": "tool_call", "toolCallId": str, "title": str, "status": str,
#      "kind": str | None, "rawInput": object}
#     {"type": "tool_call_update", "toolCallId": str, "status": str | None,
#      "output": str | None}
#     {"type": "usage", "used": int}
#     {"type": "permission_request", "requestId": int, "toolCall": {...},
#      "options": [{"optionId": str, "name": str, "kind": str}]}
#     {"type": "done"} | {"type": "error", "message": str}
#
#   client → server
#     {"type": "prompt", "text": str}
#     {"type": "permission", "requestId": int, "optionId": str | null}
#     {"type": "cancel"}


def _extract_text(content: object) -> str:
    """Pull plain text out of an ACP content-block list (best effort)."""
    parts: list[str] = []
    if isinstance(content, list):
        for block in cast("list[object]", content):
            if isinstance(block, dict):
                b = cast("JsonDict", block)
                inner = b.get("content")
                if isinstance(inner, dict):
                    inner_d = cast("JsonDict", inner)
                    if inner_d.get("type") == "text":
                        parts.append(str(inner_d.get("text") or ""))
                elif b.get("type") == "text":
                    parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


class _ChatConnection:
    """State for one browser websocket: one vibe-acp session, one prompt at a time."""

    def __init__(self, ws: WebSocket, session_id: str, queue: asyncio.Queue[JsonDict]) -> None:
        """Bind the websocket to its registered session queue."""
        self.ws = ws
        self.session_id = session_id
        self.queue = queue
        self._pending_perms: dict[int, asyncio.Future[str | None]] = {}
        self._prompt_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Receive client messages until the socket closes."""
        while True:
            data = cast("JsonDict", await self.ws.receive_json())
            kind = data.get("type")
            if kind == "prompt":
                text = str(data.get("text") or "")
                if not text:
                    continue
                # A new prompt while one is streaming cancels the old one,
                # matching the Chainlit UI's behaviour.
                await self._cancel_prompt()
                self._prompt_task = asyncio.create_task(self._run_prompt(text))
            elif kind == "permission":
                req_id = data.get("requestId")
                if isinstance(req_id, int):
                    fut = self._pending_perms.pop(req_id, None)
                    if fut is not None and not fut.done():
                        option_id = data.get("optionId")
                        fut.set_result(option_id if isinstance(option_id, str) else None)
            elif kind == "cancel":
                await self._cancel_prompt()

    async def close(self) -> None:
        """Abort any in-flight prompt and drop the session queue."""
        await self._cancel_prompt()
        _client.unregister_session(self.session_id)

    async def _cancel_prompt(self) -> None:
        """Cancel the running prompt task and tell vibe-acp to abort its loop."""
        task = self._prompt_task
        self._prompt_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            with contextlib.suppress(Exception):
                await _client.notify("session/cancel", {"session_id": self.session_id})
        for fut in self._pending_perms.values():
            if not fut.done():
                fut.set_result(None)
        self._pending_perms.clear()

    async def _send(self, msg: JsonDict) -> None:
        """Send one frame to the browser, ignoring a just-closed socket."""
        with contextlib.suppress(Exception):
            await self.ws.send_json(msg)

    async def _run_prompt(self, text: str) -> None:
        """Send one ``session/prompt`` and stream its frames to the browser.

        Mirrors the race loop in ``app.py``'s ``on_message``: wait for either
        the next inbound session frame or the prompt response, then drain
        whatever is left in the queue once the response lands.
        """
        prompt_fut = asyncio.create_task(
            _client.request(
                "session/prompt",
                {
                    "session_id": self.session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            )
        )
        try:
            while True:
                get_task: asyncio.Task[JsonDict] = asyncio.create_task(self.queue.get())
                done, _pending = await asyncio.wait(
                    {get_task, prompt_fut}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_task in done:
                    await self._forward_frame(get_task.result())
                    continue
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await get_task
                while not self.queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        await self._forward_frame(self.queue.get_nowait())
                resp = prompt_fut.result()
                if "error" in resp:
                    err = cast("JsonDict", resp["error"])
                    await self._send({"type": "error", "message": str(err.get("message", err))})
                break
        except asyncio.CancelledError:
            if not prompt_fut.done():
                prompt_fut.cancel()
            raise
        except Exception as exc:  # surface engine errors instead of a silent hang
            if not prompt_fut.done():
                prompt_fut.cancel()
            with contextlib.suppress(Exception):
                await _client.notify("session/cancel", {"session_id": self.session_id})
            await self._send({"type": "error", "message": str(exc)})
        finally:
            await self._send({"type": "done"})

    async def _forward_frame(self, msg: JsonDict) -> None:
        """Translate one inbound vibe-acp frame into a browser message."""
        method = msg.get("method")
        if method == "session/update":
            params = cast("JsonDict", msg.get("params") or {})
            update = cast("JsonDict", params.get("update") or {})
            update_type = update.get("sessionUpdate")
            if update_type == "agent_message_chunk":
                content = cast("JsonDict", update.get("content") or {})
                if content.get("type") == "text":
                    await self._send({"type": "chunk", "text": str(content.get("text") or "")})
            elif update_type == "tool_call":
                await self._send(
                    {
                        "type": "tool_call",
                        "toolCallId": str(update.get("toolCallId") or ""),
                        "title": str(update.get("title") or "tool"),
                        "status": str(update.get("status") or "pending"),
                        "kind": update.get("kind"),
                        "rawInput": update.get("rawInput"),
                    }
                )
            elif update_type == "tool_call_update":
                output = _extract_text(update.get("content"))
                raw_output = update.get("rawOutput")
                if not output and raw_output is not None:
                    output = str(raw_output)
                await self._send(
                    {
                        "type": "tool_call_update",
                        "toolCallId": str(update.get("toolCallId") or ""),
                        "status": update.get("status"),
                        "output": output[:2000] if output else None,
                    }
                )
            elif update_type == "usage_update":
                used = update.get("used")
                if isinstance(used, int):
                    await self._send({"type": "usage", "used": used})
        elif method == "session/request_permission":
            await self._handle_permission(msg)

    async def _handle_permission(self, msg: JsonDict) -> None:
        """Forward a permission request to the browser and relay the decision.

        Every decision (or timeout) is written to the ``medmcp.audit`` log,
        like the Chainlit permission prompt. A closed socket or timeout
        resolves to ``cancelled`` — never to approval.
        """
        req_id_raw = msg.get("id")
        if not isinstance(req_id_raw, int):
            return
        params = cast("JsonDict", msg.get("params") or {})
        tool_call = cast("JsonDict", params.get("toolCall") or {})
        options = cast("list[JsonDict]", params.get("options") or [])
        title = tool_call.get("title") or tool_call.get("toolCallId") or "<unknown>"

        if not options:
            _audit.warning("permission request had no options; cancelling: %s", title)
            await _client.respond(req_id_raw, {"outcome": {"outcome": "cancelled"}})
            return

        _audit.info("permission requested: %s", title)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str | None] = loop.create_future()
        self._pending_perms[req_id_raw] = fut
        await self._send(
            {
                "type": "permission_request",
                "requestId": req_id_raw,
                "toolCall": tool_call,
                "options": options,
            }
        )
        try:
            option_id = await asyncio.wait_for(fut, timeout=300)
        except TimeoutError:
            option_id = None
        finally:
            self._pending_perms.pop(req_id_raw, None)

        if option_id is None:
            _audit.warning("permission cancelled/timed out: %s", title)
            outcome: JsonDict = {"outcome": "cancelled"}
        else:
            _audit.info("permission decision: %s -> %s", title, option_id)
            outcome = {"outcome": "selected", "optionId": option_id}
        await _client.respond(req_id_raw, {"outcome": outcome})


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """Create a vibe-acp session for this socket and run the chat loop."""
    await ws.accept()
    conn: _ChatConnection | None = None
    try:
        await _client.ensure_started()
        resp = await _client.request("session/new", {"cwd": PROJECT_ROOT, "mcpServers": []})
        if "error" in resp:
            await ws.send_json({"type": "error", "message": str(resp["error"])})
            await ws.close()
            return
        result = cast("JsonDict", resp.get("result") or {})
        session_id = str(result.get("sessionId") or "")
        if not session_id:
            await ws.send_json({"type": "error", "message": "session/new returned no sessionId"})
            await ws.close()
            return
        queue = _client.register_session(session_id)
        conn = _ChatConnection(ws, session_id, queue)
        await ws.send_json({"type": "ready", "sessionId": session_id})
        await conn.run()
    except WebSocketDisconnect:
        pass
    finally:
        if conn is not None:
            await conn.close()


# ── Static frontend ────────────────────────────────────────

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/")
    async def missing_frontend() -> JSONResponse:
        """Tell the developer the frontend hasn't been built yet."""
        return JSONResponse(
            {"error": "frontend not built", "hint": "run: just workspace-build"},
            status_code=503,
        )


def main() -> None:
    """Run the workspace server on localhost (no auth — do not expose)."""
    port = int(os.environ.get("MEDMCP_WORKSPACE_PORT", str(DEFAULT_PORT)))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
