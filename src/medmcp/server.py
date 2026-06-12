"""Workspace UI server for MedMCP.

Serves the three-panel workspace frontend (file explorer, image viewer, chat)
and exposes:

- a small filesystem API rooted at ``WORKSPACE_ROOT`` (tree listing, raw file
  content for the viewer, rename/delete/mkdir/upload),
- a settings API (``/api/settings``) for the stack/workflow/feature toggles
  shared with the Chainlit UI via ``medmcp.settings``,
- a WebSocket chat endpoint that relays the vibe-acp agent loop to the browser
  (text chunks, tool calls, usage updates, and interactive permission
  requests, optionally enriched with LLM explanations and risk tags).

Run with:  medmcp-workspace  (or ``just workspace``)

SECURITY MODEL
==============
Same threat model as the Chainlit app (``app.py``):

1. The server binds to localhost only. There is no authentication — do NOT
   expose the port over a network.
2. Every tool call is gated by an interactive permission request forwarded to
   the browser; the user must click Approve before any side effect occurs.
   There is no auto-approval path. Do not add one.
3. Permission decisions are logged via the ``medmcp.audit`` logger (stderr)
   and mirrored to the session's provenance record when capture is enabled.
4. The filesystem API refuses paths that resolve outside ``WORKSPACE_ROOT``.

PROVENANCE
==========
When "Record provenance" is on, each chat session gets the same Tier-1 record
as in the Chainlit UI (manifest on first prompt, run.jsonl per tool call,
permissions.log). Caveat: workspace sessions are not registered in the
Chainlit threads DB, so the Chainlit app's orphaned-provenance GC will treat
their records as orphans and delete them at its next startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from medmcp import distill, explain, provenance, replay, settings
from medmcp.acp import PROJECT_ROOT, VIBE_HOME, JsonDict, VibeAcpClient

_audit: logging.Logger = logging.getLogger("medmcp.audit")

# Directory shown in the file explorer AND the agent's working directory (its
# tools run here, so it sees the same files as the explorer/viewer). Defaults
# to the repo's data/ directory; override with MEDMCP_WORKSPACE.
WORKSPACE_ROOT: Path = Path(
    os.environ.get("MEDMCP_WORKSPACE", str(Path(PROJECT_ROOT) / "data"))
).resolve()
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
# the Chainlit app shares one across browser tabs. The subprocess cwd must
# stay PROJECT_ROOT — `uv run` resolves the project from it; the agent's
# working directory is set per session via session/new's cwd instead.
_client: VibeAcpClient = VibeAcpClient()

# Live websocket connections, so a settings-triggered vibe restart can close
# them (each client auto-reconnects into a fresh session on the new process).
_connections: set[_ChatConnection] = set()

# Strong refs to fire-and-forget background tasks (asyncio only keeps weak ones).
_background_tasks: set[asyncio.Task[Any]] = set()


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
    # The recursive walk stats every file; run it off the event loop so a
    # large (or network-mounted) workspace can't stall the chat websockets.
    root = await asyncio.to_thread(_tree_node, WORKSPACE_ROOT, 0)
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
        await asyncio.to_thread(shutil.rmtree, target)
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
            await asyncio.to_thread(fh.write, chunk)
    return {"ok": True, "path": str(target.relative_to(WORKSPACE_ROOT))}


# ── Settings API ───────────────────────────────────────────


class ToggleEntry(BaseModel):
    """One named on/off switch as shown in the settings drawer."""

    name: str
    active: bool


class SettingsPayload(BaseModel):
    """Full settings state as submitted by the UI's settings drawer.

    ``stacks``/``workflows`` carry every entry the drawer knew about (name +
    active), so the server can tell "deactivated" apart from "unknown to this
    drawer" and preserve the state of entries created after the drawer
    fetched (e.g. a draft distilled while it was open).
    """

    explain_tools: bool
    record_provenance: bool
    workflows_enabled: bool
    stacks: list[ToggleEntry]
    workflows: list[ToggleEntry]


def _settings_state() -> JsonDict:
    """Assemble the current settings state (runs blocking discovery)."""
    stacks = settings.load_mcp_servers()
    active = settings.load_active_server_names()
    workflows = settings.discover_workflows()
    active_wf = settings.load_active_workflow_names()
    return {
        "explain_tools": settings.load_explain_enabled(),
        "record_provenance": settings.load_provenance_enabled(),
        "workflows_enabled": settings.load_workflows_enabled(),
        "stacks": [
            {"name": s["name"], "version": s.get("version"), "active": s["name"] in active}
            for s in stacks
        ],
        "workflows": [
            {
                "name": w["name"],
                "description": w["description"],
                "kind": w["kind"],
                "active": w["name"] in active_wf,
            }
            for w in workflows
        ],
    }


@app.get("/api/settings")
async def get_settings() -> JsonDict:
    """Return toggles plus the discovered stacks/workflows with active state."""
    return await asyncio.to_thread(_settings_state)


@app.put("/api/settings")
async def put_settings(payload: SettingsPayload) -> JsonDict:
    """Persist settings; restart vibe-acp when its config inputs changed.

    Stack and workflow changes are baked into ``.vibe/config.toml`` at session
    start, so applying them requires a fresh vibe-acp process. All live chat
    sockets are closed; each client auto-reconnects into a new session.
    """

    def _apply() -> bool:
        old_stacks = settings.load_active_server_names()
        old_workflows = settings.load_active_workflow_names()
        old_wf_enabled = settings.load_workflows_enabled()

        settings.save_explain_enabled(payload.explain_tools)
        settings.save_provenance_enabled(payload.record_provenance)
        settings.save_workflows_enabled(payload.workflows_enabled)
        # Merge instead of overwrite: entries the drawer never saw keep their
        # current active state instead of being silently deactivated.
        known_stacks = {t.name for t in payload.stacks}
        new_stacks = {t.name for t in payload.stacks if t.active} | (old_stacks - known_stacks)
        known_workflows = {t.name for t in payload.workflows}
        new_workflows = {t.name for t in payload.workflows if t.active} | (
            old_workflows - known_workflows
        )
        settings.save_active_server_names(new_stacks)
        settings.save_active_workflow_names(new_workflows)

        restart = (
            new_stacks != old_stacks
            or new_workflows != old_workflows
            or payload.workflows_enabled != old_wf_enabled
        )
        if restart:
            settings.sync_servers_to_vibe_config(settings.active_servers())
        return restart

    restart_needed = await asyncio.to_thread(_apply)
    if restart_needed:
        _audit.info("settings changed; restarting vibe-acp")
        await _restart_vibe()
    return {"ok": True, "restarted": restart_needed}


async def _restart_vibe() -> None:
    """Stop the shared vibe-acp process and drop every live chat socket."""
    await _client.stop()
    for conn in list(_connections):
        with contextlib.suppress(Exception):
            await conn.ws.close()


# ── Workflow API ───────────────────────────────────────────
#
# Drives the provenance→distill→replay pipeline (Tier 2/3) from the workspace
# UI: save the current chat as a draft workflow, review/promote/refine it, and
# replay its recipe deterministically (no LLM) on new inputs.


def _workflow_dir(name: str) -> Path | None:
    """Return the on-disk dir for workflow *name* (active wins over draft)."""
    for kind in ("active", "draft"):
        d = VIBE_HOME / "workflows" / kind / name
        if (d / "recipe.yaml").exists():
            return d
    return None


def _workflow_detail(name: str) -> JsonDict:
    """Load a workflow's recipe and replayability state (blocking; run in a thread)."""
    d = _workflow_dir(name)
    if d is None:
        raise FileNotFoundError(f"no workflow named {name!r}")
    recipe = distill.load_recipe(d)
    examples = {i.name: i.example for i in recipe.inputs}
    replay_error = replay.validate(recipe, examples, settings.active_servers())
    return {
        "name": recipe.name,
        "kind": d.parent.name,
        "description": recipe.description,
        "inputs": [i.to_dict() for i in recipe.inputs],
        "steps": [
            {"server": s.server, "tool": s.tool, "arguments": s.arguments} for s in recipe.steps
        ],
        "replayable": replay_error is None,
        "replay_error": replay_error,
    }


@app.get("/api/workflows")
async def get_workflows() -> JsonDict:
    """List personal workflows plus the master-toggle state."""

    def _state() -> JsonDict:
        return {
            "enabled": settings.load_workflows_enabled(),
            "workflows": settings.discover_workflows(),
        }

    return await asyncio.to_thread(_state)


class DistillPayload(BaseModel):
    """Request body for distilling a chat session into a draft workflow."""

    session_id: str


@app.post("/api/workflows/distill")
async def post_distill(payload: DistillPayload) -> JsonDict:
    """Distill a chat session into a draft workflow and return its detail.

    Runs the hybrid prose pass against the local model, so this can take a
    while; distillation itself never hard-fails on a model outage.
    """
    try:
        draft_dir = await asyncio.to_thread(distill.distill_session, payload.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit.info("workflow distilled: %s", draft_dir.name)
    return await asyncio.to_thread(_workflow_detail, draft_dir.name)


@app.get("/api/workflows/{name}")
async def get_workflow(name: str) -> JsonDict:
    """Return one workflow's recipe detail (inputs, steps, replayability)."""
    try:
        return await asyncio.to_thread(_workflow_detail, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/workflows/{name}/promote")
async def post_promote_workflow(name: str) -> JsonDict:
    """Promote a draft to active/ (loaded as a skill for new sessions)."""
    try:
        await asyncio.to_thread(distill.promote_draft, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit.info("workflow promoted: %s", name)
    return {"ok": True}


@app.post("/api/workflows/{name}/unpromote")
async def post_unpromote_workflow(name: str) -> JsonDict:
    """Move a promoted workflow back to draft/ for editing."""
    try:
        await asyncio.to_thread(distill.unpromote_workflow, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


class WorkflowRenamePayload(BaseModel):
    """Request body for renaming a draft workflow."""

    new_name: str


@app.post("/api/workflows/{name}/rename")
async def post_rename_workflow(name: str, payload: WorkflowRenamePayload) -> JsonDict:
    """Rename a draft workflow; returns the new (slugified) name."""
    try:
        new_dir = await asyncio.to_thread(distill.rename_draft, name, payload.new_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "name": new_dir.name}


class WorkflowRefinePayload(BaseModel):
    """Request body for refining a draft's narrative."""

    instruction: str


@app.post("/api/workflows/{name}/refine")
async def post_refine_workflow(name: str, payload: WorkflowRefinePayload) -> JsonDict:
    """Regenerate a draft's narrative from a plain-language instruction (LLM)."""
    try:
        await asyncio.to_thread(distill.refine_draft, name, payload.instruction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True}


@app.delete("/api/workflows/{name}")
async def delete_workflow(name: str) -> JsonDict:
    """Delete a personal workflow (draft or active). The UI confirms first."""
    try:
        await asyncio.to_thread(distill.delete_workflow, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit.info("workflow deleted: %s", name)
    return {"ok": True}


class ReplayPreviewPayload(BaseModel):
    """Request body for previewing a replay's resolved steps."""

    inputs: dict[str, str]


@app.post("/api/workflows/{name}/replay-preview")
async def post_replay_preview(name: str, payload: ReplayPreviewPayload) -> JsonDict:
    """Validate a replay and return its resolved steps for user confirmation.

    Inputs are bound now; cross-step refs (``{{stepM.*}}``) resolve at runtime,
    so they intentionally still show as placeholders in the preview.
    """

    def _preview() -> JsonDict:
        d = _workflow_dir(name)
        if d is None:
            raise FileNotFoundError(f"no workflow named {name!r}")
        recipe = distill.load_recipe(d)
        error = replay.validate(recipe, dict(payload.inputs), settings.active_servers())
        if error is not None:
            return {"ok": False, "error": error, "steps": []}
        bindings: dict[str, Any] = dict(payload.inputs)
        steps = [
            {
                "index": i,
                "server": s.server,
                "tool": s.tool,
                "arguments": replay.resolve_arguments(s.arguments, bindings),
            }
            for i, s in enumerate(recipe.steps, start=1)
        ]
        return {"ok": True, "error": None, "steps": steps}

    try:
        return await asyncio.to_thread(_preview)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/ws/replay")
async def ws_replay(ws: WebSocket) -> None:
    """Run a deterministic replay — single or batch — streaming status frames.

    First client message: ``{"name": str, "runs": [{in_N: value}, ...]}``;
    each entry is one full input binding and the recipe runs once per entry,
    sequentially. A failed item does not stop the remaining items. The server
    streams ``{"type": "step", "item": int, ...}`` per executed step,
    ``{"type": "item_result", "item": int, "ok": bool, "error": str|null,
    "outputs": [...]}`` per finished item, and a final ``{"type": "result",
    "ok": bool, "error": str|null, "outputs": [...]}`` over all items.
    Closing the socket aborts the run immediately: a concurrent watcher task
    observes the disconnect and cancels the batch, unwinding the engine's
    exit stack — which shuts down the spawned MCP servers, killing the
    in-flight tool step.

    SECURITY: the replay engine calls MCP tools directly, bypassing the
    vibe-acp permission flow — the client must show the resolved-steps preview
    (``/replay-preview``) and get an explicit confirmation before connecting.
    """
    await ws.accept()
    try:
        first = cast("JsonDict", await ws.receive_json())
        name = str(first.get("name") or "")
        runs = [
            {str(k): str(v) for k, v in cast("JsonDict", r).items()}
            for r in cast("list[object]", first.get("runs") or [])
            if isinstance(r, dict)
        ]
        d = _workflow_dir(name)
        if d is None or not runs:
            error = f"no workflow named {name!r}" if d is None else "no inputs to run"
            await ws.send_json({"type": "result", "ok": False, "error": error})
            return
        recipe = await asyncio.to_thread(distill.load_recipe, d)
        servers = await asyncio.to_thread(settings.active_servers)
        _audit.info(
            "replay started: %s (%d item(s) x %d steps)", name, len(runs), len(recipe.steps)
        )

        async def _run_batch() -> None:
            all_outputs: list[str] = []
            failed = 0
            for item, inputs in enumerate(runs):

                async def _on_step(sr: replay.StepResult, item: int = item) -> None:
                    await ws.send_json(
                        {
                            "type": "step",
                            "item": item,
                            "index": sr.index,
                            "server": sr.server,
                            "tool": sr.tool,
                            "ok": sr.ok,
                            "error": sr.error,
                            "produced": sr.produced,
                        }
                    )

                result = await replay.run(
                    recipe, inputs, servers=servers, cwd=str(WORKSPACE_ROOT), on_step=_on_step
                )
                outputs = [v for sr in result.steps for v in sr.produced.values()]
                all_outputs.extend(outputs)
                if not result.ok:
                    failed += 1
                await ws.send_json(
                    {
                        "type": "item_result",
                        "item": item,
                        "ok": result.ok,
                        "error": result.error,
                        "outputs": outputs,
                    }
                )
            ok = failed == 0
            batch_error = None if ok else f"{failed} of {len(runs)} item(s) failed"
            _audit.info("replay finished: %s ok=%s", name, ok)
            await ws.send_json(
                {"type": "result", "ok": ok, "error": batch_error, "outputs": all_outputs}
            )

        # Race the batch against a socket watcher. The client sends nothing
        # after the first message, so anything receive returns — and above all
        # a disconnect — means "abort". Cancelling the batch task unwinds
        # replay.run's exit stack, killing the in-flight MCP server and its
        # running tool; without the watcher, Stop would only take effect at
        # the next frame send, after the current step finished.
        batch_task = asyncio.create_task(_run_batch())
        watch_task = asyncio.create_task(ws.receive_text())
        try:
            done, _pending = await asyncio.wait(
                {batch_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if batch_task in done:
                watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await watch_task
                batch_task.result()  # propagate batch errors (e.g. send on closed socket)
            else:
                batch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_task
                _audit.info("replay aborted by client; in-flight step cancelled")
                with contextlib.suppress(WebSocketDisconnect):
                    await watch_task
        except asyncio.CancelledError:
            batch_task.cancel()
            watch_task.cancel()
            raise
    except WebSocketDisconnect:
        _audit.info("replay socket closed by client; run aborted")
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


# ── WebSocket chat ─────────────────────────────────────────
#
# Wire protocol (JSON messages):
#
#   server → client
#     {"type": "ready", "sessionId": str, "model": str}
#     {"type": "chunk", "text": str}
#     {"type": "tool_call", "toolCallId": str, "title": str, "status": str,
#      "kind": str | None, "rawInput": object}
#     {"type": "tool_call_update", "toolCallId": str, "status": str | None,
#      "output": str | None}
#     {"type": "usage", "used": int}
#     {"type": "permission_request", "requestId": int, "toolCall": {...},
#      "options": [{"optionId": str, "name": str, "kind": str}],
#      "explanation": str | None, "explaining": bool,
#      "risks": [{"key": str, "label": str, "severity": str}]}
#     {"type": "permission_update", "requestId": int,
#      "explanation": str | None, "risks": [...]}
#     {"type": "done"} | {"type": "error", "message": str}
#
#   client → server
#     {"type": "prompt", "text": str, "viewedPath": str | null}
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

    def __init__(
        self,
        ws: WebSocket,
        session_id: str,
        queue: asyncio.Queue[JsonDict],
        servers: list[JsonDict],
    ) -> None:
        """Bind the websocket to its registered session queue.

        ``servers`` is the active-server list captured at connect time; a
        stack change restarts vibe-acp and closes every connection, so it
        cannot go stale within a connection's lifetime.
        """
        self.ws = ws
        self.session_id = session_id
        self.queue = queue
        self.servers = servers
        self._pending_perms: dict[int, asyncio.Future[str | None]] = {}
        self._prompt_task: asyncio.Task[None] | None = None
        # Tool-call state accumulated across frames, keyed by toolCallId; feeds
        # the permission dialog backfill and the provenance run log.
        self._tool_calls: dict[str, JsonDict] = {}
        # In-flight explanation tasks, kept so they aren't GC'd mid-run.
        self._explain_tasks: set[asyncio.Task[None]] = set()
        self._prompted: bool = False

    async def run(self) -> None:
        """Receive client messages until the socket closes."""
        while True:
            data = cast("JsonDict", await self.ws.receive_json())
            kind = data.get("type")
            if kind == "prompt":
                text = str(data.get("text") or "")
                if not text:
                    continue
                viewed = data.get("viewedPath")
                viewed_path = viewed if isinstance(viewed, str) and viewed else None
                # A new prompt while one is streaming cancels the old one,
                # matching the Chainlit UI's behaviour.
                await self._cancel_prompt()
                self._prompt_task = asyncio.create_task(self._run_prompt(text, viewed_path))
            elif kind == "permission":
                req_id = data.get("requestId")
                if isinstance(req_id, int):
                    fut = self._pending_perms.pop(req_id, None)
                    if fut is not None and not fut.done():
                        option_id = data.get("optionId")
                        fut.set_result(option_id if isinstance(option_id, str) else None)
            elif kind == "cancel":
                await self._cancel_prompt()
                # A cancelled task no longer emits its own `done` (a stale one
                # would clobber a newer turn's state), so an intentional Stop
                # resets the client explicitly.
                await self._send({"type": "done"})

    async def close(self) -> None:
        """Abort any in-flight prompt and drop the session queue.

        A session that never received a prompt is purged (transcript +
        provenance), mirroring the Chainlit ``on_chat_end`` cleanup for
        abandoned tabs/refreshes.
        """
        await self._cancel_prompt()
        for task in list(self._explain_tasks):
            task.cancel()
        _client.unregister_session(self.session_id)
        if not self._prompted:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(provenance.purge_session, self.session_id)

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
            await self._drain_stale_frames()

    async def _drain_stale_frames(self) -> None:
        """Discard queued frames left over from a cancelled turn.

        Without this, chunks (or even a permission request) from the old turn
        would be forwarded into the next prompt's stream. A drained permission
        request must still be answered, or vibe-acp's agent loop would hang
        awaiting the JSON-RPC response.
        """
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if msg.get("method") == "session/request_permission":
                req_id = msg.get("id")
                if isinstance(req_id, int):
                    with contextlib.suppress(Exception):
                        await _client.respond(req_id, {"outcome": {"outcome": "cancelled"}})

    async def _send(self, msg: JsonDict) -> None:
        """Send one frame to the browser, ignoring a just-closed socket."""
        with contextlib.suppress(Exception):
            await self.ws.send_json(msg)

    async def _run_prompt(self, text: str, viewed_path: str | None = None) -> None:
        """Send one ``session/prompt`` and stream its frames to the browser.

        ``viewed_path`` is the workspace-relative file currently open in the
        viewer; it is appended to the prompt as a context note so the agent
        can resolve references like "this image". Appended to the user's text
        block (not sent as a separate content block) so it survives any
        prompt handling downstream.

        Mirrors the race loop in ``app.py``'s ``on_message``: wait for either
        the next inbound session frame or the prompt response, then drain
        whatever is left in the queue once the response lands.
        """
        if not self._prompted:
            self._prompted = True
            if settings.load_provenance_enabled():
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        lambda: provenance.write_manifest(
                            self.session_id,
                            servers=self.servers,
                            model_name=settings.OLLAMA_MODEL,
                        )
                    )
        # Frames can still trickle in between a cancel and this prompt
        # (vibe-acp processes session/cancel asynchronously); drop them now so
        # the old turn can't bleed into this one.
        await self._drain_stale_frames()
        prompt_text = text
        if viewed_path is not None:
            prompt_text += (
                f'\n\n[workspace context: the file "{viewed_path}" is currently open in the '
                'viewer; references like "this image" or "the current image" mean that file]'
            )
        prompt_fut = asyncio.create_task(
            _client.request(
                "session/prompt",
                {
                    "session_id": self.session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
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
            await self._send({"type": "done"})
        except asyncio.CancelledError:
            # Deliberately no `done` frame: when a new prompt superseded this
            # one, a stale done would reset the client's busy/permission state
            # for the wrong turn. An intentional Stop gets its done from the
            # `cancel` branch in run().
            if not prompt_fut.done():
                prompt_fut.cancel()
            raise
        except Exception as exc:  # surface engine errors instead of a silent hang
            if not prompt_fut.done():
                prompt_fut.cancel()
            with contextlib.suppress(Exception):
                await _client.notify("session/cancel", {"session_id": self.session_id})
            await self._send({"type": "error", "message": str(exc)})
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
                tc_id = str(update.get("toolCallId") or "")
                title = str(update.get("title") or "tool")
                status = str(update.get("status") or "pending")
                info = self._tool_calls.setdefault(tc_id, {"_started": time.monotonic()})
                info["title"] = title
                info["status"] = status
                if update.get("rawInput") is not None:
                    info["rawInput"] = update.get("rawInput")
                await self._send(
                    {
                        "type": "tool_call",
                        "toolCallId": tc_id,
                        "title": title,
                        "status": status,
                        "kind": update.get("kind"),
                        "rawInput": update.get("rawInput"),
                    }
                )
            elif update_type == "tool_call_update":
                tc_id = str(update.get("toolCallId") or "")
                output = _extract_text(update.get("content"))
                raw_output = update.get("rawOutput")
                if not output and raw_output is not None:
                    output = str(raw_output)
                status = update.get("status")
                info = self._tool_calls.get(tc_id)
                if info is not None:
                    if isinstance(status, str):
                        info["status"] = status
                    if raw_output is not None:
                        info["rawOutput"] = raw_output
                    elif output:
                        info["outputText"] = output
                await self._send(
                    {
                        "type": "tool_call_update",
                        "toolCallId": tc_id,
                        "status": status,
                        "output": output[:2000] if output else None,
                    }
                )
                if status in ("completed", "failed") and info is not None:
                    if settings.load_provenance_enabled():
                        event_info = info
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(
                                lambda: provenance.record_tool_event(
                                    self.session_id,
                                    tc_id,
                                    event_info,
                                    [str(s["name"]) for s in self.servers],
                                )
                            )
                    # A settled call's state (incl. its full rawOutput) is no
                    # longer needed — permission requests always precede
                    # completion — so drop it rather than letting it grow with
                    # the session.
                    self._tool_calls.pop(tc_id, None)
            elif update_type == "usage_update":
                used = update.get("used")
                if isinstance(used, int):
                    # No-I/O accessor: an inline fetch_context_window() here
                    # would stall the relay of every queued frame behind an
                    # Ollama round-trip (mirrors app.py's emit path). The
                    # cache is warmed at connect time in ws_chat.
                    size = settings.cached_context_window()
                    await self._send({"type": "usage", "used": used, "size": size})
        elif method == "session/request_permission":
            await self._handle_permission(msg)

    async def _handle_permission(self, msg: JsonDict) -> None:
        """Forward a permission request to the browser and relay the decision.

        The request is backfilled with the cached ``tool_call`` metadata
        (``request_permission`` only ships the toolCallId) and, when the user
        enabled explanations, enriched with a plain-language explanation and
        risk tags from the local model. Every decision (or timeout) is written
        to the ``medmcp.audit`` log and mirrored to the provenance record. A
        closed socket or timeout resolves to ``cancelled`` — never approval.
        """
        req_id_raw = msg.get("id")
        if not isinstance(req_id_raw, int):
            return
        params = cast("JsonDict", msg.get("params") or {})
        tool_call: JsonDict = dict(cast("JsonDict", params.get("toolCall") or {}))
        options = cast("list[JsonDict]", params.get("options") or [])
        tc_id = str(tool_call.get("toolCallId") or "")
        cached = self._tool_calls.get(tc_id, {})
        for key in ("title", "rawInput", "humanReadable", "risks"):
            if tool_call.get(key) is None and cached.get(key) is not None:
                tool_call[key] = cached[key]
        title = tool_call.get("title") or tc_id or "<unknown>"

        if not options:
            _audit.warning("permission request had no options; cancelling: %s", title)
            await _client.respond(req_id_raw, {"outcome": {"outcome": "cancelled"}})
            return

        # Reuse a previously generated explanation (vibe-acp can re-request
        # permission for the same tool call); otherwise generate one
        # concurrently after showing the dialog — the approval box must never
        # wait on the Ollama round-trip, which can take seconds while the
        # model is still busy with the agent's own generation.
        explanation = cast("str | None", tool_call.get("humanReadable"))
        risk_keys: list[str] = cast("list[str]", tool_call.get("risks") or [])
        explaining = explanation is None and settings.load_explain_enabled()

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
                "explanation": explanation,
                "explaining": explaining,
                "risks": explain.resolve_risks(risk_keys),
            }
        )
        if explaining:
            task = asyncio.create_task(self._explain_permission(req_id_raw, tool_call, cached))
            self._explain_tasks.add(task)
            task.add_done_callback(self._explain_tasks.discard)
        try:
            option_id = await asyncio.wait_for(fut, timeout=300)
        except TimeoutError:
            option_id = None
        except asyncio.CancelledError:
            # Stop / disconnect / a new prompt cancelled the prompt task while
            # the approval box was open. vibe-acp is still blocked awaiting
            # this JSON-RPC response; answer "cancelled" before propagating or
            # its agent loop hangs and rejects every later prompt. The shield
            # keeps a second cancel from abandoning the write mid-flight.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._resolve_permission(req_id_raw, title, cached, None))
            raise
        finally:
            self._pending_perms.pop(req_id_raw, None)

        await self._resolve_permission(req_id_raw, title, cached, option_id)

    async def _explain_permission(self, req_id: int, tool_call: JsonDict, cached: JsonDict) -> None:
        """Generate the LLM explanation for an already-shown permission dialog.

        Runs concurrently with the open approval box; the result is pushed as
        a ``permission_update`` frame while the request is still pending. On
        failure the frame carries a null explanation so the client clears its
        pending hint. The result is also persisted into the accumulated
        tool-call state so the provenance run log carries the explanation and
        risks alongside the decision (best effort — a fast decision can land
        before the explanation does).
        """
        explanation: str | None = None
        risk_keys: list[str] = []
        with contextlib.suppress(Exception):
            result = await explain.generate_explanation(tool_call)
            if result is not None:
                explanation, risk_keys = result
        if explanation is not None and cached:
            cached["humanReadable"] = explanation
            cached["risks"] = risk_keys
        if req_id in self._pending_perms:
            await self._send(
                {
                    "type": "permission_update",
                    "requestId": req_id,
                    "explanation": explanation,
                    "risks": explain.resolve_risks(risk_keys),
                }
            )

    async def _resolve_permission(
        self, req_id: int, title: object, cached: JsonDict, option_id: str | None
    ) -> None:
        """Audit, mirror to provenance, and answer one permission request."""
        if option_id is None:
            _audit.warning("permission cancelled/timed out: %s", title)
            decision = "cancelled"
            outcome: JsonDict = {"outcome": "cancelled"}
        else:
            _audit.info("permission decision: %s -> %s", title, option_id)
            decision = option_id
            outcome = {"outcome": "selected", "optionId": option_id}
        if cached:
            cached["decision"] = decision

        def _mirror() -> None:
            if settings.load_provenance_enabled():
                provenance.log_permission(self.session_id, title=str(title), decision=decision)

        with contextlib.suppress(Exception):
            await asyncio.to_thread(_mirror)
        await _client.respond(req_id, {"outcome": outcome})


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """Create a vibe-acp session for this socket and run the chat loop.

    The config sync before ``session/new`` is mandatory: vibe-acp reads its
    MCP server list and skill paths from ``.vibe/config.toml``, not from the
    JSON-RPC call.
    """
    await ws.accept()
    conn: _ChatConnection | None = None
    try:
        # Evaluate active_servers() inside the thread too: the first call runs
        # uv-tool stack discovery (one subprocess per stack) and must not
        # block the event loop. The list is captured for the connection so
        # per-frame provenance writes don't re-derive it.
        def _sync_config() -> list[JsonDict]:
            servers = settings.active_servers()
            settings.sync_servers_to_vibe_config(servers)
            return servers

        servers = await asyncio.to_thread(_sync_config)
        # Warm the context-window cache off the frame path; usage frames read
        # the cached value only (the badge corrects itself once this lands).
        prefetch = asyncio.create_task(settings.fetch_context_window())
        _background_tasks.add(prefetch)
        prefetch.add_done_callback(_background_tasks.discard)
        await _client.ensure_started()
        resp = await _client.request("session/new", {"cwd": str(WORKSPACE_ROOT), "mcpServers": []})
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
        conn = _ChatConnection(ws, session_id, queue, servers)
        _connections.add(conn)
        await ws.send_json(
            {"type": "ready", "sessionId": session_id, "model": settings.OLLAMA_MODEL}
        )
        await conn.run()
    except WebSocketDisconnect:
        pass
    finally:
        if conn is not None:
            _connections.discard(conn)
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
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("MEDMCP_WORKSPACE_PORT", str(DEFAULT_PORT)))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
