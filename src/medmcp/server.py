"""Workspace UI server for MedMCP.

Serves the three-panel workspace frontend (file explorer, image viewer, chat)
and exposes:

- a small filesystem API rooted at ``WORKSPACE_ROOT`` (tree listing, raw file
  content for the viewer, rename/delete/mkdir/upload),
- a settings API (``/api/settings``) for the stack/workflow/feature toggles
  (persisted via ``medmcp.settings``),
- a WebSocket chat endpoint that relays the vibe-acp agent loop to the browser
  (text chunks, tool calls, usage updates, and interactive permission
  requests, optionally enriched with LLM explanations and risk tags).

Run with:  medmcp-workspace  (or ``just workspace``)

SECURITY MODEL
==============
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
When "Record provenance" is on, each chat session gets a Tier-1 record
(manifest on first prompt, run.jsonl per tool call, permissions.log).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medmcp import (
    batchplan,
    distill,
    explain,
    provenance,
    replay,
    sessions,
    settings,
    share,
    workflow,
)
from medmcp.acp import PROJECT_ROOT, VIBE_HOME, JsonDict, VibeAcpClient
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool, BackendSpec
from medmcp.reasoning import ThoughtStripper
from medmcp.workspace_note import build_workspace_note, display_content_text, strip_workspace_note

_audit: logging.Logger = logging.getLogger("medmcp.audit")
log: logging.Logger = logging.getLogger(__name__)

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
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".uv_cache",
    }
)
_TREE_MAX_DEPTH: int = 12
_TREE_MAX_ENTRIES_PER_DIR: int = 500

# ── Persistent stack pool + broker (Layer 1; MEDMCP_STACK_POOL) ───────────────
# Created at startup only when the pool is enabled; None otherwise (the legacy
# per-call spawn). The pool lives in THIS process, so warm backends survive the
# vibe-acp restarts that stack/workflow changes trigger.
_pool: BackendPool | None = None
_broker: BackendBroker | None = None


def _resolve_backend_spec(name: str) -> BackendSpec | None:
    """Map an active stack name to its persistent-backend launch spec (pool callback)."""
    entry = settings.build_backend_registry(settings.active_servers()).get(name)
    if entry is None:
        return None
    return BackendSpec(
        name=name,
        command=str(entry["command"]),
        args=[str(a) for a in cast("list[Any]", entry["args"])],
        env={str(k): str(v) for k, v in cast("JsonDict", entry["env"]).items()},
        gpu=bool(entry["gpu"]),
        idle_ttl_sec=float(entry["idle_ttl_sec"]),
        startup_timeout_sec=float(entry["startup_timeout_sec"]),
        tool_timeout_sec=float(entry["tool_timeout_sec"]),
    )


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Start the stack pool + broker on boot (when enabled); tear them down on exit."""
    global _pool, _broker
    if settings.stack_pool_enabled():
        # Proxy children read MEDMCP_WORKSPACE for their fallback cwd; export it.
        os.environ.setdefault("MEDMCP_WORKSPACE", str(WORKSPACE_ROOT))
        _pool = BackendPool(resolve_spec=_resolve_backend_spec, cwd=str(WORKSPACE_ROOT))
        _broker = BackendBroker(_pool, settings.backend_socket_path())
        try:
            await _broker.start()
            # Proxied config + backends.json exist from boot, before any tool call.
            await asyncio.to_thread(settings.sync_servers_to_vibe_config, settings.active_servers())
            log.info("stack pool enabled; broker at %s", settings.backend_socket_path())
        except Exception:
            log.exception("backend broker failed to start; proxy will direct-spawn")
    try:
        yield
    finally:
        if _broker is not None:
            await _broker.aclose()
        if _pool is not None:
            await _pool.aclose()
        _broker = None
        _pool = None


app = FastAPI(title="MedMCP Workspace", lifespan=_lifespan)

# One vibe-acp subprocess shared by every websocket connection. The subprocess
# cwd must stay PROJECT_ROOT — `uv run` resolves the project from it; the
# agent's working directory is set per session via session/new's cwd instead.
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

    ``stacks`` carries every entry the drawer knew about (name + active), so the
    server can tell "deactivated" apart from "unknown to this drawer" and
    preserve the state of stacks installed after the drawer fetched.
    """

    explain_tools: bool
    record_provenance: bool
    # Selected GPU (CDI device id) for container stacks; "" = leave unchanged.
    gpu: str = ""
    stacks: list[ToggleEntry]


def _settings_state() -> JsonDict:
    """Assemble the current settings state (runs blocking discovery)."""
    stacks = settings.load_mcp_servers()
    active = settings.load_active_server_names()
    return {
        "explain_tools": settings.load_explain_enabled(),
        "record_provenance": settings.load_provenance_enabled(),
        "gpu": settings.load_gpu_selection(),
        "llm_gpu": settings.LLM_GPU,
        "stacks": [
            {"name": s["name"], "version": s.get("version"), "active": s["name"] in active}
            for s in stacks
        ],
    }


@app.get("/api/settings")
async def get_settings() -> JsonDict:
    """Return toggles plus the discovered stacks with active state."""
    return await asyncio.to_thread(_settings_state)


@app.put("/api/settings")
async def put_settings(payload: SettingsPayload) -> JsonDict:
    """Persist settings; restart vibe-acp when its config inputs changed.

    Stack changes are baked into ``.vibe/config.toml`` at session start, so
    applying them requires a fresh vibe-acp process. All live chat sockets are
    closed; each client auto-reconnects into a new session.
    """

    def _apply() -> tuple[bool, set[str], set[str]]:
        old_stacks = settings.load_active_server_names()

        # An empty gpu means "leave unchanged"; a new value re-pins stack containers.
        gpu_changed = bool(payload.gpu) and payload.gpu != settings.load_gpu_selection()
        if gpu_changed:
            settings.save_gpu_selection(payload.gpu)

        settings.save_explain_enabled(payload.explain_tools)
        settings.save_provenance_enabled(payload.record_provenance)
        # Merge instead of overwrite: entries the drawer never saw keep their
        # current active state instead of being silently deactivated.
        known_stacks = {t.name for t in payload.stacks}
        new_stacks = {t.name for t in payload.stacks if t.active} | (old_stacks - known_stacks)
        settings.save_active_server_names(new_stacks)

        restart = new_stacks != old_stacks or gpu_changed
        if restart:
            settings.sync_servers_to_vibe_config(settings.active_servers())
        return restart, new_stacks - old_stacks, old_stacks - new_stacks

    restart_needed, newly_active, newly_inactive = await asyncio.to_thread(_apply)
    if restart_needed:
        _audit.info("settings changed; restarting vibe-acp")
        await _restart_vibe()
    await _apply_pool_changes(newly_active, newly_inactive)
    return {"ok": True, "restarted": restart_needed}


async def _apply_pool_changes(newly_active: set[str], newly_inactive: set[str]) -> None:
    """Evict deactivated stacks and pre-warm newly-activated ones (pool enabled only)."""
    if _pool is None:
        return
    for name in newly_inactive:
        with contextlib.suppress(Exception):
            await _pool.evict(name)
    if newly_active:
        # Pre-warm in the background so the settings response stays snappy; the
        # heavy spawn/import/CUDA cost is then paid before the first real call.
        task = asyncio.create_task(_pool.prewarm(newly_active))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)


async def _restart_vibe() -> None:
    """Stop the shared vibe-acp process and drop every live chat socket."""
    await _client.stop()
    for conn in list(_connections):
        with contextlib.suppress(Exception):
            await conn.ws.close()


class StackInstallPayload(BaseModel):
    """Request body for installing a container stack from an image."""

    image: str


class StackUninstallPayload(BaseModel):
    """Request body for uninstalling a container stack by name."""

    name: str


def _apply_stack_change() -> None:
    """Re-discover stacks and re-sync vibe-acp config after an install/uninstall."""
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())


@app.get("/healthz")
async def healthz() -> JsonDict:
    """Liveness probe for container healthchecks (touches no dependencies)."""
    return {"status": "ok"}


@app.get("/api/gpus")
async def get_gpus() -> JsonDict:
    """Best-effort GPU list for the settings picker (empty if not enumerable)."""
    return {"gpus": await asyncio.to_thread(settings.list_gpus)}


@app.get("/api/stacks")
async def get_stacks() -> JsonDict:
    """List installed container stacks (from ``stacks.d`` manifests)."""
    return {"stacks": await asyncio.to_thread(settings.list_installed_stacks)}


@app.get("/api/catalog")
async def get_catalog() -> JsonDict:
    """Return the curated install catalog; each entry flagged whether it's installed."""

    def _build() -> list[JsonDict]:
        installed = {s["name"] for s in settings.list_installed_stacks()}
        return [{**e, "installed": e["name"] in installed} for e in settings.load_catalog()]

    return {"catalog": await asyncio.to_thread(_build)}


@app.post("/api/stacks/install")
async def post_stack_install(payload: StackInstallPayload) -> JsonDict:
    """Install a container stack from an image, then reload vibe-acp.

    Pulls the image if needed, reads its ``org.medmcp.stack`` label, extracts its
    skills, writes the ``stacks.d`` manifest, and restarts vibe-acp so the stack
    is available. The image is inspected, never executed, to read the label.
    """
    try:
        name = await asyncio.to_thread(settings.install_stack_image, payload.image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await asyncio.to_thread(_apply_stack_change)
    _audit.info("stack installed: %s (%s)", name, payload.image)
    await _restart_vibe()
    return {"name": name, "restarted": True}


@app.post("/api/stacks/uninstall")
async def post_stack_uninstall(payload: StackUninstallPayload) -> JsonDict:
    """Uninstall a container stack by name, then reload vibe-acp."""
    try:
        await asyncio.to_thread(settings.uninstall_stack, payload.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await asyncio.to_thread(_apply_stack_change)
    _audit.info("stack uninstalled: %s", payload.name)
    await _restart_vibe()
    return {"ok": True, "restarted": True}


@app.websocket("/ws/stacks/install")
async def ws_stack_install(ws: WebSocket) -> None:
    """Install a stack with streamed progress.

    First client message: ``{"image": "..."}``. Streams
    ``{"type":"progress","line":...}`` frames during the pull/extract, then a
    final ``{"type":"done","name":...}`` or ``{"type":"error","message":...}``.
    On success it reloads discovery and restarts vibe-acp (same as the POST path).
    """
    await ws.accept()
    try:
        first = cast("JsonDict", await ws.receive_json())
    except WebSocketDisconnect:
        return
    image = str(first.get("image", "")).strip()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[JsonDict] = asyncio.Queue()

    def on_progress(line: str) -> None:
        # Called from the install worker thread; hop back onto the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "line": line})

    async def run() -> None:
        try:
            name = await asyncio.to_thread(settings.install_stack_image, image, on_progress)
            await asyncio.to_thread(_apply_stack_change)
            _audit.info("stack installed: %s (%s)", name, image)
            await _restart_vibe()
            await queue.put({"type": "done", "name": name})
        except Exception as exc:  # relayed to the client as an error frame
            await queue.put({"type": "error", "message": str(exc)})

    task = asyncio.create_task(run())
    try:
        while True:
            frame = await queue.get()
            await ws.send_json(frame)
            if frame["type"] in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await task
        with contextlib.suppress(Exception):
            await ws.close()


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


def _requirement_statuses(recipe: workflow.Recipe, servers: list[JsonDict]) -> list[JsonDict]:
    """Each requirement enriched with its availability against the installed stacks.

    ``status`` is ``"missing"`` (the stack isn't installed), ``"mismatch"`` (it is,
    but the locally-present image digest differs from the pinned one — results may
    not reproduce), or ``"ok"``. Offline (reads already-present image digests) and
    best-effort: a digest it can't resolve is treated as ``"ok"`` rather than a
    false alarm.
    """
    by_name = {str(s.get("name")): s for s in servers}
    statuses: list[JsonDict] = []
    for req in recipe.requires:
        entry = req.to_dict()
        server = by_name.get(req.stack)
        if server is None:
            entry["status"] = "missing"
        else:
            image = provenance.docker_image_ref(server)
            local = settings.resolve_image_digest(image) if image else None
            if local:
                entry["installed_digest"] = local
            entry["status"] = "mismatch" if req.digest and local and req.digest != local else "ok"
        statuses.append(entry)
    return statuses


def _workflow_detail(name: str) -> JsonDict:
    """Load a workflow's recipe and replayability state (blocking; run in a thread)."""
    d = _workflow_dir(name)
    if d is None:
        raise FileNotFoundError(f"no workflow named {name!r}")
    recipe = distill.load_recipe(d)
    examples = {i.name: i.example for i in recipe.inputs}
    servers = settings.active_servers()
    replay_error = replay.validate(recipe, examples, servers)
    return {
        "name": recipe.name,
        "kind": d.parent.name,
        "description": recipe.description,
        "inputs": [i.to_dict() for i in recipe.inputs],
        "steps": [
            {"server": s.server, "tool": s.tool, "arguments": s.arguments} for s in recipe.steps
        ],
        "requires": _requirement_statuses(recipe, servers),
        "manual_steps": list(recipe.manual_steps),
        "replayable": replay_error is None,
        "replay_error": replay_error,
    }


@app.get("/api/workflows")
async def get_workflows() -> JsonDict:
    """List the personal workflows available to the replay engine."""
    return {"workflows": await asyncio.to_thread(settings.discover_workflows)}


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
        draft_dir = await asyncio.to_thread(
            lambda: distill.distill_session(
                payload.session_id, chain_stop_ids=_chain_stop_ids(payload.session_id)
            )
        )
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


@app.get("/api/workflows/{name}/export")
async def get_workflow_export(name: str) -> Response:
    """Export a workflow as a single self-contained ``.workflow.yaml`` download."""
    try:
        text = await asyncio.to_thread(share.export_workflow, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"{name}{share.EXPORT_SUFFIX}"
    return Response(
        content=text,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class WorkflowImportPayload(BaseModel):
    """Request body for importing a shared workflow file (its YAML text)."""

    content: str


@app.post("/api/workflows/import")
async def post_import_workflow(payload: WorkflowImportPayload) -> JsonDict:
    """Import a shared workflow envelope as a reviewable draft; return its detail."""
    try:
        draft_dir = await asyncio.to_thread(share.import_workflow, payload.content)
    except share.WorkflowShareError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit.info("workflow imported: %s", draft_dir.name)
    return await asyncio.to_thread(_workflow_detail, draft_dir.name)


def _resolve_input_path(value: str) -> str:
    """Rewrite a workspace-relative replay input to its absolute on-disk path.

    Workflow inputs reach the server as workspace-relative paths — the explorer
    tree ids and the drag payloads that populate the Run form are all relative to
    ``WORKSPACE_ROOT``. Replay, however, calls the stack tools directly and they
    resolve paths on disk, where the workspace is bind-mounted at the identical
    absolute path (path parity). A relative value naming an existing file or
    directory under the workspace is rewritten to that absolute path; an
    already-absolute path, a non-path argument (e.g. ``"cuda"``), or a value that
    doesn't exist on disk is returned unchanged — so non-path inputs pass through
    and a genuinely missing scan still surfaces as a clear tool error.
    """
    if not value or Path(value).is_absolute():
        return value
    candidate = (WORKSPACE_ROOT / value).resolve()
    if candidate.is_relative_to(WORKSPACE_ROOT) and candidate.exists():
        return str(candidate)
    return value


def _resolve_input_paths(inputs: dict[str, str]) -> dict[str, str]:
    """Apply :func:`_resolve_input_path` to every value of a replay input binding."""
    return {key: _resolve_input_path(value) for key, value in inputs.items()}


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
        inputs = _resolve_input_paths(dict(payload.inputs))
        error = replay.validate(recipe, inputs, settings.active_servers())
        if error is not None:
            return {"ok": False, "error": error, "steps": []}
        bindings: dict[str, Any] = dict(inputs)
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


class BatchFromPlanPayload(BaseModel):
    """Request body: a workspace-relative ``plan_batch`` manifest CSV."""

    plan_csv: str


@app.post("/api/workflows/{name}/batch-from-plan")
async def post_batch_from_plan(name: str, payload: BatchFromPlanPayload) -> JsonDict:
    """Turn a ``plan_batch`` manifest into ready-to-run batch bindings for a workflow.

    Reads the cohort manifest CSV (written by the cohort stack's ``plan_batch``),
    maps each ``ok`` row's resolved files onto the recipe's ``{{in_N}}`` inputs, and
    returns the per-subject runs that pre-fill the batch editor — so the user
    reviews the binding list instead of hand-entering one row per subject. Flagged
    rows (missing/ambiguous) are returned separately to be resolved, never run.
    """

    def _build() -> JsonDict:
        d = _workflow_dir(name)
        if d is None:
            raise FileNotFoundError(f"no workflow named {name!r}")
        recipe = distill.load_recipe(d)
        try:
            rows = batchplan.read_manifest(_resolve_input_path(payload.plan_csv))
        except OSError as exc:
            return {"ok": False, "error": f"cannot read plan: {exc}", "runs": [], "skipped": []}
        try:
            binding = batchplan.runs_from_manifest(recipe, rows)
        except batchplan.BatchPlanError as exc:
            return {"ok": False, "error": str(exc), "runs": [], "skipped": []}
        return {
            "ok": True,
            "error": None,
            "runs": [_resolve_input_paths(r) for r in binding.runs],
            "skipped": binding.skipped,
            "column_map": binding.column_map,
        }

    try:
        return await asyncio.to_thread(_build)
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
            _resolve_input_paths({str(k): str(v) for k, v in cast("JsonDict", r).items()})
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
            async def _on_step(item: int, sr: replay.StepResult) -> None:
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

            async def _on_item(item: int, result: replay.ReplayResult) -> None:
                await ws.send_json(
                    {
                        "type": "item_result",
                        "item": item,
                        "ok": result.ok,
                        "error": result.error,
                        "outputs": [v for sr in result.steps for v in sr.produced.values()],
                    }
                )

            # One shared set of stacks across all items (spawned once, not per item).
            results = await replay.run_batch(
                recipe,
                runs,
                servers=servers,
                cwd=str(WORKSPACE_ROOT),
                on_step=_on_step,
                on_item=_on_item,
            )
            all_outputs = [v for r in results for sr in r.steps for v in sr.produced.values()]
            failed = sum(1 for r in results if not r.ok)
            ok = failed == 0
            batch_error = None if ok else f"{failed} of {len(runs)} item(s) failed"
            _audit.info("replay finished: %s ok=%s", name, ok)
            await ws.send_json(
                {"type": "result", "ok": ok, "error": batch_error, "outputs": all_outputs}
            )

        # Race the batch against a socket watcher. The client sends nothing
        # after the first message, so anything receive returns — and above all
        # a disconnect — means "abort". Cancelling the batch task unwinds
        # run_batch's shared exit stack, killing the in-flight MCP server and its
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


# The viewer-context note sent with a prompt in _run_prompt. Its format and the
# pattern that strips it live in medmcp.workspace_note (shared with distill.py, the
# other consumer). Turns sent by vibe ≥2.23 replay with the note-free text in the
# frame's user_display_content meta; older transcripts (and titles derived from
# the stored text) still need _strip_workspace_note.
_strip_workspace_note = strip_workspace_note


def _replayed_user_text(update: JsonDict) -> str:
    """Extract the display text for a replayed ``user_message_chunk`` frame.

    Prefers the ``user_display_content`` meta (vibe ≥2.23 echoes back the
    note-free text we sent with the prompt); falls back to stripping the
    workspace note from the stored message text for older transcripts.
    """
    meta = update.get("_meta")
    if isinstance(meta, dict):
        display = cast("JsonDict", meta).get("user_display_content")
        if isinstance(display, dict):
            text = display_content_text(cast("JsonDict", display))
            if text:
                return text
    content = cast("JsonDict", update.get("content") or {})
    if content.get("type") != "text":
        return ""
    return _strip_workspace_note(str(content.get("text") or ""))


def _usage_window(update: JsonDict) -> int:
    """Pick the context-window size for a usage frame (no I/O).

    Ollama's ``num_ctx`` is the deployment truth, so the fetched value wins
    (vibe's ``size`` comes from its model registry — e.g. 200k for a Gemma
    served with a 131072 window). The frame's size only stands in before the
    first successful fetch, ahead of the static default. Never an inline
    fetch here: it would stall the relay of every queued frame behind an
    Ollama round-trip; the cache is warmed at connect time in ws_chat.
    """
    fetched = settings.fetched_context_window()
    if fetched is not None:
        return fetched
    size_raw = update.get("size")
    if isinstance(size_raw, int) and size_raw > 0:
        return size_raw
    return settings.cached_context_window()


# Permission options that approve more than the call being asked about:
# ``allow_always`` ("Allow for remainder of this session") auto-approves every
# later call of that tool in the session, and ``allow_always_permanent``
# ("Always allow") persists the approval into ``.vibe/config.toml`` so it
# outlives the session entirely. Both are auto-approval paths, which medmcp
# does not offer — every tool call is gated on its own.
_AUTO_APPROVE_OPTIONS = frozenset({"allow_always", "allow_always_permanent"})


def _visible_permission_options(options: list[JsonDict]) -> list[JsonDict]:
    """Reduce vibe's permission options to a per-call allow/deny pair.

    Every option that would auto-approve anything beyond the call in hand is
    dropped (see :data:`_AUTO_APPROVE_OPTIONS`), leaving ``allow_once`` and
    ``reject_once``. With no "always" variant left to contrast against,
    "Allow once" is relabelled to plain **"Allow"** — the scope is no longer a
    choice the user makes, so naming it only invites the question.

    Renaming here rather than in the browser keeps one source for the label:
    the frontend renders whatever ``name`` the frame carries.
    """
    visible: list[JsonDict] = []
    for option in options:
        if option.get("optionId") in _AUTO_APPROVE_OPTIONS:
            continue
        if option.get("optionId") == "allow_once":
            option = {**option, "name": "Allow"}
        visible.append(option)
    return visible


def _workspace_note(viewed_path: str) -> str:
    """Build the ``[workspace context: …]`` note for the file open in the viewer.

    The viewer reports a workspace-relative path, but the stack tools run in
    sibling containers and resolve paths on disk, where the workspace is
    bind-mounted at the identical absolute path (path parity). Handing the agent
    the **absolute** path lets its first tool call hit; a relative path misses and
    only recovers after the agent searches the filesystem. Resolution mirrors
    replay's :func:`_resolve_input_path` (an unknown/absolute value is unchanged);
    the note's text format comes from :func:`build_workspace_note`.
    """
    return build_workspace_note(_resolve_input_path(viewed_path))


class _ChatConnection:
    """State for one browser websocket: one vibe-acp session, one prompt at a time."""

    def __init__(
        self,
        ws: WebSocket,
        session_id: str,
        queue: asyncio.Queue[JsonDict],
        servers: list[JsonDict],
        *,
        resumed: bool = False,
        canonical_id: str | None = None,
    ) -> None:
        """Bind the websocket to its registered session queue.

        ``servers`` is the active-server list captured at connect time; a
        stack change restarts vibe-acp and closes every connection, so it
        cannot go stale within a connection's lifetime. ``resumed`` marks a
        session reattached via ``session/load`` — it already has a transcript,
        so it must not be purged as an abandoned empty session on close.
        ``canonical_id`` is the chain-root id the browser and provenance key
        on; it differs from ``session_id`` (the vibe RPC target) when a resume
        was mapped onto a compaction continuation (see ``ws_chat``).
        """
        self.ws = ws
        self.session_id = session_id
        self.canonical_id = canonical_id or session_id
        self.queue = queue
        self.servers = servers
        self._resumed = resumed
        # Relays unsolicited frames before the first prompt (see start_idle_pump).
        self._idle_pump: asyncio.Task[None] | None = None
        self._pending_perms: dict[int, asyncio.Future[str | None]] = {}
        self._prompt_task: asyncio.Task[None] | None = None
        # Tool-call state accumulated across frames, keyed by toolCallId; feeds
        # the permission dialog backfill and the provenance run log.
        self._tool_calls: dict[str, JsonDict] = {}
        # Strips the local model's inline <thought>…</thought> reasoning out of the
        # streamed agent text before it reaches the browser (it isn't a separate
        # ACP thought channel, so without this it renders as normal chat content).
        self._thoughts = ThoughtStripper()
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
                # The warm-up pump must not race the turn loop for the queue.
                await self._stop_idle_pump()
                # A new prompt while one is streaming cancels the old one.
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
                await self._stop_idle_pump()
                await self._cancel_prompt()
                # A cancelled task no longer emits its own `done` (a stale one
                # would clobber a newer turn's state), so an intentional Stop
                # resets the client explicitly.
                await self._send({"type": "done"})

    async def close(self) -> None:
        """Abort any in-flight prompt and drop the session queue.

        A fresh session that never received a prompt is purged (transcript +
        provenance) so abandoned tabs/refreshes don't leak session state. A
        resumed session is never purged on close — it already has history the
        user came back to view.
        """
        await self._stop_idle_pump()
        await self._cancel_prompt()
        for task in list(self._explain_tasks):
            task.cancel()
        _client.unregister_session(self.session_id)
        if not self._prompted and not self._resumed:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    lambda: provenance.purge_session(
                        self.canonical_id, stop_ids=_chain_stop_ids(self.canonical_id)
                    )
                )

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

    async def replay_history(self) -> None:
        """Relay the transcript frames vibe re-emitted during ``session/load``.

        Those ``session/update`` notifications are written before the load
        response resolves, so by the time we get here they are already sitting
        in the session queue. Draining it non-blockingly yields exactly the
        historical conversation, which we forward through the normal translator
        in ``replay`` mode (no provenance re-writes, no permission prompts —
        replayed tool calls are already settled).
        """
        while True:
            try:
                msg = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._forward_frame(msg, replay=True)

    def start_idle_pump(self) -> None:
        """Relay unsolicited frames until the first prompt arrives.

        vibe ≥2.21 pushes messages outside any turn — MCP discovery-failure and
        OAuth notices on session warm-up, plus (on resume) frames landing after
        ``replay_history``'s non-blocking drain. Without a pump they would sit
        in the queue until the first prompt, where ``_drain_stale_frames``
        discards them. The pump stops at the first prompt (or Stop) and never
        restarts: between turns the queue must stay drainable, because frames
        trickling in after a cancel belong to the dead turn and must not be
        relayed as fresh output.
        """
        self._idle_pump = asyncio.create_task(self._pump_frames())

    async def _pump_frames(self) -> None:
        while True:
            await self._forward_frame(await self.queue.get())

    async def _stop_idle_pump(self) -> None:
        task = self._idle_pump
        self._idle_pump = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run_prompt(self, text: str, viewed_path: str | None = None) -> None:
        """Send one ``session/prompt`` and stream its frames to the browser.

        ``viewed_path`` is the workspace-relative file currently open in the
        viewer; it is resolved to its absolute on-disk path and sent as a
        context-note content block (see :func:`_workspace_note`) so the agent can
        resolve references like "this image" *and* pass the path the stack tools
        expect. The block's ``automatic`` meta keeps it out of vibe's auto-title
        derivation, and the prompt's ``user_display_content`` meta records the
        note-free text vibe echoes back on transcript replay.

        Race loop: wait for either the next inbound session frame or the
        prompt response, then drain whatever is left in the queue once the
        response lands.
        """
        if not self._prompted:
            self._prompted = True
            if settings.load_provenance_enabled():
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        lambda: provenance.write_manifest(
                            self.canonical_id,
                            servers=self.servers,
                            model_name=settings.OLLAMA_MODEL,
                        )
                    )
        # Frames can still trickle in between a cancel and this prompt
        # (vibe-acp processes session/cancel asynchronously); drop them now so
        # the old turn can't bleed into this one.
        await self._drain_stale_frames()
        params: JsonDict = {
            "session_id": self.session_id,
            "prompt": [{"type": "text", "text": text}],
        }
        if viewed_path is not None:
            # The note rides as its own content block flagged `automatic`: vibe
            # orders it after the user's text and keeps it out of auto-title
            # derivation. lstrip() because vibe joins blocks with a blank line —
            # the persisted text stays byte-identical to the old appended format,
            # so strip_workspace_note keeps working on every transcript vintage.
            cast("list[JsonDict]", params["prompt"]).append(
                {
                    "type": "text",
                    "text": _workspace_note(viewed_path).lstrip(),
                    "_meta": {"automatic": True},
                }
            )
            # user_display_content is persisted with the turn and echoed back in
            # replayed user_message_chunk frames, so resumed transcripts show the
            # user's text without the note (no stripping needed on that path).
            params["_meta"] = {
                "user_display_content": {
                    "version": "1",
                    "host": "medmcp",
                    "content": [{"type": "text", "text": text}],
                }
            }
        prompt_fut = asyncio.create_task(_client.request("session/prompt", params))
        self._thoughts.reset()  # start each turn with a clean reasoning-strip state
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
            tail = self._thoughts.flush()  # emit any real text held back at a tag boundary
            if tail:
                await self._send({"type": "chunk", "text": tail})
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

    async def _forward_frame(self, msg: JsonDict, *, replay: bool = False) -> None:
        """Translate one inbound vibe-acp frame into a browser message.

        ``replay`` is set while relaying the transcript vibe re-emits during
        ``session/load`` — those tool calls already ran, so their provenance was
        recorded the first time and must not be written again.
        """
        method = msg.get("method")
        if method == "session/update":
            params = cast("JsonDict", msg.get("params") or {})
            update = cast("JsonDict", params.get("update") or {})
            update_type = update.get("sessionUpdate")
            if update_type == "agent_message_chunk":
                content = cast("JsonDict", update.get("content") or {})
                if content.get("type") == "text":
                    visible = self._thoughts.feed(str(content.get("text") or ""))
                    if visible:
                        await self._send({"type": "chunk", "text": visible})
            elif update_type == "user_message_chunk":
                # Replayed user turns (session/load) — and, since vibe 2.23, an
                # echo of the *live* prompt carrying its messageId; the browser
                # merges that echo into the locally-rendered bubble.
                text = _replayed_user_text(update)
                if text:
                    frame: JsonDict = {"type": "user", "text": text}
                    # The message id anchors per-turn actions (rewind targets).
                    mid = update.get("messageId")
                    if isinstance(mid, str) and mid:
                        frame["messageId"] = mid
                    await self._send(frame)
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
                    if not replay and settings.load_provenance_enabled():
                        event_info = info
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(
                                lambda: provenance.record_tool_event(
                                    self.canonical_id,
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
                    await self._send({"type": "usage", "used": used, "size": _usage_window(update)})
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
        options = _visible_permission_options(cast("list[JsonDict]", params.get("options") or []))
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
                provenance.log_permission(self.canonical_id, title=str(title), decision=decision)

        with contextlib.suppress(Exception):
            await asyncio.to_thread(_mirror)
        await _client.respond(req_id, {"outcome": outcome})


# ── Sessions API ───────────────────────────────────────────


def _chain_stop_ids(session_id: str) -> set[str]:
    """Registry ids that end a chain walk from *session_id* (reads a file).

    A fork carries the same ``parent_session_id`` backlink as a compaction
    continuation; its registry entry (created at fork time) marks it as its
    own chat, so the resume tip-mapping, purge, and distillation must not
    walk into it. The walked session's own entry (e.g. a renamed root) never
    stops its own chain. File I/O — call via ``asyncio.to_thread``.
    """
    return set(sessions.load_registry()) - {session_id}


def _chain_root(sid: str, parents: dict[str, str], listed: set[str]) -> str:
    """Walk ``parent_session_id`` backlinks up to the topmost *listed* ancestor."""
    seen: set[str] = set()
    cur = sid
    while cur in parents and cur not in seen:
        seen.add(cur)
        parent = parents[cur]
        if parent not in listed:
            break
        cur = parent
    return cur


def _merge_session_registry(raw: list[JsonDict]) -> list[JsonDict]:
    """Overlay UI metadata (title override, archived, provenance) onto vibe's list.

    Compaction rolls a chat over to a fresh session dir, so vibe lists one chat
    as several sessions. Continuations — entries whose ``parent_session_id``
    chain reaches another listed session — are folded into their root: hidden
    from the list, with the root carrying the newest ``updatedAt`` of the chain.
    An entry with a UI-registry record is never folded (registry entries mean
    the user sees that chat as its own — e.g. a deliberate fork — while pure
    continuations never acquire one).

    Runs off the event loop: it reads the registry file, scans session metas,
    and stats a provenance directory per session.
    """
    registry = sessions.load_registry()
    parents = provenance.vibe_session_parents()
    ids = {
        sid
        for s in raw
        if isinstance(sid := (s.get("sessionId") or s.get("session_id")), str) and sid
    }
    # First pass: fold continuations into their roots, keeping the newest stamp.
    newest_updated: dict[str, str] = {}
    hidden: set[str] = set()
    for s in raw:
        sid = s.get("sessionId") or s.get("session_id")
        if not (isinstance(sid, str) and sid) or sid in registry:
            continue
        root = _chain_root(sid, parents, ids)
        if root == sid:
            continue
        hidden.add(sid)
        updated = s.get("updatedAt") or s.get("updated_at")
        if isinstance(updated, str) and updated > newest_updated.get(root, ""):
            newest_updated[root] = updated
    out: list[JsonDict] = []
    for s in raw:
        sid = s.get("sessionId") or s.get("session_id")
        if not (isinstance(sid, str) and sid) or sid in hidden:
            continue
        entry = registry.get(sid, {})
        override = entry.get("title")
        if isinstance(override, str) and override:
            title = override
        else:
            raw_title = s.get("title")
            title = _strip_workspace_note(str(raw_title)) if raw_title else ""
        updated = s.get("updatedAt") or s.get("updated_at")
        folded = newest_updated.get(sid)
        if folded is not None and (not isinstance(updated, str) or folded > updated):
            updated = folded
        out.append(
            {
                "id": sid,
                "title": title or None,
                "updatedAt": updated,
                "archived": bool(entry.get("archived")),
                "hasProvenance": provenance.provenance_dir(sid).is_dir(),
            }
        )
    return out


@app.get("/api/sessions")
async def get_sessions() -> JsonDict:
    """List prior chat sessions for this workspace (vibe-acp ``session/list``).

    vibe-acp persists every session's transcript and can reload it; this surfaces
    the ones whose cwd is the current workspace, overlaid with the UI registry
    (custom title, archived flag) so the Chats drawer can manage them.
    """
    await _client.ensure_started()
    resp = await _client.request("session/list", {"cwd": str(WORKSPACE_ROOT)})
    if "error" in resp:
        raise HTTPException(status_code=502, detail=str(resp["error"]))
    result = cast("JsonDict", resp.get("result") or {})
    raw = cast("list[JsonDict]", result.get("sessions") or [])
    merged = await asyncio.to_thread(_merge_session_registry, raw)
    return {"sessions": merged}


class RenameSessionPayload(BaseModel):
    """Request body for setting a session's display title."""

    title: str


@app.post("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, payload: RenameSessionPayload) -> JsonDict:
    """Set (or clear, when blank) the user title override for a session."""
    await asyncio.to_thread(sessions.set_title, session_id, payload.title)
    # Write the title through to vibe's own session metadata (ext method, vibe
    # ≥2.23; ACP extension methods are underscore-prefixed on the wire) so it
    # also shows up wherever vibe surfaces the session. Targets the chain tip —
    # that is the session vibe has live/attached after a compaction. Best-effort:
    # the registry override above is what the UI reads, and vibe may not be
    # running (it starts lazily with the first chat socket).
    title = payload.title.strip()
    if title:
        with contextlib.suppress(Exception):
            tip = await asyncio.to_thread(
                lambda: provenance.vibe_chain_tip(session_id, stop_ids=_chain_stop_ids(session_id))
            )
            await _client.request("_session/set_title", {"sessionId": tip, "title": title})
    return {"ok": True}


@app.post("/api/sessions/{session_id}/fork")
async def fork_session(session_id: str) -> JsonDict:
    """Branch the live chat: fork its whole history into a new session.

    Uses vibe's ``session/fork`` (vibe ≥2.23), which copies the transcript into
    a fresh session without attaching it; the browser then opens the fork via
    the normal resume flow. The fork is registered in the UI session registry
    immediately — the entry names it *and* keeps the chain-collapse folding
    from hiding it (forks carry the same ``parent_session_id`` backlink as
    compaction continuations). Requires the chat to be open (vibe only forks
    live sessions); targets the chain tip, the id vibe holds live.
    """
    await _client.ensure_started()
    tip = await asyncio.to_thread(
        lambda: provenance.vibe_chain_tip(session_id, stop_ids=_chain_stop_ids(session_id))
    )
    resp = await _client.request(
        "session/fork", {"sessionId": tip, "cwd": str(WORKSPACE_ROOT), "mcpServers": []}
    )
    if "error" in resp:
        err = cast("JsonDict", resp["error"])
        raise HTTPException(status_code=409, detail=str(err.get("message", err)))
    result = cast("JsonDict", resp.get("result") or {})
    new_id = str(result.get("sessionId") or "")
    if not new_id:
        raise HTTPException(status_code=502, detail="fork returned no session id")
    _audit.info("chat branched: %s -> %s", session_id, new_id)

    def _register() -> None:
        src_title = sessions.load_registry().get(session_id, {}).get("title")
        if not (isinstance(src_title, str) and src_title):
            # No user title — inherit vibe's auto-title (note-stripped: old
            # transcripts derived titles from the raw prompt text).
            auto = provenance.vibe_session_title(tip)
            src_title = _strip_workspace_note(auto) if auto else None
        title = f"{src_title} (branch)" if src_title else "Branch"
        sessions.set_title(new_id, title)

    await asyncio.to_thread(_register)
    return {"id": new_id}


class RewindPayload(BaseModel):
    """Request body for previewing/performing a chat rewind."""

    message_id: str = Field(alias="messageId")
    preview: bool = False
    restore_files: bool = Field(default=True, alias="restoreFiles")


@app.post("/api/sessions/{session_id}/rewind")
async def rewind_session(session_id: str, payload: RewindPayload) -> JsonDict:
    """Preview or perform an in-place rewind of a live chat (vibe ≥2.23 ext method).

    ``preview`` returns the workspace files a rewind would restore; without it
    the conversation is truncated to before ``messageId`` (and, with
    ``restoreFiles``, the files are restored). Requires the chat to be open —
    vibe only rewinds live sessions — and targets the chain tip (the id vibe
    holds live). The UI must confirm with the user before the non-preview call:
    it truncates conversation history and rewrites workspace files.
    """
    await _client.ensure_started()
    tip = await asyncio.to_thread(
        lambda: provenance.vibe_chain_tip(session_id, stop_ids=_chain_stop_ids(session_id))
    )
    if payload.preview:
        method = "_rewind/preview"
        params: JsonDict = {"sessionId": tip, "messageId": payload.message_id}
    else:
        _audit.info("rewind requested: %s -> message %s", session_id, payload.message_id)
        method = "_rewind/to"
        params = {
            "sessionId": tip,
            "messageId": payload.message_id,
            "restoreFiles": payload.restore_files,
        }
    resp = await _client.request(method, params)
    if "error" in resp:
        err = cast("JsonDict", resp["error"])
        raise HTTPException(status_code=409, detail=str(err.get("message", err)))
    if not payload.preview:
        _audit.info("rewind performed: %s -> message %s", session_id, payload.message_id)
    return cast("JsonDict", resp.get("result") or {})


class ArchiveSessionPayload(BaseModel):
    """Request body for archiving/restoring a session."""

    archived: bool


@app.post("/api/sessions/{session_id}/archive")
async def archive_session(session_id: str, payload: ArchiveSessionPayload) -> JsonDict:
    """Archive a session (hide it from the default list) or restore it."""
    await asyncio.to_thread(sessions.set_archived, session_id, payload.archived)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> JsonDict:
    """Delete a session for good: its transcript, provenance, and UI metadata."""
    # Let vibe drop the session first (ext method, vibe ≥2.23: closes a live
    # attachment and deletes its stored copy). Targets the chain tip — after a
    # compaction that is the id vibe holds live. Best-effort — the purge below
    # sweeps whatever remains on disk (provenance plus the whole transcript
    # chain) either way.
    with contextlib.suppress(Exception):
        tip = await asyncio.to_thread(
            lambda: provenance.vibe_chain_tip(session_id, stop_ids=_chain_stop_ids(session_id))
        )
        await _client.request("_session/delete", {"sessionId": tip})

    def _delete() -> None:
        provenance.purge_session(session_id, stop_ids=_chain_stop_ids(session_id))
        sessions.remove(session_id)

    await asyncio.to_thread(_delete)
    return {"ok": True}


# ── WebSocket chat ─────────────────────────────────────────


async def _new_session() -> str:
    """Open a fresh vibe-acp session and return its id (``""`` on failure)."""
    resp = await _client.request("session/new", {"cwd": str(WORKSPACE_ROOT), "mcpServers": []})
    if "error" in resp:
        return ""
    result = cast("JsonDict", resp.get("result") or {})
    return str(result.get("sessionId") or "")


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, resume: str | None = None) -> None:
    """Create or resume a vibe-acp session for this socket and run the chat loop.

    With ``?resume=<id>`` the socket reattaches to an existing session via
    ``session/load`` (which restores the agent's context and re-emits the
    transcript); otherwise it opens a fresh one. The config sync before either
    call is mandatory: vibe-acp reads its MCP server list and skill paths from
    ``.vibe/config.toml``, not from the JSON-RPC call.
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

        # Resume: register the queue under the known id *before* session/load so
        # the transcript frames it re-emits are captured, not lost. Fall back to
        # a fresh session if the id is gone/unloadable.
        replayed = False
        session_id = ""
        canonical_id: str | None = None
        queue: asyncio.Queue[JsonDict] | None = None
        if resume:
            # Resume the chain *tip*, not the requested root: after a compaction
            # the root dir only holds the pre-compaction prefix, while the tip
            # carries the summary plus everything since. The browser keeps the
            # root id (canonical) — ready reports it, provenance keys on it —
            # and only the vibe RPC target is the tip.
            tip = await asyncio.to_thread(
                lambda: provenance.vibe_chain_tip(resume, stop_ids=_chain_stop_ids(resume))
            )
            queue = _client.register_session(tip)
            load = await _client.request(
                "session/load",
                {"cwd": str(WORKSPACE_ROOT), "mcpServers": [], "sessionId": tip},
            )
            if "error" in load:
                _client.unregister_session(tip)
                queue = None
            else:
                session_id = tip
                canonical_id = resume
                replayed = True
        if not session_id:
            session_id = await _new_session()
            if not session_id:
                await ws.send_json({"type": "error", "message": "could not open a chat session"})
                await ws.close()
                return
            queue = _client.register_session(session_id)

        assert queue is not None
        conn = _ChatConnection(
            ws, session_id, queue, servers, resumed=replayed, canonical_id=canonical_id
        )
        _connections.add(conn)
        await ws.send_json(
            {"type": "ready", "sessionId": conn.canonical_id, "model": settings.OLLAMA_MODEL}
        )
        if replayed:
            await conn.replay_history()
        conn.start_idle_pump()
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
    """Run the workspace server on localhost (no auth — do not expose).

    Binds 127.0.0.1 by default. In a container set ``MEDMCP_WORKSPACE_HOST=0.0.0.0``
    so the published port is reachable — the no-auth posture is preserved by
    publishing the port only to the host's loopback (``127.0.0.1:8100:8100``).
    """
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("MEDMCP_WORKSPACE_HOST", "127.0.0.1")
    port = int(os.environ.get("MEDMCP_WORKSPACE_PORT", str(DEFAULT_PORT)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
