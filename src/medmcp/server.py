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
from functools import partial
from pathlib import Path
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from medmcp import (
    __version__,
    batchplan,
    distill,
    explain,
    origincheck,
    pathcheck,
    pathguard,
    provenance,
    replay,
    runs,
    sessions,
    settings,
    share,
    titles,
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

# What this instance was built from, baked in by the image build: a release
# tag (v1.2.3) for a released image, a commit sha for a rolling :main one,
# empty when running from a source checkout. The version alone cannot say
# which, since main carries the released version until the next bump.
BUILD: str = os.environ.get("MEDMCP_BUILD", "")

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

# Replay runs in flight (see ``runs.py``). A run is a task in this process with a
# record on disk; sockets attach to it and detach from it without affecting it.
_runs: runs.RunManager = runs.RunManager()


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
    # A run record still marked "running" was cut off by the previous process.
    interrupted = await asyncio.to_thread(runs.reconcile_interrupted)
    if interrupted:
        _audit.warning("marked %d interrupted replay run(s) as failed", interrupted)
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
        await _runs.shutdown()
        if _broker is not None:
            await _broker.aclose()
        if _pool is not None:
            await _pool.aclose()
        _broker = None
        _pool = None


class OriginGuard:
    """Refuse browser requests that did not originate from the workspace's page.

    A pure-ASGI middleware rather than ``@app.middleware("http")`` because the
    exposure that matters most is the WebSocket one, which the HTTP decorator
    never sees: ``/ws/chat`` hands out a live agent session and ``/ws/replay``
    runs a workflow with no permission flow, and the same-origin policy does not
    apply to either — a page on any site can open them, and the server is the
    only thing that can say no. The upgrade is refused *before* ``accept``, so
    the handshake fails and no session is created.

    See :mod:`medmcp.origincheck` for what counts as acceptable and why the
    ``Host`` header is checked alongside ``Origin`` (DNS rebinding).
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap *app*, vetting every HTTP request and WebSocket upgrade."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, or refuse it without reaching the route."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        reason = origincheck.reject_reason(headers.get("host", ""), headers.get("origin", ""))
        if reason is None:
            await self.app(scope, receive, send)
            return
        _audit.warning("refused %s %s: %s", scope["type"], scope.get("path", ""), reason)
        if scope["type"] == "websocket":
            # The connect message has to be consumed before the close is sent;
            # closing before accept makes the handshake itself fail.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        response = JSONResponse({"detail": reason}, status_code=403)
        await response(scope, receive, send)


app = FastAPI(title="MedMCP Workspace", lifespan=_lifespan)
# Outermost, so a refused request never reaches a route or a static file.
app.add_middleware(OriginGuard)

# One vibe-acp subprocess shared by every websocket connection. The subprocess
# cwd must stay PROJECT_ROOT — `uv run` resolves the project from it; the
# agent's working directory is set per session via session/new's cwd instead.
# The agent process is where an external server's credential is needed; this
# process holds it so the operator does not have to plumb one into the
# deployment by hand. Passed as a provider, re-read on every (re)start.
_client: VibeAcpClient = VibeAcpClient(extra_env=settings.external_secret_env)

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
    # Set once the operator has been told the stack can reach off this machine.
    accept_network: bool = False


class StackUninstallPayload(BaseModel):
    """Request body for uninstalling a container stack by name."""

    name: str


def _apply_stack_change() -> None:
    """Re-discover stacks and re-sync vibe-acp config after an install/uninstall."""
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())


@app.get("/healthz")
async def healthz() -> JsonDict:
    """Liveness probe for container healthchecks (touches no dependencies).

    Carries the version and build identifier so an installed instance can say
    what it is — the first question any bug report has to answer.
    """
    return {"status": "ok", "version": __version__, "build": BUILD}


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
        name = await asyncio.to_thread(
            partial(settings.install_stack_image, accept_network=payload.accept_network),
            payload.image,
        )
    except settings.NetworkConsentRequiredError as exc:
        # 409, not 400: the request is well-formed and becomes valid once the
        # operator has seen what they are agreeing to.
        raise HTTPException(
            status_code=409,
            detail={"needs_network_consent": True, "name": exc.name, "image": exc.image},
        ) from exc
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

    First client message: ``{"image": "...", "accept_network": bool}``. Streams
    ``{"type":"progress","line":...}`` frames during the pull/extract, then a
    final ``{"type":"done","name":...}`` or ``{"type":"error","message":...}``.
    On success it reloads discovery and restarts vibe-acp (same as the POST path).

    An image that declares network egress ends the run with
    ``{"type":"needs_network_consent","name":...}`` instead of installing, so
    the client can put that to the operator and re-issue with consent. The
    image has been pulled and inspected by then, never executed.
    """
    await ws.accept()
    try:
        first = cast("JsonDict", await ws.receive_json())
    except WebSocketDisconnect:
        return
    image = str(first.get("image", "")).strip()
    accept_network = bool(first.get("accept_network"))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[JsonDict] = asyncio.Queue()

    def on_progress(line: str) -> None:
        # Called from the install worker thread; hop back onto the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "line": line})

    async def run() -> None:
        try:
            name = await asyncio.to_thread(
                partial(settings.install_stack_image, accept_network=accept_network),
                image,
                on_progress,
            )
            await asyncio.to_thread(_apply_stack_change)
            _audit.info("stack installed: %s (%s)", name, image)
            await _restart_vibe()
            await queue.put({"type": "done", "name": name})
        except settings.NetworkConsentRequiredError as exc:
            await queue.put({"type": "needs_network_consent", "name": exc.name, "image": exc.image})
        except Exception as exc:  # relayed to the client as an error frame
            await queue.put({"type": "error", "message": str(exc)})

    task = asyncio.create_task(run())
    try:
        while True:
            frame = await queue.get()
            await ws.send_json(frame)
            if frame["type"] in ("done", "error", "needs_network_consent"):
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
# UI: save the current chat as a workflow, rename/share it, and replay its
# recipe deterministically (no LLM) on new inputs.


def _workflow_dir(name: str) -> Path | None:
    """Return the on-disk dir for workflow *name*, or ``None``."""
    return workflow.workflow_dir(VIBE_HOME / "workflows", name)


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


# ── External MCP servers (advanced) ──────────────────────────────────────────
# Every mutation here changes what the agent can reach, so all of them are
# audit-logged and all of them re-sync the vibe config and restart the agent —
# the same treatment a stack change gets, for the same reason.


class ExternalMcpEnabledPayload(BaseModel):
    """Request body for turning external MCP support on or off."""

    enabled: bool


class ExternalServerPayload(BaseModel):
    """Request body for registering an external MCP server."""

    name: str
    transport: str
    url: str
    # The token itself, stored by this process and handed to the agent. Mutually
    # exclusive with api_key_env, which names a variable the deployment sets.
    token: str = ""
    api_key_env: str = ""
    # Optional overrides for services that don't take a bearer token; empty means
    # vibe's `Authorization: Bearer {token}` default.
    api_key_header: str = ""
    api_key_format: str = ""


class ExternalServerPatchPayload(BaseModel):
    """Request body for changing one external server: its state, its token, or both."""

    active: bool | None = None
    token: str | None = None


async def _apply_external_change() -> None:
    """Re-discover servers, re-sync the vibe config, and restart the agent."""
    await asyncio.to_thread(_apply_stack_change)
    await _restart_vibe()


@app.get("/api/external-mcp")
async def get_external_mcp() -> JsonDict:
    """Return the external-MCP state: the toggle, the acknowledgement, the servers."""

    def _state() -> JsonDict:
        state = settings.load_external_mcp()
        # Where each credential comes from, and whether it is actually there —
        # never what it is. A stored token is held by this process and injected
        # into the agent, so it is present by definition; a named variable has to
        # exist in this environment, and when it does not the request goes out
        # unauthenticated with the remote service's 401 as the only symptom.
        secrets = settings.load_external_secrets()
        servers: list[JsonDict] = []
        for srv in cast("list[JsonDict]", state["servers"]):
            managed = bool(secrets.get(str(srv.get("name", ""))))
            servers.append(
                {
                    **srv,
                    "token_managed": managed,
                    "token_present": managed
                    or settings.external_token_present(str(srv.get("api_key_env", ""))),
                }
            )
        return {
            "enabled": bool(state["enabled"]),
            "acknowledged": bool(state["acknowledged_at"]),
            "acknowledged_at": state["acknowledged_at"],
            "transports": list(settings.EXTERNAL_MCP_TRANSPORTS),
            "servers": servers,
        }

    return await asyncio.to_thread(_state)


@app.post("/api/external-mcp/acknowledge")
async def post_external_mcp_acknowledge() -> JsonDict:
    """Record that the operator accepted responsibility for external services."""
    state = await asyncio.to_thread(settings.acknowledge_external_mcp)
    _audit.info("external MCP acknowledged at %s", state["acknowledged_at"])
    return {"ok": True, "acknowledged_at": state["acknowledged_at"]}


@app.put("/api/external-mcp")
async def put_external_mcp(payload: ExternalMcpEnabledPayload) -> JsonDict:
    """Enable or disable external MCP support (enabling requires the acknowledgement)."""
    try:
        await asyncio.to_thread(settings.set_external_mcp_enabled, payload.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit.info("external MCP %s", "enabled" if payload.enabled else "disabled")
    await _apply_external_change()
    return {"ok": True, "enabled": payload.enabled, "restarted": True}


@app.post("/api/external-mcp/servers")
async def post_external_server(payload: ExternalServerPayload) -> JsonDict:
    """Register an external MCP server."""
    try:
        entry = await asyncio.to_thread(
            settings.add_external_server,
            payload.name,
            payload.transport,
            payload.url,
            payload.api_key_env,
            True,
            payload.api_key_header,
            payload.api_key_format,
            payload.token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit.info("external MCP server added: %s -> %s", entry["name"], entry["url"])
    await _apply_external_change()
    return {"ok": True, "server": entry, "restarted": True}


@app.patch("/api/external-mcp/servers/{name}")
async def patch_external_server(name: str, payload: ExternalServerPatchPayload) -> JsonDict:
    """Activate/deactivate one external server, replace its token, or both."""
    if payload.active is None and payload.token is None:
        raise HTTPException(status_code=400, detail="nothing to change")
    try:
        if payload.token is not None:
            await asyncio.to_thread(settings.replace_external_token, name, payload.token)
            # The value is deliberately absent from this line: the audit trail
            # records that a credential changed, never what it changed to.
            _audit.info("external MCP server token replaced: %s", name)
        if payload.active is not None:
            await asyncio.to_thread(settings.set_external_server_active, name, payload.active)
            _audit.info(
                "external MCP server %s: %s",
                "activated" if payload.active else "deactivated",
                name,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _apply_external_change()
    return {"ok": True, "restarted": True}


@app.delete("/api/external-mcp/servers/{name}")
async def delete_external_server(name: str) -> JsonDict:
    """Remove an external MCP server."""
    try:
        await asyncio.to_thread(settings.remove_external_server, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit.info("external MCP server removed: %s", name)
    await _apply_external_change()
    return {"ok": True, "restarted": True}


@app.get("/api/workflows")
async def get_workflows() -> JsonDict:
    """List the personal workflows available to the replay engine."""
    return {"workflows": await asyncio.to_thread(settings.discover_workflows)}


class DistillPayload(BaseModel):
    """Request body for distilling a chat session into a draft workflow."""

    session_id: str


@app.post("/api/workflows/distill")
async def post_distill(payload: DistillPayload) -> JsonDict:
    """Distill a chat session into a workflow, named after the chat, and return its detail."""
    sid = payload.session_id
    try:
        target = await asyncio.to_thread(
            lambda: distill.distill_session(
                sid,
                name_hint=sessions.chat_title(sid) or "",
                chain_stop_ids=_chain_stop_ids(sid),
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit.info("workflow distilled: %s", target.name)
    return await asyncio.to_thread(_workflow_detail, target.name)


@app.get("/api/workflows/{name}")
async def get_workflow(name: str) -> JsonDict:
    """Return one workflow's recipe detail (inputs, steps, replayability)."""
    try:
        return await asyncio.to_thread(_workflow_detail, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class WorkflowRenamePayload(BaseModel):
    """Request body for renaming a workflow."""

    new_name: str


@app.post("/api/workflows/{name}/rename")
async def post_rename_workflow(name: str, payload: WorkflowRenamePayload) -> JsonDict:
    """Rename a workflow; returns the new (slugified) name. 409 if that name is taken."""
    try:
        new_dir = await asyncio.to_thread(distill.rename_workflow, name, payload.new_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "name": new_dir.name}


@app.delete("/api/workflows/{name}")
async def delete_workflow(name: str) -> JsonDict:
    """Delete a personal workflow. The UI confirms first."""
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
    """Import a shared workflow envelope as a new workflow; return its detail."""
    try:
        target = await asyncio.to_thread(share.import_workflow, payload.content)
    except share.WorkflowShareError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit.info("workflow imported: %s", target.name)
    return await asyncio.to_thread(_workflow_detail, target.name)


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


def _confined_input_path(value: str) -> str:
    """Resolve a replay input, refusing anything that leaves the workspace.

    Replay calls stack tools directly, with no permission prompt between the
    binding and the call, so an input naming a path outside the workspace is
    either a mistake or an attempt to have a tool read something it should not.
    The container deployment already limits the damage — only the workspace is
    mounted into a stack — but a host-native stack runs as the operator and can
    read anything they can, so the confinement belongs here rather than in the
    deployment.

    Symlinks resolve before the check, so a link inside the workspace pointing
    out of it is refused too. Values that are not paths at all (``"cuda"``,
    ``"fast"``) pass through untouched, as does a path that does not exist yet —
    an output directory, or a genuinely missing scan that should surface as the
    tool's own error.

    Raises:
        ValueError: the value resolves outside :data:`WORKSPACE_ROOT`.
    """
    if not value:
        return value
    if Path(value).is_absolute():
        resolved = Path(value).resolve()
        if not resolved.is_relative_to(WORKSPACE_ROOT):
            raise ValueError(f"input path is outside the workspace: {value!r}")
        return str(resolved)
    candidate = (WORKSPACE_ROOT / value).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"input path escapes the workspace: {value!r}")
    return str(candidate) if candidate.exists() else value


def _confined_input_paths(inputs: dict[str, str]) -> dict[str, str]:
    """Apply :func:`_confined_input_path` to every value of a replay binding."""
    return {key: _confined_input_path(value) for key, value in inputs.items()}


class ReplayPreviewPayload(BaseModel):
    """Request body for previewing a replay's resolved steps.

    ``runs`` carries every item's input binding (the batch); ``inputs`` is the
    older single-item form and is treated as a one-item batch.
    """

    inputs: dict[str, str] | None = None
    runs: list[dict[str, str]] | None = None


def _input_argument_names(recipe: workflow.Recipe) -> dict[str, str]:
    """Map each ``in_N`` to the argument name it is first used as.

    The path checker classifies a path by its *parameter name* — ``input_path``
    must exist, ``output_dir`` need not — and the input's own name (``in_1``)
    says nothing. The first step argument that references the placeholder gives
    the role the tool assigns it.
    """
    names: dict[str, str] = {}
    for step in recipe.steps:
        for key, value in step.arguments.items():
            if not isinstance(value, str):
                continue
            for ref in replay.unresolved_refs(value):
                inner = ref[4:-1].strip() if ref.startswith("dir(") and ref.endswith(")") else ref
                names.setdefault(inner, key)
    return names


def _preflight_item(
    recipe: workflow.Recipe,
    inputs: dict[str, str],
    servers: list[JsonDict],
    arg_names: dict[str, str],
) -> JsonDict:
    """Everything the run screen wants to say about one item before it runs.

    ``error`` is the engine's own validation (a missing input, an uninstalled
    stack). ``findings`` are the path checks on the *bound inputs* — the files
    the person just pointed at — reported with the role the tool gives them, so
    a missing scan is an error and an output folder that already holds results
    is a warning about overwriting, not a false alarm.
    """
    error = replay.validate(recipe, inputs, servers)
    if error is not None:
        return {"ok": False, "error": error, "findings": []}
    bound = replay.apply_input_defaults(recipe, inputs)
    checked = {arg_names.get(name, name): value for name, value in bound.items()}
    findings = pathcheck.check_tool_call_paths(checked, WORKSPACE_ROOT, containerized=True)
    blocking = any(f["severity"] == "error" for f in findings)
    return {
        "ok": not blocking,
        "error": next((f["note"] for f in findings if f["severity"] == "error"), None),
        "findings": findings,
    }


@app.post("/api/workflows/{name}/replay-preview")
async def post_replay_preview(name: str, payload: ReplayPreviewPayload) -> JsonDict:
    """Validate a replay and return its resolved steps for user confirmation.

    Inputs are bound now; cross-step refs (``{{stepM.*}}``) resolve at runtime,
    so they intentionally still show as placeholders in the preview. Every item
    is pre-flighted (see :func:`_preflight_item`) so a typo in one of forty rows
    is caught here rather than by the tool, after the stacks were spawned.
    """

    def _preview() -> JsonDict:
        d = _workflow_dir(name)
        if d is None:
            raise FileNotFoundError(f"no workflow named {name!r}")
        recipe = distill.load_recipe(d)
        raw_runs = payload.runs if payload.runs is not None else [payload.inputs or {}]
        bindings_list = [_confined_input_paths(dict(r)) for r in raw_runs]
        servers = settings.active_servers()
        arg_names = _input_argument_names(recipe)
        items = [
            {"index": i, **_preflight_item(recipe, b, servers, arg_names)}
            for i, b in enumerate(bindings_list)
        ]
        first_ok = next((i for i, item in enumerate(items) if item["error"] is None), None)
        if first_ok is None and items:
            return {"ok": False, "error": items[0]["error"], "steps": [], "items": items}
        # Same defaults the run will use, so the preview shows what will actually
        # be sent rather than an unresolved placeholder the user never filled in.
        first = bindings_list[first_ok] if first_ok is not None else {}
        bindings: dict[str, Any] = replay.apply_input_defaults(recipe, first)
        steps = [
            {
                "index": i,
                "server": s.server,
                "tool": s.tool,
                "arguments": replay.resolve_arguments(s.arguments, bindings),
            }
            for i, s in enumerate(recipe.steps, start=1)
        ]
        ready = sum(1 for item in items if item["ok"])
        return {
            "ok": ready > 0,
            "error": None if ready > 0 else "no item can run",
            "steps": steps,
            "items": items,
        }

    try:
        return await asyncio.to_thread(_preview)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # an input pointing outside the workspace
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
            rows = batchplan.read_manifest(_confined_input_path(payload.plan_csv))
        except OSError as exc:
            return {"ok": False, "error": f"cannot read plan: {exc}", "runs": [], "skipped": []}
        try:
            binding = batchplan.runs_from_manifest(recipe, rows)
        except batchplan.BatchPlanError as exc:
            return {"ok": False, "error": str(exc), "runs": [], "skipped": []}
        return {
            "ok": True,
            "error": None,
            "runs": [_confined_input_paths(r) for r in binding.runs],
            "skipped": binding.skipped,
            "column_map": binding.column_map,
        }

    try:
        return await asyncio.to_thread(_build)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # a plan or binding pointing outside the workspace
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ReplayStartRequest(BaseModel):
    """The first client message on ``/ws/replay``: start a run, or attach to one."""

    name: str = ""
    runs: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    attach: str = ""


@app.websocket("/ws/replay")
async def ws_replay(ws: WebSocket) -> None:
    """Start a replay run — or attach to one — and stream its progress.

    First client message, one of:

    - ``{"name": str, "runs": [{in_N: value}, ...]}`` starts a new run;

      each entry is one full input binding and the recipe runs once per
      entry, sequentially. A failed item does not stop the remaining items.
    - ``{"attach": run_id}`` replays everything the run has done so far and then
      streams the rest (a finished run streams its record and closes).

    Frames: ``started`` (carries the ``run_id``), ``step_started``, ``step``,
    ``item_result`` and a final ``result``. **Closing the socket does not stop
    the run** — a reload or a dropped connection merely detaches, and the page
    reattaches by id. Stopping is an explicit ``{"type": "cancel"}`` message,
    which kills the in-flight tool call together with its stack.

    SECURITY: the replay engine calls MCP tools directly, bypassing the
    vibe-acp permission flow — the client must show the resolved-steps preview
    (``/replay-preview``) and get an explicit confirmation before starting.
    Inputs are confined to the workspace here regardless (see
    :func:`_confined_input_path`), since a binding arriving on this socket has
    had no permission prompt between it and the tool call.
    """
    await ws.accept()
    run_id = ""
    queue: asyncio.Queue[runs.Frame] | None = None
    try:
        try:
            first = ReplayStartRequest.model_validate(await ws.receive_json())
        except ValueError as exc:
            await ws.send_json({"type": "result", "ok": False, "error": f"bad request: {exc}"})
            return

        if first.attach:
            attached = _runs.attach(first.attach)
            if attached is None:
                await ws.send_json(
                    {"type": "result", "ok": False, "error": f"no run {first.attach!r}"}
                )
                return
            run_id = first.attach
            past, queue = attached
        else:
            try:
                bindings = [_confined_input_paths(dict(r)) for r in first.runs]
            except ValueError as exc:
                _audit.warning("replay refused: %s", exc)
                await ws.send_json({"type": "result", "ok": False, "error": str(exc)})
                return
            d = _workflow_dir(first.name)
            if d is None or not bindings:
                error = f"no workflow named {first.name!r}" if d is None else "no inputs to run"
                await ws.send_json({"type": "result", "ok": False, "error": error})
                return
            recipe = await asyncio.to_thread(distill.load_recipe, d)
            servers = await asyncio.to_thread(settings.active_servers)
            record = _runs.start(
                recipe=recipe,
                runs=bindings,
                servers=servers,
                cwd=str(WORKSPACE_ROOT),
                on_finished=lambda r: _audit.info(
                    "replay finished: %s run=%s status=%s", r.workflow, r.id, r.status
                ),
            )
            _audit.info(
                "replay started: %s run=%s (%d item(s) x %d steps)",
                recipe.name,
                record.id,
                len(bindings),
                len(recipe.steps),
            )
            run_id = record.id
            attached = _runs.attach(run_id)
            past, queue = attached if attached is not None else ([], None)

        for frame in past:
            await ws.send_json(frame)
        if queue is None:
            return  # a finished run: its record was the whole story

        # Two concurrent loops: frames out, control messages in. A disconnect
        # ends both and leaves the run alone; only an explicit cancel stops it.
        async def _pump() -> None:
            assert queue is not None
            while True:
                frame = await queue.get()
                if runs.is_end(frame):
                    return
                await ws.send_json(frame)

        async def _control() -> None:
            while True:
                msg = cast("JsonDict", await ws.receive_json())
                if msg.get("type") == "cancel":
                    _audit.info("replay cancel requested: run=%s", run_id)
                    await _runs.cancel(run_id)

        pump_task = asyncio.create_task(_pump())
        control_task = asyncio.create_task(_control())
        try:
            done, _pending = await asyncio.wait(
                {pump_task, control_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in (pump_task, control_task):
                if task not in done:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                        await task
            for task in done:
                task.result()  # surface a send failure / disconnect
        except asyncio.CancelledError:
            pump_task.cancel()
            control_task.cancel()
            raise
    except WebSocketDisconnect:
        _audit.info("replay socket closed by client; run=%s keeps running", run_id or "-")
    finally:
        if queue is not None and run_id:
            _runs.detach(run_id, queue)
        with contextlib.suppress(Exception):
            await ws.close()


@app.get("/api/runs")
async def get_runs(workflow: str | None = None, limit: int = 20) -> JsonDict:
    """Recent replay runs (newest first), optionally for one workflow.

    ``live`` lists the ids in flight in this process, so a page can reattach to
    a run it started before a reload.
    """
    records = await asyncio.to_thread(runs.list_runs, workflow=workflow, limit=max(1, limit))
    return {"runs": [r.summary() for r in records], "live": _runs.live_ids()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JsonDict:
    """One run's full record: every item, every step, what each produced."""
    record = await asyncio.to_thread(runs.load_run, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    return {**record.to_dict(), "live": _runs.is_live(run_id)}


@app.post("/api/runs/{run_id}/cancel")
async def post_cancel_run(run_id: str) -> JsonDict:
    """Stop a live run (the REST twin of the socket's ``cancel`` message)."""
    if not await _runs.cancel(run_id):
        raise HTTPException(status_code=404, detail=f"no live run {run_id!r}")
    _audit.info("replay cancelled: run=%s", run_id)
    return {"ok": True}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> JsonDict:
    """Forget a finished run's record. A live run must be stopped first."""
    if _runs.is_live(run_id):
        raise HTTPException(status_code=409, detail="run is still in progress")
    if not await asyncio.to_thread(runs.delete_run, run_id):
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    return {"ok": True}


# ── WebSocket chat ─────────────────────────────────────────
#
# Wire protocol (JSON messages):
#
#   server → client
#     {"type": "ready", "sessionId": str, "model": str, "title": str | None}
#     {"type": "chunk", "text": str}
#     {"type": "title", "title": str}            # a generated chat title landed
#     {"type": "retrying", "category": str, "detail": str}  # vibe is retrying the model
#     {"type": "notice", "text": str}            # non-error note, e.g. the turn limit
#     {"type": "tool_call", "toolCallId": str, "title": str, "status": str,
#      "kind": str | None, "rawInput": object}
#     {"type": "tool_call_update", "toolCallId": str, "status": str | None,
#      "output": str | None, "rawInput": object | None}
#     {"type": "usage", "used": int}
#     {"type": "permission_request", "requestId": int, "toolCall": {...},
#      "options": [{"optionId": str, "name": str, "kind": str}],
#      "explanation": str | None, "explaining": bool,
#      "risks": [{"key": str, "label": str, "severity": str}],
#      "paths": [{"param": str, "value": str, "role": str, "status": str,
#                 "severity": str, "note": str}]}
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
    (vibe's ``size`` comes from its model registry — e.g. 200k for a model
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


def _tool_name(update: JsonDict) -> str:
    """Underlying tool name for a tool-call frame, or "" if not advertised.

    ACP's ``title`` is prose written for a human ("Reading todos"), so it cannot
    identify a tool. vibe puts the real name in the frame's ``_meta``, which is
    what the browser needs to special-case a tool's rendering. Both spellings are
    accepted because the meta block is passed through as received rather than
    normalised like the aliased fields around it.
    """
    raw = update.get("_meta") or update.get("meta")
    if not isinstance(raw, dict):
        return ""
    meta = cast("JsonDict", raw)
    return str(meta.get("tool_name") or meta.get("toolName") or "")


def _is_pathguard_denial(output: str) -> bool:
    """True if *output* is the path guard turning a call back, not a real failure.

    vibe renders a hook denial into the tool result as "Tool 'x' was denied by hook
    '<name>': …", so the hook's own name is the marker. Matched on that rather than
    on the reason text, which is written for the model and free to change.
    """
    return f"denied by hook '{settings.PATHGUARD_HOOK_NAME}'" in output


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
        # Generated chat titles: when one is due (per completed turn) and the
        # background refresh in flight, if any (see _schedule_title).
        self._title_cadence = titles.TitleCadence()
        self._title_tasks: set[asyncio.Task[None]] = set()
        # Set once a user-chosen title is seen: generation never runs again on
        # this connection (a person's name for a chat is final).
        self._title_locked = False
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
        for task in [*self._explain_tasks, *self._title_tasks]:
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
        turn_ok = False
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
                else:
                    turn_ok = True
                    notice = _turn_limit_notice(resp)
                    if notice is not None:
                        await self._send({"type": "notice", "text": notice})
                break
            tail = self._thoughts.flush()  # emit any real text held back at a tag boundary
            if tail:
                await self._send({"type": "chunk", "text": tail})
            await self._send({"type": "done"})
            if turn_ok:
                self._schedule_title()
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

    def _schedule_title(self) -> None:
        """Start a background title refresh when one is due; never blocks the turn.

        Called after each turn that ended normally. The cadence decides (first
        completed turn, then bounded periodic refreshes); everything that reads
        or writes state happens inside the task, off the frame path.
        """
        if not titles.enabled() or self._title_locked:
            return
        ticket = self._title_cadence.begin_if_due()
        if ticket is None:
            return
        task = asyncio.create_task(self._refresh_title(ticket))
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

    async def _refresh_title(self, ticket: titles.TitleTicket) -> None:
        """Generate a title from the transcript and push it if it changed.

        A name the user typed always wins — the refresh backs off for good once
        one exists. Two records are consulted: the UI session registry, and
        vibe's own session metadata, which every rename is written through to
        and which lives on a persisted volume in a container; if the registry
        was reset (container recreate) the name is re-seeded from vibe rather
        than overwritten. A refresh that yields nothing (model outage,
        transcript not flushed yet) hands its slot back so the next turn
        retries. The result is written through to vibe's session metadata as
        well (best-effort) so its resume picker shows the same name.
        """
        cid = self.canonical_id
        try:
            entry = await asyncio.to_thread(sessions.get_entry, cid)
            if sessions.has_manual_title(entry):
                self._title_locked = True
                return
            manual = await asyncio.to_thread(provenance.vibe_manual_session_title, self.session_id)
            if manual is not None:
                self._title_locked = True
                await asyncio.to_thread(sessions.set_title, cid, manual)
                await self._send({"type": "title", "title": manual})
                return
            previous = entry.get("title")
            messages = await asyncio.to_thread(
                lambda: titles.load_session_messages(cid, stop_ids=_chain_stop_ids(cid))
            )
            title = await titles.generate_title(
                messages, previous_title=previous if isinstance(previous, str) else None
            )
            if title is None:
                self._title_cadence.restore(ticket)
                return
            changed = await asyncio.to_thread(sessions.set_auto_title, cid, title)
            if not changed:
                return
            await self._send({"type": "title", "title": title})
            with contextlib.suppress(Exception):
                await _client.request(
                    "_session/set_title", {"sessionId": self.session_id, "title": title}
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._title_cadence.restore(ticket)
            _audit.warning("chat title refresh failed", exc_info=True)

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
                        "toolName": _tool_name(update),
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
                # A tool call is announced before the model has finished streaming
                # its arguments, so the opening `tool_call` frame can carry an empty
                # or partial rawInput; vibe re-sends the completed arguments on the
                # first update whose detail changed. Overwrite (not set-if-absent)
                # or the placeholder sticks — which leaves the approval box with no
                # arguments to show and the provenance event with no `arguments`.
                raw_input = update.get("rawInput")
                info = self._tool_calls.get(tc_id)
                if info is not None:
                    if isinstance(status, str):
                        info["status"] = status
                    if raw_input is not None:
                        info["rawInput"] = raw_input
                    if raw_output is not None:
                        info["rawOutput"] = raw_output
                    elif output:
                        info["outputText"] = output
                # A call the path guard turned back never ran, and the model
                # corrects and retries on its own, so rendering it as a failure
                # would be both untrue and alarming. Flagged here rather than left
                # to the browser to pattern-match vibe's denial wording.
                await self._send(
                    {
                        "type": "tool_call_update",
                        "toolCallId": tc_id,
                        "status": status,
                        "output": output[:2000] if output else None,
                        "rawInput": raw_input,
                        "pathGuardRetry": _is_pathguard_denial(output),
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
        elif method == "_session/retrying":
            # vibe ≥2.24 announces a backend retry (rate limit, connection,
            # timeout) instead of going quiet; without this the chat looks
            # frozen for exactly as long as the retry backoff runs.
            await self._send(_retrying_frame(msg))

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

        # Existence check on the call's path arguments. Deterministic and local, so
        # unlike the explanation above it ships *with* the dialog rather than being
        # pushed in later, and it cannot itself be wrong about what is on disk.
        # Advisory only — it annotates the decision, it never makes one.
        #
        # Off the event loop for the same reason /api/tree is: these are stat calls,
        # and on a network-mounted workspace a single slow one would stall every
        # chat socket, not just this turn. Best-effort — a failure here must leave
        # the approval box exactly as it would have been without the check.
        try:
            containerized = pathguard.is_containerized_tool(str(tool_call.get("title") or ""))
            path_findings = await asyncio.to_thread(
                partial(
                    pathcheck.check_tool_call_paths,
                    tool_call.get("rawInput"),
                    WORKSPACE_ROOT,
                    containerized=containerized,
                )
            )
        except Exception:
            log.warning("path check failed for %s", title, exc_info=True)
            path_findings = []
        if any(f["severity"] == "error" for f in path_findings):
            _audit.info(
                "permission request has unresolvable paths: %s — %s",
                title,
                ", ".join(f"{f['param']}={f['value']!r} ({f['status']})" for f in path_findings),
            )

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
                "paths": path_findings,
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


def _retrying_frame(msg: JsonDict) -> JsonDict:
    """Browser frame for vibe's ``_session/retrying`` notification.

    vibe ≥2.24 sends ``{sessionId, category, detail}`` while it backs off and
    retries the model backend; ``category`` is one of rate_limited,
    server_error, timed_out, connection, unknown.
    """
    params = msg.get("params")
    params = cast("JsonDict", params) if isinstance(params, dict) else {}
    return {
        "type": "retrying",
        "category": str(params.get("category") or "unknown"),
        "detail": str(params.get("detail") or ""),
    }


def _turn_limit_notice(resp: JsonDict) -> str | None:
    """A user-facing note when a turn ended on vibe's step limit, else ``None``.

    Since vibe 2.24 hitting the per-turn limit is a normal ``session/prompt``
    response with ``stopReason: max_turn_requests`` rather than an error, so a
    turn that simply stopped mid-task would otherwise look finished.
    """
    result = resp.get("result")
    result = cast("JsonDict", result) if isinstance(result, dict) else {}
    if result.get("stopReason") == "max_turn_requests":
        return (
            "The agent stopped at the per-turn step limit before finishing. "
            "Send a follow-up (for example “continue”) to pick up where it left off."
        )
    return None


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
        # The chat's current title (user-set or generated), so a resumed chat
        # shows its name before the next refresh lands.
        current_title = await asyncio.to_thread(
            lambda: sessions.get_entry(conn.canonical_id).get("title")
        )
        await ws.send_json(
            {
                "type": "ready",
                "sessionId": conn.canonical_id,
                "model": settings.OLLAMA_MODEL,
                "title": current_title if isinstance(current_title, str) else None,
            }
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
