"""Shared MedMCP configuration: stack discovery, user toggles, vibe config sync.

Everything here is UI-agnostic (no FastAPI dependency) so the workspace
server (``server.py``) and the ``medmcp`` CLI can both drive the
same state:

- **Stack discovery** (:func:`load_mcp_servers`) — uv tool environments with a
  ``[medmcp.stacks]`` entry point, plus manual ``[[mcp_servers]]`` entries in
  ``.vibe/config.toml``.
- **Active sets** — which stacks/workflows are enabled, persisted as JSON under
  ``.vibe/`` (all active when the file is absent).
- **Feature toggles** — provenance capture, personal workflows, tool-call
  explanations; each defaults to on.
- **Config sync** (:func:`sync_servers_to_vibe_config`) — writes the resolved
  server list and workflow skill paths into ``.vibe/config.toml`` before each
  session, because vibe-acp reads that file directly.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
import tomli_w

from medmcp.acp import PROJECT_ROOT, VIBE_HOME, JsonDict

log: logging.Logger = logging.getLogger(__name__)

# Ollama endpoint + model, used for the agent's auxiliary LLM calls
# (tool-call explanations, distillation prose) and the provenance manifest.
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4-medmcp")

# Context window size used when Ollama hasn't been queried (yet) — the value
# set in Modelfile.gemma4.
DEFAULT_CONTEXT_WINDOW: int = 131_072

# Cached context window size fetched from Ollama /api/show. None = not yet
# fetched; populated lazily by fetch_context_window().
_context_window_tokens: int | None = None


async def fetch_context_window() -> int:
    """Return the active model's num_ctx by querying Ollama ``/api/show``.

    The result is cached for the process lifetime. Falls back to
    :data:`DEFAULT_CONTEXT_WINDOW` if Ollama is unreachable or the parameter
    is absent from the response.
    """
    global _context_window_tokens
    if _context_window_tokens is not None:
        return _context_window_tokens
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"model": OLLAMA_MODEL},
            )
            resp.raise_for_status()
            data = cast("JsonDict", resp.json())
            params_str = str(data.get("parameters") or "")
            for line in params_str.splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "num_ctx":
                    _context_window_tokens = int(parts[1])
                    return _context_window_tokens
    except Exception:
        log.warning("could not fetch context window size from Ollama; using fallback")
    _context_window_tokens = DEFAULT_CONTEXT_WINDOW
    return _context_window_tokens


def cached_context_window() -> int:
    """Return the cached window size without I/O (fallback when unfetched)."""
    return (
        _context_window_tokens if _context_window_tokens is not None else (DEFAULT_CONTEXT_WINDOW)
    )


# Tracks which discovered stacks are enabled; defaults to all when absent.
ACTIVE_STACKS_PATH: Path = VIBE_HOME / "active_stacks.json"
# Tracks which personal workflows are enabled (loaded as skills); all when absent.
ACTIVE_WORKFLOWS_PATH: Path = VIBE_HOME / "active_workflows.json"
# Persists the provenance-capture on/off preference; defaults to on when absent.
PROVENANCE_ENABLED_PATH: Path = VIBE_HOME / "provenance_enabled.json"
# Master on/off for the personal-workflows feature; defaults to on when absent.
WORKFLOWS_ENABLED_PATH: Path = VIBE_HOME / "workflows_enabled.json"
# Persists the explain-tool-calls on/off preference; defaults to on when absent.
EXPLAIN_ENABLED_PATH: Path = VIBE_HOME / "explain_enabled.json"

# Directory of container-stack manifests (``stacks.d/<name>.toml``). Each manifest
# declares a stack launched as a container (``command = "docker"``, ``args = [...]``)
# rather than a uv-tool-installed binary — the deployment path where stacks ship as
# images. ``${VAR}`` references in ``command``/``args``/``skills_path`` are expanded
# against the environment at load time (e.g. ``${MEDMCP_WORKSPACE}`` for path parity).
STACKS_D_PATH: Path = Path(PROJECT_ROOT) / "stacks.d"


def get_uv_tool_dir() -> Path | None:
    """Return the uv tool installation directory, or ``None`` if unavailable."""
    try:
        result = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True, timeout=5)
        return Path(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def call_entry_point(python: Path, module: str, attr: str) -> object:
    """Call ``module.attr()`` inside *python*'s environment and return the result.

    Raises ``RuntimeError`` on non-zero exit and ``ValueError`` on JSON-decode
    failure; both are caught by the caller so broken stacks are skipped cleanly.
    """
    result = subprocess.run(
        [str(python), "-c", f"import json,{module};print(json.dumps({module}.{attr}()))"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def _load_stack_manifests() -> list[JsonDict]:
    """Read container-stack manifests from :data:`STACKS_D_PATH`.

    Each ``stacks.d/<name>.toml`` declares a stack that is launched as a container
    (``command = "docker"``, ``args = [...]``) instead of a uv-tool binary — the
    deployment path where stacks ship as images. ``${VAR}`` references in
    ``command``, ``args`` and ``skills_path`` are expanded against the environment
    (notably ``${MEDMCP_WORKSPACE}`` so the bind-mount lands at the host path the
    workspace server already uses). Malformed manifests are skipped with a warning
    so one bad file never breaks discovery.

    Returns a list of server-config dicts in the same shape as the uv-tool scan.
    """
    if not STACKS_D_PATH.is_dir():
        return []
    manifests: list[JsonDict] = []
    for path in sorted(STACKS_D_PATH.glob("*.toml")):
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except Exception as exc:
            log.warning("Could not parse stack manifest %s; skipping: %s", path, exc)
            continue
        name = str(raw.get("name", "")).strip()
        command = str(raw.get("command", "")).strip()
        if not name or not command:
            log.warning("Stack manifest %s missing 'name'/'command'; skipping", path)
            continue
        args = [os.path.expandvars(str(a)) for a in cast("list[Any]", raw.get("args", []))]
        entry: JsonDict = {
            "name": name,
            "command": os.path.expandvars(command),
            "args": args,
            "env": dict(cast("dict[str, str]", raw.get("env", {}))),
        }
        if raw.get("skills_path"):
            entry["skills_path"] = os.path.expandvars(str(raw["skills_path"]))
        if raw.get("tool_timeout_sec") is not None:
            entry["tool_timeout_sec"] = raw["tool_timeout_sec"]
        manifests.append(entry)
    return manifests


@lru_cache(maxsize=1)
def load_mcp_servers() -> list[JsonDict]:
    """Discover MCP servers from uv tool environments and ``.vibe/config.toml``.

    Server configs are collected from two sources:

    1. **Installed uv tools** (authoritative) — any package installed via
       ``uv tool install`` that registers a ``[medmcp.stacks]`` entry point in
       its dist-info is auto-discovered.  The entry point must be a zero-argument
       callable returning a dict with at least ``name`` and ``command`` keys.
       The executable is resolved to its absolute path inside the isolated tool
       env, so PATH ordering never causes the wrong binary to be picked up.

    2. **Container-stack manifests** in ``stacks.d/*.toml`` (:func:`_load_stack_manifests`)
       — stacks shipped as images and launched via ``docker run -i`` (the
       deployment path); accepted only for names not already claimed by a uv tool.

    3. **Manual ``[[mcp_servers]]`` entries in ``.vibe/config.toml``** — only
       entries whose ``name`` is *not* already registered by a uv tool or a
       manifest are accepted.  This covers stacks that have not yet been installed
       via ``just install-stack``.

    Returns a list of server-config dicts ready for ``sync_servers_to_vibe_config``.
    """
    servers: dict[str, JsonDict] = {}

    # ── 1. Scan uv tool environments ─────────────────────────────────────────
    tool_dir = get_uv_tool_dir()
    if tool_dir is not None:
        cp = configparser.ConfigParser()
        cp.optionxform = lambda optionstr: optionstr  # preserve case
        for ep_file in tool_dir.glob("*/lib/python*/site-packages/*.dist-info/entry_points.txt"):
            cp.clear()
            cp.read(ep_file)
            if not cp.has_section("medmcp.stacks"):
                continue
            # tool env root: entry_points.txt → dist-info → site-packages →
            #                python3.x → lib → <tool-env>
            tool_env = ep_file.parents[4]
            python = tool_env / "bin" / "python"
            if not python.exists():
                continue
            for ep_name, ep_value in cp.items("medmcp.stacks"):
                parts = ep_value.strip().split(":")
                if len(parts) != 2:
                    continue
                module, attr = parts
                # Reject non-identifier strings before passing to a subprocess.
                if not re.fullmatch(r"[\w.]+", module) or not re.fullmatch(r"\w+", attr):
                    log.warning("Skipping malformed entry point %r = %r", ep_name, ep_value)
                    continue
                try:
                    raw = call_entry_point(python, module, attr)
                except Exception as exc:
                    log.warning("medmcp.stacks entry point %r failed: %s", ep_name, exc)
                    continue
                if not isinstance(raw, dict) or "name" not in raw:
                    log.warning(
                        "medmcp.stacks entry point %r returned an invalid config "
                        "(expected dict with 'name' key); skipping",
                        ep_name,
                    )
                    continue
                srv = cast("JsonDict", raw)
                name = str(srv.get("name", ""))
                if not name:
                    continue
                # Resolve command to the absolute executable inside the tool env
                # so vibe-acp always uses the fully-dep-installed binary.
                # Guard: an empty command must not be resolved — Path / "" collapses
                # to the parent dir, which exists, corrupting command to a dir path.
                command = str(srv.get("command", ""))
                if command:
                    candidate = tool_env / "bin" / command
                    if candidate.exists():
                        command = str(candidate)
                dist_info = ep_file.parent.name  # e.g. "medmcp_dicom-0.1.0.dist-info"
                version = dist_info.removesuffix(".dist-info").rsplit("-", 1)[-1]
                entry: JsonDict = {
                    "name": name,
                    "command": command,
                    "args": list(cast("list[Any]", srv.get("args", []))),
                    "env": dict(cast("dict[str, str]", srv.get("env", {}))),
                    "version": version,
                }
                for key in ("skills_path", "tool_timeout_sec", "startup_timeout_sec"):
                    if srv.get(key) is not None:
                        entry[key] = srv[key]
                servers[name] = entry

    # ── 2. Container-stack manifests (stacks.d/*.toml) ───────────────────────
    # Stacks shipped as images, launched via `docker run -i`. A uv-tool install
    # of the same name wins (local dev against an installed stack overrides the
    # container manifest); manifests fill in everything else.
    for entry in _load_stack_manifests():
        if entry["name"] not in servers:
            servers[entry["name"]] = entry

    # ── 3. Manual config.toml entries ────────────────────────────────────────
    # Only accepted for names NOT already claimed by a uv tool or a manifest.
    # This prevents a feedback loop where servers written to config.toml by
    # sync_servers_to_vibe_config shadow the live tool-env definitions.
    config_path = VIBE_HOME / "config.toml"
    if config_path.exists():
        try:
            with config_path.open("rb") as f:
                cfg = tomllib.load(f)
        except Exception as exc:
            log.warning("Could not parse %s; skipping manual entries: %s", config_path, exc)
            return list(servers.values())

        tool_names = set(servers)
        for srv in cfg.get("mcp_servers", []):
            name = cast("str", srv.get("name", ""))
            if not name or name in tool_names:
                continue
            command = cast("str", srv.get("command", ""))
            # Skip stale entries written by sync_servers_to_vibe_config for
            # tools that have since been uninstalled: absolute paths that no
            # longer exist on disk indicate a removed uv tool environment.
            if command and Path(command).is_absolute() and not Path(command).exists():
                log.debug(
                    "Skipping stale config.toml entry %r (command not found: %s)", name, command
                )
                continue
            servers[name] = {
                "name": name,
                "command": command,
                "args": cast("list[Any]", srv.get("args", [])),
                "env": {},
            }

    return list(servers.values())


def load_active_server_names() -> set[str]:
    """Return the set of server names currently marked active.

    When ``.vibe/active_stacks.json`` is absent (first run) every discovered
    server is considered active so behaviour is identical to the previous
    all-servers-always-on model.
    """
    all_names = {s["name"] for s in load_mcp_servers()}
    if not ACTIVE_STACKS_PATH.exists():
        return all_names
    try:
        data = cast("dict[str, Any]", json.loads(ACTIVE_STACKS_PATH.read_text()))
        return set(cast("list[str]", data.get("active", list(all_names))))
    except (json.JSONDecodeError, OSError):
        return all_names


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON state file atomically, safe across processes.

    A unique temp name (not a fixed ``.tmp`` path) so a concurrent writer in
    another process (a second server instance, the CLI) can never consume or
    interleave into this writer's temp file; ``os.replace`` makes the
    last-complete-writer win.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def save_active_server_names(names: set[str]) -> None:
    """Persist the active server set to ``.vibe/active_stacks.json``."""
    _atomic_write_json(ACTIVE_STACKS_PATH, {"active": sorted(names)})


def active_servers() -> list[JsonDict]:
    """Return only the active subset of all discovered servers."""
    active_names = load_active_server_names()
    return [s for s in load_mcp_servers() if s["name"] in active_names]


def read_skill_description(skill_md: Path) -> str:
    """Return the ``description:`` frontmatter value from a SKILL.md, or ``''``."""
    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines[:15]:
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def discover_workflows() -> list[JsonDict]:
    """Discover personal workflows from ``draft/`` and ``active/``.

    Returns ``{name, description, kind}`` dicts, deduplicated by name (an active
    workflow shadows a draft of the same name). ``kind`` is ``"active"`` for
    promoted workflows and ``"draft"`` for unpromoted ones.
    """
    found: dict[str, JsonDict] = {}
    for kind in ("active", "draft"):
        base = VIBE_HOME / "workflows" / kind
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            skill = d / "SKILL.md"
            if not d.is_dir() or not skill.is_file() or d.name in found:
                continue
            found[d.name] = {
                "name": d.name,
                "description": read_skill_description(skill),
                "kind": kind,
            }
    return list(found.values())


def load_active_workflow_names() -> set[str]:
    """Return the set of workflow names currently enabled (loaded as skills).

    When ``.vibe/active_workflows.json`` is absent every discovered workflow is
    active, so a freshly distilled/promoted workflow is on by default.
    """
    all_names = {w["name"] for w in discover_workflows()}
    if not ACTIVE_WORKFLOWS_PATH.exists():
        return all_names
    try:
        data = cast("dict[str, Any]", json.loads(ACTIVE_WORKFLOWS_PATH.read_text()))
        return set(cast("list[str]", data.get("active", list(all_names))))
    except (json.JSONDecodeError, OSError):
        return all_names


def save_active_workflow_names(names: set[str]) -> None:
    """Persist the active workflow set to ``.vibe/active_workflows.json``."""
    _atomic_write_json(ACTIVE_WORKFLOWS_PATH, {"active": sorted(names)})


def _load_flag(path: Path) -> bool:
    """Read an ``{"enabled": bool}`` preference file (default ``True`` when unset)."""
    if not path.exists():
        return True
    try:
        data = cast("dict[str, Any]", json.loads(path.read_text()))
        return bool(data.get("enabled", True))
    except (json.JSONDecodeError, OSError):
        return True


def _save_flag(path: Path, enabled: bool) -> None:
    """Persist an ``{"enabled": bool}`` preference file atomically."""
    _atomic_write_json(path, {"enabled": enabled})


def load_provenance_enabled() -> bool:
    """Return whether provenance capture is enabled (default ``True`` when unset)."""
    return _load_flag(PROVENANCE_ENABLED_PATH)


def save_provenance_enabled(enabled: bool) -> None:
    """Persist the provenance-capture on/off preference to disk."""
    _save_flag(PROVENANCE_ENABLED_PATH, enabled)


def load_workflows_enabled() -> bool:
    """Return whether the personal-workflows feature is enabled (default ``True``).

    The master switch: when off, no personal workflow is loaded as a skill and
    the UIs hide their workflow controls (see ``sync_servers_to_vibe_config``).
    """
    return _load_flag(WORKFLOWS_ENABLED_PATH)


def save_workflows_enabled(enabled: bool) -> None:
    """Persist the personal-workflows master on/off preference to disk."""
    _save_flag(WORKFLOWS_ENABLED_PATH, enabled)


def load_explain_enabled() -> bool:
    """Return whether tool-call explanations are enabled (default ``True``)."""
    return _load_flag(EXPLAIN_ENABLED_PATH)


def save_explain_enabled(enabled: bool) -> None:
    """Persist the explain-tool-calls on/off preference to disk."""
    _save_flag(EXPLAIN_ENABLED_PATH, enabled)


# Serializes the config.toml read-modify-write within this process: the
# workspace server runs the sync from a worker thread per websocket connect,
# and several connects race after a settings-triggered restart.
_config_write_lock = threading.Lock()


def sync_servers_to_vibe_config(servers: list[JsonDict]) -> None:
    """Overwrite the ``[[mcp_servers]]`` section of ``.vibe/config.toml``.

    This is the mechanism that makes entry-point discovery actually take effect:
    vibe-acp reads its server list directly from the TOML file and ignores the
    ``mcpServers`` field in JSON-RPC calls, so we must write the desired set
    here before each ``session/new`` or ``session/load``.

    Only the ``mcp_servers`` array is replaced.  All other keys (model config,
    tool permissions, etc.) are left untouched.  For servers already present in
    the file, vibe-acp-specific fields such as ``startup_timeout_sec`` are
    preserved; only ``command`` and ``args`` are updated from the discovery
    result.
    """
    with _config_write_lock:
        _sync_servers_to_vibe_config_locked(servers)


def _sync_servers_to_vibe_config_locked(servers: list[JsonDict]) -> None:
    """Body of :func:`sync_servers_to_vibe_config`; caller holds the lock."""
    config_path = VIBE_HOME / "config.toml"
    cfg: dict[str, Any] = {}
    existing_by_name: dict[str, JsonDict] = {}

    if config_path.exists():
        try:
            with config_path.open("rb") as fh:
                cfg = tomllib.load(fh)
        except Exception as exc:
            log.warning("Could not parse %s; skipping mcp_servers sync: %s", config_path, exc)
            return
        for raw in cast("list[JsonDict]", cfg.get("mcp_servers", [])):
            name = cast("str", raw.get("name", ""))
            if name:
                existing_by_name[name] = raw

    new_entries: list[JsonDict] = []
    for srv in servers:
        name = srv["name"]
        if name in existing_by_name:
            # Preserve vibe-acp-specific fields (timeouts, transport, etc.)
            # and overwrite discovery-owned fields (command, args).
            entry = dict(existing_by_name[name])
            entry["command"] = srv["command"]
            if srv.get("args"):
                entry["args"] = srv["args"]
            else:
                entry.pop("args", None)
            new_entries.append(entry)
        else:
            # Brand-new server from an entry point. Copy vibe-acp fields that
            # packages may supply (e.g. tool_timeout_sec for long-running tools),
            # then ensure required fields are set.
            passthrough = {
                "tool_timeout_sec",
                "startup_timeout_sec",
                "transport",
                "env",
                "skills_path",
            }
            entry = {"transport": "stdio"}
            for key in passthrough:
                if key in srv:
                    entry[key] = srv[key]
            entry["name"] = name
            entry["command"] = srv["command"]
            if srv.get("args"):
                entry["args"] = srv["args"]
            else:
                entry.pop("args", None)
            new_entries.append(entry)

    cfg["mcp_servers"] = new_entries

    # Collect skills_path values from discovered servers and write them to
    # skill_paths so vibe-acp loads the bundled skill docs automatically.
    skill_paths = [srv["skills_path"] for srv in servers if srv.get("skills_path")]
    # The personal-workflows feature can be turned off entirely by the master
    # toggle; when off, no workflow dir is added to skill_paths and every workflow
    # is listed in disabled_skills, so nothing personal is loaded.
    workflows_enabled = load_workflows_enabled()
    if workflows_enabled:
        # Promoted, reusable workflows live here as <name>/SKILL.md; include the
        # directory so distilled workflows are discoverable as skills (Tier 3).
        workflows_active = VIBE_HOME / "workflows" / "active"
        if workflows_active.is_dir():
            skill_paths.append(str(workflows_active))
        # Draft workflows are loaded too, so a draft can be tested (invoked as
        # `/<name>`) before it is promoted. Promotion just moves the draft into
        # active/ to keep it permanently; both dirs hold <name>/SKILL.md entries.
        workflows_draft = VIBE_HOME / "workflows" / "draft"
        if workflows_draft.is_dir():
            skill_paths.append(str(workflows_draft))
    cfg["skill_paths"] = skill_paths

    # Personal workflows toggled off in the gear are disabled by name so vibe-acp
    # skips loading them, without removing them from disk. With the master toggle
    # off, all of them are disabled. Non-workflow entries in disabled_skills (if
    # any were set manually) are preserved.
    all_workflows = {w["name"] for w in discover_workflows()}
    deactivated = (
        all_workflows if not workflows_enabled else (all_workflows - load_active_workflow_names())
    )
    existing_disabled = cast("list[str]", cfg.get("disabled_skills", []))
    preserved = [s for s in existing_disabled if s not in all_workflows]
    cfg["disabled_skills"] = sorted(set(preserved) | deactivated)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name (not a fixed .tmp path) so a concurrent writer in
    # another process can never interleave into the same file; os.replace
    # then makes last-complete-writer win.
    fd, tmp_name = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(cfg, fh)
        os.replace(tmp_name, config_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
